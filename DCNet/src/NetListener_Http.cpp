// HTTP/1.1 监听器，基于 POCO ServerSocket。
//
// 闸门顺序：accept、maxConnections 配额、读请求、过载 429、方法 405、鉴权 401、
// 路径 404、handler。两级闸门：连接级配额在 accept 期生效，含半开与慢速连接；
// 过载计数在请求完整读入后，TCP 就绪探测连接读即断，不入计。
// 生命周期不变量：worker 在 accept 期起线程前即以 _activeThreads 记账，
// stop 排水至归零才返回，分离线程不可能再访问已析构的监听器状态。
// v1 边界：仅 Content-Length 请求体；逐请求应答后关闭连接。

#include "DCNet/NetListener.h"

#include "NodeException.h"
#include "NetWire.h"
#include "NetListenerLifecycle.h"

#include <Poco/Exception.h>
#include <Poco/Net/ServerSocket.h>
#include <Poco/Net/SocketAddress.h>
#include <Poco/Net/StreamSocket.h>
#include <Poco/Timespan.h>

#include <cstdio>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <limits>
#include <cstdlib>
#include <charconv>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DC::Net {

namespace {

constexpr std::size_t kMaxHeaderBytes = 64 * 1024;		   // 请求头预算
constexpr std::size_t kMaxBodyBytes = 64ull * 1024 * 1024; // 请求体预算，超限 413
constexpr std::chrono::milliseconds kDefaultRequestTimeout{30000}; ///< 预算非法时的兜底
constexpr std::chrono::milliseconds kMinDrainGrace{5000};	// stop() 排水 grace 下限
constexpr std::chrono::milliseconds kStopPollInterval{10};	// stop() 排水轮询步长

Poco::Timespan toTimespan(const std::chrono::milliseconds& ms) {
	return Poco::Timespan(0, std::chrono::duration_cast<std::chrono::microseconds>(ms).count());
}

struct RawRequest {
	std::string method;
	std::string path;
	std::string body;
	std::size_t bodyLength = 0;
	std::unordered_map<std::string, std::string> headers; // 键小写化
};

/// 在途计数 RAII 递减。
struct InFlightGuard {
	std::atomic<std::size_t>& counter;
	~InFlightGuard() { counter.fetch_sub(1, std::memory_order_acq_rel); }
};

/// worker 退出时回收存活账，只递减。计数在 accept 期与入册同一临界区完成；
/// 若在 worker 内自增，已创建但未调度的窗口会使排水误判 0 而提前放行导致 use-after-free；
/// _inFlight 只覆盖完整读入后阶段，不能作线程存活性依据。
struct ThreadReleaseGuard {
	std::atomic<std::size_t>& counter;
	explicit ThreadReleaseGuard(std::atomic<std::size_t>& c) : counter(c) {}
	ThreadReleaseGuard(const ThreadReleaseGuard&) = delete;
	ThreadReleaseGuard& operator=(const ThreadReleaseGuard&) = delete;
	~ThreadReleaseGuard() { counter.fetch_sub(1, std::memory_order_acq_rel); }
};

/// "Bearer xxx" 或裸 key 统一去前缀比较。
std::string stripBearer(const std::string& value) {
	const std::string prefix = "Bearer ";
	if (value.rfind(prefix, 0) == 0)
		return value.substr(prefix.size());
	return value;
}

/// 回环判定：127.0.0.1 保持无认证可用；其余地址视为对外监听，强制要求 authToken。
bool isLoopbackHost(const std::string& host) {
	std::string lower;
	lower.reserve(host.size());
	for (char c : host)
		lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
	return lower == "127.0.0.1" || lower == "::1" || lower == "[::1]" || lower == "localhost";
}

thread_local const void* servingListener = nullptr;

class HttpListener final : public DcNetListener {
public:
	~HttpListener() override {
		if (servingListener == this) {
			std::fputs("DCNet fatal: listener destruction from its handler is forbidden\n", stderr);
			std::terminate();
		}
		stop();
	}

	void bind(const NetServerEndpoint& endpoint) override {
		std::lock_guard lk(_mutex);
		if (_bound)
			throw NodeException(NodeException::ErrorType::InternalError, "DcNetListener::bind",
								"listener already bound");

		// 配置期全量校验：非法范围在 bind 期 fail-fast，杜绝运行期静默异常，
		// 如负 maxInFlight 绕过 429、空白 token 形成认证绕过
		auto reject = [&](const std::string& why) {
			throw NodeException(NodeException::ErrorType::ExecutionFailed, "DcNetListener::bind", why);
		};
		if (endpoint.port < 0 || endpoint.port > 65535)
			reject("port " + std::to_string(endpoint.port) + " out of range [0,65535]");
		if (endpoint.backlog <= 0)
			reject("backlog must be > 0");
		if (endpoint.maxConnections < 0)
			reject("maxConnections must be >= 0 (0 = unlimited)");
		if (endpoint.maxInFlight <= 0)
			reject("maxInFlight must be > 0");
		if (endpoint.requestTimeout < std::chrono::milliseconds(0))
			reject("requestTimeout must be >= 0");
		if (!endpoint.authToken.empty()) {
			const std::string bare = stripBearer(endpoint.authToken);
			if (bare.empty() || bare.find_first_not_of(" \t") == std::string::npos)
				reject("authToken is blank after 'Bearer ' prefix stripping; configure a real token"
					   " or leave it empty to disable auth on loopback");
		}
		if (endpoint.maxRequestBody == 0 || endpoint.maxBufferedBodyBytes == 0)
			reject("body budgets must be positive");
		const Poco::Net::SocketAddress resolved(endpoint.listenHost, static_cast<Poco::UInt16>(endpoint.port));
		if (!resolved.host().isLoopback())
			reject("plaintext listener requires a resolved loopback address; use a trusted TLS reverse proxy");

		try {
			_socket.bind(resolved,
						 false);
			_socket.listen(endpoint.backlog);
		} catch (const Poco::Exception& e) {
			throw NodeException(NodeException::ErrorType::ExecutionFailed, "DcNetListener::bind",
								"bind " + endpoint.listenHost + ":" + std::to_string(endpoint.port) +
									" failed: " + e.displayText());
		}
		_endpoint = endpoint;
		_bound = true;
	}

	void start(RequestHandler handler) override {
		std::lock_guard lifecycle(_lifecycleMutex);
		std::lock_guard lk(_mutex);
		if (!_bound || _started.load())
			throw NodeException(NodeException::ErrorType::InternalError, "DcNetListener::start", "listener not bound or already started");
		_handler = std::move(handler);
		detail::launchListenerAccept(_started, _stopped, _handler, _acceptThread,
			[this] { return std::thread([this] { acceptLoop(); }); });
	}

	void stop() override {
		// Check thread identity before any lifecycle lock: external stop may be draining us.
		if (servingListener == this)
			throw NodeException(NodeException::ErrorType::InternalError, "DcNetListener::stop", "stop from listener handler is forbidden");
		std::lock_guard lifecycle(_lifecycleMutex);
		if (!_started.load()) return;
		_stopped.store(true);
		try { _socket.close(); } catch (...) {}
		if (_acceptThread.joinable()) _acceptThread.join();
		drainConnections();
	}

	bool alive() const override {
		return _started.load(std::memory_order_acquire) && !_stopped.load(std::memory_order_acquire);
	}

	int port() const override {
		try {
			return static_cast<int>(_socket.address().port());
		} catch (...) {
			return -1;
		}
	}

private:
	/// 连接在册 RAII 收尾：worker 退出即摘除；在册集合是配额依据与强关目标。
	struct ConnScope {
		HttpListener* self;
		std::shared_ptr<Poco::Net::StreamSocket> conn;
		~ConnScope() { self->unregisterConnection(conn); }
	};

	/// 排水：grace 内等工作线程自然退出；到期强制关闭在册连接放倒阻塞读写，
	/// 再等全部退出。本地引擎在 handler 内永久挂起时 stop 随之挂起，
	/// 宁挂起也不让分离线程访问已析构对象。
	void drainConnections() {
		const auto budget = _endpoint.requestTimeout > std::chrono::milliseconds(0) ? _endpoint.requestTimeout
																					: kDefaultRequestTimeout;
		const auto graceDeadline = std::chrono::steady_clock::now() +
									std::max(budget + std::chrono::seconds(1), kMinDrainGrace);
		while (_activeThreads.load(std::memory_order_acquire) > 0 &&
			   std::chrono::steady_clock::now() < graceDeadline)
			std::this_thread::sleep_for(kStopPollInterval);
		if (_activeThreads.load(std::memory_order_acquire) == 0)
			return;
		// grace 到期仍有在服连接：强制关闭后无条件等待
		std::vector<std::shared_ptr<Poco::Net::StreamSocket>> snapshot;
		{
			std::lock_guard lk(_connMutex);
			for (auto& conn : _conns) if (conn) forceClose(*conn);
		}
		for (auto& conn : snapshot)
			if (conn)
				forceClose(*conn);
		while (_activeThreads.load(std::memory_order_acquire) > 0)
			std::this_thread::sleep_for(kStopPollInterval);
	}

	static void forceClose(Poco::Net::StreamSocket& conn) {
		try {
			conn.shutdownReceive();
		} catch (...) {
		}
		try {
			conn.shutdownSend();
		} catch (...) {
		}
		try {
			conn.close();
		} catch (...) {
		}
	}

	/// accept 后同步入册：配额检查、存活记账与登记同一临界区，无超限窗口，
	/// 也无未调度误判归零窗口。返回 null 表示超出 maxConnections。
	std::shared_ptr<Poco::Net::StreamSocket> admitConnection(Poco::Net::StreamSocket conn) {
		auto holder = std::make_shared<Poco::Net::StreamSocket>(std::move(conn));
		{
			std::lock_guard lk(_connMutex);
			const int cap = _endpoint.maxConnections;
			if (cap > 0 && _conns.size() >= static_cast<std::size_t>(cap)) {
				forceClose(*holder); // 连接级过载：不排队、不起线程
				return nullptr;
			}
			detail::registerListenerConnection(_conns, _activeThreads, holder);
		}
		return holder;
	}

	void unregisterConnection(const std::shared_ptr<Poco::Net::StreamSocket>& holder) {
		std::lock_guard lk(_connMutex);
		for (auto it = _conns.begin(); it != _conns.end(); ++it) {
			if (it->get() == holder.get()) {
				_conns.erase(it);
				return;
			}
		}
	}

	/// 线程创建失败回滚：摘册、回收存活账并关连接，否则排水永远等待。
	void rejectAdmittedConnection(const std::shared_ptr<Poco::Net::StreamSocket>& holder) {
		forceClose(*holder);
		unregisterConnection(holder);
		_activeThreads.fetch_sub(1, std::memory_order_acq_rel);
	}

	void acceptLoop() {
		for (;;) {
			if (_stopped.load(std::memory_order_acquire))
				break;
			// 轮询而非阻塞 accept：POSIX close 不唤醒阻塞 accept
			try {
				if (!_socket.poll(Poco::Timespan(0, 50 * 1000), Poco::Net::Socket::SELECT_READ))
					continue;
				auto conn = _socket.acceptConnection();
				auto holder = admitConnection(std::move(conn));
				if (!holder)
					continue; // 连接级闸门已拦截
				auto tracked = holder; // worker 接走后仍可回滚
				std::thread worker;
				try {
					worker = std::thread([this, conn = std::move(holder)]() mutable {
						serveConnection(std::move(conn));
					});
				} catch (...) {
					rejectAdmittedConnection(tracked); // 回收 accept 期的账
					continue;
				}
				worker.detach(); // 分离：join 责任由排水承担
			} catch (const Poco::Exception&) {
				if (_stopped.load(std::memory_order_acquire))
					break;
				// 单连接异常不终止监听，不得崩溃
			} catch (...) {
				if (_stopped.load(std::memory_order_acquire))
					break;
			}
		}
	}

	void serveConnection(std::shared_ptr<Poco::Net::StreamSocket> conn) {
		// 存活账回收与在册摘除，均先于成员访问建立
		const ThreadReleaseGuard threadsGuard{_activeThreads};
		const ConnScope connScope{this, conn};
		struct IdentityGuard { const void* previous = servingListener; IdentityGuard(const void* p) { servingListener = p; } ~IdentityGuard() { servingListener = previous; } } identity{this};
		try {
			conn->setReceiveTimeout(toTimespan(_endpoint.requestTimeout));
			conn->setSendTimeout(toTimespan(_endpoint.requestTimeout));

			RawRequest req;
			const auto deadline = std::chrono::steady_clock::now() + (_endpoint.requestTimeout.count() > 0 ? _endpoint.requestTimeout : kDefaultRequestTimeout);
			const int readStatus = readRequest(*conn, req, deadline);
			if (readStatus != 0) {
				respond(*conn,
						WireResponse{readStatus, detail::wireErrorBody(readStatus == 413 ? "payload_too_large" : "bad_request",
																	   readStatus == 413 ? "request body too large"
																						 : "malformed HTTP request")});
				return;
			}
			if (req.method != "POST") {
				respond(*conn, WireResponse{405, detail::wireErrorBody("method_not_allowed", "only POST is supported")});
				return;
			}
			if (!authorized(req)) {
				respond(*conn, WireResponse{401, detail::wireErrorBody("unauthorized", "missing or invalid token")});
				return;
			}
			const std::string expected = _endpoint.basePath + _endpoint.requestPath;
			if (req.path != expected) {
				respond(*conn, WireResponse{404, detail::wireErrorBody("not_found", "no such endpoint")});
				return;
			}
			if (_inFlight.fetch_add(1, std::memory_order_acq_rel) + 1 >
				static_cast<std::size_t>(_endpoint.maxInFlight)) {
				_inFlight.fetch_sub(1, std::memory_order_acq_rel);
				respond(*conn, WireResponse{429, detail::wireErrorBody(
													"overloaded", "server overloaded: in-flight limit reached")});
				return;
			}
			InFlightGuard guard{_inFlight};

			if (req.bodyLength > _endpoint.maxRequestBody) {
				respond(*conn, WireResponse{413, detail::wireErrorBody("payload_too_large", "request body too large")});
				return;
			}
			bool reserved = false;
			{
				std::lock_guard budgetLock(_bodyMutex);
				if (req.bodyLength <= _endpoint.maxBufferedBodyBytes - _bufferedBodyBytes) {
					_bufferedBodyBytes += req.bodyLength;
					reserved = true;
				}
			}
			if (!reserved) {
				respond(*conn, WireResponse{429, detail::wireErrorBody("overloaded", "body budget exhausted")});
				return;
			}
			struct BodyLease { HttpListener* self; std::size_t bytes; ~BodyLease() { std::lock_guard lk(self->_bodyMutex); self->_bufferedBodyBytes -= bytes; } } bodyLease{this, req.bodyLength};
			if (!readBody(*conn, req, deadline)) {
				respond(*conn, WireResponse{400, detail::wireErrorBody("bad_request", "malformed HTTP request")});
				return;
			}

			WireResponse resp;
			try {
				resp = _handler(req.path, req.body);
			} catch (const std::exception& e) {
				// handler 异常不得逃逸，不得崩溃，兜底 5xx
				resp = WireResponse{500, internalError()};
			} catch (...) {
				resp = WireResponse{500, internalError()};
			}
			respond(*conn, resp);
		} catch (...) {
			// 连接中途断开：无应答，对端自行归一化超时，线程静默退出
		}
	}

	// 读取并解析请求；成功 0，失败返回待应答状态码 400 或 413。deadline 为单请求
	// 读总预算：每次读前收紧 receive 超时到剩余时间，慢速连接自行释放线程。
	static int readRequest(Poco::Net::StreamSocket& conn, RawRequest& req,
		const std::chrono::steady_clock::time_point& deadline) {
		std::string raw;
		char buf[4096];
		int n;
		// 头部：读到 \r\n\r\n
		while (raw.size() < kMaxHeaderBytes) {
			if (!armRemainingBudget(conn, deadline))
				return 400; // 读预算耗尽
			n = conn.receiveBytes(buf, sizeof(buf));
			if (n <= 0)
				return 400;
			raw.append(buf, static_cast<std::size_t>(n));
			if (raw.size() > kMaxHeaderBytes)
				return 400;
			if (raw.find("\r\n\r\n") != std::string::npos)
				break;
		}
		const std::size_t headerEnd = raw.find("\r\n\r\n");
		if (headerEnd == std::string::npos)
			return 400; // 头部超预算

		// 请求行：METHOD SP PATH SP VERSION
		const std::size_t eol = raw.find("\r\n");
		const std::string line = raw.substr(0, eol);
		const std::size_t sp1 = line.find(' ');
		const std::size_t sp2 = sp1 == std::string::npos ? std::string::npos : line.find(' ', sp1 + 1);
		if (sp1 == std::string::npos || sp2 == std::string::npos || line.substr(sp2 + 1) != "HTTP/1.1")
			return 400;
		req.method = line.substr(0, sp1);
		req.path = line.substr(sp1 + 1, sp2 - sp1 - 1);

		// 头部，键小写化
		std::size_t pos = eol + 2;
		while (pos < headerEnd) {
			const std::size_t lineEnd = raw.find("\r\n", pos);
			if (lineEnd == std::string::npos || lineEnd > headerEnd)
				break;
			const std::size_t colon = raw.find(':', pos);
			if (colon != std::string::npos && colon < lineEnd) {
				std::string key = raw.substr(pos, colon - pos);
				if (key.empty()) return 400;
				for (unsigned char c : key)
					if (!std::isalnum(c) && std::string("!#$%&'*+-.^_`|~").find(static_cast<char>(c)) == std::string::npos) return 400;
				for (std::size_t i = colon + 1; i < lineEnd; ++i) {
					const auto c = static_cast<unsigned char>(raw[i]);
					if ((c < 0x20 && c != '\t') || c == 0x7f) return 400;
				}
				for (auto& c : key)
					c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
				std::string value = raw.substr(colon + 1, lineEnd - colon - 1);
				while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
					value.erase(value.begin());
				auto [it, inserted] = req.headers.emplace(std::move(key), std::move(value));
				// 重复头拒绝：多值混淆是请求走私的经典向量
				if (!inserted)
					return 400;
			} else return 400;
			pos = lineEnd + 2;
		}

		// 请求体：Content-Length，v1 不支持 chunked；严格解析，全串十进制、
		// 无溢出，非法值一律 400，不复现 strtoull 的静默截断与回绕
		if (req.headers.count("transfer-encoding") || req.headers.count("expect")) return 400;
		std::size_t bodyLen = 0;
		if (auto it = req.headers.find("content-length"); it != req.headers.end()) {
			const std::string& v = it->second;
			if (v.empty() || v.find_first_not_of("0123456789") != std::string::npos)
				return 400;
			unsigned long long parsed = 0;
			const auto [ptr, ec] = std::from_chars(v.data(), v.data() + v.size(), parsed, 10);
			if (ec != std::errc() || ptr != v.data() + v.size() || parsed > std::numeric_limits<std::size_t>::max())
				return 400; // 溢出 / 全串未消费
			bodyLen = static_cast<std::size_t>(parsed);
		}
		req.bodyLength = bodyLen;
		req.body = raw.substr(headerEnd + 4);
		if (req.body.size() > bodyLen) return 400;
		return 0;
	}

	static bool readBody(Poco::Net::StreamSocket& conn, RawRequest& req,
		const std::chrono::steady_clock::time_point& deadline) {
		char buf[4096];
		while (req.body.size() < req.bodyLength) {
			if (!armRemainingBudget(conn, deadline)) return false;
			const int n = conn.receiveBytes(buf, static_cast<int>(std::min(sizeof(buf), req.bodyLength - req.body.size())));
			if (n <= 0) return false;
			req.body.append(buf, static_cast<std::size_t>(n));
		}
		return true;
	}

	static bool armRemainingBudget(Poco::Net::StreamSocket& conn,
								   const std::chrono::steady_clock::time_point& deadline) {
		const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
			deadline - std::chrono::steady_clock::now());
		if (remaining <= std::chrono::milliseconds(0))
			return false;
		try {
			conn.setReceiveTimeout(toTimespan(remaining));
		} catch (...) {
		}
		return true;
	}

	bool authorized(const RawRequest& req) const {
		if (_endpoint.authToken.empty())
			return true; // 未配置鉴权
		const auto it = req.headers.find("authorization");
		if (it == req.headers.end())
			return false;
		return stripBearer(it->second) == stripBearer(_endpoint.authToken);
	}

	static void respond(Poco::Net::StreamSocket& conn, const WireResponse& resp) {
		const std::string http =
			"HTTP/1.1 " + std::to_string(resp.status) + " " + detail::wireStatusText(resp.status) + "\r\n"
			"Content-Type: application/json\r\n"
			"Content-Length: " + std::to_string(resp.body.size()) + "\r\n"
			"Connection: close\r\n\r\n" + resp.body;
		try {
			// 循环补发：sendBytes 可能短写，单次发送 + 不查返回值会静默截断响应，
			// 客户端字节数与 Content-Length 不符。失败/超时仍静默关闭连接。
			std::size_t sent = 0;
			while (sent < http.size()) {
				const int n = conn.sendBytes(http.data() + sent,
											 static_cast<int>(http.size() - sent));
				if (n <= 0)
					break; // 无法继续，关闭连接
				sent += static_cast<std::size_t>(n);
			}
		} catch (...) {
		}
		try {
			conn.close();
		} catch (...) {
		}
	}

	std::string internalError() {
		const auto id = "dcnet-" + std::to_string(_errorSeq.fetch_add(1));
		if (_endpoint.diagnosticSink) { try { _endpoint.diagnosticSink(id + " internal server error stage=handler_exception"); } catch (...) {} }
		return detail::wireErrorBody("server_error", "internal server error; correlation=" + id);
	}
	std::atomic<unsigned long long> _errorSeq{0};
	std::mutex _lifecycleMutex;
	std::mutex _bodyMutex;
	std::size_t _bufferedBodyBytes = 0;
	NetServerEndpoint _endpoint;
	Poco::Net::ServerSocket _socket;
	RequestHandler _handler;
	std::thread _acceptThread;
	std::atomic<std::size_t> _inFlight{0};	   ///< 已完整读入的在途请求，仅 429 闸门
	std::atomic<std::size_t> _activeThreads{0};  ///< 在服工作线程数，stop 排水依据
	bool _bound = false;						   ///< 仅 _mutex 下访问 bind 与 start
	std::atomic<bool> _started{false};			   ///< accept 已启动，acceptLoop 与 alive 无锁读
	std::atomic<bool> _stopped{false};			   ///< stop 已发起，acceptLoop 轮询无锁读
	std::mutex _mutex; // 仅保护 bind/start/stop 状态字段，不保护请求路径
	std::mutex _connMutex; ///< 保护 _conns：accept 入册、worker 注销、stop 快照
	std::vector<std::shared_ptr<Poco::Net::StreamSocket>> _conns; ///< 在册连接，配额与强制关闭目标
};

} // namespace

std::unique_ptr<DcNetListener> makeHttpListener() {
	return std::make_unique<HttpListener>();
}

} // namespace DC::Net
