// HTTP/1.1 监听器（POCO ServerSocket；DESIGN.md §3.6）
//
// 闸门顺序（DESIGN.md §6.1）：accept →（连接级配额 maxConnections）→ 读请求 →
// 过载 429 → 方法 405 → 鉴权 401 → 路径 404 → 业务 handler → 应答。
// 连接中途断开无应答（对端自行归一化超时）。
// 两级闸门分层：连接级配额在 accept 期生效（含半开/慢速连接，它们不计入在途请求）；
// 过载计数在请求完整读入后进行——出站 connect 的 TCP 就绪探测连接（读即断）
// 不入计，避免误限流。
// 生命周期不变量：worker 线程在 accept 期（起线程之前）就以 _activeThreads 记账，
// stop() 排水等其归零后才返回——因此“已创建但尚未开始调度”的线程也在账内，
// 分离线程不可能再访问已析构的监听器状态。
// v1 边界：仅 Content-Length 请求体（不支持 chunked）；逐请求应答后关闭连接。

#include "DCNet/NetListener.h"

#include "NodeException.h"
#include "NetWire.h"

#include <Poco/Exception.h>
#include <Poco/Net/ServerSocket.h>
#include <Poco/Net/SocketAddress.h>
#include <Poco/Net/StreamSocket.h>
#include <Poco/Timespan.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdlib>
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
constexpr std::size_t kMaxBodyBytes = 64ull * 1024 * 1024; // 请求体预算（超限 413）
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
	std::unordered_map<std::string, std::string> headers; // 键小写化
};

/// 在途计数 RAII 递减（闸门放行后的所有退出路径均回收计数）。
struct InFlightGuard {
	std::atomic<std::size_t>& counter;
	~InFlightGuard() { counter.fetch_sub(1, std::memory_order_acq_rel); }
};

/// worker 线程退出时回收存活账（递减-only）：计数在 accept 期与入册同一临界区内
/// 完成（见 admitConnection）——若改为在 worker 线程体内自增，“线程已创建但
/// 尚未被调度”的窗口会使排水看到 0 而提前放行，仍是 use-after-free。
/// stop() 排水以此账为准：_inFlight 只覆盖“完整读入后”的阶段，不能充当线程存活性依据。
struct ThreadReleaseGuard {
	std::atomic<std::size_t>& counter;
	explicit ThreadReleaseGuard(std::atomic<std::size_t>& c) : counter(c) {}
	ThreadReleaseGuard(const ThreadReleaseGuard&) = delete;
	ThreadReleaseGuard& operator=(const ThreadReleaseGuard&) = delete;
	~ThreadReleaseGuard() { counter.fetch_sub(1, std::memory_order_acq_rel); }
};

/// "Bearer xxx" / 裸 key 统一去前缀比较（出站 authToken 按原样注入请求头）。
std::string stripBearer(const std::string& value) {
	const std::string prefix = "Bearer ";
	if (value.rfind(prefix, 0) == 0)
		return value.substr(prefix.size());
	return value;
}

class HttpListener final : public DcNetListener {
public:
	~HttpListener() override { stop(); }

	void bind(const NetServerEndpoint& endpoint) override {
		std::lock_guard lk(_mutex);
		if (_bound)
			throw NodeException(NodeException::ErrorType::InternalError, "DcNetListener::bind",
								"listener already bound");
		try {
			_socket.bind(Poco::Net::SocketAddress(endpoint.listenHost, static_cast<Poco::UInt16>(endpoint.port)),
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
		{
			std::lock_guard lk(_mutex);
			if (!_bound)
				throw NodeException(NodeException::ErrorType::InternalError, "DcNetListener::start",
									"bind() must be called before start()");
			if (_started.load())
				throw NodeException(NodeException::ErrorType::InternalError, "DcNetListener::start",
									"listener already started");
			_handler = std::move(handler);
			_stopped.store(false);
			_started.store(true);
		}
		_acceptThread = std::thread([this] { acceptLoop(); });
	}

	void stop() override {
		{
			std::lock_guard lk(_mutex);
			if (!_started.load() || _stopped.load())
				return;
			_stopped.store(true);
		}
		try {
			_socket.close(); // 中断 accept（轮询循环在 50ms 内感知 _stopped）
		} catch (...) {
		}
		if (_acceptThread.joinable())
			_acceptThread.join();
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
	/// 连接在册登记（RAII 收尾）：worker 退出即从在册集合摘除。
	/// 在册集合同时是 maxConnections 的配额依据与 stop() 强制关闭的目标。
	struct ConnScope {
		HttpListener* self;
		std::shared_ptr<Poco::Net::StreamSocket> conn;
		~ConnScope() { self->unregisterConnection(conn); }
	};

	/// 排水：grace 内等工作线程自然退出（已完成请求正常应答）；到期后强制关闭
	/// 在册连接，把阻塞在慢速对端读/写上的线程立即放倒，再等其全部退出。
	/// socket 层面的阻塞已由单请求读预算与本处强制关闭双重有界；若本地引擎在
	/// handler 内永久挂起，stop() 会随之挂起——宁挂起也不让分离线程访问已析构对象。
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
		// grace 到期仍有在服连接：强制关闭后无条件等待（此时 worker 不再可能阻塞）
		std::vector<std::shared_ptr<Poco::Net::StreamSocket>> snapshot;
		{
			std::lock_guard lk(_connMutex);
			snapshot = _conns; // 快照后锁外关闭：不在持锁期间做 syscall
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

	/// accept 后同步入册：配额检查、存活记账与登记同一临界区（既无超限窗口，
	/// 也无“线程已创建但未调度”时排水误判归零的窗口）。记账由 worker 退出时回收。
	/// 返回 null = 超出 maxConnections（已就地关闭，调用方不得起线程）。
	std::shared_ptr<Poco::Net::StreamSocket> admitConnection(Poco::Net::StreamSocket conn) {
		auto holder = std::make_shared<Poco::Net::StreamSocket>(std::move(conn));
		{
			std::lock_guard lk(_connMutex);
			const int cap = _endpoint.maxConnections;
			if (cap > 0 && _conns.size() >= static_cast<std::size_t>(cap)) {
				forceClose(*holder); // 连接级过载：不排队、不起线程
				return nullptr;
			}
			_activeThreads.fetch_add(1, std::memory_order_acq_rel); // 起线程前先记账
			_conns.push_back(holder);
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

	/// 线程创建失败（资源不足）回滚：摘册 + 回收 accept 期的存活账 + 关连接。
	/// 不回滚则排水会永远等一个不存在的 worker。
	void rejectAdmittedConnection(const std::shared_ptr<Poco::Net::StreamSocket>& holder) {
		forceClose(*holder);
		unregisterConnection(holder);
		_activeThreads.fetch_sub(1, std::memory_order_acq_rel);
	}

	void acceptLoop() {
		for (;;) {
			if (_stopped.load(std::memory_order_acquire))
				break;
			// 轮询而非阻塞 accept：stop 时 close+join 跨平台安全（POSIX close 不唤醒阻塞 accept）
			try {
				if (!_socket.poll(Poco::Timespan(0, 50 * 1000), Poco::Net::Socket::SELECT_READ))
					continue;
				auto conn = _socket.acceptConnection();
				auto holder = admitConnection(std::move(conn));
				if (!holder)
					continue; // 连接级闸门已拦截（429 之前的第一道防线）
				auto tracked = holder; // worker 接走 holder 后仍可回滚
				std::thread worker;
				try {
					worker = std::thread([this, conn = std::move(holder)]() mutable {
						serveConnection(std::move(conn));
					});
				} catch (...) {
					rejectAdmittedConnection(tracked); // 线程未能启动：回收 accept 期的账
					continue;
				}
				worker.detach(); // 分离：join 责任由 _activeThreads 排水承担
			} catch (const Poco::Exception&) {
				if (_stopped.load(std::memory_order_acquire))
					break; // socket 已关闭（stop）
				// 单连接异常不终止监听（不得崩溃）
			} catch (...) {
				if (_stopped.load(std::memory_order_acquire))
					break;
			}
		}
	}

	void serveConnection(std::shared_ptr<Poco::Net::StreamSocket> conn) {
		// 回收 accept 期登记的存活账；连接收尾从在册集摘除（两者均先于成员访问建立）
		const ThreadReleaseGuard threadsGuard{_activeThreads};
		const ConnScope connScope{this, conn};
		try {
			conn->setReceiveTimeout(toTimespan(_endpoint.requestTimeout));
			conn->setSendTimeout(toTimespan(_endpoint.requestTimeout));

			RawRequest req;
			const int readStatus = readRequest(*conn, req, _endpoint.requestTimeout);
			if (readStatus != 0) {
				respond(*conn,
						WireResponse{readStatus, detail::wireErrorBody(readStatus == 413 ? "payload_too_large" : "bad_request",
																	   readStatus == 413 ? "request body too large"
																						 : "malformed HTTP request")});
				return;
			}

			// 过载闸门（DESIGN.md §3.6）：请求已完整读入后计数（探测连接不入计），超出在途
			// 上限立即 wire 429——不排队、不静默丢弃
			if (_inFlight.fetch_add(1, std::memory_order_acq_rel) + 1 >
				static_cast<std::size_t>(_endpoint.maxInFlight)) {
				_inFlight.fetch_sub(1, std::memory_order_acq_rel);
				respond(*conn, WireResponse{429, detail::wireErrorBody(
													"overloaded", "server overloaded: in-flight limit reached")});
				return;
			}
			InFlightGuard guard{_inFlight};

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

			WireResponse resp;
			try {
				resp = _handler(req.path, req.body);
			} catch (const std::exception& e) {
				// handler 异常不得逃逸出服务线程（不得崩溃），兜底 5xx
				resp = WireResponse{500, detail::wireErrorBody("server_error", std::string("handler exception: ") + e.what())};
			} catch (...) {
				resp = WireResponse{500, detail::wireErrorBody("server_error", "unknown handler exception")};
			}
			respond(*conn, resp);
		} catch (...) {
			// 连接中途断开 / 请求中止：无应答（对端自行归一化超时），线程静默退出
		}
	}

	// 读取并解析请求；成功返回 0，否则返回应答的 HTTP 错误状态码（400/413）。
	// budget 为单请求读总预算：每次 receiveBytes 前把连接 receive 超时收紧到剩余
	// 时间，慢速滴灌连接到点自行释放线程（与 maxConnections 构成两道防线）。
	static int readRequest(Poco::Net::StreamSocket& conn, RawRequest& req, std::chrono::milliseconds budget) {
		const std::chrono::milliseconds total =
			budget.count() > 0 ? budget : kDefaultRequestTimeout;
		const auto deadline = std::chrono::steady_clock::now() + total;
		std::string raw;
		char buf[4096];
		int n;
		// ① 头部：读到 \r\n\r\n（受单请求总预算约束，慢速连接自行释放线程）
		while (raw.size() < kMaxHeaderBytes) {
			if (!armRemainingBudget(conn, deadline))
				return 400; // 读预算耗尽
			n = conn.receiveBytes(buf, sizeof(buf));
			if (n <= 0)
				return 400;
			raw.append(buf, static_cast<std::size_t>(n));
			if (raw.find("\r\n\r\n") != std::string::npos)
				break;
		}
		const std::size_t headerEnd = raw.find("\r\n\r\n");
		if (headerEnd == std::string::npos)
			return 400; // 头部超预算

		// ② 请求行：METHOD SP PATH SP VERSION
		const std::size_t eol = raw.find("\r\n");
		const std::string line = raw.substr(0, eol);
		const std::size_t sp1 = line.find(' ');
		const std::size_t sp2 = sp1 == std::string::npos ? std::string::npos : line.find(' ', sp1 + 1);
		if (sp1 == std::string::npos || sp2 == std::string::npos)
			return 400;
		req.method = line.substr(0, sp1);
		req.path = line.substr(sp1 + 1, sp2 - sp1 - 1);

		// ③ 头部（键小写化；仅取 Content-Length / Authorization）
		std::size_t pos = eol + 2;
		while (pos < headerEnd) {
			const std::size_t lineEnd = raw.find("\r\n", pos);
			if (lineEnd == std::string::npos || lineEnd > headerEnd)
				break;
			const std::size_t colon = raw.find(':', pos);
			if (colon != std::string::npos && colon < lineEnd) {
				std::string key = raw.substr(pos, colon - pos);
				for (auto& c : key)
					c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
				std::string value = raw.substr(colon + 1, lineEnd - colon - 1);
				while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
					value.erase(value.begin());
				req.headers.emplace(std::move(key), std::move(value));
			}
			pos = lineEnd + 2;
		}

		// ④ 请求体：Content-Length（v1 不支持 chunked）
		std::size_t bodyLen = 0;
		if (auto it = req.headers.find("content-length"); it != req.headers.end())
			bodyLen = static_cast<std::size_t>(std::strtoull(it->second.c_str(), nullptr, 10));
		if (bodyLen > kMaxBodyBytes)
			return 413;
		req.body = raw.substr(headerEnd + 4);
		if (req.body.size() > bodyLen)
			req.body.resize(bodyLen);
		while (req.body.size() < bodyLen) {
			if (!armRemainingBudget(conn, deadline))
				return 400; // 读预算耗尽（慢速滴灌请求体）
			n = conn.receiveBytes(buf, sizeof(buf));
			if (n <= 0)
				return 400;
			req.body.append(buf, static_cast<std::size_t>(n));
			if (req.body.size() > bodyLen)
				req.body.resize(bodyLen);
		}
		return 0;
	}

	/// 把连接 receive 超时收紧到剩余读预算；预算耗尽返回 false。
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
			return true; // 未配置鉴权 → 闸门放行（鉴权可选）
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
			conn.sendBytes(http.data(), static_cast<int>(http.size()));
		} catch (...) {
		}
		try {
			conn.close();
		} catch (...) {
		}
	}

	NetServerEndpoint _endpoint;
	Poco::Net::ServerSocket _socket;
	RequestHandler _handler;
	std::thread _acceptThread;
	std::atomic<std::size_t> _inFlight{0};	   ///< 已完整读入的在途请求（仅 429 闸门）
	std::atomic<std::size_t> _activeThreads{0};  ///< 在服工作线程数（stop() 排水依据）
	bool _bound = false;						   ///< 仅 _mutex 下访问（bind/start）
	std::atomic<bool> _started{false};			   ///< accept 已启动（acceptLoop/alive 无锁读）
	std::atomic<bool> _stopped{false};			   ///< stop() 已发起（acceptLoop 轮询无锁读）
	std::mutex _mutex; // 仅保护 bind/start/stop 状态字段（ADR-5/6：不保护请求路径）
	std::mutex _connMutex; ///< 保护 _conns（accept 入册 / worker 注销 / stop 快照）
	std::vector<std::shared_ptr<Poco::Net::StreamSocket>> _conns; ///< 在册连接（配额 + 强制关闭目标）
};

} // namespace

std::unique_ptr<DcNetListener> makeHttpListener() {
	return std::make_unique<HttpListener>();
}

} // namespace DC::Net
