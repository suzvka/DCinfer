// HTTP/1.1 监听器（POCO ServerSocket；MockServer 的对外契约演进，DESIGN.md §3.6）
//
// 闸门顺序（提案 §5）：accept → 读请求 → 过载 429 → 方法 405 → 鉴权 401 →
// 路径 404 → 业务 handler → 应答。连接中途断开无应答（§5 aborted 行）。
// 过载计数在请求完整读入后进行——出站 connect 的 TCP 就绪探测连接（读即断）
// 不入计，避免误限流。
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
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

namespace DC::Net {

namespace {

constexpr std::size_t kMaxHeaderBytes = 64 * 1024;		   // 请求头预算
constexpr std::size_t kMaxBodyBytes = 64ull * 1024 * 1024; // 请求体预算（超限 413）

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
			if (_started)
				throw NodeException(NodeException::ErrorType::InternalError, "DcNetListener::start",
									"listener already started");
			_handler = std::move(handler);
			_started = true;
		}
		_acceptThread = std::thread([this] { acceptLoop(); });
	}

	void stop() override {
		{
			std::lock_guard lk(_mutex);
			if (!_started || _stopped)
				return;
			_stopped = true;
		}
		try {
			_socket.close(); // 中断 accept（轮询循环在 50ms 内感知 _stopped）
		} catch (...) {
		}
		if (_acceptThread.joinable())
			_acceptThread.join();
		// graceful drain：等待在途请求完成（受 requestTimeout 约束，兜底防挂死）
		const auto deadline = std::chrono::steady_clock::now() +
							  std::max(_endpoint.requestTimeout * 2, std::chrono::milliseconds(5000));
		while (_inFlight.load(std::memory_order_acquire) > 0 && std::chrono::steady_clock::now() < deadline)
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}

	bool alive() const override {
		return _started && !_stopped;
	}

	int port() const override {
		try {
			return static_cast<int>(_socket.address().port());
		} catch (...) {
			return -1;
		}
	}

private:
	void acceptLoop() {
		for (;;) {
			if (_stopped)
				break;
			// 轮询而非阻塞 accept：stop 时 close+join 跨平台安全（POSIX close 不唤醒阻塞 accept）
			try {
				if (!_socket.poll(Poco::Timespan(0, 50 * 1000), Poco::Net::Socket::SELECT_READ))
					continue;
				auto conn = _socket.acceptConnection();
				std::thread([this, conn = std::move(conn)]() mutable {
					serveConnection(std::move(conn));
				}).detach(); // v1：worker 分离；排水以在途计数为准（stop() 等待归零）
			} catch (const Poco::Exception&) {
				if (_stopped)
					break; // socket 已关闭（stop）
				// 单连接异常不终止监听（FR-5：不得崩溃）
			} catch (...) {
				if (_stopped)
					break;
			}
		}
	}

	void serveConnection(Poco::Net::StreamSocket conn) {
		try {
			conn.setReceiveTimeout(toTimespan(_endpoint.requestTimeout));
			conn.setSendTimeout(toTimespan(_endpoint.requestTimeout));

			RawRequest req;
			const int readStatus = readRequest(conn, req);
			if (readStatus != 0) {
				respond(conn,
						WireResponse{readStatus, detail::wireErrorBody(readStatus == 413 ? "payload_too_large" : "bad_request",
																	   readStatus == 413 ? "request body too large"
																						 : "malformed HTTP request")});
				return;
			}

			// 过载闸门（FR-5）：请求已完整读入后计数（探测连接不入计），超出在途
			// 上限立即 wire 429——不排队、不静默丢弃
			if (_inFlight.fetch_add(1, std::memory_order_acq_rel) + 1 >
				static_cast<std::size_t>(_endpoint.maxInFlight)) {
				_inFlight.fetch_sub(1, std::memory_order_acq_rel);
				respond(conn, WireResponse{429, detail::wireErrorBody(
													"overloaded", "server overloaded: in-flight limit reached")});
				return;
			}
			InFlightGuard guard{_inFlight};

			if (req.method != "POST") {
				respond(conn, WireResponse{405, detail::wireErrorBody("method_not_allowed", "only POST is supported")});
				return;
			}
			if (!authorized(req)) {
				respond(conn, WireResponse{401, detail::wireErrorBody("unauthorized", "missing or invalid token")});
				return;
			}
			const std::string expected = _endpoint.basePath + _endpoint.requestPath;
			if (req.path != expected) {
				respond(conn, WireResponse{404, detail::wireErrorBody("not_found", "no such endpoint")});
				return;
			}

			WireResponse resp;
			try {
				resp = _handler(req.path, req.body);
			} catch (const std::exception& e) {
				// handler 异常不得逃逸出服务线程（FR-5：不得崩溃），兜底 5xx
				resp = WireResponse{500, detail::wireErrorBody("server_error", std::string("handler exception: ") + e.what())};
			} catch (...) {
				resp = WireResponse{500, detail::wireErrorBody("server_error", "unknown handler exception")};
			}
			respond(conn, resp);
		} catch (...) {
			// 连接中途断开 / 请求中止：无应答（提案 §5 aborted 行），线程静默退出
		}
	}

	// 读取并解析请求；成功返回 0，否则返回应答的 HTTP 错误状态码（400/413）。
	static int readRequest(Poco::Net::StreamSocket& conn, RawRequest& req) {
		std::string raw;
		char buf[4096];
		int n;
		// ① 头部：读到 \r\n\r\n（受连接读超时与预算约束，慢速连接自行释放线程）
		while (raw.size() < kMaxHeaderBytes) {
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
			n = conn.receiveBytes(buf, sizeof(buf));
			if (n <= 0)
				return 400;
			req.body.append(buf, static_cast<std::size_t>(n));
			if (req.body.size() > bodyLen)
				req.body.resize(bodyLen);
		}
		return 0;
	}

	bool authorized(const RawRequest& req) const {
		if (_endpoint.authToken.empty())
			return true; // 未配置鉴权 → 闸门放行（FR-6 P1 可选）
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
	std::atomic<std::size_t> _inFlight{0};
	bool _bound = false;
	bool _started = false;
	bool _stopped = false;
	std::mutex _mutex; // 仅保护 bind/start/stop 状态字段（ADR-5/6：不保护请求路径）
};

} // namespace

std::unique_ptr<DcNetListener> makeHttpListener() {
	return std::make_unique<HttpListener>();
}

} // namespace DC::Net
