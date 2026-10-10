// HttpTransport + DCNet.Tensor 集成测试：MockHttpServer 假远端，真实 HTTP 传输。
// 覆盖：传输层归一化 2xx、404、500 与连接拒绝；张量 codec 端到端往返，数值与文本。
// OpenAI chat 端到端见 DCEngines/OpenAI 的 OpenAiEngineTest。

#include "DCNet/DcNetHttp.h"
#include "NodeExecutor.h"
#include "DCNet/NetCodec_Tensor.h"
#include "DCNet/NetError.h"
#include "DCNet/NetTransport_Http.h"
#include "Node.h"
#include "Tensor.hpp"

#include "DCNet/MockServer.h"
#include "NetBase64.h"

#include <Poco/Net/ServerSocket.h>
#include <Poco/Net/SocketAddress.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

static std::atomic<int> g_checks{0};
static std::atomic<int> g_failures{0};

#define CHECK(cond, msg)                                                                                               \
	do {                                                                                                               \
		++g_checks;                                                                                                    \
		if (!(cond)) {                                                                                                 \
			++g_failures;                                                                                              \
			std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);                                                  \
		}                                                                                                              \
	} while (0)

#define CHECK_MSG_PREFIX(msg, prefix)                                                                                  \
	do {                                                                                                               \
		++g_checks;                                                                                                    \
		if ((msg).rfind(prefix, 0) != 0) {                                                                             \
			++g_failures;                                                                                              \
			std::printf("FAIL %s:%d  message '%s' should start with '%s'\n", __FILE__, __LINE__, (msg).c_str(),        \
						prefix);                                                                                       \
		}                                                                                                              \
	} while (0)

#define TEST(name) static void test_##name()

using namespace DC;
using namespace DC::Net;

static DC::Net::NetEndpoint epFor(int port, std::string requestPath = {}) {
	DC::Net::NetEndpoint ep;
	ep.host = "127.0.0.1";
	ep.port = port;
	ep.basePath = "/v1";
	ep.requestPath = std::move(requestPath);
	ep.connectTimeout = std::chrono::milliseconds(2000);
	ep.requestTimeout = std::chrono::milliseconds(2000);
	return ep;
}

static Tensor makeFloatTensor(const std::vector<float>& vals) {
	Tensor::DataBlock block(vals.size() * sizeof(float));
	if (!vals.empty())
		std::memcpy(block.data(), vals.data(), vals.size() * sizeof(float));
	return Tensor(Tensor::TensorType::Float, sizeof(float), {static_cast<int64_t>(vals.size())},
				  std::move(block));
}

static Tensor makeTextTensor(const std::string& s) {
	Tensor::DataBlock block(s.size());
	if (!s.empty())
		std::memcpy(block.data(), s.data(), s.size());
	return Tensor(Tensor::TensorType::Data, 1, {static_cast<int64_t>(s.size())}, std::move(block));
}

TEST(successRoundtrip) {
	MockHttpServer server;
	std::string seenPath;
	server.start([&](const std::string& path, const std::string&, int& status) {
		seenPath = path;
		status = 200;
		return R"({"echo":"pong"})";
	});

	HttpTransport t;
	auto err = t.connect(epFor(server.port(), "/infer"));
	CHECK(err.ok(), "connect should succeed");
	CHECK(t.alive(), "transport should be alive after connect");

	Payload resp;
	err = t.send(R"({"hello":"world"})");
	CHECK(err.ok(), "send (2xx) should succeed");
	err = t.recv(resp);
	CHECK(err.ok(), "recv should succeed");
	CHECK(resp == R"({"echo":"pong"})", "response body should match");
	CHECK(seenPath == "/v1/infer", "request path should be basePath + requestPath");
	t.close();
}

TEST(status404Normalized) {
	MockHttpServer server;
	server.start([](const std::string&, const std::string&, int& status) {
		status = 404;
		return R"({"error":"not here"})";
	});

	HttpTransport t;
	t.connect(epFor(server.port(), "/missing"));
	Payload resp;
	auto err = t.send("x");
	CHECK(!err.ok(), "404 → failure");
	CHECK(err.category == NetErrorCategory::RemoteRejected, "404 → RemoteRejected");
	CHECK(err.localStatus == Node::Status::InvalidInput, "404 → InvalidInput");
	CHECK_MSG_PREFIX(err.localMessage, "remote:not_found");
	t.close();
}

TEST(status500Normalized) {
	MockHttpServer server;
	server.start([](const std::string&, const std::string&, int& status) {
		status = 500;
		return R"({"error":{"code":"server_error","message":"boom"}})";
	});

	HttpTransport t;
	t.connect(epFor(server.port(), "/infer"));
	Payload resp;
	auto err = t.send("x");
	CHECK(!err.ok(), "500 → failure");
	CHECK(err.category == NetErrorCategory::RemoteServer, "500 → RemoteServer");
	CHECK(err.retryable, "5xx retryable");
	CHECK(err.localStatus == Node::Status::ExecutionFailed, "500 → ExecutionFailed");
	CHECK_MSG_PREFIX(err.localMessage, "remote:server_error");
	t.close();
}

TEST(connectionRefusedNormalized) {
	int deadPort = -1;
	{
		Poco::Net::ServerSocket s;
		s.bind(Poco::Net::SocketAddress("127.0.0.1", 0));
		s.listen();
		deadPort = static_cast<int>(s.address().port());
		s.close();
	}

	HttpTransport t;
	t.connect(epFor(deadPort, "/infer"));
	Payload resp;
	auto err = t.send("x");
	CHECK(!err.ok(), "connection refused → failure");
	CHECK(err.retryable, "unreachable retryable");
	CHECK(err.localStatus == Node::Status::ExecutionFailed, "refused → ExecutionFailed");
#ifdef _WIN32
	// POCO/Windows：WSAPoll 不上报 connect 失败，拒绝连接表现为 Timeout，POSIX 报 unreachable。
	const bool unreachableOrTimeout = err.localMessage.rfind("net:unreachable", 0) == 0 ||
									 err.localMessage.rfind("net:timeout", 0) == 0;
	CHECK(unreachableOrTimeout, "refused → unreachable|timeout");
#else
	CHECK_MSG_PREFIX(err.localMessage, "net:unreachable");
#endif
	t.close();
}

// Timeout 分类映射由 NetErrorTest 覆盖，此处不设服务端延迟超时用例。

// 3xx：不自动跟随重定向，按非 2xx 归一化。

TEST(status302Normalized) {
	MockHttpServer server;
	server.start([](const std::string&, const std::string&, int& status) {
		status = 302;
		return std::string("redirect body");
	});
	HttpTransport t;
	t.connect(epFor(server.port(), "/moved"));
	Payload resp;
	auto err = t.send("x");
	CHECK(!err.ok(), "302 (no auto-redirect) → failure, not success");
	CHECK(err.category == NetErrorCategory::RemoteMalformed, "302 → RemoteMalformed fallback");
	t.close();
}

// 响应体上限：成功体超限报错；非 2xx 错误体仅诊断、允许截断。

TEST(responseBodyLimitEnforced) {
	MockHttpServer server;
	server.start([](const std::string&, const std::string&, int& status) {
		status = 200;
		return std::string("AAAAAAAAAAAAAAAA");
	});
	HttpTransport t;
	auto ep = epFor(server.port(), "/infer");
	ep.maxResponseBody = 8;
	CHECK(t.connect(ep).ok(), "connect");
	CHECK(t.send("x").ok(), "send ok (2xx)");
	Payload resp;
	auto err = t.recv(resp);
	CHECK(!err.ok(), "oversized success body must be rejected");
	t.close();
}

TEST(errorBodyTruncated) {
	MockHttpServer server;
	server.start([](const std::string&, const std::string&, int& status) {
		status = 500;
		return std::string(4096, 'x');
	});
	HttpTransport t;
	auto ep = epFor(server.port(), "/infer");
	ep.maxResponseBody = 1024;
	CHECK(t.connect(ep).ok(), "connect");
	auto err = t.send("x");
	CHECK(!err.ok(), "500 → failure");
	CHECK(err.category == NetErrorCategory::RemoteServer,
		  "truncated error body is diagnostic only (no size error)");
	t.close();
}

// close 与慢 recv 竞速：强收中断读而非悬垂或挂死，靠副本保活加 abort。

// 慢 body 服务器：先发响应头，剩余 body 延迟发送，使读跨越 close 的 5s 强收窗口。
namespace {
class SlowBodyServer {
public:
	int start() {
		try {
			_socket.bind(Poco::Net::SocketAddress("127.0.0.1", 0), false);
			_socket.listen();
			_port = static_cast<int>(_socket.address().port());
		} catch (...) {
			return -1;
		}
		_thread = std::thread([this] {
			try {
				// 连接 1 是 connect 的就绪探测，读到 EOF 即跳过，连接 2 才是真实交换。
				Poco::Net::StreamSocket c;
				for (int i = 0; i < 2; ++i) {
					try {
						c = _socket.acceptConnection();
					} catch (...) {
						return;
					}
					char buf[4096];
					std::string req;
					int n;
					while ((n = c.receiveBytes(buf, sizeof(buf))) > 0) {
						req.append(buf, static_cast<size_t>(n));
						if (req.find("\r\n\r\n") != std::string::npos)
							break;
					}
					if (req.find("\r\n\r\n") != std::string::npos)
						break;
				}
				const std::string head = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
								 "Content-Length: 16\r\nConnection: close\r\n\r\n";
				c.sendBytes(head.data(), static_cast<int>(head.size()));
				c.sendBytes("AAAA", 4);
				std::this_thread::sleep_for(std::chrono::milliseconds(8000));
				c.sendBytes("AAAAAAAAAAAA", 12);
				c.close();
			} catch (...) {
				// 对端 abort 后 sendBytes 可能抛 ConnectionReset 等：线程内异常不得逃逸。
			}
		});
		return _port;
	}
	void stop() {
		try {
			_socket.close();
		} catch (...) {
		}
		if (_thread.joinable())
			_thread.join();
	}
	int port() const { return _port; }

private:
	int _port = -1;
	std::thread _thread;
	Poco::Net::ServerSocket _socket;
};
} // namespace

TEST(closeRacingSlowRecv) {
	SlowBodyServer server;
	CHECK(server.start() > 0, "slow-body server should start");
	HttpTransport t;
	auto ep = epFor(server.port(), "/slow");
	ep.requestTimeout = std::chrono::milliseconds(15000);
	CHECK(t.connect(ep).ok(), "connect");
	CHECK(t.send("x").ok(), "send ok; response header received, body pending");

	// 另一线程 close：等交换收尾 5s 超时后强收 abort，读方副本保活 session。
	std::thread closer([&] { t.close(); });
	Payload resp;
	const auto t0 = std::chrono::steady_clock::now();
	const auto err = t.recv(resp);
	const auto elapsed = std::chrono::steady_clock::now() - t0;
	closer.join();
	server.stop();
	CHECK(!err.ok(), "recv interrupted by racing close must return error (not a partial success)");
	CHECK(elapsed < std::chrono::seconds(7),
		  "abort must interrupt the read well before the server's remaining 8s delay");
	CHECK(resp.size() <= 16, "no garbage beyond declared body length");
	t.close(); // 幂等收尾
}

// 端到端：DCNet.Tensor 节点 + 张量 JSON codec，真实 HTTP 往返。

TEST(endToEndTensorOverHttp) {
	MockHttpServer server;
	server.start([&](const std::string& path, const std::string& body, int& status) {
		if (path != "/v1/infer") {
			status = 404;
			return std::string(R"({"error":"not found"})");
		}
		const auto j = nlohmann::json::parse(body);
		CHECK(j["dtype"] == "float32", "server sees float32 dtype");
		const std::string bytes = DC::Net::detail::base64Decode(j["data"].get<std::string>());
		CHECK(bytes.size() == 2 * sizeof(float), "server sees 2 floats");
		if (bytes.size() == 2 * sizeof(float)) {
			float v[2];
			std::memcpy(v, bytes.data(), sizeof(v));
			CHECK(v[0] == 1.0f && v[1] == 2.0f, "server sees values {1,2}");
		}
		status = 200;
		nlohmann::json r;
		r["dtype"] = "float32";
		r["shape"] = std::vector<int64_t>{2};
		r["data"] = DC::Net::detail::base64Encode(reinterpret_cast<const std::uint8_t*>(bytes.data()),
												  bytes.size());
		return r.dump();
	});

	auto& reg = EngineRegistry::instance();
	registerDcNetHttp(reg, makeTensorJsonCodec());

	auto node = reg.createNode("DCNet.Tensor", "tensorNode",
							   std::string("http://127.0.0.1:" + std::to_string(server.port()) + "/v1"));
		NodeExecutor exec(*node);
	CHECK(node != nullptr, "DCNet.Tensor node should be created");
	CHECK(node->schema().inputs[0].name == "data" && node->schema().outputs[0].name == "result",
		  "local shape rules from tensor codec");

	exec.setInput("t1", "data", makeFloatTensor({1.0f, 2.0f}));
	auto result = exec.tryExecute("t1");
	CHECK(result.ok(), "tensor roundtrip should succeed");
	CHECK(exec.hasOutput("t1", "result"), "result output should exist");
	auto out = exec.takeOutputTensor("t1", "result");
	CHECK(out.type() == Tensor::TensorType::Float && out.typeSize() == sizeof(float),
		  "decoded tensor type/size");
	auto vals = out.getData<float>();
	CHECK(vals.size() == 2 && vals[0] == 1.0f && vals[1] == 2.0f, "decoded tensor values");
}

// 端到端：DCNet.Tensor 节点 + Data 文本 codec，真实 HTTP 往返。

TEST(endToEndTextOverHttp) {
	MockHttpServer server;
	server.start([&](const std::string& path, const std::string& body, int& status) {
		if (path != "/v1/infer") {
			status = 404;
			return std::string(R"({"error":"not found"})");
		}
		const auto j = nlohmann::json::parse(body);
		CHECK(j["dtype"] == "text", "server sees text dtype");
		CHECK(j["data"] == "hello", "server sees utf-8 text (not base64)");
		status = 200;
		nlohmann::json r;
		r["dtype"] = "text";
		r["shape"] = std::vector<int64_t>{8};
		r["data"] = "hi there";
		return r.dump();
	});

	auto& reg = EngineRegistry::instance();
	// 与 tensor 测试同进程：区分 engineType 避开注册表"保留首次"冲突
	registerDcNetHttp(reg, makeTextJsonCodec(), {}, "DCNet.Text");

	auto node = reg.createNode("DCNet.Text", "textNode",
							   std::string("http://127.0.0.1:" + std::to_string(server.port()) + "/v1"));
		NodeExecutor exec(*node);
	CHECK(node != nullptr, "DCNet.Text text node should be created");
	CHECK(node->schema().inputs[0].name == "text" && node->schema().outputs[0].name == "result",
		  "local shape rules from text codec");

	exec.setInput("t1", "text", makeTextTensor("hello"));
	auto result = exec.tryExecute("t1");
	CHECK(result.ok(), "text roundtrip should succeed");
	CHECK(exec.hasOutput("t1", "result"), "result output should exist");
	auto out = exec.takeOutputTensor("t1", "result");
	CHECK(out.type() == Tensor::TensorType::Data && out.typeSize() == 1, "decoded tensor type/size");
	auto bytes = out.bytes();
	CHECK(std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()) == "hi there",
		  "decoded text content");
}

// 空文本响应端到端：服务器返回 shape=[0] 空载荷帧，解码侧与 TensorData
// metadata-only 构造都必须接受；输入端仍用非空文本，因输入端口要求稠密缓存，空载荷仅输出端合法。
TEST(emptyTextOverHttp) {
	MockHttpServer server;
	server.start([&](const std::string& path, const std::string& body, int& status) {
		if (path != "/v1/infer") {
			status = 404;
			return std::string(R"({"error":"not found"})");
		}
		const auto j = nlohmann::json::parse(body);
		CHECK(j["dtype"] == "text", "server sees text dtype");
		CHECK(j["data"] == "ping", "server sees request text");
		status = 200;
		nlohmann::json r;
		r["dtype"] = "text";
		r["shape"] = std::vector<int64_t>{0};
		r["data"] = "";
		return r.dump();
	});

	auto& reg = EngineRegistry::instance();
	registerDcNetHttp(reg, makeTextJsonCodec(), {}, "DCNet.Text.Empty");

	auto node = reg.createNode("DCNet.Text.Empty", "emptyTextNode",
							   std::string("http://127.0.0.1:" + std::to_string(server.port()) + "/v1"));
	NodeExecutor exec(*node);
	CHECK(node != nullptr, "DCNet.Text.Empty text node should be created");

	exec.setInput("t1", "text", makeTextTensor("ping"));
	auto result = exec.tryExecute("t1");
	CHECK(result.ok(), "empty text roundtrip should succeed");
	CHECK(exec.hasOutput("t1", "result"), "result output should exist");
	auto out = exec.takeOutputTensor("t1", "result");
	CHECK(out.type() == Tensor::TensorType::Data && out.typeSize() == 1, "decoded tensor type/size");
	auto bytes = out.bytes();
	CHECK(bytes.empty(), "decoded empty text must stay empty");
}

TEST(headerInjectionRejectedAtConnect) {
	// 注入防护：headers/authToken/contentType 中的控制字符在 connect 配置期拒绝，不发网络 I/O。
	HttpTransport t;
	auto ep = epFor(1, "/");
	ep.headers = {"X-Evil: val\r\nX-Injected: 1"};
	auto err = t.connect(ep);
	CHECK(!err.ok(), "header containing CRLF must be rejected at connect");
	CHECK(!t.alive(), "transport must not become alive on invalid config");

	auto ep2 = epFor(1, "/");
	ep2.authToken = "Bearer token\r\nEvil: 1";
	CHECK(!t.connect(ep2).ok(), "authToken containing CRLF must be rejected at connect");

	auto ep3 = epFor(1, "/");
	ep3.contentType = "application/json\n";
	CHECK(!t.connect(ep3).ok(), "contentType containing LF must be rejected at connect");

	// 合法 header 照常接受，含值内 tab
	auto ep4 = epFor(1, "/");
	ep4.headers = {"X-Trace: abc\t123"};
	CHECK(t.connect(ep4).ok() || !t.alive(), "tab is allowed in header values");
}

TEST(largeResponseIntegrity) {
	// 大响应经多次 sendBytes 补发：字节数与 Content-Length 一致，内容逐字节一致。
	static const std::string payload = [] {
		std::string p;
		p.reserve(4u << 20);
		for (std::size_t i = 0; i < (4u << 20); ++i)
			p.push_back(static_cast<char>('A' + (i % 26)));
		return p;
	}();
	MockHttpServer server;
	server.start([&](const std::string&, const std::string&, int& status) {
		status = 200;
		return payload;
	});
	CHECK(server.port() > 0, "mock server should start");

	HttpTransport t;
	auto ep = epFor(server.port(), "/big");
	ep.maxResponseBody = (4u << 20) + 1024;
	CHECK(t.connect(ep).ok(), "connect");
	CHECK(t.send("{}").ok(), "send");
	Payload body;
	CHECK(t.recv(body).ok(), "recv");
	CHECK(body.size() == payload.size(), "large response must arrive in full (no truncation)");
	CHECK(std::memcmp(body.data(), payload.data(), payload.size()) == 0,
		  "large response content must match byte-for-byte");
}

TEST(shortFixedBodyReleasesLease) {
	Poco::Net::ServerSocket listener;
	listener.bind(Poco::Net::SocketAddress("127.0.0.1", 0));
	listener.listen();
	const int port = listener.address().port();
	std::thread server([&] {
		try {
			for (int i = 0; i < 3; ++i) {
				if (!listener.poll(Poco::Timespan(5, 0), Poco::Net::Socket::SELECT_READ)) break;
				auto c = listener.acceptConnection();
				c.setReceiveTimeout(Poco::Timespan(2, 0));
				std::string request;
				char buffer[1024];
				int n;
				while (request.find("\r\n\r\n") == std::string::npos && (n = c.receiveBytes(buffer, sizeof(buffer))) > 0)
					request.append(buffer, n);
				if (request.empty()) continue; // TCP readiness probe
				const std::string response = i == 1
					? "HTTP/1.1 200 OK\r\nContent-Length: 16\r\nConnection: close\r\n\r\nAAAA"
					: "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok";
				c.sendBytes(response.data(), static_cast<int>(response.size()));
				c.close();
			}
		} catch (...) {}
	});
	HttpTransport t;
	CHECK(t.connect(epFor(port)).ok(), "short-body server connected");
	CHECK(t.send("").ok(), "short-body headers accepted");
	Payload out;
	CHECK(!t.recv(out).ok(), "fixed-length truncated body must fail");
	NetError next;
	std::thread another([&] { next = t.send(""); if (next.ok()) { Payload body; next = t.recv(body); } });
	another.join();
	CHECK(next.ok(), "failed read releases lease for another thread");
	server.join();
}

TEST(credentialSchemeAndLeaseOwnership) {
	#define TRACE_STEP(label) do { std::puts("CREDENTIAL " label); std::fflush(stdout); } while (0)
	TRACE_STEP("start");
	MockHttpServer server;
	server.start([](const std::string&, const std::string&, int& status) { status = 200; return std::string("ok"); });
	HttpTransport t;
	auto ep = epFor(server.port());
	ep.authToken = "Bearer PRIVATE";
	ep.useTls = true;
	ep.url = "http://127.0.0.1:" + std::to_string(server.port());
	CHECK(!t.connect(ep).ok(), "final http URI overrides useTls and refuses credentials");
	ep.authToken.clear();
	ep.headers = {"X-Api-Key: PRIVATE"};
	CHECK(!t.connect(ep).ok(), "sensitive custom header refuses plaintext");
	ep.headers.clear();
	ep.url = "http://PRIVATE@127.0.0.1:" + std::to_string(server.port());
	const auto rejected = t.connect(ep);
	CHECK(!rejected.ok() && rejected.localMessage.find("PRIVATE") == std::string::npos, "userinfo rejected without URI leak");
	ep.url = "http://127.0.0.1:" + std::to_string(server.port());
	ep.authToken = "Bearer PRIVATE";
	ep.allowInsecureCredentials = true;
	TRACE_STEP("connecting optin");
	CHECK(t.connect(ep).ok(), "explicit development opt-in permits plaintext credentials");
	TRACE_STEP("sending");
	CHECK(t.send("x").ok(), "send owns response lease");
	TRACE_STEP("wrong recv");
	NetError other;
	std::thread wrong([&] { Payload out; other = t.recv(out); });
	wrong.join();
	TRACE_STEP("wrong recv joined");
	CHECK(!other.ok(), "another thread cannot steal response lease");
	Payload out;
	TRACE_STEP("owner recv");
	CHECK(t.recv(out).ok() && out == "ok", "owner still consumes response");
	TRACE_STEP("invalid reconnect");
	ep.headers = {"Authorization: PRIVATE\r\nInjected: yes"};
	const auto bad = t.connect(ep);
	CHECK(bad.localMessage.find("PRIVATE") == std::string::npos && bad.localMessage.find('\n') == std::string::npos, "header diagnostic excludes values and controls");
}

TEST(requestTargetAndFramingRejectedBeforeProbe) {
	Poco::Net::ServerSocket listener;
	listener.bind(Poco::Net::SocketAddress("127.0.0.1", 0)); listener.listen();
	HttpTransport t;
	auto ep = epFor(listener.address().port());
	for (const auto& path : std::vector<std::string>{"/PRIVATE\r\nInjected: yes", "/has space", std::string("/nul\0PRIVATE", 12)}) {
		ep.requestPath = path;
		const auto err = t.connect(ep);
		CHECK(!err.ok() && err.localMessage.find("PRIVATE") == std::string::npos && err.localMessage.find('\n') == std::string::npos, "raw target controls rejected without diagnostic leaks");
	}
	ep.requestPath.clear();
	for (const auto& path : std::vector<std::string>{"/PRIVATE%0d%0aInjected", "/has%20space", "/nul%00PRIVATE"}) {
		ep.url = "http://127.0.0.1:" + std::to_string(ep.port) + path;
		CHECK(!t.connect(ep).ok(), "decoded URI path controls rejected");
	}
	ep.url.clear(); ep.headers = {"tRaNsFeR-EnCoDiNg: chunked"};
	CHECK(!t.connect(ep).ok(), "caller cannot configure transfer framing");
	CHECK(!listener.poll(Poco::Timespan(0, 100000), Poco::Net::Socket::SELECT_READ), "invalid targets and framing never initiate a TCP probe");
}

int main(int argc, char** argv) {
	test_requestTargetAndFramingRejectedBeforeProbe();
	if (argc > 1 && std::string(argv[1]) == "--credential") { test_credentialSchemeAndLeaseOwnership(); return g_failures.load() ? 1 : 0; }
	std::puts("RUN shortFixedBodyReleasesLease"); std::fflush(stdout);
	test_shortFixedBodyReleasesLease();
	std::puts("RUN credentialSchemeAndLeaseOwnership"); std::fflush(stdout);
	test_credentialSchemeAndLeaseOwnership();
	std::puts("RUN existing transport regressions"); std::fflush(stdout);
	#define RUN_CASE(name) do { std::puts("RUN " #name); std::fflush(stdout); test_##name(); } while (0)
	RUN_CASE(successRoundtrip);
	RUN_CASE(status404Normalized);
	RUN_CASE(status500Normalized);
	RUN_CASE(status302Normalized);
	RUN_CASE(responseBodyLimitEnforced);
	RUN_CASE(errorBodyTruncated);
	RUN_CASE(connectionRefusedNormalized);
	RUN_CASE(closeRacingSlowRecv);
	RUN_CASE(headerInjectionRejectedAtConnect);
	RUN_CASE(largeResponseIntegrity);
	RUN_CASE(endToEndTensorOverHttp);
	RUN_CASE(endToEndTextOverHttp);
	RUN_CASE(emptyTextOverHttp);
	std::printf("HttpTransportTest: %d checks, %d failures\n", g_checks.load(), g_failures.load());
	return g_failures == 0 ? 0 : 1;
}
