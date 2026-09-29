// HttpTransport + DCNet.Tensor 集成测试（MockHttpServer 假远端，真实 HTTP 传输）
//
// 覆盖（DESIGN.md §6）：
//   - 传输层：成功 2xx / 404 / 500 / 连接拒绝 / 超时 → NetError 归一化
//   - 端到端：DCNet.Tensor 节点（张量 JSON codec，数值）真实 HTTP 往返 + 张量还原
//   - 端到端：DCNet.Tensor 节点（张量 JSON codec，Data 文本）真实 HTTP 往返 + 文本还原
// 注：OpenAI chat 端到端见 DCEngines/OpenAI（OpenAiEngineTest）。

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

// ── 工具 ──

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

// ── 测试 ──

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
	// 取一个已关闭端口：绑定后立即关闭
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
	// POCO/Windows：WSAPoll 不上报 connect 失败，带超时探测下拒绝连接表现为
	// Timeout（仍为可重试 ExecutionFailed）；POSIX 报 ConnectionRefused → unreachable。
	const bool unreachableOrTimeout = err.localMessage.rfind("net:unreachable", 0) == 0 ||
									 err.localMessage.rfind("net:timeout", 0) == 0;
	CHECK(unreachableOrTimeout, "refused → unreachable|timeout");
#else
	CHECK_MSG_PREFIX(err.localMessage, "net:unreachable");
#endif
	t.close();
}

// 注：POCO 传输级超时（connect/send/receive 独立设置）已覆盖响应头等待；
// Timeout 分类/映射已由 NetErrorTest 纯单测覆盖，此处不设服务端延迟超时用例。

// ── 3xx：不自动跟随重定向，按非 2xx 归一化（文档-实现一致）──

TEST(status302Normalized) {
	MockHttpServer server;
	server.start([](const std::string&, const std::string&, int& status) {
		status = 302; // 无 Location 自动跟随：按文档语义归一化为错误
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

// ── 响应体上限：成功体超限报错；非 2xx 错误体仅诊断允许截断 ──

TEST(responseBodyLimitEnforced) {
	MockHttpServer server;
	server.start([](const std::string&, const std::string&, int& status) {
		status = 200;
		return std::string("AAAAAAAAAAAAAAAA"); // 16 字节
	});
	HttpTransport t;
	auto ep = epFor(server.port(), "/infer");
	ep.maxResponseBody = 8; // 人为压低上限
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
		return std::string(4096, 'x'); // 4KB 非 JSON 错误体
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

// ── close 与慢 recv 竞速：强收中断读而非悬垂/挂死（副本保活 + abort）──

// 慢 body 测试服务器（仅本用例）：先发响应头，剩余 body 延迟发送——
// 构造“读持续超过 close 的 5s 强收窗口”的竞速场景
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
				// 连接 1 = connect() 的就绪探测 socket（连接后立即关闭，读即 EOF）；
				// 连接 2 = 真实交换。对两者统一读请求头，探测连接读到 EOF 后跳过。
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
						break; // 真实请求：进入慢响应
					// 探测连接（req 空）：丢弃并接受下一个连接
				}
				const std::string head = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
								 "Content-Length: 16\r\nConnection: close\r\n\r\n";
				c.sendBytes(head.data(), static_cast<int>(head.size()));
				c.sendBytes("AAAA", 4); // 先给 4 字节
				std::this_thread::sleep_for(std::chrono::milliseconds(8000)); // 剩余 12 字节延迟 8s
				c.sendBytes("AAAAAAAAAAAA", 12);
				c.close();
			} catch (...) {
				// 线程函数内异常不得逃逸（std::terminate）——对端 abort 后
				// 剩余 sendBytes 可能抛 ConnectionReset 等，吞掉即可
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
	ep.requestTimeout = std::chrono::milliseconds(15000); // 覆盖慢 body 窗口
	CHECK(t.connect(ep).ok(), "connect");
	CHECK(t.send("x").ok(), "send ok; response header received, body pending");

	// 另一线程 close：等交换收尾 5s 超时 → 强收 abort（读方副本保活 session）
	std::thread closer([&] { t.close(); });
	Payload resp;
	const auto t0 = std::chrono::steady_clock::now();
	const auto err = t.recv(resp); // 阻塞读 body → 被 abort 中断 → 错误返回
	const auto elapsed = std::chrono::steady_clock::now() - t0;
	closer.join();
	server.stop();
	CHECK(!err.ok(), "recv interrupted by racing close must return error (not a partial success)");
	CHECK(elapsed < std::chrono::seconds(7),
		  "abort must interrupt the read well before the server's remaining 8s delay");
	CHECK(resp.size() <= 16, "no garbage beyond declared body length");
	t.close(); // 幂等收尾（进程不崩溃即核心断言）
}

// ── 端到端：DCNet.Tensor 节点 + 张量 JSON codec（真实 HTTP 往返）──

TEST(endToEndTensorOverHttp) {
	MockHttpServer server;
	server.start([&](const std::string& path, const std::string& body, int& status) {
		if (path != "/v1/infer") {
			status = 404;
			return std::string(R"({"error":"not found"})");
		}
		// 解码请求张量，验证浮点值，原样回显（模拟远端推理）
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

// ── 端到端：DCNet.Tensor 节点 + Data 文本 codec（真实 HTTP 往返）──

TEST(endToEndTextOverHttp) {
	MockHttpServer server;
	server.start([&](const std::string& path, const std::string& body, int& status) {
		if (path != "/v1/infer") {
			status = 404;
			return std::string(R"({"error":"not found"})");
		}
		// 解码请求文本，验证 UTF-8 直传，原样回显（模拟远端推理）
		const auto j = nlohmann::json::parse(body);
		CHECK(j["dtype"] == "text", "server sees text dtype");
		CHECK(j["data"] == "hello", "server sees utf-8 text (not base64)");
		status = 200;
		nlohmann::json r;
		r["dtype"] = "text";
		r["shape"] = std::vector<int64_t>{8}; // "hi there" 长度
		r["data"] = "hi there";
		return r.dump();
	});

	auto& reg = EngineRegistry::instance();
	// 与 tensor 测试同进程：显式区分 engineType，避免注册表"保留首次"冲突
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

int main() {
	test_successRoundtrip();
	test_status404Normalized();
	test_status500Normalized();
	test_status302Normalized();
	test_responseBodyLimitEnforced();
	test_errorBodyTruncated();
	test_connectionRefusedNormalized();
	test_closeRacingSlowRecv();
	test_endToEndTensorOverHttp();
	test_endToEndTextOverHttp();
	std::printf("HttpTransportTest: %d checks, %d failures\n", g_checks.load(), g_failures.load());
	return g_failures == 0 ? 0 : 1;
}
