// HttpTransport + DCNet.Http 集成测试（MockHttpServer 假远端，真实 HTTP 传输）
//
// 覆盖（DESIGN.md §6）：
//   - 传输层：成功 2xx / 404 / 500 / 连接拒绝 / 超时 → NetError 归一化
//   - 端到端：DCNet.Http 节点（张量 JSON codec）走真实 HTTP 往返 + 张量还原
//   - 端到端：OpenAI chat codec 走 /chat/completions 往返

#include "DCNet/DcNetHttp.h"
#include "DCNet/NetError.h"
#include "DCNet/NetTransport_Http.h"
#include "Node.h"
#include "Tensor.hpp"

#include "MockServer.h"
#include "NetBase64.h"

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
		WSADATA wsa;
		WSAStartup(MAKEWORD(2, 2), &wsa);
		SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
		sockaddr_in a{};
		a.sin_family = AF_INET;
		a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
		a.sin_port = 0;
		bind(s, reinterpret_cast<sockaddr*>(&a), sizeof(a));
		sockaddr_in b{};
		int len = sizeof(b);
		getsockname(s, reinterpret_cast<sockaddr*>(&b), &len);
		deadPort = ntohs(b.sin_port);
		closesocket(s);
		WSACleanup();
	}

	HttpTransport t;
	t.connect(epFor(deadPort, "/infer"));
	Payload resp;
	auto err = t.send("x");
	CHECK(!err.ok(), "connection refused → failure");
	CHECK(err.retryable, "unreachable retryable");
	CHECK(err.localStatus == Node::Status::ExecutionFailed, "refused → ExecutionFailed");
	CHECK_MSG_PREFIX(err.localMessage, "net:unreachable");
	t.close();
}

// 注：WinHTTP 的 dwReceiveTimeout 不覆盖响应头等待（服务端延迟响应时
// WinHttpReceiveResponse 仍会等到首字节）；传输级超时语义属 OS 行为，
// 此处不设超时测试——Timeout 分类/映射已由 NetErrorTest 纯单测覆盖。

// ── 端到端：DCNet.Http 节点 + 张量 JSON codec（真实 HTTP 往返）──

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

	auto node = reg.createNode("DCNet.Http", "tensorNode",
							   std::string("http://127.0.0.1:" + std::to_string(server.port()) + "/v1"));
	CHECK(node != nullptr, "DCNet.Http node should be created");
	CHECK(node->schema().inputs[0].name == "data" && node->schema().outputs[0].name == "result",
		  "local shape rules from tensor codec");

	node->setInput("t1", "data", makeFloatTensor({1.0f, 2.0f}));
	auto result = node->tryExecute("t1");
	CHECK(result.ok(), "tensor roundtrip should succeed");
	CHECK(node->hasOutput("t1", "result"), "result output should exist");
	auto out = node->getOutputTensor("t1", "result");
	CHECK(out.type() == Tensor::TensorType::Float && out.typeSize() == sizeof(float),
		  "decoded tensor type/size");
	auto vals = out.getData<float>();
	CHECK(vals.size() == 2 && vals[0] == 1.0f && vals[1] == 2.0f, "decoded tensor values");
}

// ── 端到端：OpenAI chat codec ──

TEST(endToEndChatOverHttp) {
	MockHttpServer server;
	server.start([&](const std::string& path, const std::string& body, int& status) {
		if (path != "/v1/chat/completions") {
			status = 404;
			return R"({"error":"not found"})";
		}
		const auto j = nlohmann::json::parse(body);
		CHECK(j["model"] == "mnist-bot", "server sees model");
		CHECK(j["messages"].back()["content"] == "hello", "server sees prompt");
		status = 200;
		return R"({"choices":[{"message":{"role":"assistant","content":"hi there"}}]})";
	});

	auto& reg = EngineRegistry::instance();
	registerDcNetHttp(reg, makeChatCodec("mnist-bot"), {}, "DCNet.HttpChat");

	auto node = reg.createNode("DCNet.HttpChat", "chatNode",
							   std::string("http://127.0.0.1:" + std::to_string(server.port()) + "/v1"));
	CHECK(node != nullptr, "DCNet.Http chat node should be created");

	node->setInput("t1", "prompt", makeTextTensor("hello"));
	auto result = node->tryExecute("t1");
	CHECK(result.ok(), "chat roundtrip should succeed");
	CHECK(node->hasOutput("t1", "response"), "response output should exist");
	auto out = node->getOutputTensor("t1", "response");
	auto bytes = out.bytes();
	CHECK(std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()) == "hi there",
		  "chat response content");
}

int main() {
	test_successRoundtrip();
	test_status404Normalized();
	test_status500Normalized();
	test_connectionRefusedNormalized();
	test_endToEndTensorOverHttp();
	test_endToEndChatOverHttp();
	std::printf("HttpTransportTest: %d checks, %d failures\n", g_checks.load(), g_failures.load());
	return g_failures == 0 ? 0 : 1;
}
