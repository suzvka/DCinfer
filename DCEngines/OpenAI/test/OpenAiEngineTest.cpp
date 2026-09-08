// DCEngine_OpenAI 集成测试（MockHttpServer 假远端，真实 HTTP 传输）
//
// 覆盖：
//   - 端到端：OpenAI 节点（chat codec）走 /chat/completions 往返
//     （prompt/system/params → response；model / stream / messages 断言）
//   - 失败路径：远端 500 → NetError 归一化（ExecutionFailed / remote:server_error）
//
// 迁移自 DCNet/test/HttpTransportTest.cpp 的 chat 端到端（2026-08）。

#include "DCEngine/OpenAiEngine.h"
#include "Node.h"
#include "Tensor.hpp"

#include "DCNet/MockServer.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

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

// ── 工具 ──

static Tensor makeTextTensor(const std::string& s) {
	Tensor::DataBlock block(s.size());
	if (!s.empty())
		std::memcpy(block.data(), s.data(), s.size());
	return Tensor(Tensor::TensorType::Data, 1, {static_cast<int64_t>(s.size())}, std::move(block));
}

// ── 测试 ──

TEST(chatRoundtrip) {
	MockHttpServer server;
	server.start([&](const std::string& path, const std::string& body, int& status) {
		if (path != "/v1/chat/completions") {
			status = 404;
			return std::string(R"({"error":"not found"})");
		}
		const auto j = nlohmann::json::parse(body);
		CHECK(j["model"] == "mnist-bot", "server sees model");
		CHECK(j["stream"] == false, "server sees stream=false");
		CHECK(j["messages"].size() == 2, "server sees system + user messages");
		CHECK(j["messages"][0]["role"] == "system" && j["messages"][0]["content"] == "be brief",
			  "server sees system prompt");
		CHECK(j["messages"][1]["role"] == "user" && j["messages"][1]["content"] == "hello",
			  "server sees user prompt");
		CHECK(j["temperature"] == 0.7, "server sees params override");
		status = 200;
		return std::string(R"({"choices":[{"message":{"role":"assistant","content":"hi there"}}]})");
	});

	auto& reg = EngineRegistry::instance();
	DC::OpenAI::registerOpenAiEngine(reg, {.model = "mnist-bot"});

	auto node = reg.createNode("OpenAI", "chatNode",
							   std::string("http://127.0.0.1:" + std::to_string(server.port()) + "/v1"));
	CHECK(node != nullptr, "OpenAI node should be created");
	CHECK(node->schema().inputs[0].name == "prompt" && node->schema().outputs[0].name == "response",
		  "local shape rules from chat codec");

	node->setInput("t1", "system", makeTextTensor("be brief"));
	node->setInput("t1", "prompt", makeTextTensor("hello"));
	node->setInput("t1", "params", makeTextTensor(R"({"temperature":0.7})"));
	auto result = node->tryExecute("t1");
	CHECK(result.ok(), "chat roundtrip should succeed");
	CHECK(node->hasOutput("t1", "response"), "response output should exist");
	auto out = node->takeOutputTensor("t1", "response");
	CHECK(out.type() == Tensor::TensorType::Data && out.typeSize() == 1, "response tensor type/size");
	auto bytes = out.bytes();
	CHECK(std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()) == "hi there",
		  "chat response content");
}

TEST(remoteServerErrorNormalized) {
	MockHttpServer server;
	server.start([](const std::string&, const std::string&, int& status) {
		status = 500;
		return std::string(R"({"error":{"code":"server_error","message":"boom"}})");
	});

	auto& reg = EngineRegistry::instance();
	// 同进程已注册 "OpenAI"（chatRoundtrip，保留首次）：此处沿用，不影响本用例
	DC::OpenAI::registerOpenAiEngine(reg, {});

	auto node = reg.createNode("OpenAI", "errNode",
							   std::string("http://127.0.0.1:" + std::to_string(server.port()) + "/v1"));
	CHECK(node != nullptr, "OpenAI error node should be created");

	node->setInput("t1", "prompt", makeTextTensor("hello"));
	auto result = node->tryExecute("t1");
	CHECK(!result.ok(), "500 → failure");
	CHECK(result.status == Node::Status::ExecutionFailed, "500 → ExecutionFailed");
	CHECK_MSG_PREFIX(result.message, "remote:server_error");
}

// ── 鉴权：Bearer Token 注入 Authorization 头（P0：云 API 接入）──
TEST(bearerTokenInjected) {
	MockHttpServer server;
	server.start([&](const std::string& path, const std::string&, int& status) {
		if (path != "/v1/chat/completions") {
			status = 404;
			return std::string("{}");
		}
		status = 200;
		return std::string(R"({"choices":[{"message":{"content":"ok"}}]})");
	});

	auto& reg = EngineRegistry::instance();
	// 注册保留首次：鉴权变体用独立 engineType，避免与 chatRoundtrip 的 "OpenAI" 冲突
	DC::OpenAI::registerOpenAiEngine(reg, {.model = "m", .engineType = "OpenAI.Auth", .authToken = "sk-test-123"});

	auto node = reg.createNode("OpenAI.Auth", "authNode",
							   std::string("http://127.0.0.1:" + std::to_string(server.port()) + "/v1"));
	CHECK(node != nullptr, "auth node should be created");
	node->setInput("t1", "prompt", makeTextTensor("hi"));
	auto result = node->tryExecute("t1");
	CHECK(result.ok(), "request with bearer token should succeed");
	CHECK(server.lastRequestHeaders().find("Authorization: Bearer sk-test-123") != std::string::npos,
		  "Authorization: Bearer <token> should be injected");
	// 敏感信息不回显：失败/错误路径均不包含 token（此处仅验证正常路径头注入）
}

// ── 输入错误不再被吞掉：非法 params JSON → InvalidInput ──
TEST(invalidParamsRejected) {
	MockHttpServer server;
	server.start([&](const std::string&, const std::string&, int& status) {
		status = 200;
		return std::string(R"({"choices":[{"message":{"content":"never"}}]})");
	});

	auto& reg = EngineRegistry::instance();
	DC::OpenAI::registerOpenAiEngine(reg, {.model = "m"});
	auto node = reg.createNode("OpenAI", "badParamsNode",
							   std::string("http://127.0.0.1:" + std::to_string(server.port()) + "/v1"));
	CHECK(node != nullptr, "node should be created");
	node->setInput("t1", "prompt", makeTextTensor("hi"));
	node->setInput("t1", "params", makeTextTensor("{not-json"));
	auto result = node->tryExecute("t1");
	CHECK(!result.ok(), "invalid params JSON should fail");
	CHECK(result.status == Node::Status::InvalidInput, "invalid params → InvalidInput");
}

// ── 响应结构异常不再被吞掉：缺 content / 非 JSON → ExecutionFailed + dcnet 诊断 ──
TEST(malformedResponseRejected) {
	MockHttpServer server;
	server.start([&](const std::string&, const std::string&, int& status) {
		status = 200;
		return std::string("not-json-at-all");
	});

	auto& reg = EngineRegistry::instance();
	DC::OpenAI::registerOpenAiEngine(reg, {.model = "m"});
	auto node = reg.createNode("OpenAI", "malformedNode",
							   std::string("http://127.0.0.1:" + std::to_string(server.port()) + "/v1"));
	node->setInput("t1", "prompt", makeTextTensor("hi"));
	auto result = node->tryExecute("t1");
	CHECK(!result.ok(), "non-JSON response should fail");
	CHECK(result.status == Node::Status::ExecutionFailed, "non-JSON → ExecutionFailed（核心枚举保持通用）");
	CHECK(result.diagnostic.has_value(), "non-JSON → 附带领域诊断");
	CHECK(result.diagnostic->domain == "dcnet", "诊断 domain=dcnet");

	// 缺 choices[0].message.content（协议漂移）→ ExecutionFailed（dcnet 诊断 code=RemoteMalformed），
	// 而非空字符串成功
	MockHttpServer server2;
	server2.start([&](const std::string&, const std::string&, int& status) {
		status = 200;
		return std::string(R"({"choices":[{"message":{"role":"assistant"}}]})");
	});
	auto node2 = reg.createNode("OpenAI", "missingContentNode",
								std::string("http://127.0.0.1:" + std::to_string(server2.port()) + "/v1"));
	node2->setInput("t1", "prompt", makeTextTensor("hi"));
	auto result2 = node2->tryExecute("t1");
	CHECK(!result2.ok(), "missing content should fail");
	CHECK(result2.status == Node::Status::ExecutionFailed, "missing content → ExecutionFailed");
	CHECK(result2.diagnostic.has_value() && result2.diagnostic->domain == "dcnet",
		  "missing content → dcnet 领域诊断");
}

// ── 合法空内容与字段缺失严格区分：content="" 成功返回 ──
TEST(emptyContentSucceeds) {
	MockHttpServer server;
	server.start([&](const std::string&, const std::string&, int& status) {
		status = 200;
		return std::string(R"({"choices":[{"message":{"content":""}}]})");
	});

	auto& reg = EngineRegistry::instance();
	DC::OpenAI::registerOpenAiEngine(reg, {.model = "m"});
	auto node = reg.createNode("OpenAI", "emptyContentNode",
							   std::string("http://127.0.0.1:" + std::to_string(server.port()) + "/v1"));
	node->setInput("t1", "prompt", makeTextTensor("hi"));
	auto result = node->tryExecute("t1");
	CHECK(result.ok(), "empty content is a legitimate response");
	auto out = node->takeOutputTensor("t1", "response");
	CHECK(out.bytes().empty(), "response should be empty string");
}

// ── maxRetries：传输级失败退避重试（500 一次 → 成功）──
TEST(transportRetryOnServerError) {
	MockHttpServer server;
	std::atomic<int> calls{0};
	server.start([&](const std::string&, const std::string&, int& status) {
		if (calls.fetch_add(1) == 0) {
			status = 500; // 第一次失败（可重试）
			return std::string(R"({"error":{"code":"server_error","message":"boom"}})");
		}
		status = 200;
		return std::string(R"({"choices":[{"message":{"content":"recovered"}}]})");
	});

	auto& reg = EngineRegistry::instance();
	// 注册保留首次：重试变体用独立 engineType，确保 maxRetries 生效
	DC::OpenAI::registerOpenAiEngine(reg, {.model = "m", .engineType = "OpenAI.Retry", .maxRetries = 1});
	auto node = reg.createNode("OpenAI.Retry", "retryNode",
							   std::string("http://127.0.0.1:" + std::to_string(server.port()) + "/v1"));
	node->setInput("t1", "prompt", makeTextTensor("hi"));
	auto result = node->tryExecute("t1");
	CHECK(result.ok(), "retry after 500 should succeed");
	CHECK(calls.load() == 2, "server should have seen exactly 2 attempts");
	auto out = node->takeOutputTensor("t1", "response");
	auto bytes = out.bytes();
	CHECK(std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()) == "recovered",
		  "recovered response content");
}

int main() {
	test_chatRoundtrip();
	test_remoteServerErrorNormalized();
	test_bearerTokenInjected();
	test_invalidParamsRejected();
	test_malformedResponseRejected();
	test_emptyContentSucceeds();
	test_transportRetryOnServerError();
	std::printf("OpenAiEngineTest: %d checks, %d failures\n", g_checks.load(), g_failures.load());
	return g_failures == 0 ? 0 : 1;
}
