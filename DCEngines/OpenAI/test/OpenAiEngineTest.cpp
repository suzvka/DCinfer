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
	auto out = node->getOutputTensor("t1", "response");
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

int main() {
	test_chatRoundtrip();
	test_remoteServerErrorNormalized();
	std::printf("OpenAiEngineTest: %d checks, %d failures\n", g_checks.load(), g_failures.load());
	return g_failures == 0 ? 0 : 1;
}
