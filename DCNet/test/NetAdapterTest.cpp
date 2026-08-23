// DCNet 适配器契约测试（M1）：FakeTransport + EchoCodec 走完整
// EngineRegistry → createNode → setInput → tryExecute 路径，不依赖真实远端。

#include "DCNet/NetAdapter.h"
#include "DCNet/NetEndpoint.h"
#include "DCNet/NetTransport.h"
#include "DCNet/NetCodec.h"

#include "EngineRegistry.h"
#include "Node.h"
#include "NodeException.h"
#include "Tensor.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

static int g_checks = 0;
static int g_failures = 0;

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

// ── 文本张量辅助（TensorType::Data 约定，DESIGN.md §3.4 / FreeToken §7）──

static Tensor makeTextTensor(const std::string& s) {
	Tensor::DataBlock block(s.size());
	if (!s.empty())
		std::memcpy(block.data(), s.data(), s.size());
	return Tensor(Tensor::TensorType::Data, 1, {static_cast<int64_t>(s.size())}, std::move(block));
}

static std::string textOf(const Tensor& t) {
	auto bytes = t.bytes();
	return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

// ── EchoCodec：request 端口 → 报文；报文 → response 端口 ──

struct EchoCodec : DcNetCodec {
	Payload encodeRequest(const Node::RunContext& ctx) override {
		const auto& val = ctx.peek("request");
		const auto* t = val.as<Tensor>();
		return t ? textOf(*t) : Payload();
	}

	void decodeResponse(Payload& payload, Node::RunContext& ctx) override {
		ctx.output("response", Value(std::make_unique<Tensor>(makeTextTensor(payload))));
	}
};

// ── FakeTransport：行为可配置，不触网；按 endpoint 登记，测试可取回 ──

struct FakeTransport : DcNetTransport, std::enable_shared_from_this<FakeTransport> {
	static inline int instances = 0;
	/// endpoint() → 实例（connect 时登记；引擎实例缓存使同一端点复用同一 transport）
	static inline std::unordered_map<std::string, std::shared_ptr<FakeTransport>> byEndpoint;

	NetError connectResult;   // None = 成功
	NetError sendResult;      // None = 成功
	NetError recvResult;      // None = 成功
	Payload sentPayload;
	Payload response;
	NetEndpoint lastEndpoint;
	int connectCalls = 0;

	FakeTransport() { ++instances; }

	NetError connect(const NetEndpoint& ep) override {
		++connectCalls;
		lastEndpoint = ep;
		byEndpoint[ep.endpoint()] = shared_from_this();
		return connectResult;
	}
	NetError send(const Payload& p) override {
		sentPayload = p;
		return sendResult;
	}
	NetError recv(Payload& out) override {
		out = response;
		return recvResult;
	}
	bool alive() const override { return true; }
	void close() override {}
};

// ── 共享注册表 ──

static EngineRegistry& g_reg = EngineRegistry::instance();

static Node::Schema makeSchema() {
	Node::Schema s;
	s.inputs = {NodePort::in<std::vector<char>>("request")};
	s.outputs = {NodePort::out<std::vector<char>>("response")};
	return s;
}

static void registerFakeAdapter(const std::string& engineType, std::shared_ptr<FakeTransport> fixedTransport = {}) {
	DcNetAdapterDesc desc;
	desc.engineType = engineType;
	desc.schema = makeSchema();
	desc.transportFactory = [fixedTransport]() -> std::shared_ptr<DcNetTransport> {
		if (fixedTransport)
			return fixedTransport;
		return std::make_shared<FakeTransport>();
	};
	desc.codec = std::make_shared<EchoCodec>();
	registerDcNetAdapter(g_reg, std::move(desc));
}

// createNode 重载歧义规避：显式 std::string 第三参（避免字符串字面量匹配 const void* 重载）
static std::unique_ptr<Node> makeNetNode(const std::string& engineType, const std::string& name,
										 const std::string& endpoint) {
	return g_reg.createNode(engineType, name, endpoint);
}

// ── 测试 ──

TEST(endpointParse) {
	auto ep = NetEndpoint::parse("http://192.168.1.10:8080/v1");
	CHECK(ep.useTls == false, "http → no tls");
	CHECK(ep.host == "192.168.1.10", "host parsed");
	CHECK(ep.port == 8080, "port parsed");
	CHECK(ep.basePath == "/v1", "basePath parsed");
	CHECK(ep.endpoint() == "http://192.168.1.10:8080/v1", "endpoint roundtrip");

	ep = NetEndpoint::parse("https://api.example.com");
	CHECK(ep.useTls, "https → tls");
	CHECK(ep.host == "api.example.com", "bare host parsed");
	CHECK(ep.endpoint() == "https://api.example.com/v1", "default basePath appended");

	ep = NetEndpoint::parse("192.168.1.5:1919");
	CHECK(ep.host == "192.168.1.5", "bare host:port parsed");
	CHECK(ep.port == 1919, "bare port parsed");
	CHECK(ep.endpoint() == "http://192.168.1.5:1919/v1", "bare form endpoint");
}

TEST(createNodeSchemaAndAffinity) {
	registerFakeAdapter("Test.Net");
	auto node = makeNetNode("Test.Net", "n1", "http://127.0.0.1:8080/v1");
	CHECK(node != nullptr, "node should be created");
	CHECK(node->type() == "Test.Net", "node type should match engine type");
	CHECK(node->modelPath() == "http://127.0.0.1:8080/v1", "modelPath should carry endpoint");
	CHECK(node->affinity() == ThreadPoolAffinity::System, "affinity should be System (I/O pool)");
	CHECK(node->schema().inputs.size() == 1 && node->schema().inputs[0].name == "request",
		  "input schema from local rules");
	CHECK(node->schema().outputs.size() == 1 && node->schema().outputs[0].name == "response",
		  "output schema from local rules");
	auto t = FakeTransport::byEndpoint["http://127.0.0.1:8080/v1"];
	CHECK(t != nullptr, "transport should be registered for endpoint");
	CHECK(t->connectCalls == 1, "createEngine should connect once");
	CHECK(t->lastEndpoint.host == "127.0.0.1", "endpoint parsed and passed to transport");
	CHECK(t->lastEndpoint.port == 8080, "endpoint port passed to transport");
}

TEST(runFlowSuccess) {
	auto node = makeNetNode("Test.Net", "n2", "http://127.0.0.1:8080/v1");
	CHECK(node != nullptr, "node should be created");
	auto t = FakeTransport::byEndpoint["http://127.0.0.1:8080/v1"];
	t->response = "hello-from-remote";

	node->setInput("t1", "request", makeTextTensor("hello"));
	auto result = node->tryExecute("t1");
	CHECK(result.ok(), "run should succeed");
	CHECK(t->sentPayload == "hello", "encoded request should reach transport");
	CHECK(node->hasOutput("t1", "response"), "response output should exist");
	auto out = node->getOutputTensor("t1", "response");
	CHECK(textOf(out) == "hello-from-remote", "decoded response should match");
}

TEST(engineInstanceCachedPerEndpoint) {
	const int before = FakeTransport::instances;
	auto a = makeNetNode("Test.Net", "c1", "http://127.0.0.1:9000/v1");
	auto b = makeNetNode("Test.Net", "c2", "http://127.0.0.1:9000/v1");
	CHECK(a && b, "both nodes created");
	CHECK(FakeTransport::instances == before + 1, "same endpoint → single shared instance");

	auto c = makeNetNode("Test.Net", "c3", "http://127.0.0.1:9001/v1");
	CHECK(c != nullptr, "third node created");
	CHECK(FakeTransport::instances == before + 2, "different endpoint → new instance");
}

TEST(connectFailureThrowsAtCreate) {
	auto bad = std::make_shared<FakeTransport>();
	bad->connectResult = normalizeTransportError(NetTransportError::ConnectionRefused, "refused");
	registerFakeAdapter("Test.NetFail", bad);

	bool threw = false;
	std::string what;
	try {
		makeNetNode("Test.NetFail", "x", "http://127.0.0.1:1/v1");
	} catch (const NodeException& e) {
		threw = true;
		what = e.what();
	}
	CHECK(threw, "connect failure should throw NodeException at createNode");
	CHECK(what.find("connect") != std::string::npos, "exception should mention connect");
	CHECK(what.find("net:unreachable") != std::string::npos, "exception should carry normalized message");
}

TEST(sendFailureNormalized) {
	auto node = makeNetNode("Test.Net", "n3", "http://127.0.0.1:8080/v1");
	CHECK(node != nullptr, "node should be created");
	auto t = FakeTransport::byEndpoint["http://127.0.0.1:8080/v1"];
	t->sendResult = normalizeTransportError(NetTransportError::Timeout, "request timed out");

	node->setInput("t1", "request", makeTextTensor("hello"));
	auto result = node->tryExecute("t1");
	CHECK(!result.ok(), "send failure → run failure");
	CHECK(result.status == Node::Status::ExecutionFailed, "timeout → ExecutionFailed");
	CHECK_MSG_PREFIX(result.message, "net:timeout");
	CHECK(result.message.find("request timed out") != std::string::npos, "detail preserved in message");
}

TEST(recvFailureNormalized) {
	auto node = makeNetNode("Test.Net", "n4", "http://127.0.0.1:8080/v1");
	CHECK(node != nullptr, "node should be created");
	auto t = FakeTransport::byEndpoint["http://127.0.0.1:8080/v1"];
	t->sendResult = {};  // 重置（sendFailure 测试可能已污染共享实例）
	t->recvResult = normalizeHttpResponse(503, R"({"error":{"code":"server_error","message":"down"}})");

	node->setInput("t1", "request", makeTextTensor("hello"));
	auto result = node->tryExecute("t1");
	CHECK(!result.ok(), "recv failure → run failure");
	CHECK(result.status == Node::Status::ExecutionFailed, "5xx → ExecutionFailed");
	CHECK_MSG_PREFIX(result.message, "remote:server_error");
}

TEST(remoteRejectedMapsToInvalidInput) {
	auto node = makeNetNode("Test.Net", "n5", "http://127.0.0.1:8080/v1");
	CHECK(node != nullptr, "node should be created");
	auto t = FakeTransport::byEndpoint["http://127.0.0.1:8080/v1"];
	t->sendResult = {};  // 重置（sendFailure 测试可能已污染共享实例）
	t->recvResult = normalizeHttpStatus(400, R"({"error":"bad request"})");

	node->setInput("t1", "request", makeTextTensor("hello"));
	auto result = node->tryExecute("t1");
	CHECK(!result.ok(), "400 → run failure");
	CHECK(result.status == Node::Status::InvalidInput, "400 → InvalidInput");
	CHECK_MSG_PREFIX(result.message, "remote:invalid_request");
}

int main() {
	test_endpointParse();
	test_createNodeSchemaAndAffinity();
	test_runFlowSuccess();
	test_engineInstanceCachedPerEndpoint();
	test_connectFailureThrowsAtCreate();
	test_sendFailureNormalized();
	test_recvFailureNormalized();
	test_remoteRejectedMapsToInvalidInput();
	std::printf("NetAdapterTest: %d checks, %d failures\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}
