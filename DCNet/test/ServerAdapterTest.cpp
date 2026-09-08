// ServerAdapter 端到端测试（M-server 节点服务化）：真实 HTTP——本地监听端 + 本地出站
//
// 覆盖（DESIGN.md §3.6 / §6.1）：
//   1. 语义一致性：远程驱动本地节点 == 本地执行（status + 输出值逐项比对）；
//   2. 错误归一化：401（鉴权）/ 415（wire 级垃圾报文）/ 404 / 429（过载）
//      → 对端经 normalizeHttpResponse 的归一化结果符合 §6.1 逆向映射表；
//   3. 跨平台：POCO 单一实现（同 HttpTransportTest）；
//   4. schema 违例一致性：类型不符输入分走「本地执行 / 远程驱动」两条路径，
//      两者均拒绝该输入（远程归一化为 InvalidInput；本地在 ValidatorRegistry
//      拒绝）。

#include "DCNet/DcNetHttp.h"
#include "DCNet/NetCodec_Tensor.h"
#include "DCNet/NetError.h"
#include "DCNet/NetServerAdapter.h"
#include "DCNet/NetTransport_Http.h"
#include "EngineRegistry.h"
#include "Node.h"
#include "NodeException.h"
#include "Tensor.hpp"

#include "NetBase64.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

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

// ── 测试引擎：Float "data" → "result"（值翻倍），RunFn 可注入延迟（429 用例）──

static std::atomic<int> g_slowMs{0};

static Node::Schema doublerSchema() {
	Node::Schema s;
	s.inputs = {NodePort::in<float>("data")};
	s.outputs = {NodePort::out<float>("result")};
	return s;
}

static Node::RunFn doublerRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		if (const int slow = g_slowMs.load(); slow > 0)
			std::this_thread::sleep_for(std::chrono::milliseconds(slow));
		const auto& v = ctx.peek("data");
		const auto* t = v.as<Tensor>();
		if (!t)
			return ctx.failure(Node::Status::InvalidInput, "input 'data' is not a tensor");
		if (t->type() != Tensor::TensorType::Float)
			return ctx.failure(Node::Status::InvalidInput, "port 'data' expects Float tensor");
		Tensor src = *t; // getData 非 const 接口，先拷贝
		const auto vals = src.getData<float>();
		std::vector<float> out;
		out.reserve(vals.size());
		for (float f : vals)
			out.push_back(f * 2.0f);
		Tensor::DataBlock block(out.size() * sizeof(float));
		if (!out.empty())
			std::memcpy(block.data(), out.data(), block.size());
		ctx.output("result", Value(std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float), src.shape(),
															std::move(block))));
		return ctx.success();
	};
}

static constexpr const char* kEngineType = "TestDoubler";
static constexpr const char* kModelRef = "test-model";

static void ensureDoublerEngine() {
	auto& reg = EngineRegistry::instance();
	if (reg.hasEngine(kEngineType))
		return;
	EngineDescriptor ed;
	ed.engineType = kEngineType;
	ed.createEngine = [](const std::string&) -> EngineInstance {
		return EngineInstance(std::make_shared<int>(0)); // 占位运行时对象（无状态引擎）
	};
	ed.getInputPorts = [](const EngineInstance&) { return doublerSchema().inputs; };
	ed.getOutputPorts = [](const EngineInstance&) { return doublerSchema().outputs; };
	ed.factory = [](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		auto node = std::make_unique<Node>(kEngineType, p.nodeName, p.schema, doublerRunFn(),
										   ThreadPoolAffinity::System);
		if (p.engineInstance)
			node->bindEngine(p.engineInstance, p.engineInstance->descriptor());
		return node;
	};
	reg.registerEngine(ed);
}

// ── 工具 ──

static NetEndpoint epFor(int port, std::string requestPath = "/infer") {
	NetEndpoint ep;
	ep.host = "127.0.0.1";
	ep.port = port;
	ep.basePath = "/v1";
	ep.requestPath = std::move(requestPath);
	ep.connectTimeout = std::chrono::milliseconds(2000);
	ep.requestTimeout = std::chrono::milliseconds(5000);
	return ep;
}

static Tensor makeFloatTensor(const std::vector<float>& vals) {
	Tensor::DataBlock block(vals.size() * sizeof(float));
	if (!vals.empty())
		std::memcpy(block.data(), vals.data(), vals.size() * sizeof(float));
	return Tensor(Tensor::TensorType::Float, sizeof(float), {static_cast<int64_t>(vals.size())}, std::move(block));
}

static Tensor makeTextTensor(const std::string& s) {
	Tensor::DataBlock block(s.size());
	if (!s.empty())
		std::memcpy(block.data(), s.data(), s.size());
	return Tensor(Tensor::TensorType::Data, 1, {static_cast<int64_t>(s.size())}, std::move(block));
}

static std::string floatPayload(const std::vector<float>& vals) {
	const Tensor t = makeFloatTensor(vals);
	nlohmann::json j;
	j["dtype"] = "float32";
	j["shape"] = t.shape();
	const auto bytes = t.bytes();
	j["data"] = DC::Net::detail::base64Encode(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
	return j.dump();
}

static std::string textPayload(const std::string& s) {
	nlohmann::json j;
	j["dtype"] = "text";
	j["shape"] = std::vector<int64_t>{static_cast<int64_t>(s.size())};
	j["data"] = s;
	return j.dump();
}

static void ensureOutbound() {
	registerDcNetHttp(EngineRegistry::instance(), makeTensorJsonCodec());
}

struct ServerHandle {
	std::shared_ptr<DcNetServerService> svc;
	int port = -1;
};

static ServerHandle startServer(std::string authToken = {}, int maxInFlight = 8) {
	DcNetServerAdapterDesc desc;
	desc.engineType = kEngineType;
	desc.localModelRef = kModelRef;
	desc.codec = makeTensorJsonServerCodec("data", "result");
	desc.endpoint.listenHost = "127.0.0.1";
	desc.endpoint.port = 0; // 随机端口
	desc.endpoint.basePath = "/v1";
	desc.endpoint.authToken = std::move(authToken);
	desc.endpoint.maxInFlight = maxInFlight;
	auto svc = registerDcNetServerAdapter(EngineRegistry::instance(), std::move(desc));
	return {svc, svc->port()};
}

static Node::Result runLocalDoubler(const std::string& taskId, std::unordered_map<std::string, Tensor> inputs,
									std::unordered_map<std::string, Tensor>& outputs) {
	auto node =
		EngineRegistry::instance().createNode(std::string(kEngineType), "local-" + taskId, std::string(kModelRef));
	if (!node)
		throw std::runtime_error("local node create failed");
	node->setInput(taskId, std::move(inputs));
	auto result = node->tryExecute(taskId);
	if (result.ok())
		outputs = node->collectOutputTensors(taskId);
	return result;
}

// ── 语义一致性对拍：远程驱动 == 本地执行 ──

TEST(paritySuccess) {
	ensureDoublerEngine();
	ensureOutbound();
	auto srv = startServer();
	CHECK(srv.svc->alive(), "service alive after register");
	CHECK(srv.port > 0, "actual port bound");

	std::unordered_map<std::string, Tensor> localOut;
	auto localRes = runLocalDoubler("p1", {{"data", makeFloatTensor({1.0f, 2.0f, 3.0f})}}, localOut);
	CHECK(localRes.ok(), "local baseline ok");
	CHECK(localOut.count("result") == 1, "local result exists");

	auto node = EngineRegistry::instance().createNode(
		"DCNet.Tensor", "remote-p1", std::string("http://127.0.0.1:" + std::to_string(srv.port) + "/v1"));
	CHECK(node != nullptr, "remote node created");
	node->setInput("p1", "data", makeFloatTensor({1.0f, 2.0f, 3.0f}));
	auto remoteRes = node->tryExecute("p1");
	CHECK(remoteRes.ok(), "remote drive ok");
	CHECK(node->hasOutput("p1", "result"), "remote result exists");

	const auto localVals = localOut["result"].getData<float>();
	const auto remoteVals = node->takeOutputTensor("p1", "result").getData<float>();
	CHECK(localVals == remoteVals, "parity: output values identical");
	CHECK(remoteVals.size() == 3 && remoteVals[0] == 2.0f && remoteVals[1] == 4.0f && remoteVals[2] == 6.0f,
		  "remote values doubled");
	srv.svc->stop();
}

// ── 鉴权闸门：无 token → 401 → RemoteAuth → InternalError ──
//    （鉴权无本地对应物，不参与「远程 == 本地」对拍，见 DESIGN.md §6.1）

TEST(authGate) {
	ensureDoublerEngine();
	ensureOutbound();
	auto srv = startServer("s3cret");

	auto node = EngineRegistry::instance().createNode(
		"DCNet.Tensor", "remote-auth", std::string("http://127.0.0.1:" + std::to_string(srv.port) + "/v1"));
	node->setInput("a1", "data", makeFloatTensor({1.0f}));
	auto res = node->tryExecute("a1");
	CHECK(!res.ok(), "request without token should fail");
	CHECK(res.status == Node::Status::InternalError, "401 → InternalError");
	CHECK_MSG_PREFIX(res.message, "remote:auth");

	// 携带 token（transport 直连注入 Authorization 头）→ 成功
	HttpTransport t;
	NetEndpoint ep = epFor(srv.port);
	ep.authToken = "Bearer s3cret";
	auto err = t.connect(ep);
	CHECK(err.ok(), "connect ok");
	err = t.send(floatPayload({1.0f}));
	CHECK(err.ok(), "send ok");
	Payload resp;
	err = t.recv(resp);
	CHECK(err.ok(), "authorized request should succeed");
	t.close();
	srv.svc->stop();
}

// ── wire 级垃圾报文：415 → RemoteMalformed → RemoteMalformed（DESIGN.md §6.1）──

TEST(malformedFrame) {
	ensureDoublerEngine();
	auto srv = startServer();
	HttpTransport t;
	t.connect(epFor(srv.port));
	// 非 2xx 错误由 send() 返回（含状态行接收）；recv 仅读 2xx 响应体
	auto err = t.send("this is not json at all");
	CHECK(!err.ok(), "malformed frame should fail");
	CHECK(err.localStatus == Node::Status::RemoteMalformed, "415 → RemoteMalformed → RemoteMalformed");
	CHECK_MSG_PREFIX(err.localMessage, "remote:malformed");
	t.close();
	srv.svc->stop();
}

// ── schema 违例一致性：类型违例输入在两条路径均被拒绝 ──
//
// 远程腿：报文可解析但类型不符本地形状规则 → 服务端输入边界校验（镜像出站
// decodeResponse 职责）→ 400 → 对端 RemoteRejected → InvalidInput（DESIGN.md
// §6.1）。
// 本地腿：同一违例输入经节点管线在 ValidatorRegistry 拒绝（tryExecute 抛
// NodeException，图级记录错误、任务失败）——「输入被拒」语义对齐；wire 面按
// §6.1 统一为 InvalidInput（若本地改产 SchemaMismatch 状态，映射无需变更）。

TEST(paritySchemaViolation) {
	ensureDoublerEngine();
	auto srv = startServer();

	// ① 本地执行：Data 文本张量注入 Float 端口 → 管线校验拒绝（异常）
	{
		auto node = EngineRegistry::instance().createNode(std::string(kEngineType), "local-p5",
															  std::string(kModelRef));
		node->setInput("p5", "data", makeTextTensor("hi"));
		bool rejected = false;
		try {
			node->tryExecute("p5");
		} catch (const NodeException& e) {
			rejected = true;
			CHECK(e.getErrorType() == NodeException::ErrorType::TypeMismatch,
				  "local: type violation aborts at ValidatorRegistry");
		}
		CHECK(rejected, "local: violating input is rejected");
	}

	// ② 远程驱动：同一违例输入（text dtype 报文）→ 400 → InvalidInput
	HttpTransport t;
	t.connect(epFor(srv.port));
	auto remoteErr = t.send(textPayload("hi"));
	t.close();
	CHECK(!remoteErr.ok(), "remote: violating input rejected");
	CHECK(remoteErr.localStatus == Node::Status::InvalidInput, "remote: 400 → InvalidInput（DESIGN.md §6.1 schema 违例）");
	CHECK_MSG_PREFIX(remoteErr.localMessage, "remote:invalid_input");
	srv.svc->stop();
}

// ── 过载限流：maxInFlight=1 + 引擎延迟 → 并发第二请求 429 ──

TEST(rateLimited) {
	ensureDoublerEngine();
	auto srv = startServer({}, /*maxInFlight*/ 1);
	g_slowMs = 300;

	Payload respA;
	std::thread ta([&] {
		HttpTransport a;
		a.connect(epFor(srv.port));
		a.send(floatPayload({1.0f}));
		a.recv(respA); // 占满唯一在途名额（300ms）
		a.close();
	});
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	HttpTransport b;
	b.connect(epFor(srv.port));
	auto errB = b.send(floatPayload({2.0f})); // 请求已完整读入 → 在途闸门 → 429
	b.close();
	ta.join();
	g_slowMs = 0;

	CHECK(errB.localStatus == Node::Status::ExecutionFailed, "429 → RemoteRateLimited → ExecutionFailed");
	CHECK(errB.retryable, "429 retryable");
	CHECK_MSG_PREFIX(errB.localMessage, "remote:rate_limited");
	srv.svc->stop();
}

// ── 未知路径：404 → RemoteRejected → InvalidInput（remote:not_found）──

TEST(unknownPath404) {
	ensureDoublerEngine();
	auto srv = startServer();
	HttpTransport t;
	t.connect(epFor(srv.port, "/nope"));
	auto err = t.send(floatPayload({1.0f}));
	CHECK(!err.ok(), "404 → failure");
	CHECK(err.localStatus == Node::Status::InvalidInput, "404 → InvalidInput");
	CHECK_MSG_PREFIX(err.localMessage, "remote:not_found");
	t.close();
	srv.svc->stop();
}

// ── 生命周期：stop → 健康镜像翻转；重复 stop 安全 ──

TEST(lifecycle) {
	ensureDoublerEngine();
	auto srv = startServer();
	CHECK(srv.svc->alive(), "service alive");
	srv.svc->stop();
	CHECK(!srv.svc->alive(), "service not alive after stop");
	srv.svc->stop(); // 重复调用安全
}

// ── 配置期出口（ADR-7）：未知 engineType → registerDcNetServerAdapter 抛 NodeException ──

TEST(configErrorsThrow) {
	DcNetServerAdapterDesc desc;
	desc.engineType = "NoSuchEngine";
	desc.localModelRef = kModelRef;
	desc.codec = makeTensorJsonServerCodec();
	bool threw = false;
	try {
		registerDcNetServerAdapter(EngineRegistry::instance(), std::move(desc));
	} catch (const NodeException&) {
		threw = true;
	}
	CHECK(threw, "unknown engineType → NodeException at config period");
}

int main() {
	struct Case {
		const char* name;
		void (*fn)();
	};
	const Case cases[] = {{"paritySuccess", test_paritySuccess},
						  {"authGate", test_authGate},
						  {"malformedFrame", test_malformedFrame},
						  {"paritySchemaViolation", test_paritySchemaViolation},
						  {"rateLimited", test_rateLimited},
						  {"unknownPath404", test_unknownPath404},
						  {"lifecycle", test_lifecycle},
						  {"configErrorsThrow", test_configErrorsThrow}};
	for (const auto& c : cases) {
		std::printf("RUN %s\n", c.name);
		std::fflush(stdout);
		c.fn();
		std::printf("OK  %s\n", c.name);
		std::fflush(stdout);
	}
	std::printf("ServerAdapterTest: %d checks, %d failures\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}
