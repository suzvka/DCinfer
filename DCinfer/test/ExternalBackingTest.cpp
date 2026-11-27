// ExternalBacking 契约测试：mock 引擎以 host DataBlock 伪装引擎侧存储，
// 覆盖直传协商 / host 兜底 / 广播共享 + COW / 拷贝移动 / 冻结 / 并发导出 / 释放回调。
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "EngineRegistry.h"
#include "Node.h"
#include "Tensor.hpp"
#include "TensorException.h"
#include "TestHarness.h"

using namespace DC;

static int failures = 0;

#define CHECK(cond, msg)                                                                                               \
	do {                                                                                                               \
		if (!(cond)) {                                                                                                 \
			std::cerr << "FAIL: " << msg << std::endl;                                                                 \
			++failures;                                                                                                \
			return;                                                                                                    \
		}                                                                                                              \
	} while (0)

#define CHECK_THROWS(stmt, exType, msg)                                                                                \
	do {                                                                                                               \
		try {                                                                                                          \
			stmt;                                                                                                      \
			std::cerr << "FAIL: " << msg << " (no exception thrown)" << std::endl;                                     \
			++failures;                                                                                                \
			return;                                                                                                    \
		} catch (const exType&) {}                                                                                     \
	} while (0)

// ---------------- mock 基建 ----------------

struct MockCore {
	int id = 0;
};

struct MockSession {
	std::string modelPath;
};

static std::shared_ptr<std::atomic<int>> g_exports = std::make_shared<std::atomic<int>>(0);
static std::shared_ptr<std::atomic<int>> g_releases = std::make_shared<std::atomic<int>>(0);
static std::atomic<int> g_directHits{0};
static std::atomic<int> g_fallbackHits{0};

static Tensor::DataBlock floatsToBytes(const std::vector<float>& vals) {
	Tensor::DataBlock bytes(vals.size() * sizeof(float));
	if (!vals.empty())
		std::memcpy(bytes.data(), vals.data(), bytes.size());
	return bytes;
}

static std::vector<float> bytesToFloats(const std::byte* p, size_t count) {
	std::vector<float> vals(count);
	if (count > 0)
		std::memcpy(vals.data(), p, count * sizeof(float));
	return vals;
}

// 构造外部张量：handle 持假设备内存，exportToHost 计数并拷贝，deleter 计释放。
static Tensor makeExternal(const std::vector<float>& vals, const Tensor::Shape& shape,
						   std::shared_ptr<const void> coreToken = nullptr, bool trackRelease = false) {
	auto* raw = new Tensor::DataBlock(floatsToBytes(vals));
	std::shared_ptr<Tensor::DataBlock> payload(raw, [trackRelease](Tensor::DataBlock* p) {
		if (trackRelease)
			g_releases->fetch_add(1);
		delete p;
	});
	ExternalBacking b;
	b.producerEngine = "ExtMock";
	b.coreToken = std::move(coreToken);
	b.handle = payload;
	b.exportToHost = [payload]() -> Tensor::DataBlock {
		g_exports->fetch_add(1);
		return *payload;
	};
	return Tensor::fromExternal(Tensor::TensorType::Float, sizeof(float), shape, std::move(b));
}

static void resetCounters() {
	g_exports->store(0);
	g_releases->store(0);
	g_directHits.store(0);
	g_fallbackHits.store(0);
}

// ---------------- Tensor 级单元测试 ----------------

static void testMetaWithoutMaterialize() {
	resetCounters();
	Tensor t = makeExternal({1, 2, 3, 4}, {4});
	CHECK(t.hasExternalBacking(), "backing present");
	CHECK(t.valid(), "external tensor is valid without host data");
	CHECK(!t.hasCache(), "no host cache before read");
	CHECK(!t.empty(), "external tensor not empty");
	CHECK(!t.isFrozen(), "not frozen");
	CHECK(t.shape() == Tensor::Shape({4}), "shape answered from meta");
	CHECK(t.type() == Tensor::TensorType::Float, "type from meta");
	CHECK(g_exports->load() == 0, "no export triggered by meta queries");
}

static void testImplicitExportOnce() {
	resetCounters();
	Tensor t = makeExternal({1, 2, 3, 4, 5, 6}, {2, 3});
	auto bytes = t.bytes();
	CHECK(bytes.size() == 6 * sizeof(float), "exported byte size");
	CHECK(g_exports->load() == 1, "first read exports once");
	CHECK(t.hasCache(), "host cache materialized");
	CHECK(t.hasExternalBacking(), "backing retained for routing after export");

	auto span = t.data<float>();
	CHECK(span.size() == 6 && std::abs(span[5] - 6.0f) < 1e-6f, "typed dense read");
	CHECK(g_exports->load() == 1, "cached reads do not re-export");

	CHECK(std::abs(t[1][2].readScalar<float>() - 6.0f) < 1e-6f, "chained view read");
	auto last = t[-1][-1].read<float>();
	CHECK(last.size() == 1 && std::abs(last[0] - 6.0f) < 1e-6f, "negative index read");
	auto row = t[0].read<float>();
	CHECK(row.size() == 3 && std::abs(row[2] - 3.0f) < 1e-6f, "sub-tensor span read");
	CHECK(g_exports->load() == 1, "view reads reuse cache");
}

static void testScalarExternal() {
	resetCounters();
	Tensor t = makeExternal({42.0f}, {});
	CHECK(t.valid(), "scalar external valid");
	CHECK(t.isScalar(), "scalar flag from meta");
	CHECK(std::abs(t.item<float>() - 42.0f) < 1e-6f, "scalar item read exports");
	CHECK(g_exports->load() == 1, "scalar export once");
}

static void testWriteDetachCOW() {
	resetCounters();
	Tensor t = makeExternal({1, 2, 3, 4}, {4});
	Tensor c = t;
	CHECK(c.hasExternalBacking(), "copy shares backing");
	CHECK(c.externalBacking() == t.externalBacking(), "same backing object");

	c[0].set<float>(9.0f);
	CHECK(!c.hasExternalBacking(), "write detaches backing (COW)");
	CHECK(g_exports->load() == 1, "COW materializes via export");
	CHECK(std::abs(c[0].readScalar<float>() - 9.0f) < 1e-6f, "copy mutated");

	CHECK(t.hasExternalBacking(), "original backing intact");
	CHECK(std::abs(t[0].readScalar<float>() - 1.0f) < 1e-6f, "original value unchanged");
	auto payload = std::static_pointer_cast<Tensor::DataBlock>(t.externalBacking()->handle);
	auto vals = bytesToFloats(payload->data(), 4);
	CHECK(std::abs(vals[0] - 1.0f) < 1e-6f, "fake device payload untouched by COW");
}

static void testFullReplaceNoExport() {
	resetCounters();
	Tensor t = makeExternal({1, 2, 3, 4}, {4});
	t.fill<float>(7.0f);
	CHECK(!t.hasExternalBacking(), "fill detaches backing");
	CHECK(g_exports->load() == 0, "full replacement skips export");
	auto span = t.data<float>();
	CHECK(span.size() == 4 && std::abs(span[3] - 7.0f) < 1e-6f, "filled values");

	Tensor u = makeExternal({1, 2, 3, 4}, {4});
	u.loadData(floatsToBytes({5, 6}), {2});
	CHECK(!u.hasExternalBacking(), "loadData detaches backing");
	CHECK(g_exports->load() == 0, "loadData skips export");
	CHECK(u.shape() == Tensor::Shape({2}), "loadData shape");

	Tensor v = makeExternal({1, 2, 3, 4}, {4});
	v = 3.0f;
	CHECK(!v.hasExternalBacking(), "scalar assignment detaches backing");
	CHECK(v.isScalar() && std::abs(v.item<float>() - 3.0f) < 1e-6f, "scalar assignment value");
	CHECK(g_exports->load() == 0, "scalar assignment skips export");
}

static void testExpandCropDetach() {
	resetCounters();
	Tensor t = makeExternal({1, 2, 3, 4}, {4});
	t.expand<float>({6}, 0.0f);
	CHECK(!t.hasExternalBacking(), "expand detaches backing");
	CHECK(g_exports->load() == 1, "expand materializes via export");
	CHECK(t.shape() == Tensor::Shape({6}), "expanded shape");
	auto span = t.data<float>();
	CHECK(std::abs(span[3] - 4.0f) < 1e-6f && std::abs(span[5] - 0.0f) < 1e-6f, "expand preserves old, fills new");

	Tensor u = makeExternal({1, 2, 3, 4}, {4});
	u.crop({2});
	CHECK(!u.hasExternalBacking(), "crop detaches backing");
	CHECK(g_exports->load() == 2, "crop materializes via export");
	auto cropped = u.data<float>();
	CHECK(cropped.size() == 2 && std::abs(cropped[1] - 2.0f) < 1e-6f, "cropped values");
}

static void testGetDataConsumes() {
	resetCounters();
	Tensor t = makeExternal({1, 2, 3, 4}, {4});
	auto vals = t.getData<float>();
	CHECK(vals.size() == 4 && std::abs(vals[3] - 4.0f) < 1e-6f, "getData values");
	CHECK(!t.hasExternalBacking(), "consuming take relinquishes backing");
	CHECK(!t.valid(), "tensor emptied after take");
	CHECK(g_exports->load() == 1, "getData exports once");
}

static void testFreezeOpaqueSemantics() {
	resetCounters();
	Tensor t = makeExternal({1, 2, 3, 4}, {4});
	t.freeze();
	CHECK(t.isFrozen(), "frozen flag set");
	CHECK(!t.hasCache(), "opaque freeze does not materialize host cache");
	CHECK(g_exports->load() == 0, "opaque freeze does not export");
	CHECK(t.hasExternalBacking(), "backing retained after freeze");

	auto bytes = t.bytes();
	CHECK(bytes.size() == 16 && g_exports->load() == 1, "frozen external still exports on host read");

	CHECK_THROWS(t[0].set<float>(1.0f), TensorException, "frozen write rejected");
	CHECK(t.hasExternalBacking(), "failed write leaves backing intact");
	CHECK_THROWS(t.fill<float>(0.0f), TensorException, "frozen fill rejected");
	CHECK_THROWS((void)t.getData<float>(), TensorException, "frozen getData rejected");
	CHECK(t.hasExternalBacking(), "failed getData leaves backing intact");
	CHECK(g_exports->load() == 1, "no extra exports from rejected writes");

	bool frozenErr = false;
	try {
		t[0].set<float>(1.0f);
	} catch (const TensorException& e) {
		frozenErr = (e.getErrorType() == TensorException::ErrorType::Frozen);
	}
	CHECK(frozenErr, "error type is Frozen");
}

static void testExportMissingThrows() {
	resetCounters();
	auto payload = std::make_shared<Tensor::DataBlock>(floatsToBytes({1, 2}));
	ExternalBacking b;
	b.producerEngine = "ExtMock";
	b.handle = payload; // exportToHost 留空 = 不支持导出
	Tensor t = Tensor::fromExternal(Tensor::TensorType::Float, sizeof(float), {2}, std::move(b));
	CHECK(t.valid(), "no-export backing still valid for routing");
	CHECK_THROWS((void)t.bytes(), TensorException, "host access without export throws");
}

static void testExportSizeMismatchThrows() {
	resetCounters();
	auto payload = std::make_shared<Tensor::DataBlock>(floatsToBytes({1, 2}));
	ExternalBacking b;
	b.producerEngine = "ExtMock";
	b.handle = payload;
	b.exportToHost = [payload]() -> Tensor::DataBlock { return *payload; }; // 8 字节 != 4*4
	Tensor t = Tensor::fromExternal(Tensor::TensorType::Float, sizeof(float), {4}, std::move(b));
	CHECK_THROWS((void)t.bytes(), std::invalid_argument, "export size mismatch surfaces contract violation");
}

static void testNullHandleRejected() {
	ExternalBacking b;
	b.producerEngine = "ExtMock";
	CHECK_THROWS((void)Tensor::fromExternal(Tensor::TensorType::Float, sizeof(float), {1}, std::move(b)),
				 std::invalid_argument, "null handle rejected");
}

static void testCopyMoveSemantics() {
	resetCounters();
	Tensor t = makeExternal({1, 2}, {2});
	const ExternalBacking* backing = t.externalBacking();

	Tensor c(t);
	CHECK(c.hasExternalBacking() && c.externalBacking() == backing, "copy ctor shares backing");
	Tensor a;
	a = t;
	CHECK(a.hasExternalBacking() && a.externalBacking() == backing, "copy assign shares backing");

	Tensor m(std::move(c));
	CHECK(m.hasExternalBacking() && m.externalBacking() == backing, "move ctor transfers backing");
	CHECK(!c.hasExternalBacking(), "moved-from loses backing");
	Tensor n;
	n = std::move(m);
	CHECK(n.hasExternalBacking() && n.externalBacking() == backing, "move assign transfers backing");
	CHECK(!m.hasExternalBacking(), "move-assigned-from loses backing");
	CHECK(g_exports->load() == 0, "copy/move never export");
}

static void testReleaseCallbackOnce() {
	resetCounters();
	{
		Tensor t = makeExternal({1, 2, 3, 4}, {4}, nullptr, true);
		Tensor a = t;
		Tensor b = t;
		Tensor c = t;
		CHECK(g_releases->load() == 0, "no release while referenced");
	}
	CHECK(g_releases->load() == 1, "release callback runs exactly once after last reference");
}

static void testConcurrentExportOnce() {
	resetCounters();
	Tensor t = makeExternal({1, 2, 3, 4}, {4});
	t.freeze();
	std::atomic<int> bad{0};
	std::vector<std::thread> threads;
	for (int i = 0; i < 4; ++i) {
		threads.emplace_back([&t, &bad]() {
			for (int k = 0; k < 50; ++k) {
				auto bytes = t.bytes();
				auto vals = bytesToFloats(bytes.data(), 4);
				if (bytes.size() != 16 || std::abs(vals[2] - 3.0f) > 1e-6f)
					bad.fetch_add(1);
			}
		});
	}
	for (auto& th : threads)
		th.join();
	CHECK(bad.load() == 0, "concurrent reads see consistent data");
	CHECK(g_exports->load() == 1, "double-checked locking exports exactly once");
}

// ---------------- mock 引擎图级测试 ----------------

static Node::Result extMockRun(Node::RunContext& ctx) {
	const auto* inst = ctx.engineInstance();
	const auto* session = inst ? static_cast<const MockSession*>(inst->get()) : nullptr;
	const std::string model = session ? session->modelPath : "";
	const auto* inT = ctx.peek("in").as<Tensor>();
	if (!inT)
		return ctx.failure(Node::Status::InvalidInput, "ExtMock: input is not a Tensor");

	if (model == "models/produce.onnx") {
		auto span = inT->data<float>();
		std::vector<float> vals(span.begin(), span.end());
		// coreToken 别名核心句柄：共享所有权防地址复用，指针即核心身份
		std::shared_ptr<const void> token;
		if (inst && inst->core()) {
			auto coreHandle = inst->core();
			token = std::shared_ptr<const void>(coreHandle, coreHandle->get());
		}
		Tensor out = makeExternal(vals, {static_cast<int64_t>(vals.size())}, std::move(token));
		ctx.output("out", Value(std::make_unique<Tensor>(std::move(out))));
		return ctx.success();
	}

	// 消费侧单边协商：同核心直传读句柄，否则 host 读触发隐式导出兜底
	const ExternalBacking* b = inT->externalBacking();
	const bool direct = b && b->coreToken && inst && inst->core()
		&& b->coreToken.get() == inst->core()->get();
	std::vector<float> vals;
	if (direct) {
		g_directHits.fetch_add(1);
		auto payload = std::static_pointer_cast<Tensor::DataBlock>(b->handle);
		vals = bytesToFloats(payload->data(), payload->size() / sizeof(float));
	} else {
		g_fallbackHits.fetch_add(1);
		auto span = inT->data<float>();
		vals.assign(span.begin(), span.end());
	}

	if (model == "models/mutate.onnx") {
		Tensor local = *inT; // 克隆共享 backing
		local[0].set<float>(99.0f); // 写侧 COW：物化 + 脱离，不影响源载荷
		ctx.output("out", Value(std::make_unique<Tensor>(std::move(local))));
		return ctx.success();
	}

	float sum = 0;
	for (float v : vals)
		sum += v;
	auto out = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
	*out = sum + 100.0f;
	ctx.output("out", Value(std::move(out)));
	return ctx.success();
}

static void registerExtEngine(EngineRegistry& reg, const std::string& type) {
	if (reg.hasEngine(type))
		return; // 单例跨测试共享，避免重复注册
	EngineDescriptor desc;
	desc.engineType = type;
	desc.converter = {[](const Tensor& dc) { return Value(std::make_unique<Tensor>(dc)); },
					  [](const void* native) { return *static_cast<const Tensor*>(native); }};
	desc.createEngineCore = []() -> EngineCore { return EngineCore(std::make_shared<MockCore>()); };
	desc.loadModel = [](const EngineCore&, const std::string& path) -> EngineInstance {
		return EngineInstance(std::make_shared<MockSession>(MockSession{path}));
	};
	desc.getInputPorts = [](const EngineInstance& inst) -> std::vector<Node::Port> {
		return inst.get() ? std::vector<Node::Port>{Node::Port::in<float>("in")} : std::vector<Node::Port>{};
	};
	desc.getOutputPorts = [](const EngineInstance& inst) -> std::vector<Node::Port> {
		return inst.get() ? std::vector<Node::Port>{Node::Port::out<float>("out")} : std::vector<Node::Port>{};
	};
	desc.factory = [](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		auto node = std::make_unique<Node>("ExtMock", p.nodeName, p.schema, extMockRun, ResourceClass::Compute);
		if (p.engineInstance)
			node->bindEngine(p.engineInstance, p.engineInstance->descriptor());
		return node;
	};
	reg.registerEngine(desc);
}

static Tensor feedVector(std::initializer_list<float> vals) {
	Tensor t = Tensor::Create<float>({static_cast<int64_t>(vals.size())});
	t.loadData(floatsToBytes(std::vector<float>(vals)), {static_cast<int64_t>(vals.size())});
	return t;
}

// 同 engineType 两模型共享核心 → 直传，零导出
static void testGraphDirect() {
	resetCounters();
	auto& reg = EngineRegistry::instance();
	registerExtEngine(reg, "ExtMock");
	TestHarness h;
	h.addNode(reg.createNode("ExtMock", "prod", std::string("models/produce.onnx")));
	h.addNode(reg.createNode("ExtMock", "cons", std::string("models/consume.onnx")));
	h.connect("prod", "out", "cons", "in");
	h.feedInput("t", "prod", "in", feedVector({1, 2, 3, 4}));
	h.submit("t", "cons", "out");
	CHECK(h.awaitCompletion("t"), "direct graph completes");
	CHECK(!h.hasErrors(), "direct graph no errors");
	Tensor out = h.getOutputTensor("t", "cons", "out");
	CHECK(std::abs(out.item<float>() - 110.0f) < 1e-4f, "direct result correct");
	CHECK(g_directHits.load() == 1, "consumer negotiated direct transfer");
	CHECK(g_exports->load() == 0, "no host export on direct path");
}

// 跨 engineType 核心互不认识 → host 兜底，行为同现状
static void testGraphFallback() {
	resetCounters();
	auto& reg = EngineRegistry::instance();
	registerExtEngine(reg, "ExtMock");
	registerExtEngine(reg, "ExtHostOnly");
	TestHarness h;
	h.addNode(reg.createNode("ExtMock", "prod", std::string("models/produce.onnx")));
	h.addNode(reg.createNode("ExtHostOnly", "cons", std::string("models/consume.onnx")));
	h.connect("prod", "out", "cons", "in");
	h.feedInput("t", "prod", "in", feedVector({1, 2, 3, 4}));
	h.submit("t", "cons", "out");
	CHECK(h.awaitCompletion("t"), "fallback graph completes");
	CHECK(!h.hasErrors(), "fallback graph no errors");
	Tensor out = h.getOutputTensor("t", "cons", "out");
	CHECK(std::abs(out.item<float>() - 110.0f) < 1e-4f, "fallback result correct");
	CHECK(g_directHits.load() == 0, "no direct transfer across cores");
	CHECK(g_fallbackHits.load() == 1, "consumer fell back to host");
	CHECK(g_exports->load() == 1, "fallback exports exactly once");
}

// 广播：冻结不物化，两消费者同核心直传；写侧克隆 COW 分离不影响另一路
static void testGraphBroadcastCOW() {
	resetCounters();
	auto& reg = EngineRegistry::instance();
	registerExtEngine(reg, "ExtMock");
	TestHarness h;
	h.addNode(reg.createNode("ExtMock", "prod", std::string("models/produce.onnx")));
	h.addNode(reg.createNode("ExtMock", "consA", std::string("models/consume.onnx")));
	h.addNode(reg.createNode("ExtMock", "consB", std::string("models/mutate.onnx")));
	h.connect("prod", "out", "consA", "in");
	h.connect("prod", "out", "consB", "in");
	h.feedInput("t", "prod", "in", feedVector({1, 2, 3, 4}));
	h.submit("t", std::vector<OutputDeclaration>{{"consA", "out", 1}, {"consB", "out", 1}});
	CHECK(h.awaitCompletion("t"), "broadcast graph completes");
	CHECK(!h.hasErrors(), "broadcast graph no errors");

	Tensor outA = h.getOutputTensor("t", "consA", "out");
	CHECK(std::abs(outA.item<float>() - 110.0f) < 1e-4f, "reader branch unaffected by writer branch");
	Tensor outB = h.getOutputTensor("t", "consB", "out");
	CHECK(outB.shape() == Tensor::Shape({4}), "mutated clone shape");
	auto spanB = outB.data<float>();
	CHECK(std::abs(spanB[0] - 99.0f) < 1e-6f, "writer branch mutation applied");
	CHECK(std::abs(spanB[1] - 2.0f) < 1e-6f && std::abs(spanB[3] - 4.0f) < 1e-6f, "COW preserved other elements");
	CHECK(!outB.hasExternalBacking(), "mutated clone detached from backing");
	CHECK(g_directHits.load() == 2, "both consumers negotiated direct");
	CHECK(g_exports->load() == 1, "only COW materialization exports");
}

// 混合链 ext → host → ext：中段兜底导出，尾段收 host 张量走现状路径
static void testGraphMixedChain() {
	resetCounters();
	auto& reg = EngineRegistry::instance();
	registerExtEngine(reg, "ExtMock");
	registerExtEngine(reg, "ExtHostOnly");
	TestHarness h;
	h.addNode(reg.createNode("ExtMock", "prod", std::string("models/produce.onnx")));
	h.addNode(reg.createNode("ExtHostOnly", "mid", std::string("models/consume.onnx")));
	h.addNode(reg.createNode("ExtMock", "tail", std::string("models/consume.onnx")));
	h.connect("prod", "out", "mid", "in");
	h.connect("mid", "out", "tail", "in");
	h.feedInput("t", "prod", "in", feedVector({1, 2, 3, 4}));
	h.submit("t", "tail", "out");
	CHECK(h.awaitCompletion("t"), "mixed chain completes");
	CHECK(!h.hasErrors(), "mixed chain no errors");
	Tensor out = h.getOutputTensor("t", "tail", "out");
	CHECK(std::abs(out.item<float>() - 210.0f) < 1e-4f, "mixed chain result correct");
	CHECK(g_exports->load() == 1, "only the ext->host edge exports");
	CHECK(g_fallbackHits.load() == 2, "mid fallback + tail host input");
	CHECK(g_directHits.load() == 0, "no direct transfer in mixed chain");
}

int main() {
	try {
		testMetaWithoutMaterialize();
		testImplicitExportOnce();
		testScalarExternal();
		testWriteDetachCOW();
		testFullReplaceNoExport();
		testExpandCropDetach();
		testGetDataConsumes();
		testFreezeOpaqueSemantics();
		testExportMissingThrows();
		testExportSizeMismatchThrows();
		testNullHandleRejected();
		testCopyMoveSemantics();
		testReleaseCallbackOnce();
		testConcurrentExportOnce();
		testGraphDirect();
		testGraphFallback();
		testGraphBroadcastCOW();
		testGraphMixedChain();
	} catch (const std::exception& e) {
		std::cerr << "UNCAUGHT EXCEPTION: " << e.what() << std::endl;
		return -1;
	}
	if (failures > 0) {
		std::cerr << failures << " check(s) failed" << std::endl;
		return -1;
	}
	std::cout << "All ExternalBacking tests passed" << std::endl;
	return 0;
}