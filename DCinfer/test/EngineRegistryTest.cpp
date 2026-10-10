// EngineRegistry 单元测试：注册、创建、转换钩子
#include <atomic>
#include "NodeExecutor.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "EngineRegistry.h"
#include "Node.h"

using namespace DC;

static Node::Schema mockSchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("in")};
	s.outputs = {Node::Port::out<float>("out")};
	return s;
}

static Node::Result mockRunImpl(Node::RunContext& ctx, int magic) {
	const auto& inNT = ctx.peek("in");
	const auto* inVal = inNT.as<Tensor>();
	auto t = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
	*t = inVal->item<float>() + static_cast<float>(magic);
	ctx.output("out", Value(std::move(t)));
	return ctx.success();
}

static Value mockToNative(const Tensor& dc) {
	return Value(std::make_unique<Tensor>(dc));
}

static Tensor mockToDC(const void* native) {
	const auto* t = static_cast<const Tensor*>(native);
	Tensor result(Tensor::TensorType::Float, sizeof(float));
	result = t->item<float>();
	return result;
}

struct MockSession {
	std::string modelPath;
};

static std::atomic<int> mockEngineCreateCount{0};
static std::atomic<int> mockFactorySchemaSeen{0};

struct LifecycleSession {
	std::string modelPath;
};

static std::atomic<int> g_lifecycleCreateCount{0};
static std::atomic<int> g_lifecycleReleaseCount{0};

static Node::Result lifecycleRunImpl(Node::RunContext& ctx) {
	const auto& inVal = ctx.peek("in");
	const auto* inT = inVal.as<Tensor>();
	auto t = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
	*t = inT->item<float>();
	ctx.output("out", Value(std::move(t)));
	return ctx.success();
}

static void registerLifecycleEngine(EngineRegistry& reg, const std::string& type) {
	if (reg.hasEngine(type))
		return; // 单例跨测试共享，避免重复注册
	EngineDescriptor desc;
	desc.engineType = type;
	desc.converter = {mockToNative, mockToDC};

	desc.loadModel = [](const EngineCore&, const std::string& path) -> EngineInstance {
		++g_lifecycleCreateCount;
		return EngineInstance(std::make_shared<LifecycleSession>(LifecycleSession{path}));
	};
	desc.releaseModel = [](void*) { ++g_lifecycleReleaseCount; };
	desc.getInputPorts = [](const EngineInstance& inst) -> std::vector<Node::Port> {
		if (!inst.get())
			return {};
		return {{"in", Tensor::TensorType::Float, sizeof(float), {}, true}};
	};
	desc.getOutputPorts = [](const EngineInstance& inst) -> std::vector<Node::Port> {
		if (!inst.get())
			return {};
		return {{"out", Tensor::TensorType::Float, sizeof(float), {}, true}};
	};
	desc.factory = [](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		auto node = std::make_unique<Node>("Lifecycle", p.nodeName, p.schema, lifecycleRunImpl,
										   ResourceClass::Operator);
		if (p.engineInstance)
			node->bindEngine(p.engineInstance, p.engineInstance->descriptor());
		return node;
	};

	reg.registerEngine(desc);
}

static std::atomic<int> g_blockEntered{0};
static std::atomic<bool> g_blockRelease{false};
static std::atomic<int> g_blockCreated{0};

static Node::Result noopRunImpl(Node::RunContext& ctx) {
	(void)ctx;
	return ctx.success();
}

// 慢路径 models/slow.onnx 阻塞至 g_blockRelease 放行：验证慢 key 不阻塞其他 key
static void registerFlightEngine(EngineRegistry& reg, const std::string& type) {
	if (reg.hasEngine(type))
		return;
	EngineDescriptor desc;
	desc.engineType = type;
	desc.converter = {mockToNative, mockToDC};
	desc.loadModel = [](const EngineCore&, const std::string& path) -> EngineInstance {
		++g_blockEntered;
		if (path == "models/slow.onnx") {
			while (!g_blockRelease.load())
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		++g_blockCreated;
		return EngineInstance(std::make_shared<LifecycleSession>(LifecycleSession{path}));
	};
	desc.getInputPorts = [](const EngineInstance& inst) -> std::vector<Node::Port> {
		return inst.get() ? std::vector<Node::Port>{{"in", Tensor::TensorType::Float, sizeof(float), {}, true}}
						  : std::vector<Node::Port>{};
	};
	desc.getOutputPorts = [](const EngineInstance& inst) -> std::vector<Node::Port> {
		return inst.get() ? std::vector<Node::Port>{{"out", Tensor::TensorType::Float, sizeof(float), {}, true}}
						  : std::vector<Node::Port>{};
	};
	desc.factory = [](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		auto node = std::make_unique<Node>("Flight", p.nodeName, p.schema, noopRunImpl,
										   ResourceClass::Operator);
		if (p.engineInstance)
			node->bindEngine(p.engineInstance, p.engineInstance->descriptor());
		return node;
	};
	reg.registerEngine(desc);
}

static std::atomic<bool> g_failNext{true};
static std::atomic<int> g_failCalls{0};

static void registerFailingEngine(EngineRegistry& reg, const std::string& type) {
	if (reg.hasEngine(type))
		return;
	EngineDescriptor desc;
	desc.engineType = type;
	desc.converter = {mockToNative, mockToDC};
	desc.loadModel = [](const EngineCore&, const std::string& path) -> EngineInstance {
		++g_failCalls;
		if (g_failNext.load())
			throw std::runtime_error("simulated engine load failure: " + path);
		return EngineInstance(std::make_shared<LifecycleSession>(LifecycleSession{path}));
	};
	desc.getInputPorts = [](const EngineInstance&) { return std::vector<Node::Port>{}; };
	desc.getOutputPorts = [](const EngineInstance&) { return std::vector<Node::Port>{}; };
	desc.factory = [](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		auto node = std::make_unique<Node>("Failing", p.nodeName, p.schema, noopRunImpl,
										   ResourceClass::Operator);
		if (p.engineInstance)
			node->bindEngine(p.engineInstance, p.engineInstance->descriptor());
		return node;
	};
	reg.registerEngine(desc);
}

struct PhaseSession {
	std::string modelPath;
};

// 注入失败点：RunFn 失败经 NodeResult 返回，其余相位以异常注入
enum class PhaseFailAt { None, PreRun, RunFn, Synchronize, PostRun };

static std::atomic<int> g_phasePreRun{0};
static std::atomic<int> g_phaseSync{0};
static std::atomic<int> g_phasePost{0};
static std::atomic<int> g_phaseOnError{0};
static std::atomic<int> g_phaseFailAt{0};
static std::atomic<bool> g_phaseOnErrorThrows{false};

static Node::Result phaseRunImpl(Node::RunContext& ctx) {
	if (static_cast<PhaseFailAt>(g_phaseFailAt.load()) == PhaseFailAt::RunFn)
		return ctx.failure(Node::Status::ExecutionFailed, "injected RunFn failure");
	const auto& inVal = ctx.peek("in");
	const auto* inT = inVal.as<Tensor>();
	auto t = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
	*t = inT->item<float>();
	ctx.output("out", Value(std::move(t)));
	return ctx.success();
}

static void resetPhaseState(PhaseFailAt failAt, bool onErrorThrows) {
	g_phasePreRun = 0;
	g_phaseSync = 0;
	g_phasePost = 0;
	g_phaseOnError = 0;
	g_phaseFailAt = static_cast<int>(failAt);
	g_phaseOnErrorThrows = onErrorThrows;
}

static void registerPhaseEngine(EngineRegistry& reg, const std::string& type) {
	if (reg.hasEngine(type))
		return;
	EngineDescriptor desc;
	desc.engineType = type;
	desc.converter = {mockToNative, mockToDC};
	desc.loadModel = [](const EngineCore&, const std::string& path) -> EngineInstance {
		return EngineInstance(std::make_shared<PhaseSession>(PhaseSession{path}));
	};
	desc.getInputPorts = [](const EngineInstance& inst) -> std::vector<Node::Port> {
		return inst.get() ? std::vector<Node::Port>{{"in", Tensor::TensorType::Float, sizeof(float), {}, true}}
						  : std::vector<Node::Port>{};
	};
	desc.getOutputPorts = [](const EngineInstance& inst) -> std::vector<Node::Port> {
		return inst.get() ? std::vector<Node::Port>{{"out", Tensor::TensorType::Float, sizeof(float), {}, true}}
						  : std::vector<Node::Port>{};
	};
	desc.factory = [](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		auto node = std::make_unique<Node>("Phase", p.nodeName, p.schema, phaseRunImpl,
										   ResourceClass::Operator);
		if (p.engineInstance)
			node->bindEngine(p.engineInstance, p.engineInstance->descriptor());
		return node;
	};
	desc.phases.preRun = [](void*) {
		++g_phasePreRun;
		if (static_cast<PhaseFailAt>(g_phaseFailAt.load()) == PhaseFailAt::PreRun)
			throw std::runtime_error("injected preRun failure");
	};
	desc.phases.synchronize = [](void*) {
		++g_phaseSync;
		if (static_cast<PhaseFailAt>(g_phaseFailAt.load()) == PhaseFailAt::Synchronize)
			throw std::runtime_error("injected synchronize failure");
	};
	desc.phases.postRun = [](void*, Node::RunContext&) {
		++g_phasePost;
		if (static_cast<PhaseFailAt>(g_phaseFailAt.load()) == PhaseFailAt::PostRun)
			throw std::runtime_error("injected postRun failure");
	};
	desc.phases.onError = [](void*) {
		++g_phaseOnError;
		if (g_phaseOnErrorThrows.load())
			throw std::runtime_error("injected onError failure");
	};
	reg.registerEngine(desc);
}

static Node::Result mockModelRunImpl(Node::RunContext& ctx) {
	(void)ctx;
	return ctx.success();
}

static void registerMockModelEngine(EngineRegistry& reg, const std::string& type) {
	EngineDescriptor desc;
	desc.engineType = type;
	desc.converter = {mockToNative, mockToDC};

	desc.loadModel = [](const EngineCore&, const std::string& path) -> EngineInstance {
		++mockEngineCreateCount;
		return EngineInstance(std::make_shared<MockSession>(MockSession{path}));
	};

	desc.getInputPorts = [](const EngineInstance& inst) -> std::vector<Node::Port> {
		if (!inst.get())
			return {};
		return {{"in", Tensor::TensorType::Float, sizeof(float), {}, true}};
	};
	desc.getOutputPorts = [](const EngineInstance& inst) -> std::vector<Node::Port> {
		if (!inst.get())
			return {};
		return {{"out", Tensor::TensorType::Float, sizeof(float), {}, true}};
	};

	desc.factory = [type](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		if (!p.schema.inputs.empty())
			++mockFactorySchemaSeen;
		auto node = std::make_unique<Node>(type, p.nodeName, p.schema, mockModelRunImpl,
										   ResourceClass::Operator);
		if (p.engineInstance)
			node->bindEngine(p.engineInstance, p.engineInstance->descriptor());
		return node;
	};

	reg.registerEngine(desc);
}

static void runTests() {
	auto& reg = EngineRegistry::instance();

	{
		EngineDescriptor desc;
		desc.engineType = "Mock";
		desc.converter = {mockToNative, mockToDC};
		desc.factory = makeNodeFactory<int>("Mock", mockSchema(), mockRunImpl);

		if (!reg.registerEngine(desc))
			throw std::runtime_error("registerEngine failed");
	}
	std::cout << "Test 1 passed: register engine" << std::endl;

	{
		EngineDescriptor desc;
		desc.engineType = "Mock";
		if (reg.registerEngine(desc))
			throw std::runtime_error("duplicate registration should fail");
	}
	std::cout << "Test 2 passed: duplicate registration rejected" << std::endl;

	{
		auto* desc = reg.find("Mock");
		if (!desc)
			throw std::runtime_error("find returned null");
		if (desc->engineType != "Mock")
			throw std::runtime_error("engineType mismatch");
		if (!desc->converter.toNative)
			throw std::runtime_error("toNative not set");
		if (!desc->converter.toDC)
			throw std::runtime_error("toDC not set");
	}
	std::cout << "Test 3 passed: find engine" << std::endl;

	{
		if (!reg.hasEngine("Mock"))
			throw std::runtime_error("hasEngine should be true");
		if (reg.hasEngine("Nonexistent"))
			throw std::runtime_error("hasEngine should be false");

		auto types = reg.engineTypes();
		if (types.empty())
			throw std::runtime_error("engineTypes should not be empty");
		bool found = false;
		for (auto& t : types)
			if (t == "Mock")
				found = true;
		if (!found)
			throw std::runtime_error("Mock not found in engineTypes");
	}
	std::cout << "Test 4 passed: hasEngine / engineTypes" << std::endl;

	{
		int magic = 42;
		auto node = reg.createNode("Mock", "testNode", &magic);
		if (!node)
			throw std::runtime_error("createNode returned null");
		if (node->type() != "Mock")
			throw std::runtime_error("node type mismatch");
		if (node->name() != "testNode")
			throw std::runtime_error("node name mismatch");
		if (node->schema().inputs.size() != 1)
			throw std::runtime_error("schema inputs count mismatch");
		if (node->schema().outputs.size() != 1)
			throw std::runtime_error("schema outputs count mismatch");
	}
	std::cout << "Test 5 passed: createNode" << std::endl;

	{
		auto node = reg.createNode("UnknownEngine", "test");
		if (node)
			throw std::runtime_error("createNode for unknown engine should return null");
	}
	std::cout << "Test 6 passed: createNode unknown engine" << std::endl;

	{
		int magic = 100;
		auto node = reg.createNode("Mock", "runner", &magic);
		if (!node)
			throw std::runtime_error("createNode failed");

		Tensor in(Tensor::TensorType::Float, sizeof(float));
		in = 50.0f;
		NodeExecutor exec(*node);
		exec.setInput("task1", "in", Value(std::make_unique<Tensor>(std::move(in))));
		exec.tryExecute("task1");

		if (!exec.hasOutput("task1", "out"))
			throw std::runtime_error("output not produced");

		auto outNT = exec.takeOutput("task1", "out");
		auto* out = outNT.as<Tensor>();
		if (std::abs(out->item<float>() - 150.0f) > 1e-6f)
			throw std::runtime_error("output value mismatch: expected 150, got " + std::to_string(out->item<float>()));
	}
	std::cout << "Test 7 passed: created node runs correctly" << std::endl;

	{
		Tensor dc(Tensor::TensorType::Float, sizeof(float));
		dc = 3.14f;

		auto native = mockToNative(dc);
		if (!native)
			throw std::runtime_error("toNative returned empty");
		auto* t = native.as<Tensor>();
		if (!t)
			throw std::runtime_error("toNative did not wrap Tensor");
		if (std::abs(t->item<float>() - 3.14f) > 1e-6f)
			throw std::runtime_error("toNative value mismatch");

		auto back = mockToDC(native.get());
		if (std::abs(back.item<float>() - 3.14f) > 1e-6f)
			throw std::runtime_error("toDC round-trip mismatch");
	}
	std::cout << "Test 8 passed: TensorConverter round-trip" << std::endl;

	{
		EngineDescriptor desc;
		desc.engineType = "";
		if (reg.registerEngine(desc))
			throw std::runtime_error("empty engineType should be rejected");
	}
	std::cout << "Test 9 passed: empty engineType rejected" << std::endl;

	{
		mockEngineCreateCount = 0;
		mockFactorySchemaSeen = 0;
		registerMockModelEngine(reg, "MockModel");

		auto node1 = reg.createNode("MockModel", "n1", std::string("models/a.onnx"));
		if (!node1)
			throw std::runtime_error("createNode(modelPath) returned null");
		if (node1->schema().inputs.size() != 1 || node1->schema().outputs.size() != 1)
			throw std::runtime_error("factory should receive framework-derived schema");
		if (node1->modelPath() != "models/a.onnx")
			throw std::runtime_error("modelPath not propagated to node");
		if (mockEngineCreateCount != 1)
			throw std::runtime_error("model should be loaded exactly once");
		if (mockFactorySchemaSeen != 1)
			throw std::runtime_error("factory should receive non-empty schema");

		auto node2 = reg.createNode("MockModel", "n2", std::string("models/a.onnx"));
		if (!node2)
			throw std::runtime_error("second createNode(modelPath) returned null");
		if (mockEngineCreateCount != 1)
			throw std::runtime_error("same modelPath should reuse cached instance");

		auto node3 = reg.createNode("MockModel", "n3", std::string("models/b.onnx"));
		if (!node3)
			throw std::runtime_error("third createNode(modelPath) returned null");
		if (mockEngineCreateCount != 2)
			throw std::runtime_error("different modelPath should create new instance");
	}
	std::cout << "Test 10 passed: createNode(modelPath) single-load + schema passing + cache" << std::endl;

	{
		mockEngineCreateCount = 0;
		mockFactorySchemaSeen = 0;
		registerMockModelEngine(reg, "LazyMockModel");

		// 声明 schema 与实例端口不同：验证原样透传、不做实例推导
		Node::Schema declared;
		declared.inputs = {Node::Port::in<float>("declaredIn")};
		declared.outputs = {Node::Port::out<float>("declaredOut")};
		auto node = reg.createLazyNode("LazyMockModel", "ln1", declared);
		if (!node)
			throw std::runtime_error("createLazyNode returned null");
		if (node->type() != "LazyMockModel")
			throw std::runtime_error("node type must be the registered engine type");
		if (node->schema().inputs.size() != 1 || node->schema().inputs[0].name != "declaredIn")
			throw std::runtime_error("declared input schema must be passed to factory verbatim");
		if (node->schema().outputs.size() != 1 || node->schema().outputs[0].name != "declaredOut")
			throw std::runtime_error("declared output schema must be passed to factory verbatim");
		if (mockEngineCreateCount != 0)
			throw std::runtime_error("createLazyNode must not create engine instances");
		if (mockFactorySchemaSeen != 1)
			throw std::runtime_error("factory must be invoked once with the declared schema");

		if (reg.createLazyNode("UnknownLazyEngine", "ln2", {}))
			throw std::runtime_error("createLazyNode for unknown engine should return null");
	}
	std::cout << "Test 10b passed: createLazyNode declared-schema materialization, zero load" << std::endl;

	{
		mockEngineCreateCount = 0;
		constexpr int kThreads = 8;
		std::atomic<int> got{0};
		std::vector<EngineHandle> handles(kThreads);
		std::vector<std::thread> threads;
		for (int i = 0; i < kThreads; ++i) {
			threads.emplace_back([&, i] {
				auto inst = reg.getOrCreateEngine("MockModel", "models/concurrent.onnx");
				if (inst && inst->get()) {
					handles[i] = inst;
					++got;
				}
			});
		}
		for (auto& t : threads)
			t.join();
		if (got != kThreads)
			throw std::runtime_error("all threads should obtain the instance");
		for (const auto& h : handles) {
			if (h != handles[0])
				throw std::runtime_error("all threads should obtain the same handle");
		}
		if (mockEngineCreateCount != 1)
			throw std::runtime_error("concurrent getOrCreateEngine should create instance once");
	}
	std::cout << "Test 11 passed: concurrent getOrCreateEngine creates once" << std::endl;

	{
		g_lifecycleCreateCount = 0;
		g_lifecycleReleaseCount = 0;
		registerLifecycleEngine(reg, "Lifecycle");

		auto node = reg.createNode("Lifecycle", "keep", std::string("models/lc.onnx"));
		if (!node)
			throw std::runtime_error("createNode(modelPath) failed");
		if (g_lifecycleCreateCount != 1)
			throw std::runtime_error("engine should be created exactly once");

		reg.releaseAllEngines();
		if (g_lifecycleReleaseCount != 0)
			throw std::runtime_error("releaseAllEngines must not destroy node-held instances");

		Tensor in(Tensor::TensorType::Float, sizeof(float));
		in = 1.0f;
		NodeExecutor exec(*node);
		exec.setInput("task1", "in", Value(std::make_unique<Tensor>(std::move(in))));
		auto result = exec.tryExecute("task1");
		if (!result.ok())
			throw std::runtime_error("node should run after releaseAllEngines");
		if (!exec.hasOutput("task1", "out"))
			throw std::runtime_error("output should be produced after releaseAllEngines");

		node.reset();
		if (g_lifecycleReleaseCount != 1)
			throw std::runtime_error("release hook should fire exactly once when last handle dies");
	}
	std::cout << "Test 12 passed: handle keeps engine alive after releaseAllEngines" << std::endl;

	{
		g_lifecycleCreateCount = 0;
		g_lifecycleReleaseCount = 0;

		auto oldNode = reg.createNode("Lifecycle", "old", std::string("models/reload.onnx"));
		if (!oldNode || g_lifecycleCreateCount != 1)
			throw std::runtime_error("initial load failed");
		auto oldHandle = reg.getOrCreateEngine("Lifecycle", "models/reload.onnx");
		if (g_lifecycleCreateCount != 1)
			throw std::runtime_error("cache hit should not reload");

		reg.releaseEngine("Lifecycle", "models/reload.onnx");
		if (g_lifecycleReleaseCount != 0)
			throw std::runtime_error("releaseEngine must not destroy node-held instances");

		auto newNode = reg.createNode("Lifecycle", "new", std::string("models/reload.onnx"));
		if (!newNode)
			throw std::runtime_error("reload after releaseEngine failed");
		if (g_lifecycleCreateCount != 2)
			throw std::runtime_error("same key should create a fresh instance after release");

		oldNode.reset();
		oldHandle.reset();
		if (g_lifecycleReleaseCount != 1)
			throw std::runtime_error("old instance should be released exactly once");

		newNode.reset();
		if (g_lifecycleReleaseCount != 1)
			throw std::runtime_error("cached handle should keep instance alive");
		reg.releaseAllEngines();
		if (g_lifecycleReleaseCount != 2)
			throw std::runtime_error("new instance should be released after cache clear");
	}
	std::cout << "Test 13 passed: cache reload creates fresh instance without disturbing holders" << std::endl;

	{
		g_blockEntered = 0;
		g_blockCreated = 0;
		g_blockRelease = false;
		registerFlightEngine(reg, "Flight");

		std::thread slowThread([&] {
			auto h = reg.getOrCreateEngine("Flight", "models/slow.onnx");
			if (!h)
				throw std::runtime_error("slow key should eventually succeed");
		});

		// 等 A 进入创建回调：此刻 A 已释放 registry 锁
		for (int i = 0; i < 5000 && g_blockEntered.load() < 1; ++i)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		if (g_blockEntered.load() < 1)
			throw std::runtime_error("slow loadModel should have been entered");

		auto fastHandle = reg.getOrCreateEngine("Flight", "models/fast.onnx");
		if (!fastHandle)
			throw std::runtime_error("fast key should not be blocked by slow key");
		if (g_blockCreated.load() != 1)
			throw std::runtime_error(
				"only fast key should have completed while slow key is still pending");

		g_blockRelease = true;
		slowThread.join();
		if (g_blockCreated.load() != 2)
			throw std::runtime_error("both keys should be created exactly once each");
	}
	std::cout << "Test 14 passed: slow key creation does not block other keys" << std::endl;

	{
		g_failNext = true;
		g_failCalls = 0;
		registerFailingEngine(reg, "Failing");

		bool threw = false;
		try {
			reg.getOrCreateEngine("Failing", "models/fail.onnx");
		} catch (const std::runtime_error&) {
			threw = true;
		}
		if (!threw)
			throw std::runtime_error("loadModel exception should propagate to leader");
		if (g_failCalls != 1)
			throw std::runtime_error("leader should invoke loadModel exactly once");

		g_failNext = false;
		auto recovered = reg.getOrCreateEngine("Failing", "models/fail.onnx");
		if (!recovered)
			throw std::runtime_error("creation should succeed after a failure");
		if (g_failCalls != 2)
			throw std::runtime_error("retry should create a fresh instance");

		g_failNext = true;
		constexpr int kThreads = 4;
		std::atomic<int> failures{0};
		std::vector<std::thread> threads;
		for (int i = 0; i < kThreads; ++i) {
			threads.emplace_back([&] {
				try {
					reg.getOrCreateEngine("Failing", "models/fail2.onnx");
				} catch (const std::runtime_error&) {
					++failures;
				}
			});
		}
		for (auto& t : threads)
			t.join();
		if (failures != kThreads)
			throw std::runtime_error("all concurrent callers should observe the failure");
	}
	std::cout << "Test 15 passed: failure propagates to followers; retry after failure" << std::endl;

	{
		registerPhaseEngine(reg, "Phase");
		auto node = reg.createNode("Phase", "p1", std::string("models/phase.onnx"));
		if (!node)
			throw std::runtime_error("createNode(modelPath) failed");

		resetPhaseState(PhaseFailAt::PreRun, false);
		Tensor in(Tensor::TensorType::Float, sizeof(float));
		in = 1.0f;
		NodeExecutor exec(*node);
		exec.setInput("t1", "in", Value(std::make_unique<Tensor>(std::move(in))));
		bool threw = false;
		try {
			(void)exec.tryExecute("t1");
		} catch (const std::runtime_error& e) {
			threw = std::string(e.what()).find("preRun") != std::string::npos;
		}
		if (!threw)
			throw std::runtime_error("preRun exception should propagate after onError reset");
		if (g_phasePreRun != 1 || g_phaseOnError != 1)
			throw std::runtime_error("preRun failure should trigger onError exactly once");
		if (g_phaseSync != 0 || g_phasePost != 0)
			throw std::runtime_error("phases after a failed phase must be skipped");
	}
	std::cout << "Test 16 passed: preRun failure triggers onError, later phases skipped" << std::endl;

	{
		auto node = reg.createNode("Phase", "p2", std::string("models/phase.onnx"));
		if (!node)
			throw std::runtime_error("createNode(modelPath) failed");

		resetPhaseState(PhaseFailAt::Synchronize, false);
		Tensor in(Tensor::TensorType::Float, sizeof(float));
		in = 2.0f;
		NodeExecutor exec(*node);
		exec.setInput("t1", "in", Value(std::make_unique<Tensor>(std::move(in))));
		bool threw = false;
		try {
			(void)exec.tryExecute("t1");
		} catch (const std::runtime_error&) {
			threw = true;
		}
		if (!threw)
			throw std::runtime_error("synchronize exception should propagate after onError reset");
		if (g_phasePreRun != 1 || g_phaseSync != 1 || g_phaseOnError != 1)
			throw std::runtime_error("synchronize failure should trigger onError exactly once");
		if (g_phasePost != 0)
			throw std::runtime_error("postRun must be skipped after synchronize failure");
	}
	std::cout << "Test 17 passed: synchronize failure triggers onError, postRun skipped" << std::endl;

	{
		auto node = reg.createNode("Phase", "p3", std::string("models/phase.onnx"));
		if (!node)
			throw std::runtime_error("createNode(modelPath) failed");

		resetPhaseState(PhaseFailAt::RunFn, false);
		Tensor in(Tensor::TensorType::Float, sizeof(float));
		in = 3.0f;
		NodeExecutor exec(*node);
		exec.setInput("t1", "in", Value(std::make_unique<Tensor>(std::move(in))));
		auto result = exec.tryExecute("t1"); // 失败经 NodeResult 返回，不抛出
		if (result.ok() || result.status != NodeStatus::ExecutionFailed)
			throw std::runtime_error("RunFn failure should surface as ExecutionFailed result");
		if (g_phaseOnError != 1)
			throw std::runtime_error("RunFn failure should trigger onError exactly once");
		if (g_phaseSync != 0 || g_phasePost != 0)
			throw std::runtime_error("synchronize/postRun must be skipped after RunFn failure");
	}
	std::cout << "Test 18 passed: RunFn failure returns via NodeResult, onError fired once" << std::endl;

	{
		auto node = reg.createNode("Phase", "p4", std::string("models/phase.onnx"));
		if (!node)
			throw std::runtime_error("createNode(modelPath) failed");

		resetPhaseState(PhaseFailAt::RunFn, true);
		Tensor in(Tensor::TensorType::Float, sizeof(float));
		in = 4.0f;
		NodeExecutor exec(*node);
		exec.setInput("t1", "in", Value(std::make_unique<Tensor>(std::move(in))));
		auto result = exec.tryExecute("t1");
		if (result.ok())
			throw std::runtime_error("original RunFn failure should still be reported");
		if (result.message.find("injected onError failure") != std::string::npos)
			throw std::runtime_error("onError's own exception must not replace the original failure");
		if (g_phaseOnError != 1)
			throw std::runtime_error("onError should be attempted exactly once");

		resetPhaseState(PhaseFailAt::PreRun, true);
		Tensor in2(Tensor::TensorType::Float, sizeof(float));
		in2 = 5.0f;
		exec.setInput("t2", "in", Value(std::make_unique<Tensor>(std::move(in2))));
		bool originalPropagated = false;
		try {
			(void)exec.tryExecute("t2");
		} catch (const std::runtime_error& e) {
			originalPropagated = std::string(e.what()).find("preRun") != std::string::npos;
		}
		if (!originalPropagated)
			throw std::runtime_error("original preRun exception should propagate, not onError's");
	}
	std::cout << "Test 19 passed: onError's own exception swallowed, no secondary propagation" << std::endl;

	{
		static std::atomic<int> g_lruCreateCount{0};
		if (!reg.hasEngine("LruMock")) {
			EngineDescriptor desc;
			desc.engineType = "LruMock";
			desc.converter = {mockToNative, mockToDC};
			desc.loadModel = [](const EngineCore&, const std::string& path) -> EngineInstance {
				++g_lruCreateCount;
				// clang-14 即最低工具链不支持 P0960 括号聚合初始化，显式构造
				return EngineInstance(std::make_shared<MockSession>(MockSession{path}));
			};
			if (!reg.registerEngine(desc))
				throw std::runtime_error("register LruMock engine failed");
		}

		// 创建 70 个实例，超上限 64：最旧条目被 LRU 逐出，句柄持有的实例保活
		std::vector<EngineHandle> recent;
		for (int i = 0; i < 70; ++i)
			recent.push_back(reg.getOrCreateEngine("LruMock", "lru-model-" + std::to_string(i)));
		if (!recent.back())
			throw std::runtime_error("latest instance must be cached");

		const int createsBefore = g_lruCreateCount.load();
		auto evicted = reg.getOrCreateEngine("LruMock", "lru-model-0");
		if (!evicted)
			throw std::runtime_error("evicted instance must be recreated on demand");
		if (g_lruCreateCount.load() != createsBefore + 1)
			throw std::runtime_error("recreating evicted instance must hit loadModel again");
		auto latest = reg.getOrCreateEngine("LruMock", "lru-model-69");
		if (latest != recent.back())
			throw std::runtime_error("recent entry must remain cached (same handle)");
		if (g_lruCreateCount.load() != createsBefore + 1)
			throw std::runtime_error("cached instance must not trigger re-creation");
	}
	std::cout << "Test 20 passed: instance cache LRU eviction at capacity limit" << std::endl;

	{
		static std::atomic<int> g_coreOnceInit{0};
		static std::atomic<int> g_coreOnceLoad{0};
		if (!reg.hasEngine("CoreOnce")) {
			EngineDescriptor desc;
			desc.engineType = "CoreOnce";
			desc.converter = {mockToNative, mockToDC};
			desc.createEngineCore = []() -> EngineCore {
				++g_coreOnceInit;
				return EngineCore(std::make_shared<int>(7));
			};
			desc.loadModel = [](const EngineCore&, const std::string& path) -> EngineInstance {
				++g_coreOnceLoad;
				return EngineInstance(std::make_shared<MockSession>(MockSession{path}));
			};
			if (!reg.registerEngine(desc))
				throw std::runtime_error("register CoreOnce engine failed");
		}

		auto h1 = reg.getOrCreateEngine("CoreOnce", "models/core-1.onnx");
		auto h2 = reg.getOrCreateEngine("CoreOnce", "models/core-2.onnx");
		if (!h1 || !h2)
			throw std::runtime_error("instances should be created");
		if (g_coreOnceInit.load() != 1)
			throw std::runtime_error("engine core should be initialized exactly once across models");
		if (g_coreOnceLoad.load() != 2)
			throw std::runtime_error("each modelPath should load exactly once");
		if (!h1->core() || !h1->core()->get())
			throw std::runtime_error("engine core should be bound and non-empty");
		if (h1->core().get() != h2->core().get())
			throw std::runtime_error("both instances should share the same engine core");

		auto core = reg.getOrCreateEngineCore("CoreOnce");
		if (!core || core.get() != h1->core().get())
			throw std::runtime_error("getOrCreateEngineCore should hit the cached core");
		if (g_coreOnceInit.load() != 1)
			throw std::runtime_error("getOrCreateEngineCore must not re-initialize");
	}
	std::cout << "Test 21 passed: engine core initialized once, shared across models" << std::endl;

	{
		static std::atomic<int> g_coreFailInit{0};
		static std::atomic<int> g_coreFailLoad{0};
		static std::atomic<int> g_coreFailMode{0}; // 0 成功；1 抛异常；2 返回空核心
		if (!reg.hasEngine("CoreFail")) {
			EngineDescriptor desc;
			desc.engineType = "CoreFail";
			desc.converter = {mockToNative, mockToDC};
			desc.createEngineCore = []() -> EngineCore {
				++g_coreFailInit;
				if (g_coreFailMode.load() == 1)
					throw std::runtime_error("simulated core init failure");
				if (g_coreFailMode.load() == 2)
					return EngineCore();
				return EngineCore(std::make_shared<int>(9));
			};
			desc.loadModel = [](const EngineCore&, const std::string& path) -> EngineInstance {
				++g_coreFailLoad;
				return EngineInstance(std::make_shared<MockSession>(MockSession{path}));
			};
			if (!reg.registerEngine(desc))
				throw std::runtime_error("register CoreFail engine failed");
		}

		g_coreFailMode = 1;
		bool threw = false;
		try {
			reg.getOrCreateEngine("CoreFail", "models/core-fail.onnx");
		} catch (const std::runtime_error&) {
			threw = true;
		}
		if (!threw)
			throw std::runtime_error("core init exception should propagate");
		if (g_coreFailLoad.load() != 0)
			throw std::runtime_error("loadModel must not run when core init fails");

		g_coreFailMode = 2;
		auto emptyResult = reg.getOrCreateEngine("CoreFail", "models/core-fail.onnx");
		if (emptyResult)
			throw std::runtime_error("empty core must surface as failure (nullptr)");
		if (g_coreFailLoad.load() != 0)
			throw std::runtime_error("loadModel must not run for empty core");

		g_coreFailMode = 0;
		auto recovered = reg.getOrCreateEngine("CoreFail", "models/core-fail.onnx");
		if (!recovered)
			throw std::runtime_error("core init should succeed after failure");
		if (g_coreFailInit.load() != 3)
			throw std::runtime_error("each failed attempt should retry core init");
		if (g_coreFailLoad.load() != 1)
			throw std::runtime_error("model should load exactly once after core success");
	}
	std::cout << "Test 22 passed: core failure semantics (throw/empty, no cache, retry)" << std::endl;

	{
		static std::atomic<int> g_lifeInit{0};
		static std::atomic<int> g_lifeLoad{0};
		static std::atomic<int> g_lifeModelRelease{0};
		static std::atomic<int> g_lifeCoreRelease{0};
		if (!reg.hasEngine("CoreLifecycle")) {
			EngineDescriptor desc;
			desc.engineType = "CoreLifecycle";
			desc.converter = {mockToNative, mockToDC};
			desc.createEngineCore = []() -> EngineCore {
				++g_lifeInit;
				return EngineCore(std::make_shared<int>(11));
			};
			desc.loadModel = [](const EngineCore&, const std::string& path) -> EngineInstance {
				++g_lifeLoad;
				return EngineInstance(std::make_shared<MockSession>(MockSession{path}));
			};
			desc.releaseModel = [](void*) { ++g_lifeModelRelease; };
			desc.releaseEngineCore = [](void*) { ++g_lifeCoreRelease; };
			if (!reg.registerEngine(desc))
				throw std::runtime_error("register CoreLifecycle engine failed");
		}

		auto h = reg.getOrCreateEngine("CoreLifecycle", "models/core-life.onnx");
		if (!h || g_lifeLoad.load() != 1)
			throw std::runtime_error("instance should be loaded once");
		auto core = h->core();
		if (!core)
			throw std::runtime_error("instance should carry its core");

		reg.releaseAllEngines();
		if (g_lifeModelRelease.load() != 0 || g_lifeCoreRelease.load() != 0)
			throw std::runtime_error("cache clear must not destroy handle-held objects");

		h.reset();
		if (g_lifeModelRelease.load() != 1)
			throw std::runtime_error("releaseModel should fire exactly once");
		if (g_lifeCoreRelease.load() != 0)
			throw std::runtime_error("core should stay alive while held by local handle");

		core.reset();
		if (g_lifeCoreRelease.load() != 1)
			throw std::runtime_error("releaseEngineCore should fire exactly once");

		auto h2 = reg.getOrCreateEngine("CoreLifecycle", "models/core-life2.onnx");
		if (!h2 || g_lifeInit.load() != 2)
			throw std::runtime_error("core should be re-initialized after full release");
	}
	std::cout << "Test 23 passed: two-level release, core outlives instance, hooks fire once" << std::endl;

	{
		static std::atomic<int> g_coreConcInit{0};
		static std::atomic<int> g_coreConcLoad{0};
		static std::atomic<bool> g_coreConcRelease{false};
		static std::atomic<int> g_coreConcEntered{0};
		if (!reg.hasEngine("CoreConcurrent")) {
			EngineDescriptor desc;
			desc.engineType = "CoreConcurrent";
			desc.converter = {mockToNative, mockToDC};
			desc.createEngineCore = []() -> EngineCore {
				++g_coreConcEntered;
				while (!g_coreConcRelease.load())
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
				++g_coreConcInit;
				return EngineCore(std::make_shared<int>(13));
			};
			desc.loadModel = [](const EngineCore&, const std::string& path) -> EngineInstance {
				++g_coreConcLoad;
				return EngineInstance(std::make_shared<MockSession>(MockSession{path}));
			};
			if (!reg.registerEngine(desc))
				throw std::runtime_error("register CoreConcurrent engine failed");
		}

		constexpr int kThreads = 8;
		std::atomic<int> got{0};
		std::vector<EngineHandle> handles(kThreads);
		std::vector<std::thread> threads;
		for (int i = 0; i < kThreads; ++i) {
			threads.emplace_back([&, i] {
				auto inst = reg.getOrCreateEngine("CoreConcurrent",
															"models/cc-" + std::to_string(i) + ".onnx");
				if (inst && inst->get()) {
					handles[i] = inst;
					++got;
				}
			});
		}
		// 等核心回调被进入：领导者已锁外执行，其余线程进入 core 等待
		for (int i = 0; i < 5000 && g_coreConcEntered.load() < 1; ++i)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		std::this_thread::sleep_for(std::chrono::milliseconds(20)); // 让其余线程抵达 core single-flight
		g_coreConcRelease = true;
		for (auto& t : threads)
			t.join();

		if (got != kThreads)
			throw std::runtime_error("all threads should obtain instances");
		if (g_coreConcEntered.load() != 1 || g_coreConcInit.load() != 1)
			throw std::runtime_error("core should initialize exactly once under concurrency");
		if (g_coreConcLoad.load() != kThreads)
			throw std::runtime_error("each distinct modelPath should load exactly once");
		for (int i = 1; i < kThreads; ++i) {
			if (handles[i]->core().get() != handles[0]->core().get())
				throw std::runtime_error("all instances should share one core");
		}
	}
	std::cout << "Test 24 passed: core single-flight under concurrent multi-model access" << std::endl;

	{
		static std::atomic<int> g_reCoreInit{0};
		static std::atomic<int> g_reCoreRelease{0};
		if (!reg.hasEngine("CoreRelease")) {
			EngineDescriptor desc;
			desc.engineType = "CoreRelease";
			desc.converter = {mockToNative, mockToDC};
			desc.createEngineCore = []() -> EngineCore {
				++g_reCoreInit;
				return EngineCore(std::make_shared<int>(17));
			};
			desc.loadModel = [](const EngineCore&, const std::string& path) -> EngineInstance {
				return EngineInstance(std::make_shared<MockSession>(MockSession{path}));
			};
			desc.releaseEngineCore = [](void*) { ++g_reCoreRelease; };
			if (!reg.registerEngine(desc))
				throw std::runtime_error("register CoreRelease engine failed");
		}

		auto h1 = reg.getOrCreateEngine("CoreRelease", "models/cr-1.onnx");
		if (!h1 || g_reCoreInit.load() != 1)
			throw std::runtime_error("initial core + model should load");
		auto oldCore = h1->core();
		if (!oldCore)
			throw std::runtime_error("h1 should carry the initial core");

		reg.releaseEngineCore("CoreRelease");
		if (g_reCoreRelease.load() != 0)
			throw std::runtime_error("releaseEngineCore must not destroy instance-held core");

		auto h2 = reg.getOrCreateEngine("CoreRelease", "models/cr-2.onnx");
		if (!h2 || g_reCoreInit.load() != 2)
			throw std::runtime_error("core should be re-initialized after releaseEngineCore");
		if (h2->core().get() == oldCore.get())
			throw std::runtime_error("new instance should carry the new core");
		if (g_reCoreRelease.load() != 0)
			throw std::runtime_error("old core must stay alive while h1 holds it");

		h1.reset();
		reg.releaseEngine("CoreRelease", "models/cr-1.onnx");
		oldCore.reset();
		if (g_reCoreRelease.load() != 1)
			throw std::runtime_error("old core release hook should fire once");
	}
	std::cout << "Test 25 passed: releaseEngineCore rebuilds core, old core kept alive by holders" << std::endl;

	std::cout << "\nAll EngineRegistry tests passed!" << std::endl;
}

int main() {
	try {
		runTests();
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Test failure: " << e.what() << std::endl;
		return -1;
	}
}
