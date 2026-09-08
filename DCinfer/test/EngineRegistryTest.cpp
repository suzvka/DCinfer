// EngineRegistry 单元测试：注册、创建、转换钩子
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "EngineRegistry.h"
#include "Node.h"

using namespace DC;

// ── Mock schema ──
static Node::Schema mockSchema() {
	Node::Schema s;
	s.inputs = {{"in", Tensor::TensorType::Float, sizeof(float), {}}};
	s.outputs = {{"out", Tensor::TensorType::Float, sizeof(float), {}}};
	return s;
}

// ── Mock 计算逻辑（magic 值来自 engineConfig）──
static Node::Result mockRunImpl(Node::RunContext& ctx, int magic) {
	const auto& inNT = ctx.peek("in");
	const auto* inVal = inNT.as<Tensor>();
	auto t = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
	*t = inVal->item<float>() + static_cast<float>(magic);
	ctx.output("out", Value(std::move(t)));
	return ctx.success();
}

// ── 模拟转换钩子 ──
static Value mockToNative(const Tensor& dc) {
	return Value(std::make_unique<Tensor>(dc));
}

static Tensor mockToDC(const void* native) {
	const auto* t = static_cast<const Tensor*>(native);
	Tensor result(Tensor::TensorType::Float, sizeof(float));
	result = t->item<float>();
	return result;
}

// ── Mock 模型引擎：验证 createNode(modelPath) 单路径 ──
// createEngine 计数验证"一次加载 + 缓存"，端口推导与 schema 传递验证"框架驱动"

struct MockSession {
	std::string modelPath;
};

static std::atomic<int> mockEngineCreateCount{0};
static std::atomic<int> mockFactorySchemaSeen{0};

// ── 生命周期计数引擎：验证 EngineHandle 句柄语义 ──
// （releaseAllEngines 后保活 / 释放钩子恰好一次 / 缓存再加载互不干扰）

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

	desc.createEngine = [](const std::string& path) -> EngineInstance {
		++g_lifecycleCreateCount;
		return EngineInstance(std::make_shared<LifecycleSession>(LifecycleSession{path}));
	};
	desc.releaseEngine = [](void*) { ++g_lifecycleReleaseCount; };
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
										   ThreadPoolAffinity::Operator);
		if (p.engineInstance)
			node->bindEngine(p.engineInstance, p.engineInstance->descriptor());
		return node;
	};

	reg.registerEngine(desc);
}

// ── single-flight 专项引擎：交错阻塞与失败传播验证 ──

static std::atomic<int> g_blockEntered{0};   // 进入 createEngine 的次数
static std::atomic<bool> g_blockRelease{false}; // 慢路径放行开关
static std::atomic<int> g_blockCreated{0};   // 完成创建的次数

static Node::Result noopRunImpl(Node::RunContext& ctx) {
	(void)ctx;
	return ctx.success();
}

/// createEngine 慢路径（models/slow.onnx）阻塞至 g_blockRelease 放行；
/// 快路径（其他 path）立即完成——用于验证慢 key 不阻塞其他 key。
static void registerFlightEngine(EngineRegistry& reg, const std::string& type) {
	if (reg.hasEngine(type))
		return;
	EngineDescriptor desc;
	desc.engineType = type;
	desc.converter = {mockToNative, mockToDC};
	desc.createEngine = [](const std::string& path) -> EngineInstance {
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
										   ThreadPoolAffinity::Operator);
		if (p.engineInstance)
			node->bindEngine(p.engineInstance, p.engineInstance->descriptor());
		return node;
	};
	reg.registerEngine(desc);
}

static std::atomic<bool> g_failNext{true};
static std::atomic<int> g_failCalls{0};

/// createEngine 按 g_failNext 抛出异常或成功——验证失败传播与重试。
static void registerFailingEngine(EngineRegistry& reg, const std::string& type) {
	if (reg.hasEngine(type))
		return;
	EngineDescriptor desc;
	desc.engineType = type;
	desc.converter = {mockToNative, mockToDC};
	desc.createEngine = [](const std::string& path) -> EngineInstance {
		++g_failCalls;
		if (g_failNext.load())
			throw std::runtime_error("simulated engine load failure: " + path);
		return EngineInstance(std::make_shared<LifecycleSession>(LifecycleSession{path}));
	};
	desc.getInputPorts = [](const EngineInstance&) { return std::vector<Node::Port>{}; };
	desc.getOutputPorts = [](const EngineInstance&) { return std::vector<Node::Port>{}; };
	desc.factory = [](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		auto node = std::make_unique<Node>("Failing", p.nodeName, p.schema, noopRunImpl,
										   ThreadPoolAffinity::Operator);
		if (p.engineInstance)
			node->bindEngine(p.engineInstance, p.engineInstance->descriptor());
		return node;
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

	desc.createEngine = [](const std::string& path) -> EngineInstance {
		++mockEngineCreateCount;
		return EngineInstance(std::make_shared<MockSession>(MockSession{path}));
	};

	// 从实例推导端口：两个 Float 标量端口
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
		if (!p.schema.inputs.empty())
			++mockFactorySchemaSeen;
		auto node = std::make_unique<Node>("MockModel", p.nodeName, p.schema, mockModelRunImpl,
										   ThreadPoolAffinity::Operator);
		if (p.engineInstance)
			node->bindEngine(p.engineInstance, p.engineInstance->descriptor());
		return node;
	};

	reg.registerEngine(desc);
}

static void runTests() {
	auto& reg = EngineRegistry::instance();

	// ── Test 1: 注册引擎（使用 makeNodeFactory）──
	{
		EngineDescriptor desc;
		desc.engineType = "Mock";
		desc.converter = {mockToNative, mockToDC};
		desc.factory = makeNodeFactory<int>("Mock", mockSchema(), mockRunImpl);

		if (!reg.registerEngine(desc))
			throw std::runtime_error("registerEngine failed");
	}
	std::cout << "Test 1 passed: register engine" << std::endl;

	// ── Test 2: 重复注册被拒绝 ──
	{
		EngineDescriptor desc;
		desc.engineType = "Mock";
		if (reg.registerEngine(desc))
			throw std::runtime_error("duplicate registration should fail");
	}
	std::cout << "Test 2 passed: duplicate registration rejected" << std::endl;

	// ── Test 3: 按名查找引擎 ──
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

	// ── Test 4: hasEngine / engineTypes ──
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

	// ── Test 5: createNode 通过工厂创建节点 ──
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

	// ── Test 6: createNode 未知引擎返回 null ──
	{
		auto node = reg.createNode("UnknownEngine", "test");
		if (node)
			throw std::runtime_error("createNode for unknown engine should return null");
	}
	std::cout << "Test 6 passed: createNode unknown engine" << std::endl;

	// ── Test 7: 创建的节点可以正常运行 ──
	{
		int magic = 100;
		auto node = reg.createNode("Mock", "runner", &magic);
		if (!node)
			throw std::runtime_error("createNode failed");

		Tensor in(Tensor::TensorType::Float, sizeof(float));
		in = 50.0f;
		node->setInput("task1", "in", Value(std::make_unique<Tensor>(std::move(in))));
		node->tryExecute("task1");

		if (!node->hasOutput("task1", "out"))
			throw std::runtime_error("output not produced");

		auto outNT = node->takeOutput("task1", "out");
		auto* out = outNT.as<Tensor>();
		if (std::abs(out->item<float>() - 150.0f) > 1e-6f)
			throw std::runtime_error("output value mismatch: expected 150, got " + std::to_string(out->item<float>()));
	}
	std::cout << "Test 7 passed: created node runs correctly" << std::endl;

	// ── Test 8: 转换钩子功能验证 ──
	{
		Tensor dc(Tensor::TensorType::Float, sizeof(float));
		dc = 3.14f;

		// DC → Native
		auto native = mockToNative(dc);
		if (!native)
			throw std::runtime_error("toNative returned empty");
		auto* t = native.as<Tensor>();
		if (!t)
			throw std::runtime_error("toNative did not wrap Tensor");
		if (std::abs(t->item<float>() - 3.14f) > 1e-6f)
			throw std::runtime_error("toNative value mismatch");

		// Native → DC
		auto back = mockToDC(native.get());
		if (std::abs(back.item<float>() - 3.14f) > 1e-6f)
			throw std::runtime_error("toDC round-trip mismatch");
	}
	std::cout << "Test 8 passed: TensorConverter round-trip" << std::endl;

	// ── Test 9: 空 engineType 注册被拒绝 ──
	{
		EngineDescriptor desc;
		desc.engineType = "";
		if (reg.registerEngine(desc))
			throw std::runtime_error("empty engineType should be rejected");
	}
	std::cout << "Test 9 passed: empty engineType rejected" << std::endl;

	// ── Test 10: createNode(modelPath) 单路径：一次加载 + schema 传递 + 缓存命中 ──
	{
		mockEngineCreateCount = 0;
		mockFactorySchemaSeen = 0;
		registerMockModelEngine(reg, "MockModel");

		// 首次建图：加载一次，schema 由框架从实例推导并传入 factory
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

		// 同 modelPath 缓存命中：不重新加载
		auto node2 = reg.createNode("MockModel", "n2", std::string("models/a.onnx"));
		if (!node2)
			throw std::runtime_error("second createNode(modelPath) returned null");
		if (mockEngineCreateCount != 1)
			throw std::runtime_error("same modelPath should reuse cached instance");

		// 不同 modelPath 创建新实例
		auto node3 = reg.createNode("MockModel", "n3", std::string("models/b.onnx"));
		if (!node3)
			throw std::runtime_error("third createNode(modelPath) returned null");
		if (mockEngineCreateCount != 2)
			throw std::runtime_error("different modelPath should create new instance");
	}
	std::cout << "Test 10 passed: createNode(modelPath) single-load + schema passing + cache" << std::endl;

	// ── Test 11: 并发 getOrCreateEngine：同 key 只创建一个实例，全部拿到同一句柄 ──
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

	// ── Test 12: 句柄保活：releaseAllEngines 后已建图节点继续可用，销毁钩子恰好一次 ──
	{
		g_lifecycleCreateCount = 0;
		g_lifecycleReleaseCount = 0;
		registerLifecycleEngine(reg, "Lifecycle");

		auto node = reg.createNode("Lifecycle", "keep", std::string("models/lc.onnx"));
		if (!node)
			throw std::runtime_error("createNode(modelPath) failed");
		if (g_lifecycleCreateCount != 1)
			throw std::runtime_error("engine should be created exactly once");

		// 释放缓存：节点持有共享句柄，实例不被销毁，释放钩子不触发
		reg.releaseAllEngines();
		if (g_lifecycleReleaseCount != 0)
			throw std::runtime_error("releaseAllEngines must not destroy node-held instances");

		// 节点继续执行成功（旧实现此处 EngineAdapter 指针已悬空）
		Tensor in(Tensor::TensorType::Float, sizeof(float));
		in = 1.0f;
		node->setInput("task1", "in", Value(std::make_unique<Tensor>(std::move(in))));
		auto result = node->tryExecute("task1");
		if (!result.ok())
			throw std::runtime_error("node should run after releaseAllEngines");
		if (!node->hasOutput("task1", "out"))
			throw std::runtime_error("output should be produced after releaseAllEngines");

		// 释放节点：最后一个句柄析构，释放钩子恰好调用一次
		node.reset();
		if (g_lifecycleReleaseCount != 1)
			throw std::runtime_error("release hook should fire exactly once when last handle dies");
	}
	std::cout << "Test 12 passed: handle keeps engine alive after releaseAllEngines" << std::endl;

	// ── Test 13: 缓存再加载：release 后同 key 重建新实例，旧句柄持有者不受影响 ──
	{
		g_lifecycleCreateCount = 0;
		g_lifecycleReleaseCount = 0;

		auto oldNode = reg.createNode("Lifecycle", "old", std::string("models/reload.onnx"));
		if (!oldNode || g_lifecycleCreateCount != 1)
			throw std::runtime_error("initial load failed");
		auto oldHandle = reg.getOrCreateEngine("Lifecycle", "models/reload.onnx");
		if (g_lifecycleCreateCount != 1)
			throw std::runtime_error("cache hit should not reload");

		// 移除缓存条目：句柄仍被 oldNode/oldHandle 持有，实例存活
		reg.releaseEngine("Lifecycle", "models/reload.onnx");
		if (g_lifecycleReleaseCount != 0)
			throw std::runtime_error("releaseEngine must not destroy node-held instances");

		// 同 key 重新加载：创建新实例，旧持有者不受影响
		auto newNode = reg.createNode("Lifecycle", "new", std::string("models/reload.onnx"));
		if (!newNode)
			throw std::runtime_error("reload after releaseEngine failed");
		if (g_lifecycleCreateCount != 2)
			throw std::runtime_error("same key should create a fresh instance after release");

		// 旧持有者全部释放：旧实例销毁，释放钩子恰好一次
		oldNode.reset();
		oldHandle.reset();
		if (g_lifecycleReleaseCount != 1)
			throw std::runtime_error("old instance should be released exactly once");

		// 新实例：节点释放后由缓存条目继续持有；清缓存后随最后句柄析构
		newNode.reset();
		if (g_lifecycleReleaseCount != 1)
			throw std::runtime_error("cached handle should keep instance alive");
		reg.releaseAllEngines();
		if (g_lifecycleReleaseCount != 2)
			throw std::runtime_error("new instance should be released after cache clear");
	}
	std::cout << "Test 13 passed: cache reload creates fresh instance without disturbing holders" << std::endl;

	// ── Test 14: single-flight 锁外创建：慢 key 不阻塞其他 key ──
	{
		g_blockEntered = 0;
		g_blockCreated = 0;
		g_blockRelease = false;
		registerFlightEngine(reg, "Flight");

		// 线程 A：慢 key 创建（阻塞至放行）
		std::thread slowThread([&] {
			auto h = reg.getOrCreateEngine("Flight", "models/slow.onnx");
			if (!h)
				throw std::runtime_error("slow key should eventually succeed");
		});

		// 等 A 进入创建回调（此刻 A 已释放 registry 锁）
		for (int i = 0; i < 5000 && g_blockEntered.load() < 1; ++i)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		if (g_blockEntered.load() < 1)
			throw std::runtime_error("slow createEngine should have been entered");

		// 线程 B：快 key 创建——不等待 A 放行即完成
		auto fastHandle = reg.getOrCreateEngine("Flight", "models/fast.onnx");
		if (!fastHandle)
			throw std::runtime_error("fast key should not be blocked by slow key");
		if (g_blockCreated.load() != 1)
			throw std::runtime_error(
				"only fast key should have completed while slow key is still pending");

		// 放行 A，完成创建
		g_blockRelease = true;
		slowThread.join();
		if (g_blockCreated.load() != 2)
			throw std::runtime_error("both keys should be created exactly once each");
	}
	std::cout << "Test 14 passed: slow key creation does not block other keys" << std::endl;

	// ── Test 15: single-flight 失败传播：异常透传、跟随者同收失败、失败后可重试 ──
	{
		g_failNext = true;
		g_failCalls = 0;
		registerFailingEngine(reg, "Failing");

		// 首个调用者：createEngine 异常透传（保持旧行为）
		bool threw = false;
		try {
			reg.getOrCreateEngine("Failing", "models/fail.onnx");
		} catch (const std::runtime_error&) {
			threw = true;
		}
		if (!threw)
			throw std::runtime_error("createEngine exception should propagate to leader");
		if (g_failCalls != 1)
			throw std::runtime_error("leader should invoke createEngine exactly once");

		// 失败清除槽位：后续调用重试创建（失败可恢复）
		g_failNext = false;
		auto recovered = reg.getOrCreateEngine("Failing", "models/fail.onnx");
		if (!recovered)
			throw std::runtime_error("creation should succeed after a failure");
		if (g_failCalls != 2)
			throw std::runtime_error("retry should create a fresh instance");

		// 并发失败：跟随者经 shared_future 收到与领导者相同的异常
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
