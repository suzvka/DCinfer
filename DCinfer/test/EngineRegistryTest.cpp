// EngineRegistry 单元测试：注册、创建、转换钩子
#include <iostream>
#include <stdexcept>

#include "EngineRegistry.h"
#include "Node.h"

using namespace DC;

// ── Mock schema ──
static Node::Schema mockSchema() {
	Node::Schema s;
	s.inputs  = {{"in",  Tensor::TensorType::Float, sizeof(float), {}}};
	s.outputs = {{"out", Tensor::TensorType::Float, sizeof(float), {}}};
	return s;
}

// ── Mock 计算逻辑（magic 值来自 engineConfig）──
static Node::Result mockRunImpl(Node::RunContext& ctx, int magic) {
	const auto& inNT = ctx.input("in");
	const auto* inVal = inNT.as<Tensor>();
	Tensor out(Tensor::TensorType::Float, sizeof(float));
	out = inVal->item<float>() + static_cast<float>(magic);
	auto* p = new Tensor(std::move(out));
	ctx.output("out", Value(p, [](Tensor* ptr) { delete ptr; }));
	return ctx.success();
}

// ── 模拟转换钩子 ──
static Value mockToNative(const Tensor& dc) {
	auto* p = new Tensor(dc);
	return Value(p, [](Tensor* ptr) { delete ptr; });
}

static Tensor mockToDC(const void* native) {
	const auto* t = static_cast<const Tensor*>(native);
	Tensor result(Tensor::TensorType::Float, sizeof(float));
	result = t->item<float>();
	return result;
}

static void runTests() {
	auto& reg = EngineRegistry::instance();

	// ── Test 1: 注册引擎（使用 makeNodeFactory）──
	{
		EngineDescriptor desc;
		desc.engineType = "Mock";
		desc.converter  = { mockToNative, mockToDC };
		desc.factory    = makeNodeFactory<int>("Mock", mockSchema(), mockRunImpl);

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
		if (!desc) throw std::runtime_error("find returned null");
		if (desc->engineType != "Mock") throw std::runtime_error("engineType mismatch");
		if (!desc->converter.toNative) throw std::runtime_error("toNative not set");
		if (!desc->converter.toDC) throw std::runtime_error("toDC not set");
	}
	std::cout << "Test 3 passed: find engine" << std::endl;

	// ── Test 4: hasEngine / engineTypes ──
	{
		if (!reg.hasEngine("Mock")) throw std::runtime_error("hasEngine should be true");
		if (reg.hasEngine("Nonexistent")) throw std::runtime_error("hasEngine should be false");

		auto types = reg.engineTypes();
		if (types.empty()) throw std::runtime_error("engineTypes should not be empty");
		bool found = false;
		for (auto& t : types) if (t == "Mock") found = true;
		if (!found) throw std::runtime_error("Mock not found in engineTypes");
	}
	std::cout << "Test 4 passed: hasEngine / engineTypes" << std::endl;

	// ── Test 5: createNode 通过工厂创建节点 ──
	{
		int magic = 42;
		auto node = reg.createNode("Mock", "testNode", &magic);
		if (!node) throw std::runtime_error("createNode returned null");
		if (node->type() != "Mock") throw std::runtime_error("node type mismatch");
		if (node->name() != "testNode") throw std::runtime_error("node name mismatch");
		if (node->schema().inputs.size() != 1) throw std::runtime_error("schema inputs count mismatch");
		if (node->schema().outputs.size() != 1) throw std::runtime_error("schema outputs count mismatch");
	}
	std::cout << "Test 5 passed: createNode" << std::endl;

	// ── Test 6: createNode 未知引擎返回 null ──
	{
		auto node = reg.createNode("UnknownEngine", "test");
		if (node) throw std::runtime_error("createNode for unknown engine should return null");
	}
	std::cout << "Test 6 passed: createNode unknown engine" << std::endl;

	// ── Test 7: 创建的节点可以正常运行 ──
	{
		int magic = 100;
		auto node = reg.createNode("Mock", "runner", &magic);
		if (!node) throw std::runtime_error("createNode failed");

		Tensor in(Tensor::TensorType::Float, sizeof(float));
		in = 50.0f;
		auto* inPtr = new Tensor(std::move(in));
		Value nt(inPtr, [](Tensor* p) { delete p; });
		node->setInput("task1", "in", std::move(nt));

		if (!node->hasOutput("task1", "out"))
			throw std::runtime_error("output not produced");

		auto outNT = node->getOutput("task1", "out");
		auto* out = outNT.as<Tensor>();
		if (std::abs(out->item<float>() - 150.0f) > 1e-6f)
			throw std::runtime_error("output value mismatch: expected 150, got " +
				std::to_string(out->item<float>()));
	}
	std::cout << "Test 7 passed: created node runs correctly" << std::endl;

	// ── Test 8: 转换钩子功能验证 ──
	{
		Tensor dc(Tensor::TensorType::Float, sizeof(float));
		dc = 3.14f;

		// DC → Native
		auto native = mockToNative(dc);
		if (!native) throw std::runtime_error("toNative returned empty");
		auto* t = native.as<Tensor>();
		if (!t) throw std::runtime_error("toNative did not wrap Tensor");
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
