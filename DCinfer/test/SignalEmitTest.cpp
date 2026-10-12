// 算法提供者发布输出信号（Phase 3）：RunContext::emitSignal 写入 <node>.<port> task 级 wire，
// 时机完全由算法提供者在 RunFn 内决定（框架不自动发布）。
#include "TestHarness.h"
#include "Node.h"
#include "Tensor.hpp"
#include "SignalStore.h"

#include <iostream>
#include <stdexcept>

using namespace DC;

namespace {

Value makeFloatTensor(float v) {
	auto t = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
	*t = v;
	return Value(std::move(t));
}

Node::Schema passSchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("x")};
	s.outputs = {Node::Port::out<float>("y")};
	return s;
}

/// A：产出后发布 sig.done（算法自决时机）
Node::Result emitRun(Node::RunContext& ctx) {
	const auto* in = ctx.peek("x").as<Tensor>();
	if (!in)
		return ctx.failure(Node::Status::InvalidInput, "missing x");
	auto t = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
	*t = in->item<float>();
	ctx.output("y", Value(std::move(t)));
	ctx.emitSignal("sig.done", true);
	return ctx.success();
}

/// B：透传 +10（仅当 gate 被 A 发布的信号放行时才会执行）
Node::Result passRun(Node::RunContext& ctx) {
	const auto* in = ctx.peek("x").as<Tensor>();
	if (!in)
		return ctx.failure(Node::Status::InvalidInput, "missing x");
	auto t = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
	*t = in->item<float>() + 10.0f;
	ctx.output("y", Value(std::move(t)));
	return ctx.success();
}

} // namespace

static void runTests() {
	// 真实消费路径：A 在 RunFn 内发布 sig.done；B 的 gate 绑定该 wire。
	// 任务内 B 的 isBlocked 读到 true → 放行 → 产出；若 A 未发布，B 终受阻，声明不可满足。
	{
		TestHarness h;
		auto a = std::make_unique<Node>("Builtin", "a", passSchema(), emitRun);
		a->attachSignalStore(h.signalStore());
		h.addNode(std::move(a));
		auto b = std::make_unique<Node>("Builtin", "b", passSchema(), passRun);
		b->bindSignal(h.signalStore(), "a.sig.done");
		h.addNode(std::move(b));
		h.connect("a", "y", "b", "x");
		h.bindOutput("y", "b", "y");
		h.feedInput("t1", "a", "x", makeFloatTensor(1.0f));
		h.submit("t1", "b", "y");
		if (!h.awaitCompletion("t1"))
			throw std::runtime_error("task should complete within timeout");
		if (!h.hasOutput("t1", "b", "y"))
			throw std::runtime_error("B must run: A's emitted sig.done released its gate");
	}
	std::cout << "Test passed: emitted signal releases downstream gate (in-task consumption)" << std::endl;

	// 未绑定仓库：emitSignal 空操作，任务仍正常完成
	{
		TestHarness h;
		h.addNode(std::make_unique<Node>("Builtin", "m", passSchema(), emitRun));
		h.bindOutput("y", "m", "y");
		h.feedInput("t2", "m", "x", makeFloatTensor(1.0f));
		h.submit("t2", "m", "y");
		if (!h.awaitCompletion("t2"))
			throw std::runtime_error("task without attached store should still complete");
		if (!h.hasOutput("t2", "m", "y"))
			throw std::runtime_error("node without attached store must still produce output");
	}
	std::cout << "Test passed: emitSignal is a no-op without attached store" << std::endl;

	std::cout << "\nAll SignalEmit tests passed!" << std::endl;
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
