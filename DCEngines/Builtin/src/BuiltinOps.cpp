#include "DCEngine/BuiltinOps.h"
#include "Tensor.hpp"

#include <memory>

namespace DC::Builtin {

// ════════════════════════════════════════════
// Add 算子：两个 Float 标量相加
// ════════════════════════════════════════════

static Node::Schema addSchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("a"), Node::Port::in<float>("b")};
	s.outputs = {Node::Port::out<float>("sum")};
	return s;
}

static Node::RunFn addRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto& aVal = ctx.peek("a");
		const auto& bVal = ctx.peek("b");
		const auto* a = aVal.as<Tensor>();
		const auto* b = bVal.as<Tensor>();
		if (!a || !b)
			return ctx.failure(Node::Status::InvalidInput, "Add: inputs must be DC::Tensor");

		float sum = a->item<float>() + b->item<float>();
		auto t = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
		*t = sum;
		ctx.output("sum", Value(std::move(t)));
		return ctx.success();
	};
}

// ════════════════════════════════════════════
// Mul 算子：两个 Float 标量相乘
// ════════════════════════════════════════════

static Node::Schema mulSchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("a"), Node::Port::in<float>("b")};
	s.outputs = {Node::Port::out<float>("product")};
	return s;
}

static Node::RunFn mulRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto& aVal = ctx.peek("a");
		const auto& bVal = ctx.peek("b");
		const auto* a = aVal.as<Tensor>();
		const auto* b = bVal.as<Tensor>();
		if (!a || !b)
			return ctx.failure(Node::Status::InvalidInput, "Mul: inputs must be DC::Tensor");

		float product = a->item<float>() * b->item<float>();
		auto t = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
		*t = product;
		ctx.output("product", Value(std::move(t)));
		return ctx.success();
	};
}

// ════════════════════════════════════════════
// Identity 算子：恒等映射
// ════════════════════════════════════════════

static Node::Schema identitySchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("x")};
	s.outputs = {Node::Port::out<float>("y")};
	return s;
}

static Node::RunFn identityRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto& xVal = ctx.peek("x");
		const auto* x = xVal.as<Tensor>();
		if (!x)
			return ctx.failure(Node::Status::InvalidInput, "Identity: input must be DC::Tensor");

		float val = x->item<float>();
		auto t = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
		*t = val;
		ctx.output("y", Value(std::move(t)));
		return ctx.success();
	};
}

// ════════════════════════════════════════════
// 注册入口
// ════════════════════════════════════════════

void registerBuiltinOperators(EngineRegistry& reg) {
	reg.registerOperator("Add", addSchema(), addRunFn());
	reg.registerOperator("Mul", mulSchema(), mulRunFn());
	reg.registerOperator("Identity", identitySchema(), identityRunFn());
}

} // namespace DC::Builtin
