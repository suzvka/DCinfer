#include "Connector.h"

#include <atomic>
#include <memory>
#include <string>
#include <stdexcept>

namespace DC::Connector {

// ════════════════════════════════════════════
// 广播连接器
// ════════════════════════════════════════════

Node::Schema broadcastSchema(size_t downstreamCount) {
	Node::Schema s;

	// 输入：任意类型的 DC::Tensor（Void + size=0 = 不校验类型）
	s.inputs = {{"in", Node::TensorType::Void, 0, {}}};

	// 输出：N 个同类型输出口
	s.outputs.reserve(downstreamCount);
	for (size_t i = 0; i < downstreamCount; ++i) {
		s.outputs.push_back({"out_" + std::to_string(i), Node::TensorType::Void, 0, {}});
	}

	return s;
}

Node::RunFn broadcastRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto& inVal = ctx.input("in");
		const auto* inTensor = inVal.as<Tensor>();
		if (!inTensor) {
			return ctx.failure(Node::Status::InvalidInput, "Broadcast: input is not a DC::Tensor");
		}

		const auto& outputs = ctx.schema().outputs;
		const size_t n = outputs.size();
		
		for (size_t i = 1; i < n; ++i) {
			ctx.output(outputs[i].name, Value(std::make_unique<Tensor>(*inTensor)));
		}
		// output[0]：直接 move 原数据
		ctx.output(outputs[0].name, ctx.takeInput("in"));
		
		return ctx.success();
	};
}

// ════════════════════════════════════════════
// 路由连接器
// ════════════════════════════════════════════

Node::Schema routingSchema(size_t downstreamCount) {
	// 路由的 Schema 与广播完全一致：1 输入 + N 输出
	return broadcastSchema(downstreamCount);
}

Node::RunFn routingRunFn() {
	// 轮询计数器：所有同类型路由连接器共享一个递增计数器
	auto roundRobin = std::make_shared<std::atomic<size_t>>(0);

	return [roundRobin](Node::RunContext& ctx) -> Node::Result {
		const auto& inVal = ctx.input("in");
		const auto* inTensor = inVal.as<Tensor>();
		if (!inTensor) {
			return ctx.failure(Node::Status::InvalidInput, "Routing: input is not a DC::Tensor");
		}

		const auto& outputs = ctx.schema().outputs;
		const size_t n = outputs.size();

		// 轮询选取一个输出口
		const size_t idx = roundRobin->fetch_add(1, std::memory_order_relaxed) % n;

		// 零拷贝 move 到选中输出口（不写其余 N-1 个口）
		ctx.output(outputs[idx].name, ctx.takeInput("in"));

		return ctx.success();
	};
}

// ════════════════════════════════════════════
// 注册到 EngineRegistry
// ════════════════════════════════════════════

void registerBuiltinConnectors(EngineRegistry& reg) {
	// 运行时通过 broadcastSchema(n) / routingSchema(n) 创建任意下游数的实例。
	reg.registerOperator("Connector.Broadcast", broadcastSchema(1), broadcastRunFn());

	reg.registerOperator("Connector.Routing", routingSchema(1), routingRunFn());
}

} // namespace DC::Connector
