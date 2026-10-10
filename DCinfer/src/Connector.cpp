#include "Connector.h"

#include <memory>
#include <string>
#include <stdexcept>

namespace DC::Connector {

Node::Schema broadcastSchema(size_t downstreamCount) {
	Node::Schema s;

	// Void 加 size=0 表示不校验类型
	s.inputs = {{"in", Node::TensorType::Void, 0, {}}};

	s.outputs.reserve(downstreamCount);
	for (size_t i = 0; i < downstreamCount; ++i) {
		s.outputs.push_back({"out_" + std::to_string(i), Node::TensorType::Void, 0, {}});
	}

	return s;
}

Node::RunFn broadcastRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		if (!ctx.peek("in").as<Tensor>()) {
			return ctx.failure(Node::Status::InvalidInput, "Broadcast: input is not a DC::Tensor");
		}

		const auto& outputs = ctx.schema().outputs;
		const size_t n = outputs.size();

		if (n == 1) {
			// 单下游：零拷贝直通，无需冻结
			ctx.output(outputs[0].name, ctx.pop("in"));
			return ctx.success();
		}

		// 多下游：一次性冻结后共享 N 份；出口经 isPublished 产出独立可变副本
		Value in = ctx.pop("in");
		if (auto* t = in.as<Tensor>())
			t->freeze();
		for (const auto& p : outputs)
			ctx.output(p.name, in.share());
		return ctx.success();
	};
}

void registerBuiltinConnectors(EngineRegistry& reg) {
	// 注册 1 对 1 占位模板；运行时按实际下游数创建实例。
	reg.registerOperator("Connector.Broadcast", broadcastSchema(1), broadcastRunFn());
}

} // namespace DC::Connector
