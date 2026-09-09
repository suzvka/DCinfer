#include "Connector.h"

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
		const auto& inVal = ctx.peek("in");
		const auto* inTensor = inVal.as<Tensor>();
		if (!inTensor) {
			return ctx.failure(Node::Status::InvalidInput, "Broadcast: input is not a DC::Tensor");
		}

		const auto& outputs = ctx.schema().outputs;
		const size_t n = outputs.size();

		if (n == 1) {
			// 单下游：零拷贝 move，等效导线直通
			ctx.output(outputs[0].name, ctx.pop("in"));
		} else {
			// 多下游：拷贝 N-1 份，最后一份 move
			for (size_t i = 1; i < n; ++i) {
				ctx.output(outputs[i].name, Value(std::make_unique<Tensor>(*inTensor)));
			}
			ctx.output(outputs[0].name, ctx.pop("in"));
		}

		return ctx.success();
	};
}

// ════════════════════════════════════════════
// 注册到 EngineRegistry
// ════════════════════════════════════════════

void registerBuiltinConnectors(EngineRegistry& reg) {
	// 注册 1→1 退化版本作为占位模板。
	// 运行时通过 broadcastSchema(n) 创建任意下游数的实例。
	reg.registerOperator("Connector.Broadcast", broadcastSchema(1), broadcastRunFn());
}

} // namespace DC::Connector
