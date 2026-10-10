#pragma once

#include <functional>
#include <string>

#include "../Node.h"
#include "../NodeExecutionGate.h"

namespace DC {

class TaskBuffer;
class SlotWorkspace;
class EngineAdapter;
struct NodeExecState;

/// @brief 无状态的执行流水线编排器（依赖全部经参数注入，独立可测）。
struct ExecutionPipeline {
	using TaskId = std::string;
	using RunFn = std::function<NodeResult(class Node::RunContext&)>;
	using CompletionFn = std::function<void(const TaskId&, const NodeResult&)>;

	/// @brief 执行完整流水线（唯一入口）。
	/// @param isCancelRequested 取消感知谓词（可选；缺省时 RunContext::isCancellationRequested 恒 false）
	/// @throws NodeException NotReady（必选输入未就绪）/ Reentrant（闸租约拒绝）
	static NodeResult execute(
		const TaskId& taskId,
		const Node& node,
		NodeExecState& exec,
		NodeExecutionGate& gate,
		std::function<bool()> isCancelRequested = {});
};

} // namespace DC
