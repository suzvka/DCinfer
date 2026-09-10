#pragma once

#include <functional>
#include <string>

#include "../Node.h"
#include "../NodeExecutionGate.h"

namespace DC {

class TaskBuffer;      // 前向声明
class SlotWorkspace;   // 前向声明
class EngineAdapter;   // 前向声明
struct NodeExecState;  // 节点 task 执行态（buffer + workspace 一致性由其构造保证）

/// @brief 无状态的 7 步执行流水线编排器 + 唯一执行入口。
///
///
/// 所有依赖通过参数注入，无状态，独立可测。task 态（buffer/workspace）
struct ExecutionPipeline {
	using TaskId = std::string;
	using RunFn = std::function<NodeResult(class Node::RunContext&)>;
	using CompletionFn = std::function<void(const TaskId&, const NodeResult&)>;

	/// @brief  执行完整流水线（唯一入口）。
	/// @param  exec 节点 task 执行态（buffer/workspace 由同一 schema 构造）
	/// @param  gate 节点执行闸（图路径来自冻结期预建闸表；单节点路径随载体）
	/// @throws NodeException(NotReady)     必选输入未就绪
	/// @throws NodeException(Reentrant)    节点正被另一 task 占用（闸租约拒绝）
	static NodeResult execute(
		const TaskId& taskId,
		const Node& node,
		NodeExecState& exec,
		NodeExecutionGate& gate);
};

} // namespace DC
