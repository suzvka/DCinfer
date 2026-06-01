#pragma once

#include <functional>
#include <string>

#include "../Node.h"

namespace DC {

class TaskBuffer;      // 前向声明
class SlotWorkspace;   // 前向声明
class EngineAdapter;   // 前向声明
class CoroutineBridge; // 前向声明

/// @brief 无状态的 7 步执行流水线编排器。
///
/// 将原 Node::_checkAndExecute 的流水线提取为纯函数：
///   ① 加载输入 → ② 清空输出 → ②½ preRun → ③ RunFn
///   → ③¼ onError → ③½ synchronize → ③¾ postRun
///   → ④ 保存输出 → ⑤ 验证完整性 → ⑥ 清理输入 → ⑥½ 通知协程 → ⑦ 回调
///
/// 所有依赖通过参数注入，无状态，独立可测。
struct ExecutionPipeline {
	using TaskId = std::string;
	using RunFn = std::function<NodeResult(class Node::RunContext&)>;
	using CompletionFn = std::function<void(const TaskId&, const NodeResult&)>;

	/// @brief  执行完整流水线。
	static NodeResult execute(
		const TaskId& taskId,
		TaskBuffer& buffer,
		SlotWorkspace& workspace,
		EngineAdapter& engine,
		const RunFn& fn,
		const NodeSchema& schema,
		CoroutineBridge& bridge,
		const CompletionFn& onComplete,
		const std::string& nodeType,
		const std::string& nodeName);
};

} // namespace DC
