#pragma once

#include "CompiledGraph.h"
#include "OutputZone.h"
#include "SignalStore.h"
#include "ErrorTracker.h"

#include <memory>

namespace DC {

class TaskExecutionDomain; // 定义见 Graph/internal/TaskExecutionState.h（模块内私有）

/// @brief 图运行时状态：异步飞行任务的共享生命周期锚点。
///
/// InferGraph 与全部飞行中的任务 lambda / TaskGate / 超时看门狗共同持有
/// 本状态的 shared_ptr：图对象先行析构时，已提交任务所需的
/// 冻结快照 / 输出区 / 信号 / 诊断组件仍然存活——异步任务的生命周期安全
/// 由所有权直接保证，不再依赖成员声明顺序与析构顺序约定。
struct GraphRuntimeState {
	GraphRuntimeState();
	~GraphRuntimeState();

	GraphRuntimeState(const GraphRuntimeState&) = delete;
	GraphRuntimeState& operator=(const GraphRuntimeState&) = delete;

	/// @brief  惰性冻结时调用：登记快照并按源图节点集合预建节点执行闸
	void attachGraph(std::shared_ptr<const CompiledGraph> snapshot);

	/// 冻结图快照（惰性冻结：首次运行期 API 时由 InferGraph 填充）。
	/// 运行期拓扑与图级签名的唯一来源——结构不可变（Build → Freeze → Execute）。
	std::shared_ptr<const CompiledGraph> graph;
	OutputZone   output;  ///< 输出聚合（纯任务态：声明 + 累加 + 结果 artifact）
	std::shared_ptr<SignalStore> signals = std::make_shared<SignalStore>(); ///< 信号仓库（节点经 bindSignal 共享持有）
	ErrorTracker errors;  ///< 异步执行诊断收集

	/// task 执行域：task → {per-node IO 缓冲 + 工作槽位} + 节点执行闸表。
	/// 原 Node 内嵌 task 态的迁入地（task 态归 task 域，Node 退化为纯 plan 对象）。
	/// 仅模块内使用（ExecutionEngine / InferGraph），故以不透明类型持有。
	std::unique_ptr<TaskExecutionDomain> exec;
};

} // namespace DC
