#pragma once

#include "CompiledGraph.h"
#include "OutputZone.h"
#include "SignalStore.h"
#include "ErrorTracker.h"

#include <atomic>
#include <memory>

namespace DC {

class TaskExecutionDomain; // 定义见 Graph/internal/TaskExecutionState.h（模块内私有）

/// @brief 图运行时状态：异步飞行任务的共享生命周期锚点。
///
/// InferGraph 与全部飞行中的任务 lambda / TaskGate 共同持有
/// 本状态的 shared_ptr：图对象先行析构时，已提交任务所需的
/// 冻结快照 / 输出区 / 信号 / 诊断组件仍然存活——异步任务的生命周期安全
/// 由所有权直接保证，不再依赖成员声明顺序与析构顺序约定。
///
/// 冻结快照发布协议（首次冻结的并发安全边界）：
/// - 快照所有权（_graph）仅由 attachGraph 在 _freezeMutex 内写入一次；
/// - 全部派生状态（task 执行域闸表等）在发布前完成；
/// - 最后以 release 语义置位 _frozen——读取方经 snapshot() acquire 读
///   标志，只能观察到"尚未发布（nullptr）"或"完整初始化"两种状态，
///   不存在"看见快照但派生状态未就绪"的中间态；不存在对同一
///   shared_ptr 的无同步读写；
/// - _frozen 置位后 _graph 值与所指对象不再变更，多读者对其做
///   const 拷贝安全（shared_ptr 并发规则）。
struct GraphRuntimeState {
	GraphRuntimeState();
	~GraphRuntimeState();

	GraphRuntimeState(const GraphRuntimeState&) = delete;
	GraphRuntimeState& operator=(const GraphRuntimeState&) = delete;

	/// @brief  读取已发布的冻结快照（acquire；未冻结返回 nullptr）。
	///         快照仅经此通道读取——禁止绕过发布协议直接访问内部成员。
	std::shared_ptr<const CompiledGraph> snapshot() const {
		if (!_frozen.load(std::memory_order_acquire))
			return nullptr;
		return _graph;
	}

	OutputZone   output;  ///< 输出聚合（纯任务态：声明 + 累加 + 结果 artifact）
	std::shared_ptr<SignalStore> signals = std::make_shared<SignalStore>(); ///< 信号仓库（节点经 bindSignal 共享持有）
	ErrorTracker errors;  ///< 异步执行诊断收集

	/// task 执行域：task → {per-node IO 缓冲 + 工作槽位} + 节点执行闸表。
	/// 仅模块内使用（ExecutionEngine / InferGraph），故以不透明类型持有。
	std::unique_ptr<TaskExecutionDomain> exec;

private:
	friend class InferGraph;

	/// @brief  惰性冻结时调用：先完成全部派生状态（闸表预建），再一次性
	///         发布快照——并发读者不可能观察到未初始化完成的中间态。
	///         仅由 InferGraph::_ensureFrozen 在 _freezeMutex 内调用（单写者）。
	void attachGraph(std::shared_ptr<const CompiledGraph> snapshot);

	std::shared_ptr<const CompiledGraph> _graph; ///< 冻结快照所有者（一次性写入；发布后只读）
	std::atomic<bool> _frozen{false};            ///< 发布标志（release 写 / acquire 读）
};

} // namespace DC
