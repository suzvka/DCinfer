#pragma once

#include "CompiledGraph.h"
#include "OutputZone.h"
#include "SignalStore.h"
#include "ErrorTracker.h"

#include <atomic>
#include <memory>

namespace DC {

class TaskExecutionDomain; // 定义见 Graph/internal/TaskExecutionState.h

/// @brief 图运行时状态：异步飞行任务的共享生命周期锚点。
///
/// InferGraph 与飞行中的任务 lambda、TaskGate 共同持有本状态：图对象先行析构时，
/// 任务所需的快照、输出区、信号与诊断仍然存活。
/// 快照发布协议：attachGraph 完成全部派生状态后以 release 语义置位 _frozen，
/// snapshot 经 acquire 读，读者只会观察到未发布或完整初始化两种状态。
struct GraphRuntimeState {
	GraphRuntimeState();
	~GraphRuntimeState();

	GraphRuntimeState(const GraphRuntimeState&) = delete;
	GraphRuntimeState& operator=(const GraphRuntimeState&) = delete;

	/// @brief 读取已发布的冻结快照；未冻结返回 nullptr。
	std::shared_ptr<const CompiledGraph> snapshot() const {
		if (!_frozen.load(std::memory_order_acquire))
			return nullptr;
		return _graph;
	}

	OutputZone output;
	std::shared_ptr<SignalStore> signals = std::make_shared<SignalStore>();
	ErrorTracker errors;

	// task 执行域：per-node IO 缓冲、工作槽位与节点执行闸表。
	std::unique_ptr<TaskExecutionDomain> exec;

private:
	friend class InferGraph;

	/// @brief 惰性冻结时调用：完成全部派生状态后一次性发布快照；单写者，在 _freezeMutex 内。
	void attachGraph(std::shared_ptr<const CompiledGraph> snapshot);

	std::shared_ptr<const CompiledGraph> _graph;
	std::atomic<bool> _frozen{false};
};

} // namespace DC
