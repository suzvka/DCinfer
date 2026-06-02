#pragma once

#include "Node.h"
#include "CoroScheduler.h"
#include "ThreadPool.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>

namespace DC {

// 前向声明
class GraphStore;
class OutputZone;
class SignalStore;
class ErrorTracker;

/// @brief 推理图执行引擎：协程驱动的数据流传播与调度。
///
/// 从 InferGraph 提取的独立组件，负责：
/// - 异步提交 task（submit）
/// - 协程驱动的节点间数据传播（_propagateFrom）
/// - task 生命周期管理（_terminate / _isTerminated）
/// - 超时看门狗与耗尽检测
/// - 同步等待（wait）
///
/// 不持有图拓扑、输出区、信号仓库、错误收集器——均通过参数化依赖注入。
class ExecutionEngine {
public:
	using TaskId = std::string;
	using TaskCompleteCallback = std::function<void(const TaskId&)>;

	/// @brief  默认最大跳数（TTL），防止循环无限传播
	static constexpr uint32_t kDefaultMaxHops = 10000;

	/// @brief  默认构造：自动创建内部协程调度器和默认线程池
	/// @param  schedulerThreads  协程调度器线程数
	explicit ExecutionEngine(size_t schedulerThreads = 2);

	/// @brief  构造引擎（传入外部协程调度器）
	/// @param  scheduler     协程调度器（非拥有引用，外部管理生命周期）
	/// @param  computeCfg    计算线程池配置
	/// @param  operatorCfg   算子线程池配置
	/// @param  systemCfg     系统线程池配置
	ExecutionEngine(CoroScheduler& scheduler,
					const PoolConfig& computeCfg = {},
					const PoolConfig& operatorCfg = {},
					const PoolConfig& systemCfg = {});

	~ExecutionEngine() = default;

	ExecutionEngine(const ExecutionEngine&) = delete;
	ExecutionEngine& operator=(const ExecutionEngine&) = delete;
	ExecutionEngine(ExecutionEngine&&) = default;
	ExecutionEngine& operator=(ExecutionEngine&&) = default;

	// ── 执行驱动 ──

	/// @brief  异步启动整张图的计算
	/// @throws GraphException(NoDeclaration) 若未事先调用 declareOutput
	void submit(const TaskId& taskId, std::chrono::milliseconds timeout, uint32_t maxHops,
				GraphStore& graph, OutputZone& output, SignalStore& signals, ErrorTracker& errors);

	// ── 同步等待 ──

	/// @brief  同步等待 task 完成
	/// @return true 在超时内完成，false 超时
	bool wait(const TaskId& taskId,
			  std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));

	// ── task 完成回调 ──

	/// @brief  设置 task 完成回调（每次 submit 前设置；_terminate 末尾触发）
	void setTaskCompleteCallback(TaskCompleteCallback cb) { _taskCompleteCb = std::move(cb); }

private:
	// ── 任务门控：shared_ptr 生命周期驱动耗尽检测 ──
	//
	// 每个活跃协程 + 超时看门狗各持有一份 shared_ptr<TaskGate>。
	// 当最后一个持有者析构时，若 task 未被终止，则触发 _onExhausted。
	struct TaskGate {
		std::atomic<bool> terminated{false};
		ExecutionEngine* engine = nullptr;
		OutputZone* output = nullptr;
		GraphStore* graph = nullptr;
		SignalStore* signals = nullptr;
		TaskId taskId;

		~TaskGate();
	};

	// ── 协程数据传播 ──
	Task<void> _propagateFrom(std::string nodeName, TaskId taskId,
							  std::shared_ptr<TaskGate> gate,
							  uint32_t remainingHops,
							  GraphStore& graph, OutputZone& output,
							  SignalStore& signals, ErrorTracker& errors);

	// ── 终止辅助 ──
	void _terminate(const TaskId& taskId,
					GraphStore& graph, OutputZone& output, SignalStore& signals);
	bool _isTerminated(const TaskId& taskId) const;
	void _exhaustedCheck(const TaskId& taskId, OutputZone& output,
						 GraphStore& graph, SignalStore& signals);

	// ── 线程池分发（消除重复的 affinity switch-case）──

	/// @brief  fire-and-forget 提交到对应线程池
	void _dispatchToPool(ThreadPoolAffinity affinity, const std::string& tag,
						 std::function<void()> task);

	/// @brief  协程友好提交：co_await 等待任务在线程池中执行完成
	PoolTicket _dispatchToPoolAsync(ThreadPoolAffinity affinity, const std::string& tag,
									std::function<void()> task);

	// ── 成员 ──
	std::unique_ptr<CoroScheduler> _ownedScheduler;
	CoroScheduler* _scheduler;

	ThreadPool _computePool;
	ThreadPool _operatorPool;
	ThreadPool _systemPool;

	std::unordered_set<TaskId> _terminatedTasks;
	mutable std::mutex _terminationMutex;

	// 超时看门狗线程（per-task），在 _terminate 时 join
	std::unordered_map<TaskId, std::jthread> _watchdogs;

	TaskCompleteCallback _taskCompleteCb;

	mutable std::mutex _completionMutex;
	mutable std::condition_variable _completionCv;
};

} // namespace DC
