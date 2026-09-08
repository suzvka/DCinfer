#pragma once

#include "Node.h"
#include "TaskStatus.h"
#include "ThreadPool.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace DC {

// 前向声明
class GraphStore;
class OutputZone;
class SignalStore;
class ErrorTracker;
struct GraphRuntimeState;

/// @brief 推理图执行引擎：事件驱动的数据流传播与调度。
///
/// 从 InferGraph 提取的独立组件，负责：
/// - 异步提交 task（submit）
/// - 节点完成事件驱动的数据传播（_propagateFrom / _submitNodeRun）
/// - task 生命周期管理（_terminate / _isTerminated）
/// - 超时看门狗与耗尽检测
/// - 同步等待（wait）
///
/// 不持有图拓扑、输出区、信号仓库、错误收集器——任务经
/// shared_ptr<GraphRuntimeState> 共享持有（GraphRuntimeState 聚合四组件，
/// 飞行任务期间状态保活），与 InferGraph 共同拥有。
///
/// 调度模型：无协程、无独立调度器线程。节点执行与数据传播打包为
/// 一个任务 lambda 提交到对应线程池，池线程执行完节点后就地传播输出，
/// 下游就绪则继续提交——数据沿图自上游向下游自然"冒泡"。
class ExecutionEngine {
public:
	using TaskId = std::string;
	using TaskCompleteCallback = std::function<void(const TaskId&)>;

	/// @brief  默认最大跳数（TTL），防止循环无限传播
	static constexpr uint32_t kDefaultMaxHops = 10000;

	/// @brief  构造引擎：自动创建三个线程池（Compute / Operator / System）
	/// @param  computeCfg    计算线程池配置
	/// @param  operatorCfg   算子线程池配置
	/// @param  systemCfg     系统线程池配置
	explicit ExecutionEngine(const PoolConfig& computeCfg = {},
							const PoolConfig& operatorCfg = {},
							const PoolConfig& systemCfg = {});

	/// @brief  析构：先在全部状态成员存活时释放残余活动门控。
	///         若留到成员析构阶段，门控析构触发的 _exhaustedCheck 将访问
	///         已析构的 _watchdogs/_blockedSkips 等状态。
	~ExecutionEngine() {
		// 先将活动门控表整体移出（锁外释放）：门控析构触发的 _exhaustedCheck
		// 可能经 _terminate 重入本表，锁内 clear 会自死锁。
		decltype(_activeGates) leftover;
		{
			std::lock_guard lk(_activeGatesMutex);
			leftover = std::move(_activeGates);
		}
		leftover.clear();
	}

	ExecutionEngine(const ExecutionEngine&) = delete;
	ExecutionEngine& operator=(const ExecutionEngine&) = delete;
	ExecutionEngine(ExecutionEngine&&) = default;
	ExecutionEngine& operator=(ExecutionEngine&&) = default;

	// ── 执行驱动 ──

	/// @brief  异步启动整张图的计算
	/// @throws GraphException(NoDeclaration) 若未事先调用 declareOutput
	/// @note   state 为图运行时状态共享句柄：任务 lambda / TaskGate / 看门狗
	///         各持一份，图对象先行析构时在飞任务所需的图组件仍存活
	void submit(const TaskId& taskId, std::chrono::milliseconds timeout, uint32_t maxHops,
				const std::shared_ptr<GraphRuntimeState>& state);

	// ── 同步等待 ──

	/// @brief  同步等待 task 终止
	/// @param  timeout 等待超时；count() <= 0 视为无限等待
	///         （与 submit 的执行超时 0=不限时约定一致）
	/// @return true 在超时内终止；false 超时，或 taskId 未知
	///         （从未提交/已释放，无限等待模式下立即返回）。任务未被取消。
	bool wait(const TaskId& taskId, std::chrono::milliseconds timeout);

	// ── task 状态与取消 ──

	/// @brief  查询 task 当前状态
	/// @return Unknown=从未提交；Running=执行中；Succeeded/Failed/TimedOut/Cancelled=已终止
	TaskStatus status(const TaskId& taskId) const;

	/// @brief  请求取消活动中的 task（幂等；未知或已终止返回 false）。
	///         协作式取消：在飞节点执行不会被中断，传播链即刻停止，
	///         节点缓冲与信号照常清理，wait() 被唤醒，状态置 Cancelled。
	bool cancel(const TaskId& taskId);

	/// @brief  释放已终止 task 的状态记录（结果与诊断由上层一并清理）
	/// @note   活动（Running）task 不可释放；释放后 status 返回 Unknown
	void releaseTask(const TaskId& taskId);

	// ── 分组限流 ──

	/// @brief  注册分组限流（组信号量由所有线程池共享，注册一次全局生效）
	/// @param  tag       分组标识
	/// @param  limit     最大并发执行数
	/// @note   组限流不区分线程池归属：信号量跨池共享、全局互斥，
	///         公开 API 不暴露模型并不区分的 affinity 维度
	void registerGroupLimit(const std::string& tag, size_t limit);

	// ── task 完成回调 ──

	/// @brief  设置 task 完成回调（每次 submit 前设置；_terminate 末尾触发）
	///         线程安全：与 _terminate 的读取之间以互斥锁同步
	void setTaskCompleteCallback(TaskCompleteCallback cb) {
		std::lock_guard lk(_cbMutex);
		_taskCompleteCb = std::move(cb);
	}

private:
	// ── 任务门控：shared_ptr 生命周期驱动耗尽检测 ──
	//
	// 每个飞行中的任务 lambda（含后续传播链）与超时看门狗各持有一份
	// shared_ptr<TaskGate>。当最后一个持有者析构时，若 task 未被终止，
	// 则触发 _exhaustedCheck。
	struct TaskGate {
		std::atomic<bool> terminated{false};
		ExecutionEngine* engine = nullptr;
		/// 图运行时状态共享句柄：TaskGate、任务 lambda、看门狗与 InferGraph
		/// 共同持有，图对象先行析构时在飞任务所需的图组件仍存活
		std::shared_ptr<GraphRuntimeState> state;
		TaskId taskId;

		~TaskGate();
	};

	// ── 事件驱动数据传播 ──

	/// @brief  提交一个节点执行任务（tryExecute + 成功后就地传播）
	/// @note   由 submit 入口与传播下游共用；任务在节点 affinity 对应线程池执行
	void _submitNodeRun(const Node* node, const std::string& nodeName, const TaskId& taskId,
						std::shared_ptr<TaskGate> gate, uint32_t remainingHops,
						const std::shared_ptr<GraphRuntimeState>& state);

	/// @brief  传播节点输出到下游（调用前提：节点已由 _submitNodeRun 执行成功）
	void _propagateFrom(std::string nodeName, TaskId taskId,
						std::shared_ptr<TaskGate> gate,
						uint32_t remainingHops,
						const std::shared_ptr<GraphRuntimeState>& state);

	// ── 终止辅助 ──
	void _terminate(const TaskId& taskId,
					const std::shared_ptr<GraphRuntimeState>& state,
					TaskStatus terminalStatus = TaskStatus::Succeeded);
	bool _isTerminated(const TaskId& taskId) const;
	void _exhaustedCheck(const TaskId& taskId,
						 const std::shared_ptr<GraphRuntimeState>& state);

	// ── 运行时诊断 ──

	/// @brief  异常终止前的诊断：将未满足声明、信号阻塞节点等信息写入 ErrorTracker。
	///         必须在 _terminate 之前调用（_terminate 会清理 OutputZone 和 _blockedSkips）。
	/// @param  reason  终止原因描述（如 "task timed out (5000ms)"）
	void _diagnoseAbnormal(const TaskId& taskId, const std::string& reason,
						   const std::shared_ptr<GraphRuntimeState>& state);

	// ── 线程池分发（消除重复的 affinity switch-case）──

	/// @brief  fire-and-forget 提交到对应线程池
	void _dispatchToPool(ThreadPoolAffinity affinity, const std::string& tag,
						 std::function<void()> task);

	// ── 成员 ──
	// 声明顺序即析构顺序约束：
	//   状态成员最先声明 → 最后析构；线程池最后声明 → 最先析构。
	// 析构顺序：池(shutdown/join worker) → 共享表 → 状态 → 在飞看门狗(join) → 退役看门狗(join)。
	// 保证池 worker 上的任务 lambda 在 join 期间访问 _isTerminated/_watchdogs
	// 等状态、以及向池提交任务时，所有对象均存活。
	// （图组件生命周期由 GraphRuntimeState shared_ptr 保证，不依赖本表顺序。）
	// task 状态表：Running → 终态（Succeeded/Failed/TimedOut/Cancelled）。
	// submit 时活动 ID 拒绝重复提交；已终止 ID 复用时清除旧状态。
	// 同时承担原 _terminatedTasks 的传播拦截与 wait 谓词职责。
	std::unordered_map<TaskId, TaskStatus> _taskStates;
	mutable std::mutex _terminationMutex;

	// 活动任务门控表：submit 注册、_terminate 移除；支撑 cancel() 定位门控。
	// 声明位置在线程池之前：析构时池先行 shutdown，残余门控的
	// _exhaustedCheck 访问的引擎状态成员（本表及上方互斥锁）仍然存活。
	std::unordered_map<TaskId, std::shared_ptr<TaskGate>> _activeGates;
	std::mutex _activeGatesMutex;

	// 看门狗退役列表：超时路径中，看门狗线程会在 _terminate 内尝试回收自身，
	// 在自身线程 join 自身将抛 resource_deadlock_would_occur，并因自 noexcept
	// 析构逃逸触发 std::terminate——此类 jthread 移入此列表，由引擎析构统一
	// join（彼时看门狗 lambda 早已返回，join 立即完成）。
	// 声明顺序约束：必须先于 _watchdogs——析构时先回收在飞看门狗（其
	// _terminate 可能仍向本列表移交自身），最后才回收本列表。
	std::vector<std::jthread> _retiredWatchdogs;

	// 超时看门狗线程（per-task），在 _terminate 时回收。
	// _watchdogsMutex 保护注册/回收：submit（提交方线程）与 _terminate
	//（看门狗线程、池 worker 线程）对该 map 的访问无其他同步。
	std::mutex _watchdogsMutex;
	std::unordered_map<TaskId, std::jthread> _watchdogs;

	// 信号阻塞追踪：记录每个 task 在传播过程中因信号阻塞而被跳过的节点名。
	// 由 _propagateFrom 写入，_diagnoseAbnormal 读取，_terminate 清理。
	std::unordered_map<TaskId, std::unordered_set<std::string>> _blockedSkips;
	mutable std::mutex _blockedSkipsMutex;

	TaskCompleteCallback _taskCompleteCb;
	mutable std::mutex _cbMutex; // 保护 _taskCompleteCb 的跨线程读写

	mutable std::mutex _completionMutex;
	mutable std::condition_variable _completionCv;

	// 跨池共享的组信号量注册表（注入三个线程池，实现混合 affinity 分组互斥）
	std::shared_ptr<GroupSemaphoreRegistry> _sharedGroups;

	ThreadPool _computePool;
	ThreadPool _operatorPool;
	ThreadPool _systemPool;
};

} // namespace DC
