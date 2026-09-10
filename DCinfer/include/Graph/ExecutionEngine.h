#pragma once

#include "Node.h"
#include "TaskStatus.h"
#include "ThreadPool.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

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
/// - 传播耗尽检测（节点自报失败 → 终止为 Failed；纯信号停滞 → 宿主护栏）
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
	///         已析构的 _blockedSkips 等状态。
	/// @note   定义于 .cpp（先移出并释放残余门控，再经成员逆序析构关闭线程池）。
	~ExecutionEngine();

	ExecutionEngine(const ExecutionEngine&) = delete;
	ExecutionEngine& operator=(const ExecutionEngine&) = delete;
	ExecutionEngine(ExecutionEngine&&) = default;
	ExecutionEngine& operator=(ExecutionEngine&&) = default;

	// ── 执行驱动 ──

	/// @brief  异步启动整张图的计算
	/// @throws GraphException(NoDeclaration) 若提交时未携带输出声明
	///         （InferGraph::submit 的 declarations / submitBound 已先行声明）
	/// @throws GraphException(UnreachableDeclaration) 声明目标在运行时视图上
	///         从已注入输入的节点集合纯拓扑不可达（构图/断链错误，提交期即暴露）
	/// @note   state 为图运行时状态共享句柄：任务 lambda / TaskGate 各持一份，
	///         图对象先行析构时在飞任务所需的图组件仍存活
	void submit(const TaskId& taskId, uint32_t maxHops,
				const std::shared_ptr<GraphRuntimeState>& state);

	// ── 同步等待 ──

	/// @brief  同步等待 task 终止
	/// @param  timeout 等待超时；count() <= 0 视为无限等待
	/// @return true 在超时内终止且声明输出已就绪可读取（wait 返回后
	///         takeOutput 必能取到已声明输出）；false 超时，或 taskId 未知
	///         （从未提交/已释放，无限等待模式下立即返回）。任务未被取消。
	bool wait(const TaskId& taskId, std::chrono::milliseconds timeout);

	// ── task 状态与取消 ──

	/// @brief  查询 task 当前状态
	/// @return Unknown=从未提交；Running=执行中；Succeeded/Failed/Cancelled=已终止
	TaskStatus status(const TaskId& taskId) const;

	/// @brief  请求取消活动中的 task（幂等；未知或已终止返回 false）。
	///         协作式取消：在飞节点执行不会被中断，传播链即刻停止，
	///         节点缓冲与信号照常清理，wait() 被唤醒，状态置 Cancelled。
	bool cancel(const TaskId& taskId);

	/// @brief  释放已终止 task 的状态记录（结果与诊断由上层一并清理）
	/// @note   活动（Running）task 不可释放；释放后 status 返回 Unknown
	void releaseTask(const TaskId& taskId);

	// ── task 完成回调 ──

	/// @brief  设置 task 完成回调（每次 submit 前设置；_terminate 步骤② 触发，
	///         先于结果就绪发布——回调可安全读取 task 缓冲中尚存的输出）
	///         线程安全：与 _terminate 的读取之间以互斥锁同步
	/// @note   回调内不得对同一 taskId 调用 wait()：回调先于 resultsReady
	///         置位执行，等待将自我阻塞；回调应只读取/捕获数据
	void setTaskCompleteCallback(TaskCompleteCallback cb) {
		std::lock_guard lk(_cbMutex);
		_taskCompleteCb = std::move(cb);
	}

private:
	// ── 任务门控：在飞计数驱动耗尽检测 ──
	//
	// 每次节点执行 lambda 提交前 inflight+1、lambda 收尾（RAII）时 -1；
	// 归零且 task 未终止时由最后完成的 lambda 触发 _exhaustedCheck——取代原
	// "最后一个持有者析构"方案（活动门控表强持有 gate 至 _terminate，
	// 传播耗尽时析构实际不会发生，耗尽检测依赖看门狗收尾）。
	struct TaskGate {
		std::atomic<bool> terminated{false};
		/// 在飞节点执行 lambda 计数（submit 入口与传播下游提交时 +1）
		std::atomic<uint32_t> inflight{0};
		ExecutionEngine* engine = nullptr;
		/// 图运行时状态共享句柄：TaskGate、任务 lambda 与 InferGraph 共同持有，
		/// 图对象先行析构时在飞任务所需的图组件仍存活
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
	/// @brief  结果可读判定：声明输出已全部抢救进 OutputZone（wait 谓词绑定点）
	bool _resultsReady(const TaskId& taskId) const;
	void _exhaustedCheck(const TaskId& taskId,
						 const std::shared_ptr<GraphRuntimeState>& state);

	// ── 运行时诊断 ──

	/// @brief  异常终止前的诊断：将未满足声明、信号阻塞节点等信息写入 ErrorTracker。
	///         必须在 _terminate 之前调用（_terminate 会清理 OutputZone 和 _blockedSkips）。
	/// @param  reason  终止原因描述（如 "propagation exhausted with failed node 'x'"）
	void _diagnoseAbnormal(const TaskId& taskId, const std::string& reason,
						   const std::shared_ptr<GraphRuntimeState>& state);

	// ── 线程池分发（消除重复的 affinity switch-case）──

	/// @brief  fire-and-forget 提交到对应线程池
	void _dispatchToPool(ThreadPoolAffinity affinity,
						 std::function<void()> task);

	// ── 成员 ──
	// 声明顺序即析构顺序约束：
	//   状态成员最先声明 → 最后析构；线程池最后声明 → 最先析构。
	// 析构顺序：池(shutdown/join worker) → 共享表 → 状态。
	// 保证池 worker 上的任务 lambda 在 join 期间访问 _isTerminated
	// 等状态、以及向池提交任务时，所有对象均存活。
	// （图组件生命周期由 GraphRuntimeState shared_ptr 保证，不依赖本表顺序。）
	// task 状态表：Running → 终态（Succeeded/Failed/Cancelled）。
	// 终态发布（status 迁移）与"结果可读"是两个完成点：resultsReady 在
	// _terminate 完成声明输出抢救（步骤⑥）后置位，wait() 谓词绑定它，
	// 保证 wait 返回后经 takeOutput 必能读到声明输出。
	// submit 时活动 ID 拒绝重复提交；已终止 ID 复用时清除旧记录（含 resultsReady）。
	struct TaskStateRecord {
		TaskStatus status = TaskStatus::Running;
		bool resultsReady = false;
	};
	std::unordered_map<TaskId, TaskStateRecord> _taskStates;
	mutable std::mutex _terminationMutex;

	// 活动任务门控表：submit 注册、_terminate 移除；支撑 cancel() 定位门控。
	// 声明位置在线程池之前：析构时池先行 shutdown，残余门控的
	// _exhaustedCheck 访问的引擎状态成员（本表及上方互斥锁）仍然存活。
	std::unordered_map<TaskId, std::shared_ptr<TaskGate>> _activeGates;
	std::mutex _activeGatesMutex;

	// 信号阻塞追踪：记录每个 task 在传播过程中因信号阻塞而被跳过的节点名。
	// 由 _propagateFrom 写入，_diagnoseAbnormal 读取，_terminate 清理。
	std::unordered_map<TaskId, std::unordered_set<std::string>> _blockedSkips;
	mutable std::mutex _blockedSkipsMutex;

	TaskCompleteCallback _taskCompleteCb;
	mutable std::mutex _cbMutex; // 保护 _taskCompleteCb 的跨线程读写

	mutable std::mutex _completionMutex;
	mutable std::condition_variable _completionCv;

	ThreadPool _computePool;
	ThreadPool _operatorPool;
	ThreadPool _systemPool;
};

} // namespace DC
