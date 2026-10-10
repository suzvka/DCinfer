#pragma once

#include "Node.h"
#include "OutputZone.h"
#include "TaskStatus.h"
#include "ResourceScheduler.h"

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
#include <vector>

namespace DC {

class GraphStore;
class OutputZone;
class SignalStore;
class ErrorTracker;
class TaskExecutionState;
struct GraphRuntimeState;

/// @brief 推理图执行引擎：事件驱动的数据流传播与调度。
class ExecutionEngine {
public:
	using TaskId = std::string;
	using TaskCompleteCallback = std::function<void(const TaskId&)>;

	/// @brief  默认最大跳数（TTL），防止循环无限传播
	static constexpr uint32_t kDefaultMaxHops = 10000;

	/// @brief 构造引擎：绑定资源调度器（不得为空）。
	explicit ExecutionEngine(std::shared_ptr<ResourceScheduler> scheduler);

	/// @brief 析构：自排水——等待在飞与已排队任务 lambda 全部退出后再释放轮次表。
	/// @note  必须由外部非 worker 方销毁：从自身 RunFn/完成回调/writeOp 中销毁
	///        本引擎或 InferGraph 会打印诊断并 std::terminate。
	~ExecutionEngine();

	ExecutionEngine(const ExecutionEngine&) = delete;
	ExecutionEngine& operator=(const ExecutionEngine&) = delete;
	ExecutionEngine(ExecutionEngine&&) = delete;
	ExecutionEngine& operator=(ExecutionEngine&&) = delete;

	/// @brief 资源调度器共享句柄。
	const std::shared_ptr<ResourceScheduler>& scheduler() const { return _scheduler; }

	/// @brief 异步启动整张图的计算（原子提交事务：并发同 ID 提交恰一方成功）。
	/// @throws GraphException NoDeclaration（声明为空）/ UnreachableDeclaration
	///         （拓扑不可达）/ DuplicateTask（同 taskId 在飞或收尾窗口内）。
	void submit(const TaskId& taskId, uint32_t maxHops,
				const std::shared_ptr<GraphRuntimeState>& state,
				std::vector<OutputDeclaration> declarations);

	/// @brief 受收尾协议保护的 task 状态写入（feedInput 专用）。
	/// @return true = writeOp 已执行；false = 收尾窗口拒绝（调用方应抛错/重试）。
	/// @note   锁序：_roundsMutex → round->m → 域/缓冲锁 单向，禁止反向。
	bool tryWriteTaskState(const TaskId& taskId, const std::function<void()>& writeOp);

	/// @brief 同步等待 task 终止。
	/// @param timeout 等待超时；<= 0 视为无限等待。
	/// @return true = 已终止且声明输出可读（takeOutput 必能取到）；false = 超时或 taskId 未知。
	bool wait(const TaskId& taskId, std::chrono::milliseconds timeout);

	/// @brief 查询 task 当前状态。
	TaskStatus status(const TaskId& taskId) const;

	/// @brief 查询 task 是否处于"终态已发布、结果收尾未完成"的窗口（此窗口注入输入会静默丢失）。
	bool isFinalizing(const TaskId& taskId) const;

	/// @brief 请求取消活动中的 task（幂等；未知或已终止返回 false）。
	///        协作式：在飞节点不中断，传播即刻停止，wait 被唤醒，状态置 Cancelled。
	bool cancel(const TaskId& taskId);

	/// @brief 释放已终止 task 的状态记录（仅"终态 + 收尾完成"可释放）。
	/// @return true = 已释放；false = 未知/活动/收尾窗口未完成。
	bool releaseTask(const TaskId& taskId);

	/// @brief 弃置在飞 task 的托管句柄：不取消任务，终态收尾时自动回收状态/结果/诊断；
	///        已终止 → 等价 releaseTask；未知 → no-op。
	void detachTask(const TaskId& taskId);

	/// @brief 设置 task 完成回调（_terminate 触发，先于结果就绪发布；回调可安全读取输出缓冲）。
	/// @note  回调内不得对同一 taskId 调用 wait()（自我阻塞），不得 submit/feedInput/
	///        releaseTask/detachTask，不得销毁本引擎/InferGraph；回调应只读取数据。
	void setTaskCompleteCallback(TaskCompleteCallback cb) {
		std::lock_guard lk(_cbMutex);
		_taskCompleteCb = std::move(cb);
	}

private:
	// 每轮 submit 对应一个 TaskGate：终态/结果可读/等待协议由自身 mutex+cv 承载；
	// execState 提交时捕获，同 ID 复用后旧轮次 lambda 无法读写新一轮执行态；
	// inflight 归零且未终止时由最后完成的 lambda 触发 _exhaustedCheck。
	struct TaskGate {
		std::atomic<bool> terminated{false};
		std::atomic<uint32_t> inflight{0};
		ExecutionEngine* engine = nullptr;
		std::shared_ptr<GraphRuntimeState> state;
		TaskId taskId;
		std::shared_ptr<TaskExecutionState> execState;

		std::mutex m;
		std::condition_variable cv;
		TaskStatus terminalStatus = TaskStatus::Running;
		bool resultsReady = false;
		bool autoRelease = false;

		~TaskGate();
	};

	/// @brief 提交一个节点执行任务（执行成功后就地传播下游）。
	void _submitNodeRun(const Node* node, const std::string& nodeName,
						std::shared_ptr<TaskGate> round, uint32_t remainingHops);

	/// @brief 传播节点输出到下游（前提：节点已执行成功）。
	void _propagateFrom(std::string nodeName, std::shared_ptr<TaskGate> round,
						uint32_t remainingHops);

	std::shared_ptr<TaskGate> _findRound(const TaskId& taskId) const;

	void _terminate(const std::shared_ptr<TaskGate>& round,
					TaskStatus terminalStatus = TaskStatus::Succeeded);
	void _exhaustedCheck(const std::shared_ptr<TaskGate>& round);

	/// @brief 弃置轮次（autoRelease）的完成回收（身份校验防误清新一轮状态）。
	void _finalizeDetached(const std::shared_ptr<TaskGate>& round);

	/// @brief 异常终止前诊断：未满足声明、信号阻塞节点写入 ErrorTracker（须在 _terminate 前调用）。
	void _diagnoseAbnormal(const std::shared_ptr<TaskGate>& round, const std::string& reason);

	// 轮次表：taskId → 当前轮次 TaskGate；复用提交时替换、释放时移除。
	// 锁序约束：仅在持 _roundsMutex 时嵌套获取 round->m，禁止反向获取。
	std::unordered_map<TaskId, std::shared_ptr<TaskGate>> _rounds;
	mutable std::mutex _roundsMutex;

	// 信号阻塞追踪：_propagateFrom 写入，_diagnoseAbnormal 读取，_terminate 清理。
	std::unordered_map<TaskId, std::unordered_set<std::string>> _blockedSkips;
	mutable std::mutex _blockedSkipsMutex;

	TaskCompleteCallback _taskCompleteCb;
	mutable std::mutex _cbMutex;

	std::shared_ptr<ResourceScheduler> _scheduler;

	// 析构自排水：派发在 _drainMutex 内登记并签发 DrainTicket（RAII 随任务
	// lambda 转移），lambda 执行完毕或被池弃置析构时回收计数，归零唤醒析构等待者。
	std::mutex _drainMutex;
	std::condition_variable _drainCv;
	bool _shuttingDown = false;
	std::atomic<uint32_t> _pendingRuns{0};

	struct DrainTicket;
};

} // namespace DC
