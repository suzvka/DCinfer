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

// 前向声明
class GraphStore;
class OutputZone;
class SignalStore;
class ErrorTracker;
class TaskExecutionState;
struct GraphRuntimeState;

/// @brief 推理图执行引擎：事件驱动的数据流传播与调度。
///
/// 从 InferGraph 提取的独立组件，负责：
/// - 异步提交 task（submit）
/// - 节点完成事件驱动的数据传播（_propagateFrom / _submitNodeRun）
/// - task 生命周期管理（_terminate 终态收尾 / _exhaustedCheck 耗尽检测）
/// - 传播耗尽检测（节点自报失败 → 终止为 Failed；纯信号停滞 → 宿主护栏）
/// - 同步等待（wait）
class ExecutionEngine {
public:
	using TaskId = std::string;
	using TaskCompleteCallback = std::function<void(const TaskId&)>;

	/// @brief  默认最大跳数（TTL），防止循环无限传播
	static constexpr uint32_t kDefaultMaxHops = 10000;

	/// @brief  构造引擎：绑定资源调度器（进程级共享的执行载体）。
	/// @param  scheduler 调度器共享句柄（不得为空）
	/// @throws std::invalid_argument scheduler 为空
	explicit ExecutionEngine(std::shared_ptr<ResourceScheduler> scheduler);

	/// @brief  析构：自排水——停止新派发、标记全部轮次终止、等待在飞与
	///         已排队任务 lambda 全部退出（引擎回访归零），再移出并释放
	///         轮次表。共享调度器不受影响（仅引擎自身退出）。
	/// @note   定义于 .cpp；析构不得从调度器 worker 线程上调用（任务
	///         lambda 内析构图会自我等待）。
	~ExecutionEngine();

	ExecutionEngine(const ExecutionEngine&) = delete;
	ExecutionEngine& operator=(const ExecutionEngine&) = delete;
	// 含不可移动成员（线程池/互斥），移动操作显式删除（此前 = default 实为删除）
	ExecutionEngine(ExecutionEngine&&) = delete;
	ExecutionEngine& operator=(ExecutionEngine&&) = delete;

	// ── 执行载体 ──

	/// @brief  资源调度器共享句柄（执行派发载体；跨图共享承载进程预算）
	const std::shared_ptr<ResourceScheduler>& scheduler() const { return _scheduler; }

	// ── 执行驱动 ──

	/// @brief  异步启动整张图的计算（原子提交事务）。
	/// @param  declarations 期望产出：{nodeName, portName, count} 列表。声明清理与
	///         写入、准入检查、轮次登记在同一临界区完成（H-3/H-5）——并发同 ID
	///         提交恰有一方成功，败者抛错且零副作用，不可能破坏在飞轮次状态。
	/// @throws GraphException(NoDeclaration) declarations 为空
	/// @throws GraphException(UnreachableDeclaration) 声明目标在运行时视图上
	///         从已注入输入的节点集合纯拓扑不可达（构图/断链错误，提交期即暴露）
	/// @throws GraphException(DuplicateTask) 同 taskId 轮次仍在运行，或处于
	///         "终态已发布、结果可读（wait 返回）前"的收尾窗口——复用/释放/
	///         再喂输入均须待 waitForResult 返回后
	/// @note   state 为图运行时状态共享句柄：任务 lambda / TaskGate 各持一份，
	///         图对象先行析构时在飞任务所需的图组件仍存活
	void submit(const TaskId& taskId, uint32_t maxHops,
				const std::shared_ptr<GraphRuntimeState>& state,
				std::vector<OutputDeclaration> declarations);

	/// @brief  受收尾协议保护的 task 状态写入（feedInput 专用）。
	///         writeOp 在「任务仍可接收输入」时于轮次锁内执行：
	///         - 无轮次（从未提交/已释放）：不存在并发收尾，直接执行；
	///         - 有轮次且非收尾窗口（Running，或收尾已完成）：锁内执行——
	///           与 _terminate 的结果抢救段（clearTaskState）同一互斥，
	///           写入不可能在检查后被收尾摘除；
	///         - 收尾窗口（终态已发布、resultsReady 未置）：拒绝。
	/// @return true = writeOp 已执行；false = 收尾窗口拒绝（调用方应抛错/重试）
	/// @note   锁序约定：_roundsMutex → round->m → 域/缓冲锁 单向；
	///         round->m 内绝不获取 _roundsMutex。
	bool tryWriteTaskState(const TaskId& taskId, const std::function<void()>& writeOp);

	// ── 同步等待 ──

	/// @brief  同步等待 task 终止
	/// @param  timeout 等待超时；count() <= 0 视为无限等待
	/// @return true 在超时内终止且声明输出已就绪可读取（wait 返回后
	///         takeOutput 必能取到已声明输出）；false 超时，或 taskId 未知
	///         （从未提交/已释放，立即返回）。任务未被取消。
	/// @note   谓词（终态 + 结果可读）与状态发布绑定同一把轮次锁——
	///         不存在"已终止但通知丢失"的丢失唤醒窗口。
	bool wait(const TaskId& taskId, std::chrono::milliseconds timeout);

	// ── task 状态与取消 ──

	/// @brief  查询 task 当前状态
	/// @return Unknown=从未提交；Running=执行中；Succeeded/Failed/Cancelled=已终止
	TaskStatus status(const TaskId& taskId) const;

	/// @brief  查询 task 是否处于"终态已发布、结果收尾未完成"的窗口。
	///         供上层（feedInput 准入）区分"可复用"与"收尾中"——收尾窗口内
	///         注入输入会写入随旧轮 clearTaskState 摘除的执行态，静默丢失。
	bool isFinalizing(const TaskId& taskId) const;

	/// @brief  请求取消活动中的 task（幂等；未知或已终止返回 false）。
	///         协作式取消：在飞节点执行不会被中断，传播链即刻停止，
	///         节点缓冲与信号照常清理，wait() 被唤醒，状态置 Cancelled。
	bool cancel(const TaskId& taskId);

	/// @brief  释放已终止 task 的状态记录（结果与诊断由上层一并清理）
	/// @return true 已释放；false 未释放（未知 taskId、活动 Running 任务，
	///         或终态已发布但结果收尾尚未完成的窗口）
	/// @note   仅"终态 + 收尾完成"可释放；释放后 status 返回 Unknown
	bool releaseTask(const TaskId& taskId);

	/// @brief  弃置在飞 task 的托管句柄（高-层句柄析构路径）：不取消任务，
	///         完成收尾（_terminate）时自动回收状态表条目 / OutputZone 结果 / 诊断。
	///         已终止且收尾完成 → 立即等价 releaseTask；未知 → no-op；
	///         Running 或收尾中 → 登记自动回收（不打断在飞/收尾语义）。
	void detachTask(const TaskId& taskId);

	// ── task 完成回调 ──

	/// @brief  设置 task 完成回调（每次 submit 前设置；_terminate 步骤① 触发，
	///         先于结果就绪发布——回调可安全读取 task 缓冲中尚存的输出）
	///         线程安全：与 _terminate 的读取之间以互斥锁同步
	/// @note   回调内不得对同一 taskId 调用 wait()：回调先于 resultsReady
	///         置位执行，等待将自我阻塞；亦不得 submit()/feedInput()/
	///         releaseTask()/detachTask()——收尾窗口内复用/输入注入/释放
	///         均被拒绝（DuplicateTask/无操作，H-3）。回调应只读取/捕获数据，
	///         新一轮提交请在 waitForResult 返回后进行。
	void setTaskCompleteCallback(TaskCompleteCallback cb) {
		std::lock_guard lk(_cbMutex);
		_taskCompleteCb = std::move(cb);
	}

private:
	// ── 任务轮次门控：每轮一次 submit 对应一个 TaskGate ──
	//
	// 自包含轮次状态容器：终态 / 结果可读 / 等待协议由本对象自身的
	// mutex + condition_variable 承载（单锁协议，无丢失唤醒窗口）；
	// 执行态（execState）在 submit 时捕获——调度 lambda 与传播链全部经
	// 本对象寻址，不再按可复用 taskId 重新查表，同 ID 复用后旧轮次
	// lambda 无法读写新一轮执行态。
	//
	// 每次节点执行 lambda 提交前 inflight+1、lambda 收尾（RAII）时 -1；
	// 归零且本轮未终止时由最后完成的 lambda 触发 _exhaustedCheck。
	struct TaskGate {
		std::atomic<bool> terminated{false};
		/// 在飞节点执行 lambda 计数（submit 入口与传播下游提交时 +1）
		std::atomic<uint32_t> inflight{0};
		ExecutionEngine* engine = nullptr;
		/// 图运行时状态共享句柄：TaskGate、任务 lambda 与 InferGraph 共同持有，
		/// 图对象先行析构时在飞任务所需的图组件仍存活
		std::shared_ptr<GraphRuntimeState> state;
		TaskId taskId;
		/// 本轮专属执行态（submit 时捕获）：终止清理仅从域表摘除条目，
		/// 在飞 lambda 经本句柄保活其引用的执行态至流水线结束
		std::shared_ptr<TaskExecutionState> execState;

		// ── 轮次锁：终态 / 结果可读发布与等待谓词绑定同一把锁 ──
		std::mutex m;
		std::condition_variable cv;
		TaskStatus terminalStatus = TaskStatus::Running; ///< 终态；Running=未终止（由 m 保护）
		bool resultsReady = false;                       ///< 声明输出已抢救进 OutputZone（由 m 保护）
		bool autoRelease = false;                        ///< 弃置句柄的完成后自动回收（由 m 保护）

		~TaskGate();
	};

	// ── 事件驱动数据传播 ──

	/// @brief  提交一个节点执行任务（tryExecute + 成功后就地传播）
	/// @note   由 submit 入口与传播下游共用；执行态经 round->execState
	///         捕获寻址（同 ID 复用后旧 lambda 不消费新一轮输入）；
	///         任务在节点亲和资源类（affinity）对应的调度器槽位执行
	void _submitNodeRun(const Node* node, const std::string& nodeName,
						std::shared_ptr<TaskGate> round, uint32_t remainingHops);

	/// @brief  传播节点输出到下游（调用前提：节点已由 _submitNodeRun 执行成功）
	void _propagateFrom(std::string nodeName, std::shared_ptr<TaskGate> round,
						uint32_t remainingHops);

	// ── 轮次辅助 ──

	/// @brief  取指定 taskId 当前轮次（表锁下取副本；未知返回 nullptr）
	std::shared_ptr<TaskGate> _findRound(const TaskId& taskId) const;

	// ── 终止辅助 ──
	void _terminate(const std::shared_ptr<TaskGate>& round,
					TaskStatus terminalStatus = TaskStatus::Succeeded);
	void _exhaustedCheck(const std::shared_ptr<TaskGate>& round);
	/// @brief  弃置轮次（autoRelease）的完成回收：表内仍是本轮时移除并清理
	///         输出区结果与诊断（身份校验防误清新一轮复用后的状态）
	void _finalizeDetached(const std::shared_ptr<TaskGate>& round);

	// ── 运行时诊断 ──

	/// @brief  异常终止前的诊断：将未满足声明、信号阻塞节点等信息写入 ErrorTracker。
	///         必须在 _terminate 之前调用（_terminate 会清理 OutputZone 和 _blockedSkips）。
	/// @param  reason  终止原因描述（如 "propagation exhausted with failed node 'x'"）
	void _diagnoseAbnormal(const std::shared_ptr<TaskGate>& round, const std::string& reason);

	// ── 成员 ──
	// 析构顺序约定：析构函数体（自排水）在所有成员存活时执行——排水等待
	// 归零后，无任务 lambda 再访问引擎状态；成员逆序析构仅作兜底。
	// （图组件生命周期由 GraphRuntimeState shared_ptr 保证，不依赖本表顺序。）
	//
	// 轮次表：taskId → 当前轮次 TaskGate（含终态轮次，供 status/wait/
	// releaseTask 查询；复用提交时替换、释放时移除）。每轮终态、结果可读
	// 与等待协议由 TaskGate 自身的 m/cv 承载（单锁协议）；本表锁仅保护
	// 映射结构——锁序约束：仅在持 _roundsMutex 时可嵌套获取 round->m，
	// round->m 内绝不获取 _roundsMutex。
	std::unordered_map<TaskId, std::shared_ptr<TaskGate>> _rounds;
	mutable std::mutex _roundsMutex;

	// 信号阻塞追踪：记录每个 task 在传播过程中因信号阻塞而被跳过的节点名。
	// 由 _propagateFrom 写入，_diagnoseAbnormal 读取，_terminate 清理。
	std::unordered_map<TaskId, std::unordered_set<std::string>> _blockedSkips;
	mutable std::mutex _blockedSkipsMutex;

	TaskCompleteCallback _taskCompleteCb;
	mutable std::mutex _cbMutex; // 保护 _taskCompleteCb 的跨线程读写

	// 执行载体：进程级共享资源调度器（跨图共享，承载资源隔离预算）
	std::shared_ptr<ResourceScheduler> _scheduler;

	// ── 析构自排水 ──
	// 共享调度器下不能靠“关闭自己的池” join 在飞任务，改为显式排水：
	// 派发登记（_shuttingDown 检查 + _pendingRuns +1）在 _drainMutex 内
	// 原子完成；每次派发同时签发一张排水票据（DrainTicket）随任务 lambda
	// 转移——lambda 执行完成（worker 线程）或被池弃置（调度器 shutdown 清队，
	// function 随之析构，发生在 shutdown 调用者线程）时票据析构，在排水锁
	// 内递减计数、归零者唤醒析构等待者（递减与 notify 同锁——归零观察
	// 不可能越过票据的最后一次引擎访问）。两条路径必经其一（RAII）→
	// 析构可无条件等待归零，无需对调度器关停状态做逃逸判定。
	std::mutex _drainMutex;
	std::condition_variable _drainCv;       ///< 排水等待（谓词：_pendingRuns == 0）
	bool _shuttingDown = false;             ///< 由 _drainMutex 保护；置位后新派发早退
	std::atomic<uint32_t> _pendingRuns{0};  ///< 已登记未回收的任务 lambda 计数（票据制）

	/// @brief 排水票据：派发登记时签发，随任务 lambda 转移；lambda（含被
	///        弃置未执行的）析构时回收计数。定义于 ExecutionEngine.cpp
	///        （回收需访问本类私有排水状态）。
	struct DrainTicket;
};

} // namespace DC
