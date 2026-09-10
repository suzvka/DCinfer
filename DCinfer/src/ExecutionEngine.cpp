#include "ExecutionEngine.h"
#include "GraphRuntimeState.h"
#include "Graph/internal/TaskExecutionState.h"
#include "Graph/internal/TimerService.h"
#include "GraphStore.h"
#include "OutputZone.h"
#include "SignalStore.h"
#include "ErrorTracker.h"
#include "GraphException.h"
#include "NodeException.h"
#include "Node/internal/ExecutionPipeline.h"

#include <chrono>

namespace DC {

// ════════════════════════════════════════════
// TaskGate 析构
// ════════════════════════════════════════════

ExecutionEngine::TaskGate::~TaskGate() {
	if (!terminated.load(std::memory_order_acquire) && engine && state) {
		engine->_exhaustedCheck(taskId, state);
	}
}

// ════════════════════════════════════════════
// 构造 / 析构
// ════════════════════════════════════════════

ExecutionEngine::ExecutionEngine(const PoolConfig& computeCfg,
								 const PoolConfig& operatorCfg,
								 const PoolConfig& systemCfg)
	: _computePool(computeCfg),
	  _operatorPool(operatorCfg),
	  _systemPool(systemCfg),
	  _timer(std::make_unique<TimerService>()) {}

ExecutionEngine::~ExecutionEngine() {
	// 先将活动门控表整体移出（锁外释放）：门控析构触发的 _exhaustedCheck
	// 可能经 _terminate 重入本表，锁内 clear 会自死锁。
	// 成员随后逆序析构：定时器线程最先停止（早于池 shutdown），
	// 之后线程池 join 全部 worker，状态成员最后释放。
	decltype(_activeGates) leftover;
	{
		std::lock_guard lk(_activeGatesMutex);
		leftover = std::move(_activeGates);
	}
	leftover.clear();
}

// ════════════════════════════════════════════
// 线程池分发
// ════════════════════════════════════════════

void ExecutionEngine::_dispatchToPool(ThreadPoolAffinity affinity,
									  std::function<void()> task) {
	switch (affinity) {
	case ThreadPoolAffinity::Compute:
		_computePool.submit(std::move(task));
		break;
	case ThreadPoolAffinity::Operator:
		_operatorPool.submit(std::move(task));
		break;
	case ThreadPoolAffinity::System:
		_systemPool.submit(std::move(task));
		break;
	}
}

void ExecutionEngine::_submitNodeRun(const Node* node, const std::string& nodeName,
									 const TaskId& taskId,
									 std::shared_ptr<TaskGate> gate, uint32_t remainingHops,
									 const std::shared_ptr<GraphRuntimeState>& state) {
	// 捕获 state 共享句柄：图拓扑/输出区/信号/诊断的存活期由引用计数保证，
	// 与图对象析构顺序无关（gate 与 lambda 各持一份）
	_dispatchToPool(node->affinity(),
					[this, node, nodeName, taskId, gate, remainingHops, state] {
		auto& errors = state->errors;
		NodeResult result;
		// task 态取自 task 执行域（原 Node 内嵌态），执行闸按节点名定位：
		// 节点本身只读（const），可变状态全部在 task 域；lambda 持
		// shared_ptr 副本，终止清理不会回收在飞执行态。
		// taskExec 就地复用（终止后输出判定共用同一句柄），
		// 不再重复走 findTaskState 加锁查找
		auto taskExec = state->exec->taskState(taskId);
		try {
			auto& exec = taskExec->ensure(nodeName, node->schema());
			result = ExecutionPipeline::execute(taskId, *node, exec,
												state->exec->gateFor(nodeName));
		} catch (const NodeException& e) {
			// 就绪判定竞态（NotReady/Reentrant）：记录错误，跳过传播
			errors.recordError(taskId, nodeName, "ExecutionEngine::_submitNodeRun",
							   "NodeException in tryExecute: " + std::string(e.what()));
			return;
		}

		// 任务已终止（超时/取消/同 ID 复用竞态）→ 丢弃本轮结果，不传播。
		// gate 级检查而非 taskId 级：复用后旧任务的 lambda 不得污染新任务。
		if (gate->terminated.load(std::memory_order_acquire))
			return;

		// 完成判定与原节点完成事件语义一致：只要产生了任何输出即视为成功传播。
		// 部分输出场景（如用户自定义路由节点仅产出一个输出口）允许继续传播；
		// 完全无输出的节点记录错误并跳过传播。
		bool hasAnyOutput = false;
		{
			auto* ns = taskExec->find(nodeName);
			if (ns) {
				for (const auto& p : node->schema().outputs) {
					if (ns->buffer.hasOutput(taskId, p.name)) {
						hasAnyOutput = true;
						break;
					}
				}
			}
		}
		if (!hasAnyOutput) {
			errors.recordError(taskId, nodeName, "ExecutionEngine::_submitNodeRun",
							   "Node execution failed: " + result.message, result.diagnostic);
			return; // 失败不传播
		}

		// 节点执行成功 → 就地传播输出到下游（池线程内，与调度点同上下文）
		_propagateFrom(nodeName, taskId, gate, remainingHops, state);
	});
}

// ════════════════════════════════════════════
// 异步提交：事件驱动的数据传播
// ════════════════════════════════════════════

void ExecutionEngine::submit(const TaskId& taskId, std::chrono::milliseconds timeout,
							 uint32_t maxHops, const std::shared_ptr<GraphRuntimeState>& state) {
	auto& output = state->output;
	auto& graph = state->graph->runtimeView();

	// 校验：必须已声明输出
	if (!output.hasDeclaration(taskId)) {
		throw GraphException(GraphException::ErrorType::NoDeclaration, "ExecutionEngine::submit",
							 "no output declarations for task '" + taskId
								 + "'. Call declareOutput() before submit().");
	}

	// 校验与登记：同一 taskId 活动期间禁止重复提交；已终止的 ID 允许复用
	//（清除上一轮终止状态，wait() 谓词与传播拦截随新任务重新生效）
	{
		std::lock_guard lk(_terminationMutex);
		auto it = _taskStates.find(taskId);
		if (it != _taskStates.end()) {
			if (it->second.status == TaskStatus::Running)
				throw GraphException(GraphException::ErrorType::DuplicateTask, "ExecutionEngine::submit",
									 "task '" + taskId + "' is still running; duplicate submit rejected");
			_taskStates.erase(it);
			// 节点执行态无需在此清理：所有终态必经 _terminate（唯一清理点），
			// 且本分支之后调用方可能重新 feedInput——此时清理会抹掉新输入
		}
		_taskStates.emplace(taskId, TaskStateRecord{});
	}

	// 创建任务门控：任务 lambda 链、超时触发路径与 cancel() 共享；
	// 注册到活动表供 cancel() 定位与 _onWatchdogFired 身份校验，_terminate 时移除
	auto gate = std::make_shared<TaskGate>();
	gate->engine = this;
	gate->state = state; // 与图对象共享图运行时状态（在飞任务保活）
	gate->taskId = taskId;
	{
		std::lock_guard lk(_activeGatesMutex);
		_activeGates[taskId] = gate;
	}

	// 超时看门狗：引擎级共享 TimerService 条目（无 per-task 线程；
	// 到点处理与仲裁链见 _onWatchdogFired）
	_scheduleWatchdog(taskId, timeout, state, gate);

	// 扫描全图（运行时视图），对所有已就绪的节点提交执行任务（执行完成后再传播下游）。
	// 就绪查询仅针对已有 task 态条目（feedInput 时创建）：无条目 = 无暂存输入 = 未就绪
	auto taskExec = state->exec->findTaskState(taskId);
	for (const auto& [nodeName, nodePtr] : graph.nodes) {
		auto* ns = taskExec ? taskExec->find(nodeName) : nullptr;
		if (!ns || !nodePtr->isReady(taskId, ns->buffer))
			continue;

		// 入口节点：执行 + 完成后就地传播（_submitNodeRun 内部处理）
		_submitNodeRun(nodePtr, nodeName, taskId, gate, maxHops, state);
	}
}

// ════════════════════════════════════════════
// 事件驱动数据传播核心
// ════════════════════════════════════════════

void ExecutionEngine::_propagateFrom(std::string nodeName, TaskId taskId,
									 std::shared_ptr<TaskGate> gate,
									 uint32_t remainingHops,
									 const std::shared_ptr<GraphRuntimeState>& state) {
	auto& graph = state->graph->runtimeView();
	auto& output = state->output;
	auto& signals = *state->signals;
	auto& errors = state->errors;
	// 调用前提：节点已由 _submitNodeRun 执行成功（失败路径已记录错误并跳过传播），
	// 输出已写入 TaskBuffer 输出槽位，本函数在池线程内就地执行。

	// [检查点 0] TTL 耗尽：主动终止 task（不依赖 _exhaustedCheck）
	if (remainingHops == 0) {
		std::string reason = "propagation hops exhausted (TTL=0) at node '" + nodeName
							 + "': cycle or excessively deep graph detected";
		errors.recordError(taskId, nodeName, "ExecutionEngine::_propagateFrom", reason);
		gate->terminated.store(true, std::memory_order_release);
		_diagnoseAbnormal(taskId, reason, state);
		_terminate(taskId, state, TaskStatus::Failed);
		return;
	}

	// [检查点 1] 入口：若本轮任务已终止（超时/取消/同 ID 复用竞态），直接返回。
	// gate 级检查而非 taskId 级 _isTerminated：复用后旧 lambda 不得继续传播。
	if (gate->terminated.load(std::memory_order_acquire)) {
		return;
	}

	auto* src = graph.node(nodeName);
	if (!src)
		return;

	// 本节点 task 态（调用前提：本节点已执行成功，条目必已存在）
	auto execState = state->exec->findTaskState(taskId);
	NodeExecState* srcNs = execState ? execState->find(nodeName) : nullptr;

	// [检查点 2] 节点完成 → 三步流水线：打卡 → OutputZone 搬运 → 边搬运
	//
	// 第一步：打卡 — 所有产出端口统一累加计数，不论目的地
	for (const auto& outPort : src->schema().outputs) {
		if (!srcNs || !srcNs->buffer.hasOutput(taskId, outPort.name))
			continue;
		// 终止复查：打卡写入前再确认本轮未被终止（取消/超时可能在检查点 1 之后触发）
		if (gate->terminated.load(std::memory_order_acquire))
			return;
		if (output.accumulateAndCheck(nodeName, outPort.name, taskId)) {
			gate->terminated.store(true, std::memory_order_release);
			_terminate(taskId, state, TaskStatus::Succeeded);
			return;
		}
	}

	// 第二步：OutputZone 目的地搬运 — OutputZone 绑定端口消费后自然空
	for (const auto& outPort : src->schema().outputs) {
		if (!srcNs || !srcNs->buffer.hasOutput(taskId, outPort.name))
			continue;
		if (state->graph->signature().isOutputBound(nodeName, outPort.name)) {
			Value data = srcNs->buffer.takeOutput(taskId, outPort.name);
			output.append(taskId, nodeName, outPort.name, std::move(data),
						  {nodeName, outPort.name, taskId});
		}
	}

	// 第三步：边目的地搬运 — 已在第二步消费的端口 hasOutput=false，自动跳过
	// （运行时视图：Broadcast(1) wire 已被 lowering 擦除，边为改写后的直连边）
	for (const auto& edge : graph.edges) {
		if (edge.srcNode != nodeName)
			continue;

		if (!srcNs || !srcNs->buffer.hasOutput(taskId, edge.srcPort))
			continue;

		// 阻塞检查：下游节点被信号阻塞时跳过此边，不消费上游输出
		// 数据留在上游 task 缓冲中等待其他出边消费或自然背压释放
		auto* dst = graph.node(edge.dstNode);
		if (!dst)
			continue;
		if (dst->isBlocked(taskId)) {
			// 记录被跳过的节点，供 _diagnoseAbnormal 在异常终止时报告
			std::lock_guard lk(_blockedSkipsMutex);
			_blockedSkips[taskId].insert(edge.dstNode);
			continue;
		}

		Value data = srcNs->buffer.takeOutput(taskId, edge.srcPort);

		// [检查点 3] 写入下游前再确认一次本轮未被终止（gate 级，同 ID 复用安全）
		if (gate->terminated.load(std::memory_order_acquire))
			return;

		// 下游 task 态：惰性创建（原 Node 内嵌 TaskBuffer 的 set 输入语义）。
		// dstExec/dstNs 就地复用（isReady 判定共用同一执行态），不再重复走
		// findTaskState + find 加锁查找；ensure 产出条目由 unique_ptr 承载
		// 地址稳定，dstExec 副本保证其存活至本轮传播结束
		std::shared_ptr<TaskExecutionState> dstExec;
		NodeExecState* dstNs = nullptr;
		try {
			dstExec = state->exec->taskState(taskId);
			dstNs = &dstExec->ensure(edge.dstNode, dst->schema());
			dstNs->buffer.setInput(taskId, edge.dstPort, std::move(data), dst->schema());
		} catch (const NodeException& e) {
			errors.recordError(taskId, edge.dstNode, "ExecutionEngine::_propagateFrom",
							   "NodeException in setInput for port '" + edge.dstPort
								   + "': " + std::string(e.what()));
			continue;
		}

		// 下游就绪 → 提交执行 + 完成后继续传播（数据冒泡）
		if (dst->isReady(taskId, dstNs->buffer)) {
			_submitNodeRun(dst, edge.dstNode, taskId, gate, remainingHops - 1, state);
		}
	}
}

// ════════════════════════════════════════════
// 终止辅助
// ════════════════════════════════════════════

bool ExecutionEngine::_isTerminated(const TaskId& taskId) const {
	std::lock_guard lk(_terminationMutex);
	auto it = _taskStates.find(taskId);
	return it != _taskStates.end() && it->second.status != TaskStatus::Running;
}

bool ExecutionEngine::_resultsReady(const TaskId& taskId) const {
	std::lock_guard lk(_terminationMutex);
	auto it = _taskStates.find(taskId);
	return it != _taskStates.end() && it->second.resultsReady;
}

void ExecutionEngine::_terminate(const TaskId& taskId,
								 const std::shared_ptr<GraphRuntimeState>& state,
								 TaskStatus terminalStatus) {
	auto& graph = state->graph->runtimeView();
	auto& output = state->output;
	auto& signals = *state->signals;
	{
		std::lock_guard lk(_terminationMutex);
		// 防止重复终止（幂等）：仅 Running → 终态迁移一次有效。
		// 此处仅发布终态（T1），结果可读（T2）由步骤⑥' 后置发布
		auto it = _taskStates.find(taskId);
		if (it == _taskStates.end() || it->second.status != TaskStatus::Running)
			return;
		it->second.status = terminalStatus;
	}

	// ① 失效超时条目（若存在）
	//    引擎级 TimerService：O(1) 作废，无线程可回收——原 per-task 看门狗
	//    的 join 回收与 _retiredWatchdogs 自 join 补丁随之移除。
	_cancelWatchdog(taskId);

	// ② 触发 task 完成回调（数据仍在，回调可安全读取并捕获输出）
	//    锁内拷贝、锁外调用：避免回调重入死锁
	TaskCompleteCallback cb;
	{
		std::lock_guard lk(_cbMutex);
		cb = _taskCompleteCb;
	}
	if (cb) {
		cb(taskId);
	}

	// ③（生命周期变更）不再清理 OutputZone：结果保留至下一次同 ID submit
	//    或 releaseTask() —— 支持 submit → wait → takeOutput 的同步取结果用法

	// ④ 清理该 task 的所有 task 级信号（防止泄漏）
	signals.clearTask(taskId);

	// ⑤ 清理信号阻塞追踪记录
	{
		std::lock_guard lk(_blockedSkipsMutex);
		_blockedSkips.erase(taskId);
	}

	// ⑥ 抢救结果 + 清理 task 执行态：
	//    终止路径在打卡满足后直接返回，声明端口的数据仍留在 task 缓冲。
	//    先把这些未及搬运的数据转移到 OutputZone（保证 wait → takeOutput 可取，
	//    含看门狗/取消路径的部分结果），再整体清除该 task 的节点执行态。
	//    回调在②已先行触发，其消费过的端口 hasOutput=false 自然跳过。
	if (auto taskExec = state->exec->findTaskState(taskId)) {
		for (const auto& decl : output.declarationsOf(taskId)) {
			auto* ns = taskExec->find(decl.nodeName);
			if (!ns || !ns->buffer.hasOutput(taskId, decl.portName))
				continue;
			Value data = ns->buffer.takeOutput(taskId, decl.portName);
			output.append(taskId, decl.nodeName, decl.portName, std::move(data),
						  {decl.nodeName, decl.portName, taskId});
		}
	}
	state->exec->clearTaskState(taskId);

	// ⑥' 发布"结果可读"：声明输出已全部抢救进 OutputZone。wait() 谓词绑定
	//     本标志且先于 notify 生效——此后返回的等待者必能取到结果，
	//     消除"终态已发布、结果未抢救"的窗口（终态/可读/清理三完成点中，
	//     wait 绑定中间点；notify 仍最后发出）。
	{
		std::lock_guard lk(_terminationMutex);
		if (auto it = _taskStates.find(taskId); it != _taskStates.end())
			it->second.resultsReady = true;
	}

	// ⑦ 移除活动门控并通知同步等待者（结果仍保留，供 wait 后 takeOutput 取用）
	{
		std::lock_guard lk(_activeGatesMutex);
		_activeGates.erase(taskId);
	}
	_completionCv.notify_all();
}

// ════════════════════════════════════════════
// 耗尽检测：TaskGate 析构或超时触发
// ════════════════════════════════════════════

void ExecutionEngine::_exhaustedCheck(const TaskId& taskId,
									  const std::shared_ptr<GraphRuntimeState>& state) {
	auto& output = state->output;
	auto& graph = state->graph->runtimeView();
	// 已终止则跳过
	if (_isTerminated(taskId)) {
		return;
	}

	// 检查声明是否已满足
	bool allMet = output.checkAllSatisfied(taskId);

	if (allMet) {
		// 声明已满足 → 正常终止（守护路径，主路径在 OutputZone::accumulateAndCheck 中处理）
		_terminate(taskId, state);
		return;
	}

	// 声明未满足：传播链已耗尽但输出声明未达成。
	// 写入诊断警告，但不主动终止——留给看门狗（若已配置）处理真正的死锁。
	// 若未配置看门狗（timeout=0），调用方需自行处理 wait() 超时。
	_diagnoseAbnormal(taskId, "propagation chain exhausted with unsatisfied output declarations",
					  state);
}

// ════════════════════════════════════════════
// 超时定时器（引擎级共享 TimerService）
// ════════════════════════════════════════════

void ExecutionEngine::_scheduleWatchdog(const TaskId& taskId,
										std::chrono::milliseconds timeout,
										const std::shared_ptr<GraphRuntimeState>& state,
										const std::shared_ptr<TaskGate>& gate) {
	if (timeout.count() <= 0)
		return; // 不限时：与原实现一致，不设防

	// 同 ID 残留条目先失效：活动 ID 重复提交已在 _taskStates 校验拒绝，
	// 此处兜底已终止 ID 复用路径上的旧条目
	_cancelWatchdog(taskId);

	auto deadline = std::chrono::steady_clock::now() + timeout;
	// 回调捕获本提交 gate 与 state 共享句柄：图组件存活期由引用计数保证
	//（同原看门狗线程体）；到点处理见 _onWatchdogFired
	uint64_t handle = _timer->schedule(deadline, [this, taskId, timeout, state, gate] {
		_onWatchdogFired(taskId, timeout, state, gate);
	});

	std::lock_guard lk(_timerHandlesMutex);
	_timerHandles[taskId] = handle;
}

void ExecutionEngine::_cancelWatchdog(const TaskId& taskId) {
	uint64_t handle = 0;
	{
		std::lock_guard lk(_timerHandlesMutex);
		auto it = _timerHandles.find(taskId);
		if (it == _timerHandles.end())
			return;
		handle = it->second;
		_timerHandles.erase(it);
	}
	// 锁外调用：_timerHandlesMutex 与 timer 内部锁不嵌套持有
	_timer->cancel(handle);
}

void ExecutionEngine::_onWatchdogFired(const TaskId& taskId,
									   std::chrono::milliseconds timeout,
									   const std::shared_ptr<GraphRuntimeState>& state,
									   const std::shared_ptr<TaskGate>& gate) {
	// ① 提交唯一性校验：仅当本提交的 gate 仍是活动门控时才继续。
	//    同 ID 复用后注册的是新 gate，旧条目到点在此失配退出——
	//    无 per-task 线程可 join，这道校验取代原 join 带来的唯一性保证。
	{
		std::lock_guard lk(_activeGatesMutex);
		auto it = _activeGates.find(taskId);
		if (it == _activeGates.end() || it->second != gate)
			return;
	}

	// ② gate 仲裁：与 cancel()/正常完成竞争唯一终止权（原看门狗语义）
	if (gate->terminated.exchange(true, std::memory_order_acq_rel))
		return;

	// ③ 诊断 + 终止（顺序与原看门狗线程体一致）
	std::string reason = "task timed out (" + std::to_string(timeout.count())
						 + "ms) without meeting output declarations";
	state->errors.recordError(taskId, "<watchdog>", "ExecutionEngine::submit", reason);
	_diagnoseAbnormal(taskId, reason, state);
	_terminate(taskId, state, TaskStatus::TimedOut);
}

// ════════════════════════════════════════════
// 运行时诊断
// ════════════════════════════════════════════

void ExecutionEngine::_diagnoseAbnormal(const TaskId& taskId, const std::string& reason,
										const std::shared_ptr<GraphRuntimeState>& state) {
	auto& output = state->output;
	auto& graph = state->graph->runtimeView();
	auto& errors = state->errors;
	// ① 报告未满足的输出声明
	auto unsatisfied = output.unsatisfiedDeclarations(taskId);
	for (const auto& u : unsatisfied) {
		errors.recordWarning(taskId, u.decl.nodeName, "ExecutionEngine::_diagnoseAbnormal",
							 "declared output '" + u.decl.nodeName + ":" + u.decl.portName
								 + "' not satisfied (expected " + std::to_string(u.decl.count)
								 + ", got " + std::to_string(u.current) + "); reason: " + reason);
	}

	// ② 报告传播过程中因信号阻塞而被跳过的节点
	std::unordered_set<std::string> blocked;
	{
		std::lock_guard lk(_blockedSkipsMutex);
		auto it = _blockedSkips.find(taskId);
		if (it != _blockedSkips.end())
			blocked = it->second; // 拷贝，不 move（_terminate 负责清理）
	}
	for (const auto& nodeName : blocked) {
		errors.recordWarning(taskId, nodeName, "ExecutionEngine::_diagnoseAbnormal",
							 "node was signal-blocked during propagation; "
							 "data may have been prevented from reaching declared outputs");
	}

	// ③ 报告从未被到达的声明输出节点（无 task 态条目）
	auto taskExec = state->exec->findTaskState(taskId);
	for (const auto& u : unsatisfied) {
		auto* node = graph.node(u.decl.nodeName);
		auto* ns = taskExec ? taskExec->find(u.decl.nodeName) : nullptr;
		if (node && !ns) {
			errors.recordWarning(taskId, u.decl.nodeName, "ExecutionEngine::_diagnoseAbnormal",
								 "node was never reached during propagation "
								 "(no task-level IO buffer was created)");
		}
	}
}

// ════════════════════════════════════════════
// 同步等待
// ════════════════════════════════════════════

bool ExecutionEngine::wait(const TaskId& taskId, std::chrono::milliseconds timeout) {
	// timeout <= 0 视为无限等待（与 submit 的执行超时 0=不限时约定一致）。
	// 谓词绑定"终态 + 结果可读"：_terminate 先发布终态、后抢救声明输出
	// （步骤⑥），仅查终态会让等待者早于结果就绪返回，破坏
	// "wait 返回即可读"契约。
	// 未知 taskId（从未提交或已 releaseTask）不可终止，立即返回 false，
	// 防止无限等待模式下误拼写 taskId 挂死。
	if (timeout.count() <= 0 && status(taskId) == TaskStatus::Unknown)
		return false;
	std::unique_lock lk(_completionMutex);
	if (timeout.count() <= 0) {
		_completionCv.wait(lk, [this, &taskId] { return _isTerminated(taskId) && _resultsReady(taskId); });
		return true;
	}
	return _completionCv.wait_for(lk, timeout, [this, &taskId] {
		return _isTerminated(taskId) && _resultsReady(taskId);
	});
}

// ════════════════════════════════════════════
// task 状态与取消
// ════════════════════════════════════════════

TaskStatus ExecutionEngine::status(const TaskId& taskId) const {
	std::lock_guard lk(_terminationMutex);
	auto it = _taskStates.find(taskId);
	return it != _taskStates.end() ? it->second.status : TaskStatus::Unknown;
}

bool ExecutionEngine::cancel(const TaskId& taskId) {
	std::shared_ptr<TaskGate> gate;
	{
		std::lock_guard lk(_activeGatesMutex);
		if (auto it = _activeGates.find(taskId); it != _activeGates.end())
			gate = it->second;
	}
	if (!gate)
		return false; // 未知或已终止（幂等）
	if (gate->terminated.exchange(true, std::memory_order_acq_rel))
		return false; // 已被正常路径/看门狗终止
	// 传播链在下个检查点停止；缓冲与信号由 _terminate 照常清理
	_terminate(taskId, gate->state, TaskStatus::Cancelled);
	return true;
}

void ExecutionEngine::releaseTask(const TaskId& taskId) {
	std::lock_guard lk(_terminationMutex);
	auto it = _taskStates.find(taskId);
	if (it == _taskStates.end() || it->second.status == TaskStatus::Running)
		return; // 未知或活动任务不可释放
	_taskStates.erase(it);
}

} // namespace DC
