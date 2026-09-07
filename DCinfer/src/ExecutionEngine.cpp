#include "ExecutionEngine.h"
#include "GraphStore.h"
#include "OutputZone.h"
#include "SignalStore.h"
#include "ErrorTracker.h"
#include "GraphException.h"
#include "NodeException.h"

#include <thread>
#include <chrono>

namespace DC {

// ════════════════════════════════════════════
// TaskGate 析构
// ════════════════════════════════════════════

ExecutionEngine::TaskGate::~TaskGate() {
	if (!terminated.load(std::memory_order_acquire) && engine && output && graph && signals && errors) {
		engine->_exhaustedCheck(taskId, *output, *graph, *signals, *errors);
	}
}

// ════════════════════════════════════════════
// 构造
// ════════════════════════════════════════════

ExecutionEngine::ExecutionEngine(const PoolConfig& computeCfg,
								 const PoolConfig& operatorCfg,
								 const PoolConfig& systemCfg)
	: _sharedGroups(std::make_shared<GroupSemaphoreRegistry>()),
	  _computePool(computeCfg, _sharedGroups),
	  _operatorPool(operatorCfg, _sharedGroups),
	  _systemPool(systemCfg, _sharedGroups) {}

// ════════════════════════════════════════════
// 线程池分发
// ════════════════════════════════════════════

void ExecutionEngine::_dispatchToPool(ThreadPoolAffinity affinity, const std::string& tag,
									  std::function<void()> task) {
	switch (affinity) {
	case ThreadPoolAffinity::Compute:
		_computePool.submit(tag, std::move(task));
		break;
	case ThreadPoolAffinity::Operator:
		_operatorPool.submit(tag, std::move(task));
		break;
	case ThreadPoolAffinity::System:
		_systemPool.submit(tag, std::move(task));
		break;
	}
}

void ExecutionEngine::_submitNodeRun(Node* node, const std::string& nodeName, const TaskId& taskId,
									 std::shared_ptr<TaskGate> gate, uint32_t remainingHops,
									 GraphStore& graph, OutputZone& output,
									 SignalStore& signals, ErrorTracker& errors) {
	// 捕获裸指针：图拓扑的存活期必须覆盖全部飞行中的任务（与 gate 相同约束）
	_dispatchToPool(node->affinity(), node->tag(),
					[this, node, nodeName, taskId, gate, remainingHops,
					 &graph, &output, &signals, &errors] {
		NodeResult result;
		try {
			result = node->tryExecute(taskId);
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
		// 部分输出场景（如 Routing 连接器仅路由到一个输出口）允许继续传播；
		// 完全无输出的节点记录错误并跳过传播。
		bool hasAnyOutput = false;
		for (const auto& p : node->schema().outputs) {
			if (node->hasOutput(taskId, p.name)) {
				hasAnyOutput = true;
				break;
			}
		}
		if (!hasAnyOutput) {
			errors.recordError(taskId, nodeName, "ExecutionEngine::_submitNodeRun",
							   "Node execution failed: " + result.message);
			return; // 失败不传播
		}

		// 节点执行成功 → 就地传播输出到下游（池线程内，与调度点同上下文）
		_propagateFrom(nodeName, taskId, gate, remainingHops, graph, output, signals, errors);
	});
}

// ════════════════════════════════════════════
// 异步提交：事件驱动的数据传播
// ════════════════════════════════════════════

void ExecutionEngine::submit(const TaskId& taskId, std::chrono::milliseconds timeout,
							 uint32_t maxHops, GraphStore& graph, OutputZone& output,
							 SignalStore& signals, ErrorTracker& errors) {
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
			if (it->second == TaskStatus::Running)
				throw GraphException(GraphException::ErrorType::DuplicateTask, "ExecutionEngine::submit",
									 "task '" + taskId + "' is still running; duplicate submit rejected");
			_taskStates.erase(it);
		}
		_taskStates.emplace(taskId, TaskStatus::Running);
	}

	// 创建任务门控：任务 lambda 链、看门狗与 cancel() 共享；
	// 注册到活动表供 cancel() 定位，_terminate 时移除
	auto gate = std::make_shared<TaskGate>();
	gate->engine = this;
	gate->output = &output;
	gate->graph = &graph;
	gate->signals = &signals;
	gate->errors = &errors;
	gate->taskId = taskId;
	{
		std::lock_guard lk(_activeGatesMutex);
		_activeGates[taskId] = gate;
	}

	// 超时看门狗（std::jthread + stop_token，生命周期由 ExecutionEngine 管理）
	if (timeout.count() > 0) {
		auto deadline = std::chrono::steady_clock::now() + timeout;
		auto watchdog = std::jthread(
			[this, taskId, timeout, deadline, gate, &graph, &output, &signals, &errors](
				std::stop_token stoken) {
				// 轮询 sleep，支持 stop_token 提前取消
				while (!stoken.stop_requested()
					   && std::chrono::steady_clock::now() < deadline) {
					std::this_thread::sleep_for(std::chrono::milliseconds(100));
				}
				if (stoken.stop_requested())
					return; // task 正常完成，_terminate 已请求停止

				if (!gate->terminated.exchange(true, std::memory_order_acq_rel)) {
					std::string reason = "task timed out (" + std::to_string(timeout.count())
										 + "ms) without meeting output declarations";
					errors.recordError(taskId, "<watchdog>", "ExecutionEngine::submit", reason);
					_diagnoseAbnormal(taskId, reason, output, graph, errors);
					_terminate(taskId, graph, output, signals, TaskStatus::TimedOut);
				}
			});

		// 注册到看门狗表：_watchdogs 无其他同步，须与 _terminate 的回收、
		// 并发 submit 互斥。同 taskId 重复提交时，旧看门狗移出后在锁外
		// 回收，避免在锁内 join。
		std::jthread replaced;
		{
			std::lock_guard lk(_watchdogsMutex);
			if (auto it = _watchdogs.find(taskId); it != _watchdogs.end()) {
				replaced = std::move(it->second);
				_watchdogs.erase(it);
			}
			_watchdogs.emplace(taskId, std::move(watchdog));
		}
		// replaced（若存在）在此析构：request_stop + join，位于锁外
	}

	// 扫描全图，对所有已就绪的节点提交执行任务（执行完成后再传播下游）
	for (const auto& [nodeName, nodePtr] : graph.nodes()) {
		if (!nodePtr->isReady(taskId))
			continue;

		// 入口节点：执行 + 完成后就地传播（_submitNodeRun 内部处理）
		_submitNodeRun(nodePtr.get(), nodeName, taskId, gate, maxHops,
					   graph, output, signals, errors);
	}
}

// ════════════════════════════════════════════
// 事件驱动数据传播核心
// ════════════════════════════════════════════

void ExecutionEngine::_propagateFrom(std::string nodeName, TaskId taskId,
									 std::shared_ptr<TaskGate> gate,
									 uint32_t remainingHops,
									 GraphStore& graph, OutputZone& output,
									 SignalStore& signals, ErrorTracker& errors) {
	// 调用前提：节点已由 _submitNodeRun 执行成功（失败路径已记录错误并跳过传播），
	// 输出已写入 TaskBuffer 输出槽位，本函数在池线程内就地执行。

	// [检查点 0] TTL 耗尽：主动终止 task（不依赖 _exhaustedCheck）
	if (remainingHops == 0) {
		std::string reason = "propagation hops exhausted (TTL=0) at node '" + nodeName
							 + "': cycle or excessively deep graph detected";
		errors.recordError(taskId, nodeName, "ExecutionEngine::_propagateFrom", reason);
		gate->terminated.store(true, std::memory_order_release);
		_diagnoseAbnormal(taskId, reason, output, graph, errors);
		_terminate(taskId, graph, output, signals, TaskStatus::Failed);
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

	// [检查点 2] 节点完成 → 三步流水线：打卡 → OutputZone 搬运 → 边搬运
	//
	// 第一步：打卡 — 所有产出端口统一累加计数，不论目的地
	for (const auto& outPort : src->schema().outputs) {
		if (!src->hasOutput(taskId, outPort.name))
			continue;
		// 终止复查：打卡写入前再确认本轮未被终止（取消/超时可能在检查点 1 之后触发）
		if (gate->terminated.load(std::memory_order_acquire))
			return;
		if (output.accumulateAndCheck(nodeName, outPort.name, taskId)) {
			gate->terminated.store(true, std::memory_order_release);
			_terminate(taskId, graph, output, signals, TaskStatus::Succeeded);
			return;
		}
	}

	// 第二步：OutputZone 目的地搬运 — OutputZone 绑定端口消费后自然空
	for (const auto& outPort : src->schema().outputs) {
		if (!src->hasOutput(taskId, outPort.name))
			continue;
		if (output.isBound(nodeName, outPort.name)) {
			Value data = src->getOutput(taskId, outPort.name);
			output.append(taskId, nodeName, outPort.name, std::move(data),
						  {nodeName, outPort.name, taskId});
		}
	}

	// 第三步：边目的地搬运 — 已在第二步消费的端口 hasOutput=false，自动跳过
	for (const auto& edge : graph.edges()) {
		if (edge.srcNode != nodeName)
			continue;

		if (!src->hasOutput(taskId, edge.srcPort))
			continue;

		// 阻塞检查：下游节点被信号阻塞时跳过此边，不消费上游输出
		// 数据留在上游输出槽中等待其他出边消费或自然背压释放
		auto* dst = graph.node(edge.dstNode);
		if (!dst)
			continue;
		if (dst->isBlocked(taskId)) {
			// 记录被跳过的节点，供 _diagnoseAbnormal 在异常终止时报告
			std::lock_guard lk(_blockedSkipsMutex);
			_blockedSkips[taskId].insert(edge.dstNode);
			continue;
		}

		Value data = src->getOutput(taskId, edge.srcPort);

		// [检查点 3] 写入下游前再确认一次本轮未被终止（gate 级，同 ID 复用安全）
		if (gate->terminated.load(std::memory_order_acquire))
			return;

		try {
			dst->setInput(taskId, edge.dstPort, std::move(data));
		} catch (const NodeException& e) {
			errors.recordError(taskId, edge.dstNode, "ExecutionEngine::_propagateFrom",
							   "NodeException in setInput for port '" + edge.dstPort
								   + "': " + std::string(e.what()));
			continue;
		}

		// 下游就绪 → 提交执行 + 完成后继续传播（数据冒泡）
		if (dst->isReady(taskId)) {
			_submitNodeRun(dst, edge.dstNode, taskId, gate, remainingHops - 1,
						   graph, output, signals, errors);
		}
	}
}

// ════════════════════════════════════════════
// 终止辅助
// ════════════════════════════════════════════

bool ExecutionEngine::_isTerminated(const TaskId& taskId) const {
	std::lock_guard lk(_terminationMutex);
	auto it = _taskStates.find(taskId);
	return it != _taskStates.end() && it->second != TaskStatus::Running;
}

void ExecutionEngine::_terminate(const TaskId& taskId,
								 GraphStore& graph, OutputZone& output,
								 SignalStore& signals, TaskStatus terminalStatus) {
	{
		std::lock_guard lk(_terminationMutex);
		// 防止重复终止（幂等）：仅 Running → 终态迁移一次有效
		auto it = _taskStates.find(taskId);
		if (it == _taskStates.end() || it->second != TaskStatus::Running)
			return;
		it->second = terminalStatus;
	}

	// ① 取消并回收超时看门狗（若存在）
	//    持锁移出、锁外回收：与 submit 的注册及并发 _terminate 互斥，
	//    join 不在锁内，避免阻塞其他线程的注册/回收。
	//    若调用线程正是该看门狗自身（超时路径），join 自身将抛
	//    resource_deadlock_would_occur，并因自 noexcept 析构逃逸触发
	//    std::terminate——此时移交退役列表，由引擎析构统一 join。
	std::jthread finished;
	{
		std::lock_guard lk(_watchdogsMutex);
		if (auto it = _watchdogs.find(taskId); it != _watchdogs.end()) {
			if (it->second.get_id() == std::this_thread::get_id())
				_retiredWatchdogs.push_back(std::move(it->second));
			else
				finished = std::move(it->second);
			_watchdogs.erase(it);
		}
	}
	if (finished.joinable()) {
		finished.request_stop();
		finished.join();
	}

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
	//    或 releaseTask() —— 支持 submit → wait → getOutput 的同步取结果用法

	// ④ 清理该 task 的所有 task 级信号（防止泄漏）
	signals.clearTask(taskId);

	// ⑤ 清理信号阻塞追踪记录
	{
		std::lock_guard lk(_blockedSkipsMutex);
		_blockedSkips.erase(taskId);
	}

	// ⑥ 抢救结果 + 清理节点缓冲：
	//    终止路径在打卡满足后直接返回，声明端口的数据仍留在节点缓冲。
	//    先把这些未及搬运的数据转移到 OutputZone（保证 wait → getOutput 可取，
	//    含看门狗/取消路径的部分结果），再清理缓冲。回调在②已先行触发，
	//    其消费过的端口 hasOutput=false 自然跳过。
	for (auto& [name, nodePtr] : graph.nodes()) {
		if (!nodePtr->hasTask(taskId))
			continue;
		for (const auto& decl : output.declarationsOf(taskId)) {
			if (decl.nodeName != name)
				continue;
			if (!nodePtr->hasOutput(taskId, decl.portName))
				continue;
			Value data = nodePtr->getOutput(taskId, decl.portName);
			output.append(taskId, decl.nodeName, decl.portName, std::move(data),
						  {decl.nodeName, decl.portName, taskId});
		}
		nodePtr->terminateTask(taskId);
	}

	// ⑦ 移除活动门控并通知同步等待者（结果仍保留，供 wait 后 getOutput 取用）
	{
		std::lock_guard lk(_activeGatesMutex);
		_activeGates.erase(taskId);
	}
	_completionCv.notify_all();
}

// ════════════════════════════════════════════
// 耗尽检测：TaskGate 析构或超时触发
// ════════════════════════════════════════════

void ExecutionEngine::_exhaustedCheck(const TaskId& taskId, OutputZone& output,
									  GraphStore& graph, SignalStore& signals, ErrorTracker& errors) {
	// 已终止则跳过
	if (_isTerminated(taskId)) {
		return;
	}

	// 检查声明是否已满足
	bool allMet = output.checkAllSatisfied(taskId);

	if (allMet) {
		// 声明已满足 → 正常终止（守护路径，主路径在 OutputZone::accumulateAndCheck 中处理）
		_terminate(taskId, graph, output, signals);
		return;
	}

	// 声明未满足：传播链已耗尽但输出声明未达成。
	// 写入诊断警告，但不主动终止——留给看门狗（若已配置）处理真正的死锁。
	// 若未配置看门狗（timeout=0），调用方需自行处理 wait() 超时。
	_diagnoseAbnormal(taskId, "propagation chain exhausted with unsatisfied output declarations",
					  output, graph, errors);
}

// ════════════════════════════════════════════
// 运行时诊断
// ════════════════════════════════════════════

void ExecutionEngine::_diagnoseAbnormal(const TaskId& taskId, const std::string& reason,
										OutputZone& output, GraphStore& graph, ErrorTracker& errors) {
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

	// ③ 报告从未被到达的声明输出节点（无 task 级 IO 缓冲区）
	for (const auto& u : unsatisfied) {
		auto* node = graph.node(u.decl.nodeName);
		if (node && !node->hasTask(taskId)) {
			errors.recordWarning(taskId, u.decl.nodeName, "ExecutionEngine::_diagnoseAbnormal",
								 "node was never reached during propagation "
								 "(no task-level IO buffer was created)");
		}
	}
}

// ════════════════════════════════════════════
// 分组限流
// ════════════════════════════════════════════

void ExecutionEngine::registerGroupLimit(ThreadPoolAffinity /*affinity*/, const std::string& tag,
										 size_t limit) {
	// 语义升级：组信号量由三个线程池共享，注册一次全局生效（跨池互斥）
	_sharedGroups->setLimit(tag, limit);
}

// ════════════════════════════════════════════
// 同步等待
// ════════════════════════════════════════════

bool ExecutionEngine::wait(const TaskId& taskId, std::chrono::milliseconds timeout) {
	std::unique_lock lk(_completionMutex);
	return _completionCv.wait_for(lk, timeout, [this, &taskId] {
		return _isTerminated(taskId);
	});
}

// ════════════════════════════════════════════
// task 状态与取消
// ════════════════════════════════════════════

TaskStatus ExecutionEngine::status(const TaskId& taskId) const {
	std::lock_guard lk(_terminationMutex);
	auto it = _taskStates.find(taskId);
	return it != _taskStates.end() ? it->second : TaskStatus::Unknown;
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
	_terminate(taskId, *gate->graph, *gate->output, *gate->signals, TaskStatus::Cancelled);
	return true;
}

void ExecutionEngine::releaseTask(const TaskId& taskId) {
	std::lock_guard lk(_terminationMutex);
	auto it = _taskStates.find(taskId);
	if (it == _taskStates.end() || it->second == TaskStatus::Running)
		return; // 未知或活动任务不可释放
	_taskStates.erase(it);
}

} // namespace DC
