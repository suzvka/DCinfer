#include "ExecutionEngine.h"
#include "GraphRuntimeState.h"
#include "Graph/internal/TaskExecutionState.h"
#include "GraphStore.h"
#include "OutputZone.h"
#include "SignalProbe.h"
#include "SignalStore.h"
#include "ErrorTracker.h"
#include "GraphException.h"
#include "NodeException.h"
#include "Node/internal/ExecutionPipeline.h"

#include <chrono>
#include <cstdio>
#include <exception>
#include <stdexcept>

namespace DC {

namespace {
	// 侵入式 TLS 栈：零分配；嵌套调用仍可枚举所有活动引擎身份。
	struct EngineScope {
		const ExecutionEngine* engine;
		EngineScope* previous;
		static thread_local EngineScope* top;
		explicit EngineScope(const ExecutionEngine* e) noexcept : engine(e), previous(top) { top = this; }
		~EngineScope() { top = previous; }
		static bool contains(const ExecutionEngine* e) noexcept {
			for (auto* p = top; p; p = p->previous)
				if (p->engine == e) return true;
			return false;
		}
	};
	thread_local EngineScope* EngineScope::top = nullptr;
}


ExecutionEngine::TaskGate::~TaskGate() {
	// 纯资源回收：引用归零只发生在终态收尾完成或引擎析构之后；
	// 两类场景均无需触发耗尽检测。
}

struct ExecutionEngine::DrainTicket {
	ExecutionEngine* engine = nullptr;
	// 登记配对标志：仅当对应的 _pendingRuns +1 已在排水锁内完成后置位；
	// 未登记路径析构不递减，不产生计数泄漏。
	bool armed = false;

	~DrainTicket() {
		if (!armed)
			return;
		// 票据回收：lambda 完成或被池弃置的公共必经点，此后不可能再回访引擎。
		// 递减必须在排水锁内：归零与加锁之间若存在窗口，析构者可先行销毁
		// _drainMutex/_drainCv，worker 触碰即 UAF（先减后锁的旧写法即此缺陷）。
		std::lock_guard lk(engine->_drainMutex);
		if (engine->_pendingRuns.fetch_sub(1, std::memory_order_acq_rel) == 1)
			engine->_drainCv.notify_all();
	}
};

ExecutionEngine::ExecutionEngine(std::shared_ptr<ResourceScheduler> scheduler)
	: _scheduler(std::move(scheduler)) {
	if (!_scheduler)
		throw std::invalid_argument("ExecutionEngine: scheduler must not be null");
}

ExecutionEngine::~ExecutionEngine() {
	if (EngineScope::contains(this)) {
		// stderr 重定向后可能全缓冲，崩溃路径不经 flush 会丢诊断；显式 flush 兜底。
		std::fputs("ExecutionEngine: prohibited reentrant destruction from own node/API/callback; retain external ownership\n", stderr);
		std::fflush(stderr);
		std::terminate();
	}
	// 自排水：共享调度器下不能靠关闭自己的池来 join 在飞任务。
	// ① 停止新派发：登记在 _drainMutex 内原子完成，置位后不再产生新登记。
	{
		std::lock_guard lk(_drainMutex);
		_shuttingDown = true;
	}

	// ② 标记全部轮次终止：排队 lambda 经 terminated 早退，不消费输入。
	{
		std::lock_guard lk(_roundsMutex);
		for (auto& entry : _rounds)
			entry.second->terminated.store(true, std::memory_order_release);
	}

	// ③ 等待全部已登记 lambda 回收：执行完或清队弃置均经票据归零；
	//    shutdown 先弃置排队票据再 join 在飞任务，本等待必然终止，
	//    wait_for 周期复查仅作防御兜底。
	{
		std::unique_lock lk(_drainMutex);
		while (_pendingRuns.load(std::memory_order_acquire) != 0) {
			_drainCv.wait_for(lk, std::chrono::milliseconds(100));
		}
	}

	// ④ 锁外释放轮次表：析构中的轮次不再触发引擎回访。
	decltype(_rounds) leftover;
	{
		std::lock_guard lk(_roundsMutex);
		leftover = std::move(_rounds);
	}
	leftover.clear();
}

void ExecutionEngine::_submitNodeRun(const Node* node, const std::string& nodeName,
									 std::shared_ptr<TaskGate> round, uint32_t remainingHops) {
	// 调度点检查：本轮已终止则不再提交新执行
	if (round->terminated.load(std::memory_order_acquire))
		return;

	// 在飞计数 +1：派发失败或抛异常时按失败语义回滚收尾，否则计数泄漏
	// 使耗尽检测永不触发。
	// 排水登记：_shuttingDown 检查与 _pendingRuns +1 在 _drainMutex 内原子完成。
	// 票据先构造后登记（分配失败零副作用），登记成功才 arm，随 lambda 移交，
	// 析构恰好一次递减。
	auto ticket = std::make_shared<DrainTicket>();
	ticket->engine = this;
	{
		std::lock_guard lk(_drainMutex);
		if (_shuttingDown)
			return; // 引擎析构中：不再派发（轮次已全终止，队列将被排空）
		_pendingRuns.fetch_add(1, std::memory_order_acq_rel);
		ticket->armed = true;
	}
	round->inflight.fetch_add(1, std::memory_order_acq_rel);
	bool dispatched = false;
	try {
		dispatched = _scheduler->submit(node->affinity(),
									 [this, node, nodeName, round, remainingHops, ticket] {
		EngineScope active{this}; // 先于 RunDone 声明：done 析构触发耗尽检测时作用域仍有效
		// 在飞计数收尾（RAII）：归零且本轮未终止时触发耗尽检测；
		// 票据递减晚于全部引擎回访，析构等待归零即保证无残留回访。
		struct RunDone {
			std::shared_ptr<TaskGate> round;
			ExecutionEngine* engine;

			~RunDone() {
				if (round->inflight.fetch_sub(1, std::memory_order_acq_rel) == 1
					&& !round->terminated.load(std::memory_order_acquire)) {
					engine->_exhaustedCheck(round);
				}
			}
		} done{round, this};

		// 排队期间本轮已终止则禁止执行：不入流水线、不消费输入
		if (round->terminated.load(std::memory_order_acquire))
			return;

		auto& state = round->state;
		const TaskId& taskId = round->taskId;
		auto& errors = state->errors;
		NodeResult result;
		// 执行态取自轮次副本：复用后旧轮次 lambda 只写旧执行态，不消费新一轮输入
		auto execState = round->execState;
		try {
			auto& exec = execState->ensure(nodeName, node->schema());
			result = ExecutionPipeline::execute(
				taskId, *node, exec, state->exec->gateFor(nodeName),
				[round] { return round->terminated.load(std::memory_order_acquire); });
		} catch (const NodeException& e) {
			switch (e.getErrorType()) {
			case NodeException::ErrorType::Reentrant:
				// 节点正忙：登记重投，闸释放时自动重放，不再判死任务；
				// 重投入口经 terminated 检查丢弃已终止轮次。
				state->exec->gateFor(nodeName).enqueueRetry(
					round.get(), [this, node, nodeName, round, remainingHops] {
						_submitNodeRun(node, nodeName, round, remainingHops);
					});
				return;
			case NodeException::ErrorType::NotReady:
				// 就绪竞态降级为 Warning：跳过本次提交，不判死任务
				errors.recordWarning(taskId, nodeName, "ExecutionEngine::_submitNodeRun",
									 "NotReady race in tryExecute (duplicate trigger skipped): "
										 + std::string(e.what()));
				return;
			default:
				errors.recordError(taskId, nodeName, "ExecutionEngine::_submitNodeRun",
								   "NodeException in tryExecute: " + std::string(e.what()));
				return;
			}
		} catch (const std::exception& e) {
			// 统一记录 Error 诊断，由耗尽检测收束为 Failed（失败闭环）
			errors.recordError(taskId, nodeName, "ExecutionEngine::_submitNodeRun",
							   "non-NodeException escaped node execution: " + std::string(e.what()));
			return;
		} catch (...) {
			errors.recordError(taskId, nodeName, "ExecutionEngine::_submitNodeRun",
							   "unknown non-standard exception escaped node execution");
			return;
		}

		// 本轮已终止则丢弃结果不传播；轮次级检查保证旧轮 lambda 不污染新轮
		if (round->terminated.load(std::memory_order_acquire))
			return;

		// 自报失败优先于输出存在性：失败不得因产生过部分输出而被成功传播；
		// 仅成功且至少一个输出时继续传播。
		if (!result.ok()) {
			errors.recordError(taskId, nodeName, "ExecutionEngine::_submitNodeRun",
							   "Node execution failed: " + result.message, result.diagnostic);
			return;
		}
		bool hasAnyOutput = false;
		{
			auto* ns = execState->find(nodeName);
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
							   "Node execution produced no outputs: " + result.message, result.diagnostic);
			return;
		}

		// 执行成功：池线程内就地传播输出到下游
		_propagateFrom(nodeName, round, remainingHops);
		});
	} catch (...) {
		// 分配失败按拒绝处理
	}
	if (!dispatched) {
		// 派发被拒：回滚计数 + 诊断 + 触发耗尽检测，经 Error 诊断收尾为 Failed
		auto& errors = round->state->errors;
		errors.recordError(round->taskId, nodeName, "ExecutionEngine::_submitNodeRun",
						   "task dispatch rejected (scheduler stopped or memory pressure)");
		if (round->inflight.fetch_sub(1, std::memory_order_acq_rel) == 1
			&& !round->terminated.load(std::memory_order_acquire)) {
			_exhaustedCheck(round);
		}
		// 排水计数由票据回收：晚于上方全部回访，维持回访先于递减的不变量
	}
}

void ExecutionEngine::submit(const TaskId& taskId, uint32_t maxHops,
							 const std::shared_ptr<GraphRuntimeState>& state,
							 std::vector<OutputDeclaration> declarations) {
	EngineScope active{this};
	// 快照经发布协议读取（acquire）；本地句柄钉住存活期
	auto snap = state->snapshot();
	auto& graph = snap->runtimeView();

	// 必须声明输出；声明随提交事务在下方临界区原子写入
	if (declarations.empty()) {
		throw GraphException(GraphException::ErrorType::NoDeclaration, "ExecutionEngine::submit",
							 "no output declarations for task '" + taskId
								 + "'; pass declarations via InferGraph::submit(...) / submitBound(...) first");
	}

	// 提交期拓扑守卫，先于任何状态登记：声明目标须从已注入输入的节点 ∪
	// 输入绑定节点纯拓扑可达；信号阻断属合法运行期状态，不参与判定。
	std::vector<std::string> starts;
	if (auto taskExec = state->exec->findTaskState(taskId))
		starts = taskExec->nodeNames(); // 已注入输入的节点
	for (const auto& b : snap->signature().inputs)
		starts.push_back(b.nodeName);
	std::unordered_set<std::string> targets;
	for (const auto& d : declarations) {
		// 声明坐标校验：拼错名称在提交期即暴露，否则任务悬置且无诊断
		auto* declared = graph.node(d.nodeName);
		if (!declared)
			throw GraphException(GraphException::ErrorType::NodeNotFound, "ExecutionEngine::submit",
								 "declared output node '" + d.nodeName + "' not found in graph");
		if (!declared->schema().findOutput(d.portName))
			throw GraphException(GraphException::ErrorType::PortNotFound, "ExecutionEngine::submit",
								 "declared output port '" + d.portName + "' not found on node '"
									 + d.nodeName + "'");
		targets.insert(d.nodeName);
	}
	if (!starts.empty() && !targets.empty()
		&& !canSatisfyTopologically(snap->runtimeView(), starts, targets)) {
		throw GraphException(GraphException::ErrorType::UnreachableDeclaration,
							 "ExecutionEngine::submit",
							 "declared output is topologically unreachable from any fed input "
							 "node (cycle/break in graph construction)");
	}

	// 构建本轮轮次（含终态/等待协议）；执行态自此捕获，后续调度与清理
	// 全部经本对象寻址，不再按可复用 taskId 查表。
	auto round = std::make_shared<TaskGate>();
	round->engine = this;
	round->state = state; // 与图对象共享图运行时状态（在飞任务保活）
	round->taskId = taskId;

	// 提交事务（单临界区）：准入检查、声明写入、执行态捕获、轮次登记原子完成。
	// 复用准入须终态已发布且清理完成，收尾窗口内一律拒绝；执行态捕获置于准入
	// 之后，不与旧轮共享；并发同 ID 提交恰一方成功。
	{
		std::lock_guard lk(_roundsMutex);
		auto it = _rounds.find(taskId);
		if (it != _rounds.end()) {
			std::lock_guard lk2(it->second->m);
			if (it->second->terminalStatus == TaskStatus::Running || !it->second->resultsReady)
				throw GraphException(GraphException::ErrorType::DuplicateTask, "ExecutionEngine::submit",
									 "task '" + taskId
										 + "' is still running or finalizing; duplicate submit rejected");
		}
		round->execState = state->exec->taskState(taskId);
		state->errors.clearTask(taskId); // 上一轮诊断不残留
		state->output.clearTask(taskId); // 清掉上一轮声明/累加/结果
		state->output.declare(taskId, std::move(declarations));
		_rounds[taskId] = round; // 替换旧轮；旧轮由在飞 lambda 保活
	}
	// 节点执行态无需在此清理：所有终态必经 _terminate。

	// 扫描全图提交所有已就绪节点；就绪查询仅针对本轮执行态条目，
	// 无条目即未就绪（条目由 feedInput 创建）。
	for (const auto& [nodeName, nodePtr] : graph.nodes) {
		auto* ns = round->execState ? round->execState->find(nodeName) : nullptr;
		if (!ns || !nodePtr->isReady(taskId, ns->buffer))
			continue;

		// 入口节点：执行 + 完成后就地传播
		_submitNodeRun(nodePtr, nodeName, round, maxHops);
	}
}

bool ExecutionEngine::tryWriteTaskState(const TaskId& taskId, const std::function<void()>& writeOp) {
	EngineScope active{this};
	auto round = _findRound(taskId);
	if (!round) {
		// 无轮次则无并发收尾，直接写入
		writeOp();
		return true;
	}
	// 与 _terminate 抢救段同一互斥：收尾窗口拒绝写入，其余窗口写入不会被摘除
	std::lock_guard lk(round->m);
	if (round->terminalStatus != TaskStatus::Running && !round->resultsReady)
		return false; // 收尾窗口：终态已发布、结果可读未发布
	writeOp();
	return true;
}

void ExecutionEngine::_propagateFrom(std::string nodeName, std::shared_ptr<TaskGate> round,
									 uint32_t remainingHops) {
	auto& state = round->state;
	const TaskId& taskId = round->taskId;
	// 快照经发布协议读取（acquire）；本地句柄钉住存活期，池线程不再读共享成员
	auto snap = state->snapshot();
	auto& graph = snap->runtimeView();
	auto& output = state->output;
	auto& errors = state->errors;
	// 调用前提：节点已执行成功、输出已入 TaskBuffer 槽位；池线程内就地执行。

	// [检查点 0] TTL 耗尽：主动终止任务，不依赖 _exhaustedCheck
	if (remainingHops == 0) {
		std::string reason = "propagation hops exhausted (TTL=0) at node '" + nodeName
							 + "': cycle or excessively deep graph detected";
		errors.recordError(taskId, nodeName, "ExecutionEngine::_propagateFrom", reason);
		round->terminated.store(true, std::memory_order_release);
		_diagnoseAbnormal(round, reason);
		_terminate(round, TaskStatus::Failed);
		return;
	}

	// [检查点 1] 本轮已终止则直接返回；轮次级检查保证旧轮 lambda 不继续传播
	if (round->terminated.load(std::memory_order_acquire)) {
		return;
	}

	auto* src = graph.node(nodeName);
	if (!src)
		return;

	// 本轮执行态：submit 时捕获；节点已执行成功，条目必已存在
	const auto& execState = round->execState;
	NodeExecState* srcNs = execState ? execState->find(nodeName) : nullptr;

	// [检查点 2] 三步流水线：打卡 → OutputZone 搬运 → 边搬运
	// 第一步：打卡，所有产出端口统一累加计数
	for (const auto& outPort : src->schema().outputs) {
		if (!srcNs || !srcNs->buffer.hasOutput(taskId, outPort.name))
			continue;
		// 打卡写入前复查终止标记（取消可能在检查点 1 之后触发）
		if (round->terminated.load(std::memory_order_acquire))
			return;
		if (output.accumulateAndCheck(nodeName, outPort.name, taskId)) {
			round->terminated.store(true, std::memory_order_release);
			_terminate(round, TaskStatus::Succeeded);
			return;
		}
	}

	// 第二步：OutputZone 搬运；绑定端口消费后自然空
	for (const auto& outPort : src->schema().outputs) {
		if (!srcNs)
			continue;
		if (snap->signature().isOutputBound(nodeName, outPort.name)) {
			// OutputZone 写入前复查终止标记
			if (round->terminated.load(std::memory_order_acquire))
				return;
			// 一次加锁完成检查+取数：与抢救并发时后到者得 nullopt，不抛异常
			auto data = srcNs->buffer.tryTakeOutput(taskId, outPort.name);
			if (!data)
				continue;
			output.append(taskId, nodeName, outPort.name, std::move(*data),
						  {nodeName, outPort.name, taskId});
		}
	}

	// 第三步：边搬运；已消费端口自动跳过，运行时视图为改写后的直连边
	for (const auto& edge : graph.edges) {
		if (edge.srcNode != nodeName)
			continue;

		if (!srcNs || !srcNs->buffer.hasOutput(taskId, edge.srcPort))
			continue;

		// 下游被信号阻塞则跳过此边，不消费上游输出；数据留在缓冲等待其他出边
		auto* dst = graph.node(edge.dstNode);
		if (!dst)
			continue;
		if (dst->isBlocked(taskId)) {
			// 记录被跳过的节点，供 _diagnoseAbnormal 报告
			std::lock_guard lk(_blockedSkipsMutex);
			_blockedSkips[taskId].insert(edge.dstNode);
			continue;
		}

		// 一次加锁完成检查+取数：前置 hasOutput 仅为快速过滤，后到者得 nullopt
		auto dataOpt = srcNs->buffer.tryTakeOutput(taskId, edge.srcPort);
		if (!dataOpt)
			continue;
		Value data = std::move(*dataOpt);

		// [检查点 3] 写入下游前复查终止标记
		if (round->terminated.load(std::memory_order_acquire))
			return;

		// 下游执行态经本轮执行态惰性创建；存活至本轮传播结束
		NodeExecState* dstNs = nullptr;
		bool ready = false;
		try {
			dstNs = &execState->ensure(edge.dstNode, dst->schema());
			if (dst->hasReadyOverride()) {
				// 覆盖路径保留 setInput + isReady 分离，残余竞态归 override 实现方
				dstNs->buffer.setInput(taskId, edge.dstPort, std::move(data), dst->schema());
				ready = dst->isReady(taskId, dstNs->buffer);
			} else {
				// 原子路径：写入与就绪判定同临界区，仅最后写入者观察到就绪，双触发消除
				ready = dstNs->buffer.setInputAndCheckReady(taskId, edge.dstPort,
															std::move(data), dst->schema());
			}
		} catch (const NodeException& e) {
			errors.recordError(taskId, edge.dstNode, "ExecutionEngine::_propagateFrom",
							   "NodeException in setInput for port '" + edge.dstPort
								   + "': " + std::string(e.what()));
			continue;
		}

		// 下游就绪：提交执行，完成后继续传播
		if (ready) {
			_submitNodeRun(dst, edge.dstNode, round, remainingHops - 1);
		}
	}
}

std::shared_ptr<ExecutionEngine::TaskGate> ExecutionEngine::_findRound(const TaskId& taskId) const {
	std::lock_guard lk(_roundsMutex);
	auto it = _rounds.find(taskId);
	return it != _rounds.end() ? it->second : nullptr;
}

void ExecutionEngine::_finalizeDetached(const std::shared_ptr<TaskGate>& round) {
	const TaskId& taskId = round->taskId;
	// 身份校验：表内仍是本轮才移除清理；已被 release 或新一轮替换则不动
	bool owned = false;
	{
		std::lock_guard lk(_roundsMutex);
		auto it = _rounds.find(taskId);
		if (it != _rounds.end() && it->second == round) {
			_rounds.erase(it);
			owned = true;
		}
	}
	if (owned) {
		round->state->output.clearTask(taskId);
		round->state->errors.clearTask(taskId);
	}
}

void ExecutionEngine::_terminate(const std::shared_ptr<TaskGate>& round, TaskStatus terminalStatus) {
	EngineScope active{this};
	auto& state = round->state;
	const TaskId& taskId = round->taskId;
	auto& output = state->output;
	auto& signals = *state->signals;

	// 幂等：仅 Running → 终态迁移一次有效；迁移成功即置 terminated 标记。
	// 确立「已收尾 ⇒ 已终止」不变量：迟到重试与在飞传播经 terminated 检查全部丢弃。
	// 此处仅发布终态，结果可读由收尾守卫统一发布；cancel/TTL/打卡共用此迁移点。
	{
		std::lock_guard lk(round->m);
		if (round->terminalStatus != TaskStatus::Running)
			return;
		round->terminalStatus = terminalStatus;
		round->terminated.store(true, std::memory_order_release);
	}

	// 收尾守卫（RAII）：任何步骤异常不得跳过结果可读发布、唤醒与弃置回收。
	struct FinishGuard {
		std::shared_ptr<TaskGate> round;
		ExecutionEngine* engine;

		~FinishGuard() {
			bool autoRel;
			{
				std::lock_guard lk(round->m);
				round->resultsReady = true;
				autoRel = round->autoRelease;
			}
			round->cv.notify_all();
			if (autoRel)
				engine->_finalizeDetached(round);
		}
	} finish{round, this};

	// ① 触发完成回调（数据仍在，可安全读取）；锁内拷贝、锁外调用，
	//    回调异常隔离为 Warning，不影响终态判定与资源回收。
	TaskCompleteCallback cb;
	{
		std::lock_guard lk(_cbMutex);
		cb = _taskCompleteCb;
	}
	if (cb) {
		try {
			cb(taskId);
		} catch (const std::exception& e) {
			state->errors.recordWarning(taskId, "", "ExecutionEngine::_terminate",
										"task complete callback threw: " + std::string(e.what()));
		} catch (...) {
			state->errors.recordWarning(taskId, "", "ExecutionEngine::_terminate",
										"task complete callback threw: unknown exception");
		}
	}

	// ② 清理 task 级信号，防止泄漏
	signals.clearTask(taskId);

	// ③ 清理阻塞追踪记录
	{
		std::lock_guard lk(_blockedSkipsMutex);
		_blockedSkips.erase(taskId);
	}

	// ④ 抢救结果 + 清理执行态：数据先转移至 OutputZone 再从域表清除
	//    （在飞 lambda 经 round->execState 保活）；clearTaskState 经守卫必然执行。
	//    锁协议：本段在 round->m 内，与 feedInput 写入同互斥；取数经
	//    tryTakeOutput 后到者得 nullopt，异常不从 _terminate 逃逸。
	{
		std::lock_guard lk(round->m);
		struct ClearGuard {
			TaskExecutionDomain& domain;
			const TaskId& taskId;

			~ClearGuard() { domain.clearTaskState(taskId); }
		} clearGuard{*state->exec, taskId};

		if (round->execState) {
			for (const auto& decl : output.declarationsOf(taskId)) {
				auto* ns = round->execState->find(decl.nodeName);
				if (!ns)
					continue;
				auto data = ns->buffer.tryTakeOutput(taskId, decl.portName);
				if (!data)
					continue;
				output.append(taskId, decl.nodeName, decl.portName, std::move(*data),
							  {decl.nodeName, decl.portName, taskId});
			}
		}
	}

	// ⑤⑥ 由 finish 守卫发布结果可读并唤醒等待者：wait 谓词与两个发布点
	// 绑定同一把轮次锁，不存在已终止但通知丢失的窗口。
}

void ExecutionEngine::_exhaustedCheck(const std::shared_ptr<TaskGate>& round) {
	auto& state = round->state;
	const TaskId& taskId = round->taskId;
	auto& output = state->output;
	{
		std::lock_guard lk(round->m);
		if (round->terminalStatus != TaskStatus::Running)
			return;
	}

	bool allMet = output.checkAllSatisfied(taskId);

	if (allMet) {
		// 声明已满足：正常终止；主路径在 OutputZone::accumulateAndCheck
		_terminate(round);
		return;
	}

	// 声明未满足且传播链已耗尽：存在 Error 诊断则终止为 Failed；
	// 仅 Warning 或无诊断则保持挂起，由宿主 wait+cancel 解围。
	auto errors = state->errors.taskErrors(taskId);
	bool hasError = false;
	for (const auto& e : errors) {
		if (e.level == DiagnosticLevel::Error) {
			hasError = true;
			break;
		}
	}
	if (hasError) {
		_diagnoseAbnormal(round,
						  "propagation chain exhausted with error-level diagnostics "
						  "(node-reported failure)");
		_terminate(round, TaskStatus::Failed);
		return;
	}

	// 不主动终止：保持挂起，宿主护栏接管
	_diagnoseAbnormal(round, "propagation chain exhausted with unsatisfied output declarations");
}

void ExecutionEngine::_diagnoseAbnormal(const std::shared_ptr<TaskGate>& round,
										const std::string& reason) {
	auto& state = round->state;
	const TaskId& taskId = round->taskId;
	// 快照经发布协议读取（acquire）；本地句柄钉住存活期
	auto snap = state->snapshot();
	auto& output = state->output;
	auto& graph = snap->runtimeView();
	auto& errors = state->errors;
	// ① 报告未满足的输出声明
	auto unsatisfied = output.unsatisfiedDeclarations(taskId);
	for (const auto& u : unsatisfied) {
		errors.recordWarning(taskId, u.decl.nodeName, "ExecutionEngine::_diagnoseAbnormal",
							 "declared output '" + u.decl.nodeName + ":" + u.decl.portName
								 + "' not satisfied (expected " + std::to_string(u.decl.count)
								 + ", got " + std::to_string(u.current) + "); reason: " + reason);
	}

	// ② 报告因信号阻塞被跳过的节点
	std::unordered_set<std::string> blocked;
	{
		std::lock_guard lk(_blockedSkipsMutex);
		auto it = _blockedSkips.find(taskId);
		if (it != _blockedSkips.end())
			blocked = it->second; // 拷贝，不 move：清理归 _terminate
	}
	for (const auto& nodeName : blocked) {
		errors.recordWarning(taskId, nodeName, "ExecutionEngine::_diagnoseAbnormal",
							 "node was signal-blocked during propagation; "
							 "data may have been prevented from reaching declared outputs");
	}

	// ③ 报告从未被到达的声明输出节点：本轮无执行态条目
	for (const auto& u : unsatisfied) {
		auto* node = graph.node(u.decl.nodeName);
		auto* ns = round->execState ? round->execState->find(u.decl.nodeName) : nullptr;
		if (node && !ns) {
			errors.recordWarning(taskId, u.decl.nodeName, "ExecutionEngine::_diagnoseAbnormal",
								 "node was never reached during propagation "
								 "(no task-level IO buffer was created)");
		}
	}
}

bool ExecutionEngine::wait(const TaskId& taskId, std::chrono::milliseconds timeout) {
	// 未知 taskId（从未提交或已释放）不可终止，立即返回 false，防拼写错误挂死
	auto round = _findRound(taskId);
	if (!round)
		return false;
	// timeout <= 0 视为无限等待（宿主护栏：只放弃等待，非执行超时语义）。
	// 单锁等待协议：谓词绑定终态 + 结果可读，与两个发布点同持 round->m，不会睡过通知。
	std::unique_lock lk(round->m);
	auto ready = [&round] {
		return round->terminalStatus != TaskStatus::Running && round->resultsReady;
	};
	if (timeout.count() <= 0) {
		round->cv.wait(lk, ready);
		return true;
	}
	return round->cv.wait_for(lk, timeout, ready);
}

TaskStatus ExecutionEngine::status(const TaskId& taskId) const {
	auto round = _findRound(taskId);
	if (!round)
		return TaskStatus::Unknown;
	std::lock_guard lk(round->m);
	return round->terminalStatus;
}

bool ExecutionEngine::isFinalizing(const TaskId& taskId) const {
	auto round = _findRound(taskId);
	if (!round)
		return false;
	std::lock_guard lk(round->m);
	return round->terminalStatus != TaskStatus::Running && !round->resultsReady;
}

bool ExecutionEngine::cancel(const TaskId& taskId) {
	EngineScope active{this};
	auto round = _findRound(taskId);
	if (!round)
		return false; // 未知或已释放
	if (round->terminated.exchange(true, std::memory_order_acq_rel))
		return false; // 已被正常路径终止
	{
		std::lock_guard lk(round->m);
		if (round->terminalStatus != TaskStatus::Running)
			return false; // 正常路径已收束，迁移幂等
	}
	// 传播链在下个检查点停止；缓冲与信号由 _terminate 清理
	_terminate(round, TaskStatus::Cancelled);
	return true;
}

bool ExecutionEngine::releaseTask(const TaskId& taskId) {
	std::lock_guard lk(_roundsMutex);
	auto it = _rounds.find(taskId);
	if (it == _rounds.end())
		return false; // 从未提交或已释放
	{
		std::lock_guard lk2(it->second->m);
		if (it->second->terminalStatus == TaskStatus::Running || !it->second->resultsReady)
			return false; // 活动任务或收尾中：不可释放
	}
	_rounds.erase(it); // 在飞 lambda 经 shared_ptr 副本保活
	return true;
}

void ExecutionEngine::detachTask(const TaskId& taskId) {
	auto round = _findRound(taskId);
	if (!round)
		return; // 未知：无托管对象；输入由上层 discardUnsubmitted 清理
	{
		std::lock_guard lk(round->m);
		if (round->terminalStatus == TaskStatus::Running || !round->resultsReady) {
			// 在飞或收尾中：登记自动回收，不取消任务
			round->autoRelease = true;
			return;
		}
	}
	// 已终止且收尾完成：立即释放；身份校验防误清新一轮状态
	bool owned = false;
	{
		std::lock_guard lk(_roundsMutex);
		auto it = _rounds.find(taskId);
		if (it != _rounds.end() && it->second == round) {
			_rounds.erase(it);
			owned = true;
		}
	}
	if (owned) {
		round->state->output.clearTask(taskId);
		round->state->errors.clearTask(taskId);
	}
}

} // namespace DC
