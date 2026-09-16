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

namespace DC {

// ════════════════════════════════════════════
// TaskGate 析构
// ════════════════════════════════════════════

ExecutionEngine::TaskGate::~TaskGate() {
	// 纯资源回收（无副作用）：轮次生命周期由 shared_ptr 驱动。
	// 引用归零只发生在两类场景——① 终态收尾（_terminate）完成后表引用
	// 移除且无在飞 lambda / 等待者；② 引擎析构时轮次表整体移出。
	// 两类场景均无需（也不应）再触发耗尽检测：前者终态已发布，后者
	// 线程池即将 shutdown/join，不再有并发消费者。
}

// ════════════════════════════════════════════
// 构造 / 析构
// ════════════════════════════════════════════

ExecutionEngine::ExecutionEngine(const PoolConfig& computeCfg,
								 const PoolConfig& operatorCfg,
								 const PoolConfig& systemCfg)
	: _computePool(computeCfg),
	  _operatorPool(operatorCfg),
	  _systemPool(systemCfg) {}

ExecutionEngine::~ExecutionEngine() {
	// 先将轮次表整体移出（锁外释放）：析构中的轮次对象不再触发任何引擎回访，
	// 锁内 clear 保留防御性（未来若恢复回访路径，锁内 clear 会自死锁）。
	// 成员随后逆序析构：线程池先 shutdown/join，状态成员最后释放。
	decltype(_rounds) leftover;
	{
		std::lock_guard lk(_roundsMutex);
		leftover = std::move(_rounds);
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
									 std::shared_ptr<TaskGate> round, uint32_t remainingHops) {
	// 调度点检查：本轮已终止（取消/复用替换/收束）则不再提交新执行
	if (round->terminated.load(std::memory_order_acquire))
		return;

	// 在飞计数 +1：submit 入口扫描与传播下游提交统一经本函数促发
	round->inflight.fetch_add(1, std::memory_order_acq_rel);
	_dispatchToPool(node->affinity(),
					[this, node, nodeName, round, remainingHops] {
		// 在飞计数收尾（RAII）：任何退出路径均经本析构递减；归零且本轮
		// 未终止时由最后完成的 lambda 触发耗尽检测。
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

		// 排队期间本轮已被取消/收束 → 旧轮次禁止执行（不入流水线、不消费输入）
		if (round->terminated.load(std::memory_order_acquire))
			return;

		auto& state = round->state;
		const TaskId& taskId = round->taskId;
		auto& errors = state->errors;
		NodeResult result;
		// 执行态取自本轮轮次（submit 时捕获，lambda 持副本）：同 ID 复用后
		// 旧轮次 lambda 只写旧执行态，不再按 taskId 重新寻址消费新一轮输入。
		// 节点本身只读（const），可变状态全部在轮次执行域。
		auto execState = round->execState;
		try {
			auto& exec = execState->ensure(nodeName, node->schema());
			result = ExecutionPipeline::execute(
				taskId, *node, exec, state->exec->gateFor(nodeName),
				[round] { return round->terminated.load(std::memory_order_acquire); });
		} catch (const NodeException& e) {
			switch (e.getErrorType()) {
			case NodeException::ErrorType::Reentrant:
				// 节点正忙（另一任务/另一提交持有执行租约）：登记重投——
				// 闸释放时自动重放本提交，节点上的并发任务排队执行，
				// "节点正忙"不再记为 Error 判死任务（H-1）。重投入口天然经
				// round->terminated 检查丢弃已终止轮次。
				state->exec->gateFor(nodeName).enqueueRetry(
					round.get(), [this, node, nodeName, round, remainingHops] {
						_submitNodeRun(node, nodeName, round, remainingHops);
					});
				return;
			case NodeException::ErrorType::NotReady:
				// 就绪竞态（重复提交已由原子就绪路径消除，此处仅剩边缘场景）：
				// 降级为 Warning，跳过本次提交，不判死任务（H-1）。
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
			// 非 NodeException（引擎钩子/缓冲层抛出）此前穿过调度层逃逸到池 worker
			// （仅 stderr），任务因无诊断而永久挂起——现统一记录 Error 诊断，
			// 由耗尽检测收束为 Failed（H-2 失败闭环）。
			errors.recordError(taskId, nodeName, "ExecutionEngine::_submitNodeRun",
							   "non-NodeException escaped node execution: " + std::string(e.what()));
			return;
		} catch (...) {
			errors.recordError(taskId, nodeName, "ExecutionEngine::_submitNodeRun",
							   "unknown non-standard exception escaped node execution");
			return;
		}

		// 本轮已终止（取消/同 ID 复用竞态）→ 丢弃本轮结果，不传播。
		// 轮次级检查而非 taskId 级：复用后旧轮次的 lambda 不得污染新轮次。
		if (round->terminated.load(std::memory_order_acquire))
			return;

		// 完成判定与失败闭环（F05）：节点自报失败（含必需输出缺失
		// 归一化的 InternalError）必须优先于输出存在性——失败结果不得因
		// 产生过部分输出而被当作成功传播。失败记录完整诊断后跳过传播；
		// 仅成功且产出至少一个输出时继续传播（部分输出场景允许）。
		if (!result.ok()) {
			errors.recordError(taskId, nodeName, "ExecutionEngine::_submitNodeRun",
							   "Node execution failed: " + result.message, result.diagnostic);
			return; // 失败不传播
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
			return; // 无输出不传播
		}

		// 节点执行成功 → 就地传播输出到下游（池线程内，与调度点同上下文）
		_propagateFrom(nodeName, round, remainingHops);
	});
}

// ════════════════════════════════════════════
// 异步提交：事件驱动的数据传播
// ════════════════════════════════════════════

void ExecutionEngine::submit(const TaskId& taskId, uint32_t maxHops,
							 const std::shared_ptr<GraphRuntimeState>& state,
							 std::vector<OutputDeclaration> declarations) {
	// 快照经发布协议读取（acquire）：进入本函数前 facade 已 _ensureFrozen，
	// 本地句柄同时把快照存活期钉在本次调用内（state 亦持有）
	auto snap = state->snapshot();
	auto& graph = snap->runtimeView();

	// 校验：必须声明输出（声明随提交事务传入，由下方临界区原子写入）
	if (declarations.empty()) {
		throw GraphException(GraphException::ErrorType::NoDeclaration, "ExecutionEngine::submit",
							 "no output declarations for task '" + taskId
								 + "'; pass declarations via InferGraph::submit(...) / submitBound(...) first");
	}

	// 提交期拓扑守卫（F10：先于任何状态登记——失败不留 Running 残留，可重试）：
	// 声明目标必须从"已注入输入的节点 ∪ 输入绑定节点"纯拓扑可达。忽略信号——
	// 信号阻断属合法运行期状态，由宿主 wait+cancel 解围；构图/断链等确定性
	// 错误才在提交期立即暴露。检查不依赖轮次表/门控，可在登记前完成。
	// 目标集取自本次提交参数（声明写入移入下方事务，与登记同临界区）。
	std::vector<std::string> starts;
	if (auto taskExec = state->exec->findTaskState(taskId))
		starts = taskExec->nodeNames(); // 已注入输入的节点
	for (const auto& b : snap->signature().inputs)
		starts.push_back(b.nodeName);
	std::unordered_set<std::string> targets;
	for (const auto& d : declarations)
		targets.insert(d.nodeName);
	if (!starts.empty() && !targets.empty()
		&& !canSatisfyTopologically(snap->runtimeView(), starts, targets)) {
		throw GraphException(GraphException::ErrorType::UnreachableDeclaration,
							 "ExecutionEngine::submit",
							 "declared output is topologically unreachable from any fed input "
							 "node (cycle/break in graph construction)");
	}

	// 构建本轮轮次（TaskGate 自包含终态/结果可读/等待协议与本轮执行态）。
	// 执行态自此捕获：后续调度 lambda、传播链与终止清理全部经本对象寻址，
	// 不再按可复用 taskId 重新查表。
	auto round = std::make_shared<TaskGate>();
	round->engine = this;
	round->state = state; // 与图对象共享图运行时状态（在飞任务保活）
	round->taskId = taskId;

	// ── 提交事务（单临界区，H-3/H-5）──
	// 复用准入检查、声明清理与写入、执行态捕获、轮次登记原子完成：
	// - 并发同 ID 提交：恰一方通过检查并完成登记，败者抛 DuplicateTask 且
	//   零副作用（不再出现"先清声明、后拒提交"的破坏窗口）；
	// - 复用准入 = "终态已发布 + 清理完成（resultsReady）"：收尾窗口内
	//   （回调/结果抢救/clearTaskState 仍在执行）的 ID 一律拒绝——旧轮
	//   清理不可能再触碰新一轮的声明/累加/结果/执行态；
	// - 执行态捕获置于准入之后：复用时必为全新对象（旧轮 clearTaskState
	//   已将其摘除），不存在新旧轮共享执行态的跨轮污染。
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
		state->errors.clearTask(taskId); // 上一轮诊断不残留（影响 taskStatus 归一化）
		state->output.clearTask(taskId); // 复用同 ID：清掉上一轮声明/累加/结果
		state->output.declare(taskId, std::move(declarations));
		_rounds[taskId] = round; // 登记（替换旧轮；旧轮由在飞 lambda 保活至收尾）
	}
	// 节点执行态无需在此清理：所有终态必经 _terminate（唯一清理点）。

	// 扫描全图（运行时视图），对所有已就绪的节点提交执行任务（执行完成后再传播下游）。
	// 就绪查询仅针对本轮执行态条目（feedInput 时创建）：无条目 = 无暂存输入 = 未就绪
	for (const auto& [nodeName, nodePtr] : graph.nodes) {
		auto* ns = round->execState ? round->execState->find(nodeName) : nullptr;
		if (!ns || !nodePtr->isReady(taskId, ns->buffer))
			continue;

		// 入口节点：执行 + 完成后就地传播（_submitNodeRun 内部处理）
		_submitNodeRun(nodePtr, nodeName, round, maxHops);
	}
}

// ════════════════════════════════════════════
// 事件驱动数据传播核心
// ════════════════════════════════════════════

void ExecutionEngine::_propagateFrom(std::string nodeName, std::shared_ptr<TaskGate> round,
									 uint32_t remainingHops) {
	auto& state = round->state;
	const TaskId& taskId = round->taskId;
	// 快照经发布协议读取（acquire）；本地句柄钉住存活期，池线程不再
	// 无同步读取共享成员
	auto snap = state->snapshot();
	auto& graph = snap->runtimeView();
	auto& output = state->output;
	auto& errors = state->errors;
	// 调用前提：节点已由 _submitNodeRun 执行成功（失败路径已记录错误并跳过传播），
	// 输出已写入 TaskBuffer 输出槽位，本函数在池线程内就地执行。

	// [检查点 0] TTL 耗尽：主动终止 task（不依赖 _exhaustedCheck）
	if (remainingHops == 0) {
		std::string reason = "propagation hops exhausted (TTL=0) at node '" + nodeName
							 + "': cycle or excessively deep graph detected";
		errors.recordError(taskId, nodeName, "ExecutionEngine::_propagateFrom", reason);
		round->terminated.store(true, std::memory_order_release);
		_diagnoseAbnormal(round, reason);
		_terminate(round, TaskStatus::Failed);
		return;
	}

	// [检查点 1] 入口：若本轮任务已终止（取消/同 ID 复用竞态），直接返回。
	// 轮次级检查而非 taskId 级：复用后旧轮次 lambda 不得继续传播。
	if (round->terminated.load(std::memory_order_acquire)) {
		return;
	}

	auto* src = graph.node(nodeName);
	if (!src)
		return;

	// 本轮执行态（submit 时捕获；调用前提：本节点已执行成功，条目必已存在）
	const auto& execState = round->execState;
	NodeExecState* srcNs = execState ? execState->find(nodeName) : nullptr;

	// [检查点 2] 节点完成 → 三步流水线：打卡 → OutputZone 搬运 → 边搬运
	//
	// 第一步：打卡 — 所有产出端口统一累加计数，不论目的地
	for (const auto& outPort : src->schema().outputs) {
		if (!srcNs || !srcNs->buffer.hasOutput(taskId, outPort.name))
			continue;
		// 终止复查：打卡写入前再确认本轮未被终止（取消可能在检查点 1 之后触发）
		if (round->terminated.load(std::memory_order_acquire))
			return;
		if (output.accumulateAndCheck(nodeName, outPort.name, taskId)) {
			round->terminated.store(true, std::memory_order_release);
			_terminate(round, TaskStatus::Succeeded);
			return;
		}
	}

	// 第二步：OutputZone 目的地搬运 — OutputZone 绑定端口消费后自然空
	for (const auto& outPort : src->schema().outputs) {
		if (!srcNs || !srcNs->buffer.hasOutput(taskId, outPort.name))
			continue;
		if (snap->signature().isOutputBound(nodeName, outPort.name)) {
			// [检查点 2] 终止复查：OutputZone 写入前再确认本轮未被终止
			// （取消/复用可能在检查点 1 与打卡复查之后触发）
			if (round->terminated.load(std::memory_order_acquire))
				return;
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

		// [检查点 3] 写入下游前再确认一次本轮未被终止（轮次级，同 ID 复用安全）
		if (round->terminated.load(std::memory_order_acquire))
			return;

		// 下游执行态：经本轮执行态惰性创建（写入缓冲，不触发执行）。
		// execState 由轮次持有，存活至本轮传播结束
		NodeExecState* dstNs = nullptr;
		bool ready = false;
		try {
			dstNs = &execState->ensure(edge.dstNode, dst->schema());
			if (dst->hasReadyOverride()) {
				// 就绪语义由覆盖方定义（无法与写入合并判定）：保留 setInput +
				// isReady 分离路径，残余竞态归 override 实现方。
				dstNs->buffer.setInput(taskId, edge.dstPort, std::move(data), dst->schema());
				ready = dst->isReady(taskId, dstNs->buffer);
			} else {
				// 原子路径（H-1）：写入与就绪判定同临界区——多上游并发传播时
				// 仅"最后写入者"观察到就绪并返回 true，双触发自根上消除。
				ready = dstNs->buffer.setInputAndCheckReady(taskId, edge.dstPort,
															std::move(data), dst->schema());
			}
		} catch (const NodeException& e) {
			errors.recordError(taskId, edge.dstNode, "ExecutionEngine::_propagateFrom",
							   "NodeException in setInput for port '" + edge.dstPort
								   + "': " + std::string(e.what()));
			continue;
		}

		// 下游就绪 → 提交执行 + 完成后继续传播（数据冒泡）
		if (ready) {
			_submitNodeRun(dst, edge.dstNode, round, remainingHops - 1);
		}
	}
}

// ════════════════════════════════════════════
// 终止辅助
// ════════════════════════════════════════════

// ════════════════════════════════════════════
// 轮次辅助
// ════════════════════════════════════════════

std::shared_ptr<ExecutionEngine::TaskGate> ExecutionEngine::_findRound(const TaskId& taskId) const {
	std::lock_guard lk(_roundsMutex);
	auto it = _rounds.find(taskId);
	return it != _rounds.end() ? it->second : nullptr;
}

void ExecutionEngine::_finalizeDetached(const std::shared_ptr<TaskGate>& round) {
	const TaskId& taskId = round->taskId;
	// 身份校验：表内仍是本轮时移除并清理（已被显式 release 或新一轮替换则不动——
	// 新一轮的声明/结果由其自身 submit 流程清理）
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
	auto& state = round->state;
	const TaskId& taskId = round->taskId;
	auto& output = state->output;
	auto& signals = *state->signals;

	// 防止重复终止（幂等）：仅 Running → 终态迁移一次有效。
	// 此处仅发布终态；"结果可读"由收尾守卫在 notify 前统一发布。
	{
		std::lock_guard lk(round->m);
		if (round->terminalStatus != TaskStatus::Running)
			return;
		round->terminalStatus = terminalStatus;
	}

	// 收尾守卫（RAII，F07）：任何后续步骤异常不得跳过"结果可读发布 + 唤醒
	// 等待者"与弃置回收——终止事务必须完整（回调异常已在下方隔离，此处
	// 防御其余异常路径）。
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

	// ① 触发 task 完成回调（数据仍在，回调可安全读取并捕获输出）
	//    锁内拷贝、锁外调用；用户回调异常在此隔离（记录 Warning 后继续——
	//    回调失败不影响终态判定、结果发布与资源回收，终止事务不被中断）。
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

	// ② 清理该 task 的所有 task 级信号（防止泄漏）
	signals.clearTask(taskId);

	// ③ 清理信号阻塞追踪记录
	{
		std::lock_guard lk(_blockedSkipsMutex);
		_blockedSkips.erase(taskId);
	}

	// ④ 抢救结果 + 清理 task 执行态：
	//    终止路径在打卡满足后直接返回，声明端口的数据仍留在 task 缓冲。
	//    先把这些未及搬运的数据转移到 OutputZone（保证 wait → takeOutput 可取，
	//    含取消/失败路径的部分结果），再从域表清除本轮的节点执行态
	//    （在飞 lambda 经 round->execState 保活至流水线结束）。
	//    回调在①已先行触发，其消费过的端口 hasOutput=false 自然跳过。
	if (round->execState) {
		for (const auto& decl : output.declarationsOf(taskId)) {
			auto* ns = round->execState->find(decl.nodeName);
			if (!ns || !ns->buffer.hasOutput(taskId, decl.portName))
				continue;
			Value data = ns->buffer.takeOutput(taskId, decl.portName);
			output.append(taskId, decl.nodeName, decl.portName, std::move(data),
						  {decl.nodeName, decl.portName, taskId});
		}
	}
	state->exec->clearTaskState(taskId);

	// ⑤⑥ 由 finish 守卫发布"结果可读"并唤醒全部等待者：
	//    wait 谓词（终态 + resultsReady）与两个发布点绑定同一把轮次锁
	//    （round->m 单锁协议），不存在"已终止但通知丢失"的窗口。
}

// ════════════════════════════════════════════
// 耗尽检测：在飞 lambda 归零触发（TaskGate 析构仅作引擎析构兜底）
// ════════════════════════════════════════════

void ExecutionEngine::_exhaustedCheck(const std::shared_ptr<TaskGate>& round) {
	auto& state = round->state;
	const TaskId& taskId = round->taskId;
	auto& output = state->output;
	// 已终止则跳过
	{
		std::lock_guard lk(round->m);
		if (round->terminalStatus != TaskStatus::Running)
			return;
	}

	// 检查声明是否已满足
	bool allMet = output.checkAllSatisfied(taskId);

	if (allMet) {
		// 声明已满足 → 正常终止（守护路径，主路径在 OutputZone::accumulateAndCheck 中处理）
		_terminate(round);
		return;
	}

	// 声明未满足：传播链已耗尽。区分两类结局——
	// ① 存在 Error 级诊断（节点自报失败）：实现方拥有时间/失败解释权，
	//    任务终止为 Failed（节点失败闭环）。
	// ② 仅有 Warning 或无诊断（如信号阻塞停滞）：保持挂起，宿主 wait(t)+cancel 解围。
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

	// 无 Error：不主动终止，保持挂起，宿主护栏接管
	_diagnoseAbnormal(round, "propagation chain exhausted with unsatisfied output declarations");
}

// ════════════════════════════════════════════
// 运行时诊断
// ════════════════════════════════════════════

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

	// ③ 报告从未被到达的声明输出节点（本轮无执行态条目）
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

// ════════════════════════════════════════════
// 同步等待
// ════════════════════════════════════════════

bool ExecutionEngine::wait(const TaskId& taskId, std::chrono::milliseconds timeout) {
	// 未知 taskId（从未提交或已释放）不可终止，立即返回 false——
	// 无限等待模式下防误拼写 taskId 挂死；有限等待同样如实立即失败。
	auto round = _findRound(taskId);
	if (!round)
		return false;
	// timeout <= 0 视为无限等待（宿主护栏：只放弃等待，不作为执行超时语义）。
	// 单锁等待协议（F06）：谓词绑定"终态 + 结果可读"，与两个发布点
	// （_terminate 的终态迁移与收尾守卫的 resultsReady）绑定同一把
	// round->m——生产者更新状态后 notify，等待者不可能睡过通知。
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

// ════════════════════════════════════════════
// task 状态与取消
// ════════════════════════════════════════════

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
	auto round = _findRound(taskId);
	if (!round)
		return false; // 未知或已释放（幂等）
	if (round->terminated.exchange(true, std::memory_order_acq_rel))
		return false; // 已被正常路径终止
	{
		std::lock_guard lk(round->m);
		if (round->terminalStatus != TaskStatus::Running)
			return false; // 正常路径已收束（终态已发布，迁移幂等）
	}
	// 传播链在下个检查点停止；缓冲与信号由 _terminate 照常清理
	_terminate(round, TaskStatus::Cancelled);
	return true;
}

bool ExecutionEngine::releaseTask(const TaskId& taskId) {
	std::lock_guard lk(_roundsMutex);
	auto it = _rounds.find(taskId);
	if (it == _rounds.end())
		return false; // 未知（从未提交或已释放）：无释放资格
	{
		std::lock_guard lk2(it->second->m);
		if (it->second->terminalStatus == TaskStatus::Running || !it->second->resultsReady)
			return false; // 活动任务或收尾中（清理未完成）：不可释放
	}
	_rounds.erase(it); // 终态轮次移除；在飞 lambda 经 shared_ptr 副本保活，不受影响
	return true;
}

void ExecutionEngine::detachTask(const TaskId& taskId) {
	auto round = _findRound(taskId);
	if (!round)
		return; // 未知：无托管对象（feed 输入由上层 discardUnsubmitted 清理）
	{
		std::lock_guard lk(round->m);
		if (round->terminalStatus == TaskStatus::Running || !round->resultsReady) {
			// 在飞或收尾中：登记完成后自动回收（不取消任务，保留在飞/收尾语义）
			round->autoRelease = true;
			return;
		}
	}
	// 已终止且收尾完成：立即等价释放（身份校验防误清新一轮复用后的状态）
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
