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
	if (!terminated.load(std::memory_order_acquire) && engine && output && graph && signals) {
		engine->_exhaustedCheck(taskId, *output, *graph, *signals);
	}
}

// ════════════════════════════════════════════
// 构造
// ════════════════════════════════════════════

ExecutionEngine::ExecutionEngine(size_t schedulerThreads)
	: _ownedScheduler(std::make_unique<CoroScheduler>(schedulerThreads)),
	  _scheduler(_ownedScheduler.get()),
	  _computePool({}), _operatorPool({}), _systemPool({}) {}

ExecutionEngine::ExecutionEngine(CoroScheduler& scheduler,
								 const PoolConfig& computeCfg,
								 const PoolConfig& operatorCfg,
								 const PoolConfig& systemCfg)
	: _scheduler(&scheduler),
	  _computePool(computeCfg), _operatorPool(operatorCfg), _systemPool(systemCfg) {}

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

PoolTicket ExecutionEngine::_dispatchToPoolAsync(ThreadPoolAffinity affinity, const std::string& tag,
												 std::function<void()> task) {
	switch (affinity) {
	case ThreadPoolAffinity::Compute:
		return _computePool.submitAsync(tag, std::move(task));
	case ThreadPoolAffinity::Operator:
		return _operatorPool.submitAsync(tag, std::move(task));
	case ThreadPoolAffinity::System:
		return _systemPool.submitAsync(tag, std::move(task));
	}
	return _systemPool.submitAsync(tag, std::move(task));
}

// ════════════════════════════════════════════
// 异步提交：协程驱动的数据传播
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

	// 创建任务门控：协程链与看门狗共享，最后一个持有者析构时触发耗尽检测
	auto gate = std::make_shared<TaskGate>();
	gate->engine = this;
	gate->output = &output;
	gate->graph = &graph;
	gate->signals = &signals;
	gate->taskId = taskId;

	// 超时看门狗（std::jthread + stop_token，生命周期由 ExecutionEngine 管理）
	if (timeout.count() > 0) {
		auto deadline = std::chrono::steady_clock::now() + timeout;
		_watchdogs[taskId] = std::jthread(
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
					errors.recordError(taskId, "<watchdog>", "ExecutionEngine::submit",
									   "task timed out (" + std::to_string(timeout.count())
										   + "ms) without meeting output declarations");
					_terminate(taskId, graph, output, signals);
				}
			});
	}

	// 扫描全图，对所有已就绪的节点创建传播协程
	for (const auto& [nodeName, nodePtr] : graph.nodes()) {
		if (!nodePtr->isReady(taskId))
			continue;

		// 根据 affinity 将入口节点提交到对应线程池
		_dispatchToPool(nodePtr->affinity(), nodePtr->tag(),
						[nodePtr = nodePtr.get(), taskId] {
			try {
				nodePtr->tryExecute(taskId);
			} catch (const NodeException& e) {
				// 线程池中无法直接调用 recordError，错误通过协程链中 whenComplete 的 result 传播
			}
		});

		// 创建协程：等待节点完成后自动传播数据到下游
		_scheduler->spawnTask(
			_propagateFrom(nodeName, taskId, gate, maxHops, graph, output, signals, errors));
	}
}

// ════════════════════════════════════════════
// 协程数据传播核心
// ════════════════════════════════════════════

Task<void> ExecutionEngine::_propagateFrom(std::string nodeName, TaskId taskId,
										   std::shared_ptr<TaskGate> gate,
										   uint32_t remainingHops,
										   GraphStore& graph, OutputZone& output,
										   SignalStore& signals, ErrorTracker& errors) {
	// [检查点 0] TTL 耗尽：主动终止 task（不依赖 _onExhausted）
	if (remainingHops == 0) {
		errors.recordError(taskId, nodeName, "ExecutionEngine::_propagateFrom",
						   "propagation hops exhausted (TTL=0) at node '" + nodeName
							   + "': cycle or excessively deep graph detected");
		gate->terminated.store(true, std::memory_order_release);
		_terminate(taskId, graph, output, signals);
		co_return;
	}

	// [检查点 1] 入口：若 task 已终止，直接返回
	if (_isTerminated(taskId)) {
		co_return;
	}

	auto* src = graph.findNode(nodeName);
	if (!src)
		co_return;

	// co_await 挂起当前协程，等待该节点执行完成
	auto result = co_await src->whenComplete(taskId);
	if (!result.ok()) {
		errors.recordError(taskId, nodeName, "ExecutionEngine::_propagateFrom",
						   "Node execution failed: " + result.message);
		co_return; // 失败则不传播
	}

	// [检查点 2] 节点完成 → 三步流水线：打卡 → OutputZone 搬运 → 边搬运
	//
	// 第一步：打卡 — 所有产出端口统一累加计数，不论目的地
	for (const auto& outPort : src->schema().outputs) {
		if (!src->hasOutput(taskId, outPort.name))
			continue;
		if (output.accumulateAndCheck(nodeName, outPort.name, taskId)) {
			gate->terminated.store(true, std::memory_order_release);
			_terminate(taskId, graph, output, signals);
			co_return;
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
		auto* dst = graph.findNode(edge.dstNode);
		if (!dst)
			continue;
		if (dst->isBlocked(taskId))
			continue;

		Value data = src->getOutput(taskId, edge.srcPort);

		// [检查点 3] 写入下游前再确认一次未被终止
		if (_isTerminated(taskId))
			co_return;

		try {
			dst->setInput(taskId, edge.dstPort, std::move(data));
		} catch (const NodeException& e) {
			errors.recordError(taskId, edge.dstNode, "ExecutionEngine::_propagateFrom",
							   "NodeException in setInput for port '" + edge.dstPort
								   + "': " + std::string(e.what()));
			continue;
		}

		// 下游就绪 → 按 affinity 提交到相应线程池
		if (dst->isReady(taskId)) {
			auto dstNode = edge.dstNode;
			co_await _dispatchToPoolAsync(dst->affinity(), dst->tag(),
				[this, dst, taskId, dstNode, &errors] {
					try {
						dst->tryExecute(taskId);
					} catch (const NodeException& e) {
						errors.recordError(taskId, dstNode,
										   "ExecutionEngine::_propagateFrom",
										   "NodeException in tryExecute: "
											   + std::string(e.what()));
					}
				});

			// [检查点 1b] 提交完确认未被终止再 spawn 下游
			if (_isTerminated(taskId))
				co_return;

			// 创建下游传播协程（非阻塞：fire-and-forget spawn）
			_scheduler->spawnTask(
				_propagateFrom(edge.dstNode, taskId, gate, remainingHops - 1,
							   graph, output, signals, errors));
		}
	}
}

// ════════════════════════════════════════════
// 终止辅助
// ════════════════════════════════════════════

bool ExecutionEngine::_isTerminated(const TaskId& taskId) const {
	std::lock_guard lk(_terminationMutex);
	return _terminatedTasks.contains(taskId);
}

void ExecutionEngine::_terminate(const TaskId& taskId,
								 GraphStore& graph, OutputZone& output,
								 SignalStore& signals) {
	{
		std::lock_guard lk(_terminationMutex);
		// 防止重复终止
		if (!_terminatedTasks.insert(taskId).second)
			return;
	}

	// ① 取消并 join 超时看门狗（若存在）
	//    erase 触发 std::jthread 析构 → request_stop() → join()
	_watchdogs.erase(taskId);

	// ② 触发 task 完成回调（数据仍在，回调可安全读取并捕获输出）
	if (_taskCompleteCb) {
		_taskCompleteCb(taskId);
	}

	// ③ 清理输出区（同一 taskId 可安全重新提交）
	output.clearTask(taskId);

	// ④ 清理该 task 的所有 task 级信号（防止泄漏）
	signals.clearTask(taskId);

	// ⑤ 遍历所有节点，清理此 taskId 的 IO 缓冲区并通知等待者
	for (auto& [name, nodePtr] : graph.nodes()) {
		if (nodePtr->hasTask(taskId))
			nodePtr->terminateTask(taskId);
	}

	// ⑥ 通知同步等待者（所有清理已完成，数据应由回调预先捕获）
	_completionCv.notify_all();
}

// ════════════════════════════════════════════
// 耗尽检测：TaskGate 析构或超时触发
// ════════════════════════════════════════════

void ExecutionEngine::_exhaustedCheck(const TaskId& taskId, OutputZone& output,
									  GraphStore& graph, SignalStore& signals) {
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

	// 声明未满足，数据可能残留在阻塞路径上（节点 isBlocked=true 导致边被跳过）——
	// 这是正常工作流的一部分，不是异常。不报错、不终止，留给看门狗检测真正的死锁。
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

} // namespace DC
