#include "InferGraph.h"
#include "NodeException.h"
#include "GraphException.h"
#include "Graph/internal/TaskExecutionState.h"
#include "SignalProbe.h"

#include <unordered_map>

namespace DC {

// ════════════════════════════════════════════
// 构造
// ════════════════════════════════════════════

InferGraph::InferGraph(const PoolConfig& computeCfg,
					   const PoolConfig& operatorCfg, const PoolConfig& systemCfg)
	: _state(std::make_shared<GraphRuntimeState>()),
	  _engine(std::make_unique<ExecutionEngine>(computeCfg, operatorCfg, systemCfg)) {}

// ════════════════════════════════════════════
// 数据注入
// ════════════════════════════════════════════

void InferGraph::feedInput(const TaskId& taskId, const std::string& nodeName,
						   const std::string& portName, Value data) {
	_ensureFrozen(); // 惰性冻结：运行期 API 入口统一编译（此后拓扑不可变）

	// 收尾窗口（终态已发布、结果抢救未完成）拒绝注入：此时旧轮执行态容器
	// 仍挂在域表上、随后将被 clearTaskState 摘除——注入的输入会静默丢失。
	// 复用语义：waitForResult 返回（结果可读）后再 feed。
	if (_engine->isFinalizing(taskId))
		throw GraphException(GraphException::ErrorType::DuplicateTask, "InferGraph::feedInput",
							 "task '" + taskId
								 + "' is finalizing; feed input after waitForResult/release");

	auto* n = _topology().node(nodeName);
	if (!n)
		throw GraphException(GraphException::ErrorType::NodeNotFound, "InferGraph::feedInput",
							 "node '" + nodeName + "' not found");
	try {
		// 输入写入 task 执行域的 per-node 缓冲；
		// shared_ptr 先落局部量，防止临时量析构导致引用悬垂
		auto ts = _state->exec->taskState(taskId);
		auto& ns = ts->ensure(nodeName, n->schema());
		ns.buffer.setInput(taskId, portName, std::move(data), n->schema());
	} catch (const NodeException& e) {
		_state->errors.recordError(taskId, nodeName, "InferGraph::feedInput",
							"NodeException in setInput for port '" + portName
								+ "': " + std::string(e.what()));
		throw GraphException(GraphException::ErrorType::FeedFailed, "InferGraph::feedInput",
							"failed to feed input '" + portName + "' on node '" + nodeName
								+ "': " + std::string(e.what()));
	}
}

void InferGraph::feedInput(const TaskId& taskId, const std::string& nodeName,
						   const std::string& portName, Tensor data) {
	feedInput(taskId, nodeName, portName, Value(std::make_unique<Tensor>(std::move(data))));
}

void InferGraph::submitBound(const TaskId& taskId, uint32_t maxHops) {
	std::vector<OutputDeclaration> declarations;
	for (const auto& ob : _outputBindingsView())
		declarations.push_back({ob.nodeName, ob.portName, 1});
	if (declarations.empty())
		throw GraphException(GraphException::ErrorType::NoDeclaration, "InferGraph::submitBound",
							 "no output bindings; call bindOutput(alias, nodeName, portName) first");
	submit(taskId, std::move(declarations), maxHops);
}

// ════════════════════════════════════════════
// 结果获取
// ════════════════════════════════════════════

Value InferGraph::takeOutput(const TaskId& taskId, const std::string& nodeName,
							const std::string& portName) {
	// 优先查 OutputZone（OutputZone 绑定端口的数据在 _propagateFrom 第二步已搬运至此）
	auto ozVal = _state->output.take(taskId, nodeName, portName);
	if (ozVal)
		return std::move(*ozVal);

	auto* n = _topology().node(nodeName);
	if (!n) {
		throw GraphException(GraphException::ErrorType::NodeNotFound, "InferGraph::takeOutput",
							 "node '" + nodeName + "' not found");
	}
	// 回退查 task 执行域的 per-node 缓冲
	auto taskExec = _state->exec->findTaskState(taskId);
	auto* ns = taskExec ? taskExec->find(nodeName) : nullptr;
	if (!ns)
		throw NodeException(NodeException::ErrorType::TaskNotFound, "TaskBuffer::takeOutput",
							"task '" + taskId + "' not found");
	return ns->buffer.takeOutput(taskId, portName);
}

Tensor InferGraph::takeOutputTensor(const TaskId& taskId, const std::string& nodeName,
								   const std::string& portName) {
	// 优先查 OutputZone
	auto ozVal = _state->output.take(taskId, nodeName, portName);
	if (ozVal) {
		auto* t = ozVal->as<Tensor>();
		if (t)
			return std::move(*t);
		throw GraphException(GraphException::ErrorType::Other, "InferGraph::takeOutputTensor",
							 "OutputZone artifact for '" + nodeName + "." + portName
								 + "' is not a DC::Tensor");
	}

	auto* n = _topology().node(nodeName);
	if (!n) {
		throw GraphException(GraphException::ErrorType::NodeNotFound, "InferGraph::takeOutputTensor",
							 "node '" + nodeName + "' not found");
	}
	auto taskExec = _state->exec->findTaskState(taskId);
	auto* ns = taskExec ? taskExec->find(nodeName) : nullptr;
	if (!ns)
		throw NodeException(NodeException::ErrorType::TaskNotFound, "TaskBuffer::takeOutput",
							"task '" + taskId + "' not found");
	auto nt = ns->buffer.takeOutput(taskId, portName);
	auto* t = nt.as<Tensor>();
	if (!t) {
		throw NodeException(NodeException::ErrorType::TypeMismatch, "Node::takeOutputTensor",
							"output '" + portName + "' is not a DC::Tensor (innerType=" +
								std::to_string(static_cast<uint32_t>(nt.innerType())) + ")");
	}
	return std::move(*t);
}

bool InferGraph::hasOutput(const TaskId& taskId, const std::string& nodeName,
						   const std::string& portName) const {
	// 优先查 OutputZone
	if (_state->output.hasOutput(taskId, nodeName, portName))
		return true;

	auto* n = _topology().node(nodeName);
	if (!n)
		return false;
	auto taskExec = _state->exec->findTaskState(taskId);
	auto* ns = taskExec ? taskExec->find(nodeName) : nullptr;
	return ns && ns->buffer.hasOutput(taskId, portName);
}

// ════════════════════════════════════════════
// task 生命周期：状态 / 结构化等待 / 资源回收
// ════════════════════════════════════════════

TaskStatus InferGraph::taskStatus(const TaskId& taskId) const {
	auto st = _engine->status(taskId);
	if (st == TaskStatus::Succeeded) {
		// 正常终止但存在 Error 级诊断 → 归一化为 Failed（部分节点执行失败）
		for (const auto& e : _state->errors.taskErrors(taskId)) {
			if (e.level == DiagnosticLevel::Error)
				return TaskStatus::Failed;
		}
	}
	return st;
}

TaskResult InferGraph::waitForResult(const TaskId& taskId) {
	return waitForResult(taskId, std::chrono::milliseconds(0)); // 0 = 无限等待
}

TaskResult InferGraph::waitForResult(const TaskId& taskId, std::chrono::milliseconds timeout) {
	const bool ready = _engine->wait(taskId, timeout); // timeout <= 0 视为无限等待
	TaskResult result;
	if (!ready) {
		// 等待未满足（超时，或 taskId 未知/已释放）：如实返回 {Running / Unknown}。
		// 终态已迁移但收尾（结果抢救）尚未完成的竞争窗口同样按未完成报告——
		// "wait 返回 true 才保证结果可读"契约不因状态表提前可见而失真，
		// 调用方可继续 wait 或按 Running 语义处理。
		const TaskStatus st = taskStatus(taskId);
		result.status = (st == TaskStatus::Unknown) ? TaskStatus::Unknown : TaskStatus::Running;
		result.errors = _state->errors.taskErrors(taskId);
		return result;
	}
	result.status = taskStatus(taskId);
	result.errors = _state->errors.taskErrors(taskId);
	return result;
}

void InferGraph::releaseTask(const TaskId& taskId) {
	if (!_engine->releaseTask(taskId))
		return; // 未知或活动任务：引擎拒绝释放，结果/诊断保持不动（待终态后再释放）
	_state->output.clearTask(taskId);   // 释放结果 artifact
	_state->errors.clearTask(taskId);   // 释放诊断记录
}

void InferGraph::discardUnsubmitted(const TaskId& taskId) {
	// 仅清理"已喂数据但从未成功提交"（Unknown 状态）的输入执行态/声明/诊断。
	// 活动或已终止任务不受影响：前者须保留输入（在飞执行），后者分别由
	// detachTask（弃置回收）与 releaseTask（显式释放）路径管理。
	if (_engine->status(taskId) != TaskStatus::Unknown)
		return;
	_state->exec->clearTaskState(taskId);
	_state->output.clearTask(taskId);
	_state->errors.clearTask(taskId);
}

void InferGraph::detachTask(const TaskId& taskId) {
	// 转发引擎：不取消在飞任务；完成收尾时自动回收
	// （状态表条目 / OutputZone 结果 / 诊断）；已终止立即释放；未知 no-op。
	_engine->detachTask(taskId);
}

// ════════════════════════════════════════════
// 图导出：将完整推理图包装为可嵌入父图的 Node
// ════════════════════════════════════════════

std::unique_ptr<Node> InferGraph::exportNode(const std::string& nodeName, uint32_t maxHops) {
	// 单栈命名法则：端口名在导出时扁平化为子图接口端口名（父图按该名寻址），
	// 接口层端口名必须唯一——同名端口（跨节点同名/同一端口重复绑定）拒绝导出。
	// 运行时寻址 (nodeName, portName) 在普通拓扑下天然消歧，仅此扁平化时刻需要
	// 唯一性；不校验会让同名端口静默折叠到同一槽位（数据串扰）。

	// ① 从 InputZone 推导输入 Schema
	Node::Schema inSchema;
	std::unordered_map<std::string, std::string> inPortSource; // 端口名 → 来源 "node.port"
	for (auto& b : _inputBindingsView()) {
		auto* n = _topology().node(b.nodeName);
		if (!n) continue;
		auto* port = n->schema().findInput(b.portName);
		if (!port) continue;
		const auto srcTag = b.nodeName + "." + b.portName;
		auto [it, inserted] = inPortSource.try_emplace(port->name, srcTag);
		if (!inserted)
			throw GraphException(GraphException::ErrorType::DuplicatePort, "InferGraph::exportNode",
								 "subgraph export rejected: duplicate input port '" + port->name +
									 "' (from '" + it->second + "' and '" + srcTag +
									 "'); rename ports to unique names before export");
		inSchema.inputs.push_back(*port);
	}

	// ② 从输出绑定推导输出 Schema（跳过连接器；端口名同样要求唯一）
	Node::Schema outSchema;
	std::unordered_map<std::string, std::string> outPortSource; // 端口名 → 来源 "node.port"
	for (auto& b : _outputBindingsView()) {
		auto* n = _topology().node(b.nodeName);
		if (!n || n->isConnector()) continue;
		auto* port = n->schema().findOutput(b.portName);
		if (!port) continue;
		const auto srcTag = b.nodeName + "." + b.portName;
		auto [it, inserted] = outPortSource.try_emplace(port->name, srcTag);
		if (!inserted)
			throw GraphException(GraphException::ErrorType::DuplicatePort, "InferGraph::exportNode",
								 "subgraph export rejected: duplicate output port '" + port->name +
									 "' (from '" + it->second + "' and '" + srcTag +
									 "'); rename ports to unique names before export");
		outSchema.outputs.push_back(*port);
	}

	Node::Schema fullSchema;
	fullSchema.inputs = std::move(inSchema.inputs);
	fullSchema.outputs = std::move(outSchema.outputs);

	// ③ 构造 RunFn：捕获 this + maxHops + 生命周期哨兵（CORE-04）
	//    调用者必须保证 this 在 Node 生命周期内有效；哨兵为 best-effort 检测——
	//    子图先析构再执行时返回 ExecutionFailed 而非悬垂段错误。
	std::weak_ptr<void> lifeToken = _lifeToken; // RunFn 外先建 weak，避免 lambda 捕获表达式过于复杂
	auto runFn = [this, maxHops, lifeToken](Node::RunContext& ctx) -> Node::Result {
		if (lifeToken.expired()) {
			return ctx.failure(Node::Status::ExecutionFailed,
							   "subgraph owner InferGraph was destroyed before the exported node executed; "
							   "the subgraph must outlive every run of its exported node");
		}
		// 子图 task ID = 父任务 ID：taskId 空间贯穿父子边界
		// （blockedOverride/SignalProbe 同空间寻址；并发父任务天然互不冲突）。
		// 注：同一父任务内同一子图的多个导出节点并发调用不受支持
		//（同一父任务下复用同一子图 task 空间，将以 DuplicateTask 显式失败）。
		const std::string tid = ctx.taskId();

		// 将 RunContext 的输入注入子图
		for (auto& ib : _inputBindingsView()) {
			const auto& inVal = ctx.peek(ib.portName);
			if (!inVal.as<Tensor>()) {
				continue;
			}
			auto val = ctx.pop(ib.portName);
			feedInput(tid, ib.nodeName, ib.portName, std::move(val));
		}

		// 收集输出声明
		std::vector<OutputDeclaration> declarations;
		for (auto& ob : _outputBindingsView()) {
			declarations.push_back({ob.nodeName, ob.portName, 1});
		}

		// 驱动子图（无执行超时：时间语义归节点实现方，由 TTL 与宿主护栏兜底）。
		// wait 返回即 task 已终止：_terminate 已把声明输出抢救至 OutputZone，
		// 此后声明输出必可经 takeOutput 取出。
		// 分段等待 + 父轮感知（H-4）：每 100ms 轮询父轮是否已被请求取消/已终止
		// （宿主 cancel / TTL 耗尽 / 收束 / 同 ID 复用替换）——子图信号阻塞令
		// 声明无法满足时，等待不再令池线程永久挂起：检测到父轮终止即主动
		// cancel 子图 task 并解围返回（父子取消边界打通）。
		submit(tid, std::move(declarations), maxHops);
		static constexpr auto kParentPollInterval = std::chrono::milliseconds(100);
		while (!_engine->wait(tid, kParentPollInterval)) {
			if (!ctx.isCancellationRequested())
				continue;
			cancel(tid); // 子图任务解围（幂等；已终态时无操作）
			_engine->wait(tid, std::chrono::seconds(1)); // 有限等待收尾（结果不再使用）
			return ctx.failure(Node::Status::ExecutionFailed,
							   "parent task terminated while awaiting subgraph completion");
		}

		// 检查本 task 的错误诊断（task 级判定，不读全局 hasErrors/clearErrors）
		auto errors = taskErrors(tid);
		if (!errors.empty()) {
			return ctx.failure(Node::Status::ExecutionFailed, errors[0].message);
		}

		// 收集输出到 RunContext（终止后结果保留在 OutputZone，直至下一次同 ID submit）
		for (auto& ob : _outputBindingsView()) {
			if (hasOutput(tid, ob.nodeName, ob.portName)) {
				ctx.output(ob.portName, takeOutput(tid, ob.nodeName, ob.portName));
			}
		}

		return ctx.success();
	};

	// 构造 GraphNode：注册状态委托（内部声明通路检测 → 资源中介语义）
	auto graphNode = std::make_unique<Node>(
		"GraphNode", nodeName, fullSchema,
		std::move(runFn),
		ThreadPoolAffinity::Compute);

	// blockedOverride：内部无通路满足输出声明时，子图节点向父级应答阻塞。
	// isReady 保持边界缓冲语义（父级数据齐即可进入执行）。
	// 注：本回调由父图执行线程调用——首次冻结可能在此触发（并发安全：
	// 快照经发布协议一次性就位，见 GraphRuntimeState 发布协议）。
	graphNode->setBlockedOverride([this, lifeToken](const Node::TaskId& tid) {
		if (lifeToken.expired()) {
			// 子图已析构：不阻塞，让节点进入执行后由 RunFn 哨兵显式失败
			return false;
		}
		auto snap = _ensureFrozen(); // 通路检测读图级签名（冻结快照）
		return !canSatisfyDeclarations(snap->store(), snap->signature(), tid);
	});

	return graphNode;
}

} // namespace DC
