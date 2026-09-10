#include "InferGraph.h"
#include "NodeException.h"
#include "GraphException.h"
#include "Graph/internal/TaskExecutionState.h"
#include "SignalProbe.h"

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
	auto* n = _topology().node(nodeName);
	if (!n)
		throw GraphException(GraphException::ErrorType::NodeNotFound, "InferGraph::feedInput",
							 "node '" + nodeName + "' not found");
	try {
		// 输入写入 task 执行域的 per-node 缓冲（原 Node 内嵌 TaskBuffer）；
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
	_engine->wait(taskId, timeout);   // timeout <= 0 视为无限等待
	TaskResult result;
	result.status = taskStatus(taskId);
	result.errors = _state->errors.taskErrors(taskId);
	return result;
}

void InferGraph::releaseTask(const TaskId& taskId) {
	_engine->releaseTask(taskId);     // 仅终止态可释放（活动任务拒绝）
	_state->output.clearTask(taskId);   // 释放结果 artifact
	_state->errors.clearTask(taskId);       // 释放诊断记录
}

// ════════════════════════════════════════════
// 图导出：将完整推理图包装为可嵌入父图的 Node
// ════════════════════════════════════════════

std::unique_ptr<Node> InferGraph::exportNode(const std::string& nodeName, uint32_t maxHops) {
	// ① 从 InputZone 推导输入 Schema
	Node::Schema inSchema;
	for (auto& b : _inputBindingsView()) {
		auto* n = _topology().node(b.nodeName);
		if (!n) continue;
		auto* port = n->schema().findInput(b.portName);
		if (port) inSchema.inputs.push_back(*port);
	}

	// ② 从 OutputZone 推导输出 Schema（跳过连接器）
	Node::Schema outSchema;
	for (auto& b : _outputBindingsView()) {
		auto* n = _topology().node(b.nodeName);
		if (!n || n->isConnector()) continue;
		auto* port = n->schema().findOutput(b.portName);
		if (port) outSchema.outputs.push_back(*port);
	}

	Node::Schema fullSchema;
	fullSchema.inputs = std::move(inSchema.inputs);
	fullSchema.outputs = std::move(outSchema.outputs);

	// ③ 构造 RunFn：捕获 this + maxHops
	//    调用者必须保证 this 在 Node 生命周期内有效
	auto runFn = [this, maxHops](Node::RunContext& ctx) -> Node::Result {
		const std::string tid = ctx.name();

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

		// 驱动子图（不设执行超时：时间语义归节点实现方；由 TTL 与宿主护栏兜底）。
		// wait 返回即 task 已终止：_terminate 已把声明输出抢救至 OutputZone，
		// 此后声明输出必可经 takeOutput 取出
		submit(tid, std::move(declarations), maxHops);
		_engine->wait(tid, std::chrono::milliseconds(0)); // 内部路径：无限等待至子图 task 终止

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
	graphNode->setBlockedOverride([this](const Node::TaskId& tid) {
		_ensureFrozen(); // 通路检测读图级签名（冻结快照）
		return !canSatisfyDeclarations(_state->graph->store(), _state->graph->signature(), tid);
	});

	return graphNode;
}

} // namespace DC
