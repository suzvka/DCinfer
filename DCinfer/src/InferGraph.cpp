#include "InferGraph.h"
#include "NodeException.h"
#include "GraphException.h"
#include "Graph/internal/TaskExecutionState.h"

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
	if (ozVal) {
		// 发布残留载荷（广播共享/冻结）：产出独立可变副本
		if (ozVal->isPublished())
			return ozVal->cloneOwned();
		return std::move(*ozVal);
	}

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
	Value v = ns->buffer.takeOutput(taskId, portName);
	// 发布残留载荷（广播共享/冻结）：产出独立可变副本
	if (v.isPublished())
		return v.cloneOwned();
	return v;
}

Tensor InferGraph::takeOutputTensor(const TaskId& taskId, const std::string& nodeName,
								   const std::string& portName) {
	// 优先查 OutputZone
	auto ozVal = _state->output.take(taskId, nodeName, portName);
	if (ozVal) {
		auto* t = ozVal->as<Tensor>();
		if (t) {
			// 发布残留载荷（广播共享/冻结）：深拷贝产出独立可变副本
			if (ozVal->isPublished())
				return Tensor(*t);
			return std::move(*t);
		}
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
	// 发布残留载荷（广播共享/冻结）：深拷贝产出独立可变副本
	if (nt.isPublished())
		return Tensor(*t);
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

} // namespace DC
