#include "InferGraph.h"
#include "NodeException.h"
#include "GraphException.h"
#include "Graph/internal/TaskExecutionState.h"

namespace DC {

InferGraph::InferGraph(std::shared_ptr<ResourceScheduler> scheduler)
	: _state(std::make_shared<GraphRuntimeState>()),
	  _scheduler(scheduler ? std::move(scheduler) : ResourceScheduler::instance()),
	  _engine(std::make_unique<ExecutionEngine>(_scheduler)) {}

void InferGraph::feedInput(const TaskId& taskId, const std::string& nodeName,
						   const std::string& portName, Value data) {
	_ensureFrozen(); // 惰性冻结：运行期入口统一编译，拓扑此后不可变

	// 复用语义：waitForResult 返回后再 feed；收尾窗口写入与抢救同锁互斥，无静默丢失。
	if (_engine->isFinalizing(taskId))
		throw GraphException(GraphException::ErrorType::DuplicateTask, "InferGraph::feedInput",
							 "task '" + taskId
								 + "' is finalizing; feed input after waitForResult/release");

	auto* n = _topology().node(nodeName);
	if (!n)
		throw GraphException(GraphException::ErrorType::NodeNotFound, "InferGraph::feedInput",
							 "node '" + nodeName + "' not found");
	try {
		// 轮次锁内写入：收尾窗口拒绝时不给旧轮写输入；taskState 先落局部量防悬垂
		const bool accepted = _engine->tryWriteTaskState(taskId, [&] {
			auto ts = _state->exec->taskState(taskId);
			auto& ns = ts->ensure(nodeName, n->schema());
			ns.buffer.setInput(taskId, portName, std::move(data), n->schema());
		});
		if (!accepted)
			throw GraphException(GraphException::ErrorType::DuplicateTask, "InferGraph::feedInput",
								 "task '" + taskId
									 + "' is finalizing; feed input after waitForResult/release");
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

Value InferGraph::takeOutput(const TaskId& taskId, const std::string& nodeName,
							const std::string& portName) {
	// 优先查 OutputZone：绑定端口数据已由传播搬运至此
	auto ozVal = _state->output.take(taskId, nodeName, portName);
	if (ozVal) {
		// 共享/冻结载荷产出独立可变副本
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
	// 共享/冻结载荷产出独立可变副本
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
			// 共享/冻结载荷深拷贝
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
	// 共享/冻结载荷深拷贝
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

TaskStatus InferGraph::taskStatus(const TaskId& taskId) const {
	auto st = _engine->status(taskId);
	if (st == TaskStatus::Succeeded) {
		// 存在 Error 级诊断时归一化为 Failed
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
		// 超时/未知/收尾未完成时如实返回 Running 或 Unknown；wait 返回 true 才保证结果可读。
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
		return; // 引擎拒绝释放；结果/诊断保持不动
	_state->output.clearTask(taskId);
	_state->errors.clearTask(taskId);
}

void InferGraph::discardUnsubmitted(const TaskId& taskId) {
	// 仅清理未知状态任务；活动与已终止任务由 detachTask/releaseTask 路径管理。
	if (_engine->status(taskId) != TaskStatus::Unknown)
		return;
	_state->exec->clearTaskState(taskId);
	_state->output.clearTask(taskId);
	_state->errors.clearTask(taskId);
}

void InferGraph::detachTask(const TaskId& taskId) {
	// 转发引擎：不取消在飞任务；收尾自动回收，已终止立即释放，未知 no-op。
	_engine->detachTask(taskId);
}

} // namespace DC
