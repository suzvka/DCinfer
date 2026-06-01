#include "Node.h"
#include "Node/internal/SignalGate.h"
#include "Node/internal/TaskBuffer.h"
#include "Node/internal/SlotWorkspace.h"
#include "Node/internal/CoroutineBridge.h"
#include "Node/internal/EngineAdapter.h"
#include "Node/internal/ExecutionPipeline.h"
#include "EngineRegistry.h"
#include "SignalStore.h"

namespace DC {

// ── NodeSchema 方法实现 ──

const NodePort* NodeSchema::find(const std::vector<NodePort>& ports, const std::string& name) {
	for (const auto& port : ports) {
		if (port.name == name)
			return &port;
	}
	return nullptr;
}

bool NodeSchema::hasUniqueNames(const std::vector<NodePort>& ports) {
	std::unordered_set<std::string> names;
	names.reserve(ports.size());
	for (const auto& port : ports) {
		if (!names.insert(port.name).second)
			return false;
	}
	return true;
}

const NodePort* NodeSchema::findInput(const std::string& name) const {
	return find(inputs, name);
}

const NodePort* NodeSchema::findOutput(const std::string& name) const {
	return find(outputs, name);
}

bool NodeSchema::valid() const {
	if (!hasUniqueNames(inputs) || !hasUniqueNames(outputs))
		return false;
	auto checkTypeSize = [](const std::vector<NodePort>& ports) {
		for (const auto& p : ports) {
			if (p.type != Tensor::TensorType::Void && p.typeSize == 0)
				return false;
		}
		return true;
	};
	if (!checkTypeSize(inputs) || !checkTypeSize(outputs))
		return false;
	for (const auto& p : inputs) {
		if (p.defaultValue.has_value()) {
			const auto& dv = p.defaultValue.value();
			if (dv.type() != p.type || dv.typeSize() != p.typeSize)
				return false;
		}
	}
	for (const auto& p : inputs) {
		if (p.shapeAnchor.has_value()) {
			const auto& anchorName = p.shapeAnchor.value();
			if (anchorName == p.name)
				return false;
			if (!findInput(anchorName))
				return false;
		}
	}
	return true;
}

// ── 构造/析构 ──

Node::Node(std::string type, std::string name, Schema schema, RunFn fn, EngineInstance* engineInstance,
		   ThreadPoolAffinity affinity, const EngineDescriptor* engineDesc)
	: _fn(std::move(fn)) {
	_meta.type = std::move(type);
	_meta.name = std::move(name);
	_meta.affinity = affinity;
	_meta.engineDescriptor = engineDesc;
	_meta.schema = std::move(schema);

	_buffer = std::make_unique<TaskBuffer>();
	_workspace = std::make_unique<SlotWorkspace>(_meta.schema);
	_bridge = std::make_unique<CoroutineBridge>();
	_signal = std::make_unique<SignalGate>();
	_engine = std::make_unique<EngineAdapter>(engineInstance, engineDesc);
}

Node::~Node() = default;

// ── 回调注册 ──

void Node::setCompletionCallback(CompletionFn fn) {
	_onComplete = std::move(fn);
}

bool Node::hasCompletionCallback() const {
	return static_cast<bool>(_onComplete);
}

// ── 信号绑定 ──

void Node::bindSignal(std::shared_ptr<SignalStore> store, std::string name) {
	_signal->bind(std::move(store), std::move(name));
}

bool Node::isBlocked() const {
	return _signal->isBlocked();
}

bool Node::isBlocked(const TaskId& taskId) const {
	return _signal->isBlocked(taskId);
}

// ── 单端口输入 ──

void Node::setInput(const TaskId& taskId, const std::string& portName, Value data) {
	_buffer->setInput(taskId, portName, std::move(data), _meta.schema);
}

void Node::setInput(const TaskId& taskId, std::unordered_map<std::string, TaskData> inputs) {
	_buffer->setInputBatch(taskId, std::move(inputs), _meta.schema);
}

// ── 任务级输出 ──

bool Node::hasOutput(const TaskId& taskId, const std::string& name) const {
	return _buffer->hasOutput(taskId, name);
}

Value Node::getOutput(const TaskId& taskId, const std::string& name) {
	return _buffer->getOutput(taskId, name);
}

const Value& Node::peekOutput(const TaskId& taskId, const std::string& name) const {
	return _buffer->peekOutput(taskId, name);
}

std::unordered_map<std::string, Node::TaskData> Node::collectOutputs(const TaskId& taskId) {
	return _buffer->collectOutputs(taskId);
}

// ── 任务生命周期 ──

bool Node::hasTask(const TaskId& taskId) const {
	return _buffer->hasTask(taskId);
}

void Node::clearTask(const TaskId& taskId) {
	_buffer->clearTask(taskId);
	_bridge->clearCompleted(taskId);
}

void Node::terminateTask(const TaskId& taskId) {
	_buffer->clearTask(taskId);
	_bridge->terminateTask(taskId);
}

size_t Node::taskCount() const {
	return _buffer->taskCount();
}

// ── 调度接口 ──

bool Node::isReady(const TaskId& taskId) const {
	return _buffer->isReady(taskId, _meta.schema);
}

void Node::tryExecute(const TaskId& taskId) {
	if (!_buffer->isReady(taskId, _meta.schema)) {
		throw NodeException(NodeException::ErrorType::NotReady, "Node::tryExecute",
							"task '" + taskId + "' is not ready");
	}

	if (!_workspace->tryAcquire()) {
		throw NodeException(NodeException::ErrorType::Reentrant, "Node::tryExecute",
							"node '" + _meta.name + "' is busy executing another task");
	}

	_workspace->setCurrentTask(taskId);

	try {
		ExecutionPipeline::execute(taskId, *_buffer, *_workspace, *_engine,
								   _fn, _meta.schema, *_bridge, _onComplete,
								   _meta.type, _meta.name);
	} catch (...) {
		_workspace->clearCurrentTask();
		_workspace->release();
		throw;
	}

	_workspace->clearCurrentTask();
	_workspace->release();
}

std::optional<Node::TaskId> Node::currentTaskId() const {
	return _workspace->currentTask();
}

// ════════════════════════════════════════════
// Tensor 便捷接口
// ════════════════════════════════════════════

void Node::setInput(const TaskId& taskId, const std::string& portName, Tensor data) {
	setInput(taskId, portName, Value(std::make_unique<Tensor>(std::move(data))));
}

void Node::setInput(const TaskId& taskId, std::unordered_map<std::string, Tensor> inputs) {
	std::unordered_map<std::string, TaskData> wrapped;
	wrapped.reserve(inputs.size());
	for (auto& [name, t] : inputs) {
		wrapped.emplace(name, Value(std::make_unique<Tensor>(std::move(t))));
	}
	setInput(taskId, std::move(wrapped));
}

Tensor Node::getOutputTensor(const TaskId& taskId, const std::string& name) {
	auto nt = getOutput(taskId, name);
	auto* t = nt.as<Tensor>();
	if (!t) {
		throw NodeException(NodeException::ErrorType::TypeMismatch, "Node::getOutputTensor",
							"output '" + name + "' is not a DC::Tensor (innerType=" +
								std::to_string(static_cast<uint32_t>(nt.innerType())) + ")");
	}
	return std::move(*t);
}

std::unordered_map<std::string, Tensor> Node::collectOutputTensors(const TaskId& taskId) {
	auto outputs = collectOutputs(taskId);
	std::unordered_map<std::string, Tensor> result;
	result.reserve(outputs.size());
	for (auto& [name, nt] : outputs) {
		auto* t = nt.as<Tensor>();
		if (t) {
			result.emplace(name, std::move(*t));
		}
	}
	return result;
}

// ── 槽位暴露 ──

const std::unordered_map<std::string, TensorSlot>& Node::inputSlots() const {
	return _workspace->inputSlots();
}

const std::unordered_map<std::string, TensorSlot>& Node::outputSlots() const {
	return _workspace->outputSlots();
}

// ── 协程支持 ──

NodeCompletion Node::whenComplete(const TaskId& taskId) {
	return _bridge->whenComplete(taskId, *_buffer, _meta.schema);
}

// ── RunContext 方法实现 ──

const Value& Node::RunContext::peek(const std::string& name) const {
	return _workspace.peekInput(name);
}

Value Node::RunContext::pop(const std::string& name) {
	return _workspace.popInput(name);
}

void Node::RunContext::output(const std::string& name, Value tensor) {
	_workspace.writeOutput(name, std::move(tensor));
}

const Value* Node::RunContext::outputRaw(const std::string& name) const {
	return _workspace.peekOutputRaw(name);
}

Node::Result Node::RunContext::success(std::string message) const {
	Node::Result r;
	r.status = Node::Status::Ok;
	r.message = std::move(message);
	return r;
}

Node::Result Node::RunContext::failure(Node::Status status, std::string message) const {
	Node::Result r;
	r.status = status;
	r.message = std::move(message);
	return r;
}

const TensorConverter* Node::RunContext::converter() const {
	return _engine.converter();
}

const EngineDescriptor* Node::RunContext::engineDescriptor() const {
	return _engine.descriptor();
}

const EngineInstance* Node::RunContext::engineInstance() const {
	return _engine.instance();
}

void* Node::RunContext::engine() const {
	return _engine.engine();
}

const Node::Schema& Node::RunContext::schema() const {
	return _schema;
}

const std::string& Node::RunContext::type() const {
	return _type;
}

const std::string& Node::RunContext::name() const {
	return _name;
}

Node::RunContext::RunContext(SlotWorkspace& workspace, EngineAdapter& engine,
							 const Node::Schema& schema, const std::string& type, const std::string& name)
	: _workspace(workspace), _engine(engine), _schema(schema), _type(type), _name(name) {}

// ── NodeCompletion 实现 ──

bool NodeCompletion::await_ready() const {
	if (!buffer || !schema)
		return false;
	for (const auto& port : schema->outputs) {
		if (buffer->hasOutput(taskId, port.name))
			return true;
	}
	return false;
}

void NodeCompletion::await_suspend(std::coroutine_handle<> h) {
	handle = h;

	if (!bridge) {
		h.resume();
		return;
	}

	std::lock_guard lk(bridge->_mutex);

	if (bridge->_completedTasks.contains(taskId)) {
		h.resume();
		return;
	}

	bridge->_waiters[taskId].push_back(h);
}

Node::Result NodeCompletion::await_resume() const {
	if (!buffer || !schema) {
		Node::Result r;
		r.status = Node::Status::ExecutionFailed;
		r.message = "NodeCompletion: missing buffer or schema";
		return r;
	}

	for (const auto& port : schema->outputs) {
		if (buffer->hasOutput(taskId, port.name)) {
			Node::Result r;
			r.status = Node::Status::Ok;
			return r;
		}
	}

	Node::Result r;
	r.status = Node::Status::ExecutionFailed;
	r.message = "NodeCompletion: task '" + taskId + "' completed without outputs";
	return r;
}

} // namespace DC
