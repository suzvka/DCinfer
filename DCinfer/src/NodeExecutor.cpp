#include "NodeExecutor.h"
#include "Node/internal/NodeExecState.h"
#include "Node/internal/ExecutionPipeline.h"

namespace DC {

struct NodeExecutor::Impl {
	NodeExecState exec;
	NodeExecutionGate gate;

	explicit Impl(const Node& node) : exec(node.schema()) {}
};

NodeExecutor::NodeExecutor(const Node& node)
	: _impl(std::make_unique<Impl>(node)), _node(&node) {}

NodeExecutor::~NodeExecutor() = default;

// ── task 级输入 ──

void NodeExecutor::setInput(const TaskId& taskId, const std::string& portName, Value data) {
	_impl->exec.buffer.setInput(taskId, portName, std::move(data), _node->schema());
}

void NodeExecutor::setInput(const TaskId& taskId, std::unordered_map<std::string, Value> inputs) {
	_impl->exec.buffer.setInputBatch(taskId, std::move(inputs), _node->schema());
}

// ── task 级输出 ──

bool NodeExecutor::isReady(const TaskId& taskId) const {
	return _node->isReady(taskId, _impl->exec.buffer);
}

bool NodeExecutor::hasOutput(const TaskId& taskId, const std::string& name) const {
	return _impl->exec.buffer.hasOutput(taskId, name);
}

Value NodeExecutor::takeOutput(const TaskId& taskId, const std::string& name) {
	return _impl->exec.buffer.takeOutput(taskId, name);
}

const Value& NodeExecutor::peekOutput(const TaskId& taskId, const std::string& name) const {
	return _impl->exec.buffer.peekOutput(taskId, name);
}

std::unordered_map<std::string, Value> NodeExecutor::collectOutputs(const TaskId& taskId) {
	return _impl->exec.buffer.collectOutputs(taskId);
}

// ── task 生命周期 ──

bool NodeExecutor::hasTask(const TaskId& taskId) const {
	return _impl->exec.buffer.hasTask(taskId);
}

void NodeExecutor::clearTask(const TaskId& taskId) {
	_impl->exec.buffer.clearTask(taskId);
}

size_t NodeExecutor::taskCount() const {
	return _impl->exec.buffer.taskCount();
}

// ── 执行 ──

NodeResult NodeExecutor::tryExecute(const TaskId& taskId) {
	return ExecutionPipeline::execute(taskId, *_node, _impl->exec, _impl->gate);
}

// ════════════════════════════════════════════
// Tensor 便捷接口
// ════════════════════════════════════════════

void NodeExecutor::setInput(const TaskId& taskId, const std::string& portName, Tensor data) {
	setInput(taskId, portName, Value(std::make_unique<Tensor>(std::move(data))));
}

void NodeExecutor::setInput(const TaskId& taskId, std::unordered_map<std::string, Tensor> inputs) {
	std::unordered_map<std::string, Value> wrapped;
	wrapped.reserve(inputs.size());
	for (auto& [name, t] : inputs) {
		wrapped.emplace(name, Value(std::make_unique<Tensor>(std::move(t))));
	}
	setInput(taskId, std::move(wrapped));
}

Tensor NodeExecutor::takeOutputTensor(const TaskId& taskId, const std::string& name) {
	auto nt = takeOutput(taskId, name);
	auto* t = nt.as<Tensor>();
	if (!t) {
		throw NodeException(NodeException::ErrorType::TypeMismatch, "NodeExecutor::takeOutputTensor",
							"output '" + name + "' is not a DC::Tensor (innerType=" +
								std::to_string(static_cast<uint32_t>(nt.innerType())) + ")");
	}
	return std::move(*t);
}

std::unordered_map<std::string, Tensor> NodeExecutor::collectOutputTensors(const TaskId& taskId) {
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

} // namespace DC
