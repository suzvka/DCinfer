#include "GraphInterface.h"
#include "InferGraph.h"
#include "GraphException.h"

#include <atomic>
#include <cstdint>
#include <utility>

namespace DC {

namespace {

/// @brief 别名列表描述（供错误消息列出全部可用别名）
template <typename Bindings>
std::string describeAliases(const Bindings& bindings) {
	if (bindings.empty())
		return "(none)";
	std::string out;
	for (const auto& b : bindings) {
		if (!out.empty())
			out += ", ";
		out += "'" + b.alias + "'";
	}
	return out;
}

} // namespace

// ════════════════════════════════════════════
// 图公开接口：工厂与构造
// ════════════════════════════════════════════

GraphInterface InferGraph::interface() {
	freeze(); // 取接口即定型：别名 → 坐标一次性解析以冻结签名为准
	return GraphInterface(*this, inputBindings(), outputBindings());
}

GraphInterface::GraphInterface(InferGraph& graph, std::vector<InputBinding> inputs,
							   std::vector<OutputBinding> outputs)
	: _graph(&graph), _inputs(std::move(inputs)), _outputs(std::move(outputs)) {
	// 坐标校验（fail-fast）：绑定必须指向存在的节点与方向正确的端口
	const InferGraph& g = graph; // 冻结后只读访问（非 const node() 是构建期 API）
	for (const auto& b : _inputs) {
		const Node* n = g.node(b.nodeName);
		if (!n)
			throw GraphException(GraphException::ErrorType::NodeNotFound, "GraphInterface",
								 "input binding '" + b.alias + "' references node '" +
									 b.nodeName + "' not found");
		if (!n->schema().findInput(b.portName))
			throw GraphException(GraphException::ErrorType::PortNotFound, "GraphInterface",
								 "input binding '" + b.alias + "' references input port '" +
									 b.nodeName + "." + b.portName + "' not found in schema");
	}
	for (const auto& b : _outputs) {
		const Node* n = g.node(b.nodeName);
		if (!n)
			throw GraphException(GraphException::ErrorType::NodeNotFound, "GraphInterface",
								 "output binding '" + b.alias + "' references node '" +
									 b.nodeName + "' not found");
		if (!n->schema().findOutput(b.portName))
			throw GraphException(GraphException::ErrorType::PortNotFound, "GraphInterface",
								 "output binding '" + b.alias + "' references output port '" +
									 b.nodeName + "." + b.portName + "' not found in schema");
	}
}

std::vector<std::string> GraphInterface::inputAliases() const {
	std::vector<std::string> aliases;
	aliases.reserve(_inputs.size());
	for (const auto& b : _inputs)
		aliases.push_back(b.alias);
	return aliases;
}

std::vector<std::string> GraphInterface::outputAliases() const {
	std::vector<std::string> aliases;
	aliases.reserve(_outputs.size());
	for (const auto& b : _outputs)
		aliases.push_back(b.alias);
	return aliases;
}

const InputBinding& GraphInterface::_resolveInput(const std::string& alias, const char* api) const {
	for (const auto& b : _inputs)
		if (b.alias == alias)
			return b;
	throw GraphException(GraphException::ErrorType::InvalidBinding, api,
						 "unknown input alias '" + alias +
							 "'; available inputs: " + describeAliases(_inputs));
}

const OutputBinding& GraphInterface::_resolveOutput(const std::string& alias, const char* api) const {
	for (const auto& b : _outputs)
		if (b.alias == alias)
			return b;
	throw GraphException(GraphException::ErrorType::InvalidBinding, api,
						 "unknown output alias '" + alias +
							 "'; available outputs: " + describeAliases(_outputs));
}

GraphInterface::Task GraphInterface::createTask() & {
	static std::atomic<uint64_t> nextTaskNumber{0};
	const auto n = nextTaskNumber.fetch_add(1, std::memory_order_relaxed) + 1;
	return Task(*this, "iface-task-" + std::to_string(n));
}

// ════════════════════════════════════════════
// 任务句柄：生命周期与别名转发
// ════════════════════════════════════════════

GraphInterface::Task::Task(GraphInterface& iface, std::string taskId)
	: _iface(&iface), _taskId(std::move(taskId)) {}

GraphInterface::Task::Task(Task&& other) noexcept
	: _iface(other._iface), _taskId(std::move(other._taskId)) {
	other._iface = nullptr;
}

GraphInterface::Task& GraphInterface::Task::operator=(Task&& other) noexcept {
	if (this != &other) {
		_releaseIfTerminated();
		_iface = other._iface;
		_taskId = std::move(other._taskId);
		other._iface = nullptr;
	}
	return *this;
}

GraphInterface::Task::~Task() { _releaseIfTerminated(); }

void GraphInterface::Task::_releaseIfTerminated() noexcept {
	if (!_iface)
		return;
	const TaskStatus st = _iface->_graph->taskStatus(_taskId);
	if (st == TaskStatus::Succeeded || st == TaskStatus::Failed || st == TaskStatus::Cancelled)
		_iface->_graph->releaseTask(_taskId);
}

GraphInterface::Task& GraphInterface::Task::feed(const std::string& alias, Value data) {
	const auto& b = _iface->_resolveInput(alias, "GraphInterface::Task::feed");
	_iface->_graph->feedInput(_taskId, b.nodeName, b.portName, std::move(data));
	return *this;
}

GraphInterface::Task& GraphInterface::Task::feed(const std::string& alias, Tensor data) {
	const auto& b = _iface->_resolveInput(alias, "GraphInterface::Task::feed");
	_iface->_graph->feedInput(_taskId, b.nodeName, b.portName, std::move(data));
	return *this;
}

// ── 执行：同步与异步同级（全部转发 InferGraph 运行期 API）──

void GraphInterface::Task::submit() {
	_iface->_graph->submitBound(_taskId, InferGraph::kDefaultMaxHops);
}

TaskResult GraphInterface::Task::run() {
	return run(std::chrono::milliseconds(0)); // 0 = 无限等待（引擎约定）
}

TaskResult GraphInterface::Task::run(std::chrono::milliseconds timeout) {
	submit();
	return wait(timeout);
}

TaskResult GraphInterface::Task::wait() {
	return _iface->_graph->waitForResult(_taskId);
}

TaskResult GraphInterface::Task::wait(std::chrono::milliseconds timeout) {
	return _iface->_graph->waitForResult(_taskId, timeout);
}

TaskStatus GraphInterface::Task::status() const {
	return _iface->_graph->taskStatus(_taskId);
}

bool GraphInterface::Task::cancel() {
	return _iface->_graph->cancel(_taskId);
}

bool GraphInterface::Task::has(const std::string& alias) const {
	const auto& b = _iface->_resolveOutput(alias, "GraphInterface::Task::has");
	return _iface->_graph->hasOutput(_taskId, b.nodeName, b.portName);
}

std::vector<TaskError> GraphInterface::Task::errors() const {
	return _iface->_graph->taskErrors(_taskId);
}

Value GraphInterface::Task::take(const std::string& alias) {
	const auto& b = _iface->_resolveOutput(alias, "GraphInterface::Task::take");
	return _iface->_graph->takeOutput(_taskId, b.nodeName, b.portName);
}

Tensor GraphInterface::Task::takeTensor(const std::string& alias) {
	const auto& b = _iface->_resolveOutput(alias, "GraphInterface::Task::takeTensor");
	return _iface->_graph->takeOutputTensor(_taskId, b.nodeName, b.portName);
}

} // namespace DC
