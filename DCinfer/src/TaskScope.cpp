#include "TaskScope.h"

namespace DC {

// ════════════════════════════════════════════
// 生命周期：构造 / 析构 / 移动
// ════════════════════════════════════════════

TaskScope::TaskScope(InferGraph& graph, TaskId taskId)
	: _graph(&graph), _id(std::move(taskId)) {}

TaskScope::~TaskScope() {
	_cleanup();
}

TaskScope::TaskScope(TaskScope&& other) noexcept
	: _graph(other._graph), _id(std::move(other._id)) {
	other._graph = nullptr;
}

TaskScope& TaskScope::operator=(TaskScope&& other) noexcept {
	if (this != &other) {
		_cleanup(); // 先清理自身持有的 taskId，再接管对方
		_graph = other._graph;
		_id = std::move(other._id);
		other._graph = nullptr;
	}
	return *this;
}

void TaskScope::_cleanup() noexcept {
	if (!_graph)
		return;
	// 未终止（Running）→ 先取消：cancel 是同步终止路径（发布终态、清理 task 态、
	// 唤醒等待者），不阻塞；在飞节点执行不被中断，其迟到结果经 gate 检查丢弃。
	if (_graph->taskStatus(_id) == TaskStatus::Running)
		_graph->cancel(_id);
	_graph->releaseTask(_id); // 仅终止态生效；未知/已释放幂等 no-op
	_graph = nullptr;         // 防重复清理（幂等）
}

// ════════════════════════════════════════════
// 组装与异步路径（转发）
// ════════════════════════════════════════════

TaskScope& TaskScope::feed(const std::string& nodeName, const std::string& portName,
						   Value data) {
	_graph->feedInput(_id, nodeName, portName, std::move(data));
	return *this;
}

TaskScope& TaskScope::feed(const std::string& nodeName, const std::string& portName,
						   Tensor data) {
	_graph->feedInput(_id, nodeName, portName, std::move(data));
	return *this;
}

void TaskScope::submit(uint32_t maxHops) {
	_graph->submitBound(_id, maxHops);
}

bool TaskScope::cancel() {
	return _graph->cancel(_id);
}

TaskStatus TaskScope::status() const {
	return _graph->taskStatus(_id);
}

TaskResult TaskScope::wait() {
	return _graph->waitForResult(_id);
}

TaskResult TaskScope::wait(std::chrono::milliseconds timeout) {
	return _graph->waitForResult(_id, timeout);
}

Value TaskScope::take(const std::string& nodeName, const std::string& portName) {
	return _graph->takeOutput(_id, nodeName, portName);
}

Tensor TaskScope::takeTensor(const std::string& nodeName, const std::string& portName) {
	return _graph->takeOutputTensor(_id, nodeName, portName);
}

// ════════════════════════════════════════════
// 同步便捷路径
// ════════════════════════════════════════════

TaskScope::Result TaskScope::run() {
	return run(std::chrono::milliseconds(0)); // 0 = 无限等待（引擎约定）
}

TaskScope::Result TaskScope::run(std::chrono::milliseconds timeout) {
	Result result;
	_graph->submitBound(_id);

	TaskResult terminal = _graph->waitForResult(_id, timeout);
	result.status = terminal.status;
	result.errors = std::move(terminal.errors);

	// 宿主等待超时未终止：不取不释放，作用域仍持有任务
	if (terminal.status == TaskStatus::Running)
		return result;

	// 收集全部绑定输出（未被产出/被取消的任务允许部分为空）
	for (const auto& binding : _graph->outputBindings()) {
		if (_graph->hasOutput(_id, binding.nodeName, binding.portName))
			result.outputs.push_back({binding, _graph->takeOutput(_id, binding.nodeName, binding.portName)});
	}

	// 终态已达成：释放任务资源（作用域析构随之成为空操作）
	_graph->releaseTask(_id);
	return result;
}

// ════════════════════════════════════════════
// Result 取数（消费式）
// ════════════════════════════════════════════

Value TaskScope::Result::take(const std::string& nodeName, const std::string& portName) {
	for (auto it = outputs.begin(); it != outputs.end(); ++it) {
		if (it->binding.nodeName == nodeName && it->binding.portName == portName) {
			Value value = std::move(it->value);
			outputs.erase(it);
			return value;
		}
	}
	throw GraphException(GraphException::ErrorType::Other, "TaskScope::Result::take",
						 "no output for '" + nodeName + "." + portName + "'");
}

Tensor TaskScope::Result::takeTensor(const std::string& nodeName, const std::string& portName) {
	Value value = take(nodeName, portName);
	auto* t = value.as<Tensor>();
	if (!t)
		throw GraphException(GraphException::ErrorType::Other, "TaskScope::Result::takeTensor",
							 "output '" + nodeName + "." + portName + "' is not a DC::Tensor");
	return std::move(*t);
}

} // namespace DC
