#include "Node.h"
#include "EngineRegistry.h"
#include "NodeException.h"

#include <iostream>

namespace DC {

// ── Schema 方法实现 ──

const Node::Port* Node::Schema::find(const std::vector<Port>& ports, const std::string& name) {
	for (const auto& port : ports) {
		if (port.name == name)
			return &port;
	}
	return nullptr;
}

bool Node::Schema::hasUniqueNames(const std::vector<Port>& ports) {
	std::unordered_set<std::string> names;
	names.reserve(ports.size());
	for (const auto& port : ports) {
		if (!names.insert(port.name).second)
			return false;
	}
	return true;
}

const Node::Port* Node::Schema::findInput(const std::string& name) const {
	return find(inputs, name);
}

const Node::Port* Node::Schema::findOutput(const std::string& name) const {
	return find(outputs, name);
}

bool Node::Schema::valid() const {
	if (!hasUniqueNames(inputs) || !hasUniqueNames(outputs))
		return false;
	auto checkTypeSize = [](const std::vector<Port>& ports) {
		for (const auto& p : ports) {
			if (p.type != TensorType::Void && p.typeSize == 0)
				return false;
		}
		return true;
	};
	if (!checkTypeSize(inputs) || !checkTypeSize(outputs))
		return false;
	// 校验默认值类型一致性
	for (const auto& p : inputs) {
		if (p.defaultValue.has_value()) {
			const auto& dv = p.defaultValue.value();
			if (dv.type() != p.type || dv.typeSize() != p.typeSize)
				return false;
		}
	}
	return true;
}

// ── 构造：从 Schema 构建工作槽位 ──
Node::Node(std::string type, std::string name, Schema schema, RunFn fn, EngineInstance* engineInstance,
		   ThreadPoolAffinity affinity)
	: _type(std::move(type)), _name(std::move(name)), _schema(std::move(schema)), _affinity(affinity),
	  _fn(std::move(fn)), _engineInstance(engineInstance) {
	for (const auto& port : _schema.inputs) {
		TensorSlot::Config cfg;
		cfg.setPosition(TensorSlot::Config::Position::Input);
		_inputSlots.emplace(port.name, TensorSlot(port.name, port.type, port.typeSize, port.shape, cfg));
	}

	for (const auto& port : _schema.outputs) {
		TensorSlot::Config cfg;
		cfg.setPosition(TensorSlot::Config::Position::Output);
		_outputSlots.emplace(port.name, TensorSlot(port.name, port.type, port.typeSize, port.shape, cfg));
	}
}

Node::~Node() = default;

// ── 回调注册 ──
void Node::setCompletionCallback(CompletionFn fn) {
	_onComplete = std::move(fn);
}

bool Node::hasCompletionCallback() const {
	return static_cast<bool>(_onComplete);
}

// ── 单端口输入 ──
void Node::setInput(const TaskId& taskId, const std::string& portName, Value data) {
	if (!_schema.findInput(portName)) {
		throw NodeException(NodeException::ErrorType::PortNotFound, "Node::setInput",
							"port '" + portName + "' not found in schema");
	}

	std::unique_lock lk(_bufferMutex);
	_ensureTaskExists(taskId);
	_taskInputs[taskId].at(portName) = std::move(data);
}

// ── 批量输入 ──
void Node::setInput(const TaskId& taskId, std::unordered_map<std::string, TaskData> inputs) {
	// 预校验：所有端口名必须存在
	for (const auto& [name, data] : inputs) {
		if (!_schema.findInput(name)) {
			throw NodeException(NodeException::ErrorType::PortNotFound, "Node::setInput",
								"port '" + name + "' not found in schema");
		}
	}

	std::unique_lock lk(_bufferMutex);
	_ensureTaskExists(taskId);

	for (auto& [name, data] : inputs) {
		_taskInputs[taskId].at(name) = std::move(data);
	}
}

// ── 任务级输出 ──
bool Node::hasOutput(const TaskId& taskId, const std::string& name) const {
	std::shared_lock lk(_bufferMutex);

	auto taskIt = _taskOutputs.find(taskId);
	if (taskIt == _taskOutputs.end())
		return false;
	auto slotIt = taskIt->second.find(name);
	if (slotIt == taskIt->second.end())
		return false;
	return slotIt->second.has_value();
}

Value Node::getOutput(const TaskId& taskId, const std::string& name) {
	std::unique_lock lk(_bufferMutex);
	auto taskIt = _taskOutputs.find(taskId);
	if (taskIt == _taskOutputs.end()) {
		throw NodeException(NodeException::ErrorType::TaskNotFound, "Node::getOutput",
							"task '" + taskId + "' not found");
	}
	auto& optVal = taskIt->second.at(name);
	if (!optVal.has_value()) {
		throw NodeException(NodeException::ErrorType::OutputNotProduced, "Node::getOutput",
							"output '" + name + "' is empty");
	}
	Value result = std::move(optVal.value());
	optVal.reset();
	return result;
}

const Value& Node::peekOutput(const TaskId& taskId, const std::string& name) const {
	std::shared_lock lk(_bufferMutex);
	auto taskIt = _taskOutputs.find(taskId);
	if (taskIt == _taskOutputs.end()) {
		throw NodeException(NodeException::ErrorType::TaskNotFound, "Node::peekOutput",
							"task '" + taskId + "' not found");
	}
	auto& optVal = taskIt->second.at(name);
	if (!optVal.has_value()) {
		throw NodeException(NodeException::ErrorType::OutputNotProduced, "Node::peekOutput",
							"output '" + name + "' is empty");
	}
	return optVal.value();
}

Node::TaskPortMap::mapped_type Node::collectOutputs(const TaskId& taskId) {
	std::unique_lock lk(_bufferMutex);
	std::unordered_map<std::string, TaskData> result;
	auto taskIt = _taskOutputs.find(taskId);
	if (taskIt == _taskOutputs.end())
		return result;

	for (auto& [name, optVal] : taskIt->second) {
		if (optVal.has_value()) {
			result.emplace(name, std::move(optVal.value()));
			optVal.reset(); // 消费
		}
	}
	return result;
}

// ── 任务生命周期 ──
bool Node::hasTask(const TaskId& taskId) const {
	std::shared_lock lk(_bufferMutex);
	return _taskInputs.contains(taskId) || _taskOutputs.contains(taskId);
}

void Node::clearTask(const TaskId& taskId) {
	std::unique_lock lk(_bufferMutex);
	_taskInputs.erase(taskId);
	_taskOutputs.erase(taskId);
	lk.unlock();
	{
		std::lock_guard lk(_waitersMutex);
		_completedTasks.erase(taskId);
	}
}

void Node::terminateTask(const TaskId& taskId) {
	// 1. 清除 IO 缓冲区
	{
		std::unique_lock lk(_bufferMutex);
		_taskInputs.erase(taskId);
		_taskOutputs.erase(taskId);
	}

	// 2. 通知所有等待此 task 的协程（在锁外 resume）
	std::vector<std::coroutine_handle<>> handles;
	{
		std::lock_guard lk(_waitersMutex);
		_completedTasks.erase(taskId);
		auto it = _waiters.find(taskId);
		if (it != _waiters.end()) {
			handles = std::move(it->second);
			_waiters.erase(it);
		}
	}
	for (auto h : handles) {
		if (h)
			h.resume();
	}
}

size_t Node::taskCount() const {
	std::shared_lock lk(_bufferMutex);
	return _taskInputs.size();
}

// ── 调度接口 ──
bool Node::isReady(const TaskId& taskId) const {
	std::shared_lock lk(_bufferMutex);
	return _isTaskReady(taskId);
}

void Node::tryExecute(const TaskId& taskId) {
	if (!_isTaskReady(taskId)) {
		throw NodeException(NodeException::ErrorType::NotReady, "Node::tryExecute",
							"task '" + taskId + "' is not ready");
	}

	// 保护共享工作槽位 _inputSlots / _outputSlots，同一时刻仅允许一个 task 使用
	if (_executionGuard.test_and_set(std::memory_order_acquire)) {
		throw NodeException(NodeException::ErrorType::Reentrant, "Node::tryExecute",
							"node '" + _name + "' is busy executing another task");
	}

	_currentTaskId = taskId;

	try {
		_checkAndExecute(taskId);
	} catch (...) {
		_currentTaskId.reset();
		_executionGuard.clear(std::memory_order_release);
		throw;
	}

	_currentTaskId.reset();
	_executionGuard.clear(std::memory_order_release);
}

std::optional<Node::TaskId> Node::currentTaskId() const {
	return _currentTaskId;
}

// ════════════════════════════════════════════
// Tensor 便捷接口（自动完成 Tensor ↔ Value 包装）
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

// ── RunFn 内部使用的计算 API ──
const Value& Node::_inputImpl(const std::string& name) const {
	auto it = _inputSlots.find(name);
	if (it == _inputSlots.end()) {
		throw NodeException(NodeException::ErrorType::PortNotFound, "Node::_inputImpl",
							"input '" + name + "' not found");
	}
	const auto* nt = it->second.peek<Value>();
	if (!nt) {
		throw NodeException(NodeException::ErrorType::TypeMismatch, "Node::_inputImpl",
							"input '" + name + "' is not a Value");
	}
	return *nt;
}

void Node::_outputImpl(const std::string& name, Value tensor) {
	auto it = _outputSlots.find(name);
	if (it == _outputSlots.end()) {
		throw NodeException(NodeException::ErrorType::PortNotFound, "Node::_outputImpl",
							"output '" + name + "' not found");
	}
	it->second.store(std::move(tensor));
}

// ── 转换钩子访问器 ──
const TensorConverter* Node::_converter() const {
	auto* desc = _engineDescriptor();
	if (!desc)
		return nullptr;
	return &desc->converter;
}

const EngineDescriptor* Node::_engineDescriptor() const {
	if (_engineInstance)
		return _engineInstance->descriptor();
	return EngineRegistry::instance().find(_type);
}

void Node::_synchronizeEngine() const {
	if (!_engineInstance)
		return;
	auto* desc = _engineInstance->descriptor();
	if (desc && desc->synchronize) {
		desc->synchronize(_engineInstance->get());
	}
}

// ── RunContext out-of-line 实现（需要 EngineInstance 完整类型）──
const EngineInstance* Node::RunContext::engineInstance() const {
	return _node._engineInstance;
}

void* Node::RunContext::engine() const {
	return _node._engineInstance ? _node._engineInstance->get() : nullptr;
}

const Value* Node::RunContext::outputRaw(const std::string& name) const {
	auto it = _node._outputSlots.find(name);
	if (it == _node._outputSlots.end())
		return nullptr;
	return it->second.peek<Value>();
}

// ── 结果辅助 ──
Node::Result Node::_makeSuccess(std::string message) const {
	Result r;
	r.status = Status::Ok;
	r.message = std::move(message);
	return r;
}

Node::Result Node::_makeFailure(Status status, std::string message) const {
	Result r;
	r.status = status;
	r.message = std::move(message);
	return r;
}

// ════════════════════════════════════════════
// 内部方法（调用时已持锁）
// ════════════════════════════════════════════

void Node::_ensureTaskExists(const TaskId& taskId) {
	if (_taskInputs.contains(taskId))
		return;

	TaskBuffer inputs;
	for (const auto& port : _schema.inputs) {
		// 所有端口初始为空（nullopt），数据由 setInput 填充
		inputs.emplace(port.name, std::nullopt);
	}
	_taskInputs.emplace(taskId, std::move(inputs));
}

bool Node::_isTaskReady(const TaskId& taskId) const {
	auto it = _taskInputs.find(taskId);
	if (it == _taskInputs.end())
		return false;

	for (const auto& port : _schema.inputs) {
		if (!port.required)
			continue;
		if (port.defaultValue.has_value())
			continue; // 默认值视为已就绪

		auto slotIt = it->second.find(port.name);
		if (slotIt == it->second.end())
			return false;
		if (!slotIt->second.has_value())
			return false;
	}
	return true;
}

void Node::_checkAndExecute(const TaskId& taskId) {
	{
		std::shared_lock lk(_bufferMutex);
		if (!_isTaskReady(taskId))
			return;
	}

	Result result;

	try {
		// ① 加载输入：_taskInputs[taskId] → _inputSlots (move)
		{
			std::unique_lock lk(_bufferMutex);
			_loadTaskToWorkingSlots(taskId);
		}

		// ② 清空上一轮工作输出
		_clearWorkingOutputs();

		// ②½ preRun 钩子：推理前准备（I/O 绑定、warmup 等）
		if (_engineInstance) {
			auto* desc = _engineInstance->descriptor();
			if (desc && desc->preRun) {
				desc->preRun(_engineInstance->get());
			}
		}

		// ③ 执行 RunFn（通过 RunContext 隔离接口）
		try {
			RunContext ctx(*this);
			result = _fn(ctx);
		} catch (const std::exception& e) {
			result = _makeFailure(Status::ExecutionFailed, e.what());
		} catch (...) {
			result = _makeFailure(Status::ExecutionFailed, "Unknown exception in RunFn");
		}

		// ③¼ onError 钩子：执行失败时重置引擎状态
		if (!result.ok() && _engineInstance) {
			auto* desc = _engineInstance->descriptor();
			if (desc && desc->onError) {
				desc->onError(_engineInstance->get());
			}
		}

		// ③½ 同步：确保异步引擎计算已完成
		if (result.ok()) {
			_synchronizeEngine();
		}

		// ③¾ postRun 钩子：同步后的后处理（D2H 传输、输出格式转换等）
		if (result.ok() && _engineInstance) {
			auto* desc = _engineInstance->descriptor();
			if (desc && desc->postRun) {
				RunContext ctx(*this);
				desc->postRun(_engineInstance->get(), ctx);
			}
		}

		// ④ 保存输出：_outputSlots → _taskOutputs[taskId]
		{
			std::unique_lock lk(_bufferMutex);
			_collectAndSaveOutputs(taskId);

			// ⑤ 验证输出完整性（持锁读取 _taskOutputs）
			if (result.ok()) {
				for (const auto& port : _schema.outputs) {
					auto outIt = _taskOutputs[taskId].find(port.name);
					if (outIt == _taskOutputs[taskId].end() || !outIt->second.has_value()) {
						result = _makeFailure(Status::InternalError, "Output '" + port.name + "' was not produced by RunFn");
						break;
					}
				}
			}
		}

		// Diagnostic logging for failures to aid debugging
		if (!result.ok()) {
			try {
				std::cerr << "Node[" << _name << "] task '" << taskId
						  << "' failed: status=" << static_cast<int>(result.status) << ", message='" << result.message
						  << "'" << std::endl;
			} catch (...) {
				// swallow any logging errors
			}
		}

		// ⑥ 清理输入缓冲（输出缓冲保留，供调用方拉取）
		{
			std::unique_lock lk(_bufferMutex);
			_taskInputs.erase(taskId);
		}

		// ⑥½ 通知等待协程（在回调之前，保证协程能看到结果）
		_notifyWaiters(taskId, result);

		// ⑦ 调用回调（仅通知）
		if (_onComplete) {
			_onComplete(taskId, result);
		}
	} catch (const std::exception& e) {
		// 加载阶段或执行阶段抛出未捕获异常（如终止后 inputs 被清空）
		// 必须通知等待者，避免协程句柄泄漏
		result = _makeFailure(Status::ExecutionFailed, e.what());
		_notifyWaiters(taskId, result);
		if (_onComplete) {
			_onComplete(taskId, result);
		}
		throw; // 重新抛出给 tryExecute（由其清理 _executionGuard）
	}
}

void Node::_loadTaskToWorkingSlots(const TaskId& taskId) {
	auto& taskInputs = _taskInputs[taskId];

	for (const auto& port : _schema.inputs) {
		auto taskIt = taskInputs.find(port.name);
		auto& workSlot = _inputSlots.at(port.name);

		if (taskIt != taskInputs.end() && taskIt->second.has_value()) {
			auto& nativeData = taskIt->second.value();

			// 对 DCTensor 类型的 Value 进行类型校验
			if (nativeData.innerType() == SlotDataType::DCTensor && port.type != TensorType::Void) {
				const auto* t = static_cast<const Tensor*>(nativeData.get());
				if (!t || !t->valid()) {
					throw NodeException(NodeException::ErrorType::InternalError, "Node::_loadTaskToWorkingSlots",
										"invalid Tensor in Value for port '" + port.name + "'");
				}
				if (t->type() != port.type) {
					throw NodeException(NodeException::ErrorType::TypeMismatch, "Node::_loadTaskToWorkingSlots",
										"type mismatch for port '" + port.name + "'");
				}
			}

			workSlot.store(std::move(taskIt->second.value()));
			taskIt->second.reset();
		} else if (port.defaultValue.has_value()) {
			workSlot.store(Value(std::make_unique<Tensor>(port.defaultValue.value())));
		}
	}
}

void Node::_clearWorkingOutputs() {
	for (auto& [name, slot] : _outputSlots) {
		slot.clear();
	}
}

void Node::_collectAndSaveOutputs(const TaskId& taskId) {
	if (!_taskOutputs.contains(taskId)) {
		TaskBuffer outputs;
		for (const auto& port : _schema.outputs) {
			outputs.emplace(port.name, std::nullopt);
		}
		_taskOutputs.emplace(taskId, std::move(outputs));
	}

	auto& taskOutputs = _taskOutputs[taskId];
	for (auto& [name, workSlot] : _outputSlots) {
		if (!workSlot.hasData())
			continue;
		auto taskIt = taskOutputs.find(name);
		if (taskIt == taskOutputs.end())
			continue;
		taskIt->second = workSlot.take<Value>();
	}
}

// ════════════════════════════════════════════
// 协程支持
// ════════════════════════════════════════════

NodeCompletion Node::whenComplete(const TaskId& taskId) {
	return NodeCompletion(*this, taskId);
}

void Node::_notifyWaiters(const TaskId& taskId, const Result& result) {
	std::vector<std::coroutine_handle<>> handles;
	{
		std::lock_guard lk(_waitersMutex);
		// 先标记完成：即使 _waiters 中尚无等待者，
		// await_suspend 也能通过此标记发现任务已结束，直接 resume
		_completedTasks.insert(taskId);
		auto it = _waiters.find(taskId);
		if (it == _waiters.end())
			return;
		handles = std::move(it->second);
		_waiters.erase(it);
	}
	// 在锁外 resume，避免协程恢复后可能的死锁
	for (auto h : handles) {
		if (h)
			h.resume();
	}
}

// ── NodeCompletion 实现 ──

bool NodeCompletion::await_ready() const {
	// 如果任务已完成（输出已存在），不需要挂起
	std::shared_lock lk(_node->_bufferMutex);
	auto taskIt = _node->_taskOutputs.find(_taskId);
	if (taskIt == _node->_taskOutputs.end())
		return false;
	// 检查是否有任意端口已有输出数据
	for (const auto& [name, optVal] : taskIt->second) {
		if (optVal.has_value())
			return true;
	}
	return false;
}

void NodeCompletion::await_suspend(std::coroutine_handle<> h) {
	_handle = h;

	std::lock_guard lk(_node->_waitersMutex);

	// 关键：await_ready() 到 await_suspend() 之间存在竞态窗口——
	// 目标 task 可能已在 ThreadPool 中完成且 _notifyWaiters 已执行完毕。
	// _completedTasks 在 _waitersMutex 保护下充当原子"完成标记"，
	// 避免协程句柄被孤儿化（永远无人 resume）。
	if (_node->_completedTasks.contains(_taskId)) {
		h.resume();
		return;
	}

	_node->_waiters[_taskId].push_back(h);
}

Node::Result NodeCompletion::await_resume() const {
	// 协程恢复后，从 task 输出中收集结果状态
	std::shared_lock lk(_node->_bufferMutex);
	auto taskIt = _node->_taskOutputs.find(_taskId);
	if (taskIt != _node->_taskOutputs.end()) {
		// 检查所有输出是否已产生（有任意输出即认为完成）
		for (const auto& port : _node->_schema.outputs) {
			auto outIt = taskIt->second.find(port.name);
			if (outIt != taskIt->second.end() && outIt->second.has_value()) {
				return _node->_makeSuccess();
			}
		}
	}
	return _node->_makeFailure(Node::Status::ExecutionFailed,
							   "NodeCompletion: task '" + _taskId + "' completed without outputs");
}

} // namespace DC
