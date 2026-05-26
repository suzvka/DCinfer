#include "Node.h"
#include "EngineRegistry.h"
#include "TensorException.h"

#include <stdexcept>
#include <iostream>

namespace DC {

// ── 构造：从 Schema 构建工作槽位 ──
Node::Node(std::string type, std::string name, Schema schema, RunFn fn,
                     EngineInstance* engineInstance)
	: _type(std::move(type))
	, _name(std::move(name))
	, _schema(std::move(schema))
	, _fn(std::move(fn))
	, _engineInstance(engineInstance)
{
	for (const auto& port : _schema.inputs) {
		TensorSlot::Config cfg;
		cfg.setPosition(TensorSlot::Config::Position::Input);
		_inputSlots.emplace(port.name, TensorSlot(
			port.name, port.type, port.typeSize, port.shape, cfg));
	}

	for (const auto& port : _schema.outputs) {
		TensorSlot::Config cfg;
		cfg.setPosition(TensorSlot::Config::Position::Output);
		_outputSlots.emplace(port.name, TensorSlot(
			port.name, port.type, port.typeSize, port.shape, cfg));
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
bool Node::setInput(const TaskId& taskId, const std::string& portName, Value data) {
	if (!_schema.findInput(portName)) {
		return false;
	}

	_ensureTaskExists(taskId);
	_taskInputs[taskId].at(portName) = std::move(data);

	try {
		_checkAndExecute(taskId);
	} catch (const TensorException&) {
		return false;
	}
	return true;
}

// ── 批量输入 ──
bool Node::setInput(const TaskId& taskId,
                          std::unordered_map<std::string, TaskData> inputs) {
	// 预校验：所有端口名必须存在
	for (const auto& [name, data] : inputs) {
		if (!_schema.findInput(name)) {
			return false;
		}
	}

	_ensureTaskExists(taskId);

	for (auto& [name, data] : inputs) {
		_taskInputs[taskId].at(name) = std::move(data);
	}

	try {
		_checkAndExecute(taskId);
	} catch (const TensorException&) {
		return false;
	}
	return true;
}

// ── 任务级输出 ──
bool Node::hasOutput(const TaskId& taskId, const std::string& name) const {

	auto taskIt = _taskOutputs.find(taskId);
	if (taskIt == _taskOutputs.end()) return false;
	auto slotIt = taskIt->second.find(name);
	if (slotIt == taskIt->second.end()) return false;
	return slotIt->second.has_value();
}

Value Node::getOutput(const TaskId& taskId, const std::string& name) {
	auto taskIt = _taskOutputs.find(taskId);
	if (taskIt == _taskOutputs.end()) {
		throw std::out_of_range("Node::getOutput: task '" + taskId + "' not found");
	}
	auto& optVal = taskIt->second.at(name);
	if (!optVal.has_value()) {
		throw std::out_of_range("Node::getOutput: output '" + name + "' is empty");
	}
	Value result = std::move(optVal.value());
	optVal.reset();
	return result;
}

const Value& Node::peekOutput(const TaskId& taskId, const std::string& name) const {
	auto taskIt = _taskOutputs.find(taskId);
	if (taskIt == _taskOutputs.end()) {
		throw std::out_of_range("Node::peekOutput: task '" + taskId + "' not found");
	}
	auto& optVal = taskIt->second.at(name);
	if (!optVal.has_value()) {
		throw std::out_of_range("Node::peekOutput: output '" + name + "' is empty");
	}
	return optVal.value();
}

Node::TaskPortMap::mapped_type
Node::collectOutputs(const TaskId& taskId) {
	std::unordered_map<std::string, TaskData> result;
	auto taskIt = _taskOutputs.find(taskId);
	if (taskIt == _taskOutputs.end()) return result;

	for (auto& [name, optVal] : taskIt->second) {
		if (optVal.has_value()) {
			result.emplace(name, std::move(optVal.value()));
			optVal.reset();  // 消费
		}
	}
	return result;
}

// ── 阻塞式一次执行 ──
Value Node::execute(const std::string& outputName,
                                std::unordered_map<std::string, Value> inputs) {
	static size_t execCounter = 0;
	auto taskId = "__exec_" + std::to_string(++execCounter);

	// 暂存并清空外部回调，避免 execute 内部被外部回调干扰
	auto savedCallback = std::move(_onComplete);
	_onComplete = nullptr;

	try {
		if (!setInput(taskId, std::move(inputs))) {
			_onComplete = std::move(savedCallback);
			throw std::runtime_error("Node::execute: setInputs failed");
		}
	} catch (...) {
		_onComplete = std::move(savedCallback);
		throw;
	}

	_onComplete = std::move(savedCallback);

	// 同步执行已在 setInputs → _checkAndExecute 内完成，直接读取输出
	auto taskIt = _taskOutputs.find(taskId);
	if (taskIt == _taskOutputs.end()) {
		throw std::runtime_error("Node::execute: no output produced for task '" + taskId + "'");
	}

	auto outIt = taskIt->second.find(outputName);
	if (outIt == taskIt->second.end() || !outIt->second.has_value()) {
		_taskOutputs.erase(taskId);
		throw std::runtime_error("Node::execute: output '" + outputName + "' not found");
	}

	Value result = std::move(outIt->second.value());
	_taskOutputs.erase(taskId);
	return result;
}

// ── 任务生命周期 ──
void Node::clearTask(const TaskId& taskId) {
	_taskInputs.erase(taskId);
	_taskOutputs.erase(taskId);
}

size_t Node::taskCount() const {
	return _taskInputs.size();
}

// ════════════════════════════════════════════
// Tensor 便捷接口（自动完成 Tensor ↔ Value 包装）
// ════════════════════════════════════════════

bool Node::setInput(const TaskId& taskId, const std::string& portName, Tensor data) {
	auto* p = new Tensor(std::move(data));
	return setInput(taskId, portName, Value(p, [](Tensor* ptr) { delete ptr; }));
}

bool Node::setInput(const TaskId& taskId,
                                std::unordered_map<std::string, Tensor> inputs) {
	std::unordered_map<std::string, TaskData> wrapped;
	wrapped.reserve(inputs.size());
	for (auto& [name, t] : inputs) {
		auto* p = new Tensor(std::move(t));
		wrapped.emplace(name, Value(p, [](Tensor* ptr) { delete ptr; }));
	}
	return setInput(taskId, std::move(wrapped));
}

Tensor Node::getOutputTensor(const TaskId& taskId, const std::string& name) {
	auto nt = getOutput(taskId, name);
	auto* t = nt.as<Tensor>();
	if (!t) {
		throw std::out_of_range("Node::getOutputTensor: output '" + name
			+ "' is not a DC::Tensor (innerType="
			+ std::to_string(static_cast<uint32_t>(nt.innerType())) + ")");
	}
	return std::move(*t);
}

std::unordered_map<std::string, Tensor>
Node::collectOutputTensors(const TaskId& taskId) {
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

Tensor Node::executeTensor(const std::string& outputName,
                                std::unordered_map<std::string, Tensor> inputs) {
	std::unordered_map<std::string, Value> wrapped;
	wrapped.reserve(inputs.size());
	for (auto& [name, t] : inputs) {
		auto* p = new Tensor(std::move(t));
		wrapped.emplace(name, Value(p, [](Tensor* ptr) { delete ptr; }));
	}

	auto nt = execute(outputName, std::move(wrapped));
	auto* t = nt.as<Tensor>();
	if (!t) {
		throw std::runtime_error("Node::executeTensor: output '" + outputName
			+ "' is not a DC::Tensor (innerType="
			+ std::to_string(static_cast<uint32_t>(nt.innerType())) + ")");
	}
	return std::move(*t);
}

// ── RunFn 内部使用的计算 API ──
const Value& Node::_inputImpl(const std::string& name) const {
	auto it = _inputSlots.find(name);
	if (it == _inputSlots.end()) {
		throw std::out_of_range("Node::_inputImpl: input '" + name + "' not found");
	}
	const auto* nt = it->second.peek<Value>();
	if (!nt) {
		throw std::runtime_error("Node::_inputImpl: input '" + name + "' is not a Value");
	}
	return *nt;
}

void Node::_outputImpl(const std::string& name, Value tensor) {
	auto it = _outputSlots.find(name);
	if (it == _outputSlots.end()) {
		throw std::out_of_range("Node::_outputImpl: output '" + name + "' not found");
	}
	it->second.store(std::move(tensor));
}

// ── 转换钩子访问器 ──
const TensorConverter* Node::_converter() const {
	auto* desc = _engineDescriptor();
	if (!desc) return nullptr;
	return &desc->converter;
}

const EngineDescriptor* Node::_engineDescriptor() const {
	if (_engineInstance) return _engineInstance->descriptor();
	return EngineRegistry::instance().find(_type);
}

void Node::_synchronizeEngine() const {
	if (!_engineInstance) return;
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
	if (it == _node._outputSlots.end()) return nullptr;
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
	if (_taskInputs.contains(taskId)) return;

	TaskBuffer inputs;
	for (const auto& port : _schema.inputs) {
		// 所有端口初始为空（nullopt），数据由 setInput 填充
		inputs.emplace(port.name, std::nullopt);
	}
	_taskInputs.emplace(taskId, std::move(inputs));
}

bool Node::_isTaskReady(const TaskId& taskId) const {
	auto it = _taskInputs.find(taskId);
	if (it == _taskInputs.end()) return false;

	for (const auto& port : _schema.inputs) {
		if (!port.required) continue;
		if (port.defaultValue.has_value()) continue;  // 默认值视为已就绪

		auto slotIt = it->second.find(port.name);
		if (slotIt == it->second.end()) return false;
		if (!slotIt->second.has_value()) return false;
	}
	return true;
}

void Node::_checkAndExecute(const TaskId& taskId) {
	if (!_isTaskReady(taskId)) return;

	// ① 加载输入：_taskInputs[taskId] → _inputSlots (move)
	_loadTaskToWorkingSlots(taskId);

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
	Result result;
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
	_collectAndSaveOutputs(taskId);

	// ⑤ 验证输出完整性
	if (result.ok()) {
		for (const auto& port : _schema.outputs) {
			auto outIt = _taskOutputs[taskId].find(port.name);
			if (outIt == _taskOutputs[taskId].end() || !outIt->second.has_value()) {
				result = _makeFailure(Status::InternalError,
					"Output '" + port.name + "' was not produced by RunFn");
				break;
			}
		}
	}

	// Diagnostic logging for failures to aid debugging
	if (!result.ok()) {
		try {
			std::cerr << "Node[" << _name << "] task '" << taskId
					  << "' failed: status=" << static_cast<int>(result.status)
					  << ", message='" << result.message << "'" << std::endl;
		} catch (...) {
			// swallow any logging errors
		}
	}

	// ⑥ 清理输入缓冲（输出缓冲保留，供调用方拉取）
	_taskInputs.erase(taskId);

	// ⑦ 调用回调（仅通知）
	if (_onComplete) {
		_onComplete(taskId, result);
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
					throw TensorException(TensorException::ErrorType::Other,
						"Node::_loadTaskToWorkingSlots: invalid Tensor in Value for port '" + port.name + "'");
				}
				if (t->type() != port.type) {
					throw TensorException(TensorException::ErrorType::TypeMismatch,
						"Node::_loadTaskToWorkingSlots: type mismatch for port '" + port.name + "'");
				}
			}

			workSlot.store(std::move(taskIt->second.value()));
			taskIt->second.reset();
		} else if (port.defaultValue.has_value()) {
			auto* p = new Tensor(port.defaultValue.value());
			workSlot.store(Value(p, [](Tensor* ptr) { delete ptr; }));
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
		if (!workSlot.hasData()) continue;
		auto taskIt = taskOutputs.find(name);
		if (taskIt == taskOutputs.end()) continue;
		taskIt->second = workSlot.take<Value>();
	}
}

} // namespace DC
