#include "InferNode.h"
#include "EngineRegistry.h"
#include "TensorException.h"

#include <stdexcept>
#include <iostream>

namespace DC {

// ── 构造：从 Schema 构建工作槽位 ──
InferNode::InferNode(std::string type, std::string name, Schema schema, RunFn fn)
	: _type(std::move(type))
	, _name(std::move(name))
	, _schema(std::move(schema))
	, _fn(std::move(fn))
{
	for (const auto& port : _schema.inputs) {
		TensorSlotBase::Config cfg;
		cfg.setPosition(TensorSlotBase::Config::Position::Input);
		_inputSlots.emplace(port.name, TensorSlotBase(
			port.name, port.type, port.typeSize, port.shape, cfg));
	}

	for (const auto& port : _schema.outputs) {
		TensorSlotBase::Config cfg;
		cfg.setPosition(TensorSlotBase::Config::Position::Output);
		_outputSlots.emplace(port.name, TensorSlotBase(
			port.name, port.type, port.typeSize, port.shape, cfg));
	}
}

InferNode::~InferNode() = default;

// ── 回调注册 ──
void InferNode::setCompletionCallback(CompletionFn fn) {
	std::lock_guard lock(_mutex);
	_onComplete = std::move(fn);
}

bool InferNode::hasCompletionCallback() const {
	std::lock_guard lock(_mutex);
	return static_cast<bool>(_onComplete);
}

// ── 单端口输入 ──
bool InferNode::setInput(const TaskId& taskId, const std::string& portName, NativeTensor data) {
	std::unique_lock lock(_mutex);

	if (!_schema.findInput(portName)) {
		return false;
	}

	_ensureTaskExists(taskId);
	_taskInputs[taskId].at(portName) = std::move(data);

	try {
		_checkAndExecute(taskId, lock);
	} catch (const TensorException&) {
		return false;
	}
	return true;
}

// ── 批量输入 ──
bool InferNode::setInputs(const TaskId& taskId,
                          std::unordered_map<std::string, TaskData> inputs) {
	std::unique_lock lock(_mutex);

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
		_checkAndExecute(taskId, lock);
	} catch (const TensorException&) {
		return false;
	}
	return true;
}

// ── 任务级输出 ──
bool InferNode::hasOutput(const TaskId& taskId, const std::string& name) const {
	std::lock_guard lock(_mutex);

	auto taskIt = _taskOutputs.find(taskId);
	if (taskIt == _taskOutputs.end()) return false;
	auto slotIt = taskIt->second.find(name);
	if (slotIt == taskIt->second.end()) return false;
	return slotIt->second.has_value();
}

InferNode::TaskPortMap::mapped_type
InferNode::collectOutputs(const TaskId& taskId) {
	std::lock_guard lock(_mutex);

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

// ── 任务生命周期 ──
void InferNode::clearTask(const TaskId& taskId) {
	std::lock_guard lock(_mutex);
	_taskInputs.erase(taskId);
	_taskOutputs.erase(taskId);
}

size_t InferNode::taskCount() const {
	std::lock_guard lock(_mutex);
	return _taskInputs.size();
}

// ── RunFn 内部使用的计算 API ──
const Tensor& InferNode::input(const std::string& name) const {
	auto it = _inputSlots.find(name);
	if (it == _inputSlots.end()) {
		throw std::out_of_range("InferNode::input: input '" + name + "' not found");
	}
	return it->second.view();
}

void InferNode::output(const std::string& name, const Tensor& tensor) {
	auto it = _outputSlots.find(name);
	if (it == _outputSlots.end()) {
		throw std::out_of_range("InferNode::output: output '" + name + "' not found");
	}
	it->second.store(tensor);
}

void InferNode::output(const std::string& name, Tensor&& tensor) {
	auto it = _outputSlots.find(name);
	if (it == _outputSlots.end()) {
		throw std::out_of_range("InferNode::output: output '" + name + "' not found");
	}
	it->second.store(std::move(tensor));
}

// ── 转换钩子访问器 ──
const TensorConverter* InferNode::converter() const {
	auto* desc = engineDescriptor();
	if (!desc) return nullptr;
	return &desc->converter;
}

const EngineDescriptor* InferNode::engineDescriptor() const {
	return EngineRegistry::instance().find(_type);
}

// ── 结果辅助 ──
InferNode::Result InferNode::success(std::string message) const {
	Result r;
	r.status = Status::Ok;
	r.message = std::move(message);
	return r;
}

InferNode::Result InferNode::failure(Status status, std::string message) const {
	Result r;
	r.status = status;
	r.message = std::move(message);
	return r;
}

// ════════════════════════════════════════════
// 内部方法（调用时已持锁）
// ════════════════════════════════════════════

void InferNode::_ensureTaskExists(const TaskId& taskId) {
	if (_taskInputs.contains(taskId)) return;

	TaskBuffer inputs;
	for (const auto& port : _schema.inputs) {
		// 所有端口初始为空（nullopt），数据由 setInput 填充
		inputs.emplace(port.name, std::nullopt);
	}
	_taskInputs.emplace(taskId, std::move(inputs));
}

bool InferNode::_isTaskReady(const TaskId& taskId) const {
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

void InferNode::_checkAndExecute(const TaskId& taskId, std::unique_lock<std::mutex>& lock) {
	if (!_isTaskReady(taskId)) return;

	// ① 加载输入：_taskInputs[taskId] → _inputSlots (move)
	_loadTaskToWorkingSlots(taskId);

	// ② 清空上一轮工作输出
	_clearWorkingOutputs();

	// ③ 执行 RunFn
	Result result;
	try {
		result = _fn(*this);
	} catch (const std::exception& e) {
		result = failure(Status::ExecutionFailed, e.what());
	} catch (...) {
		result = failure(Status::ExecutionFailed, "Unknown exception in RunFn");
	}

	// ④ 保存输出：_outputSlots → _taskOutputs[taskId]
	_collectAndSaveOutputs(taskId);

	// ⑤ 验证输出完整性
	if (result.ok()) {
		for (const auto& port : _schema.outputs) {
			auto outIt = _taskOutputs[taskId].find(port.name);
			if (outIt == _taskOutputs[taskId].end() || !outIt->second.has_value()) {
				result = failure(Status::InternalError,
					"Output '" + port.name + "' was not produced by RunFn");
				break;
			}
		}
	}

	// Diagnostic logging for failures to aid debugging
	if (!result.ok()) {
		try {
			std::cerr << "InferNode[" << _name << "] task '" << taskId
					  << "' failed: status=" << static_cast<int>(result.status)
					  << ", message='" << result.message << "'" << std::endl;
		} catch (...) {
			// swallow any logging errors
		}
	}

	// ⑥ 清理输入缓冲（输出缓冲保留，供调用方拉取）
	_taskInputs.erase(taskId);

	// ⑦ 释放锁
	lock.unlock();

	// ⑧ 在锁外调用回调（仅通知）
	if (_onComplete) {
		_onComplete(taskId, result);
	}
}

void InferNode::_loadTaskToWorkingSlots(const TaskId& taskId) {
	auto& taskInputs = _taskInputs[taskId];
	auto* nodeConverter = converter();

	for (const auto& port : _schema.inputs) {
		auto taskIt = taskInputs.find(port.name);
		auto& workSlot = _inputSlots.at(port.name);

		if (taskIt != taskInputs.end() && taskIt->second.has_value()) {
			// 用户通过 NativeTensor 提供了数据
			auto& nativeData = taskIt->second.value();

			if (nodeConverter && nodeConverter->toDC) {
				// 通过转换钩子：NativeTensor → Tensor → 工作槽位
				Tensor t = nodeConverter->toDC(nativeData.get());
				workSlot.store(std::move(t));
			} else {
				// 无转换器（引擎原生节点）：直接存入 NativeTensor
				workSlot.store(std::move(nativeData));
			}
			taskIt->second.reset();  // 已消费
		} else if (port.defaultValue.has_value()) {
			// 回退到默认值（默认值始终以 Tensor 形式存入）
			workSlot.store(Tensor(port.defaultValue.value()));
		}
	}
}

void InferNode::_clearWorkingOutputs() {
	for (auto& [name, slot] : _outputSlots) {
		slot.clear();
	}
}

void InferNode::_collectAndSaveOutputs(const TaskId& taskId) {
	// 确保输出 task bucket 存在
	if (!_taskOutputs.contains(taskId)) {
		TaskBuffer outputs;
		for (const auto& port : _schema.outputs) {
			outputs.emplace(port.name, std::nullopt);
		}
		_taskOutputs.emplace(taskId, std::move(outputs));
	}

	auto* nodeConverter = converter();
	auto& taskOutputs = _taskOutputs[taskId];
	for (auto& [name, workSlot] : _outputSlots) {
		if (!workSlot.hasData()) continue;

		auto taskIt = taskOutputs.find(name);
		if (taskIt == taskOutputs.end()) continue;

		switch (workSlot.storedType()) {
		case SlotDataType::DCTensor: {
			auto t = workSlot.take<Tensor>();
			if (nodeConverter && nodeConverter->toNative) {
				// 通过转换钩子：Tensor → NativeTensor → 输出缓冲
				taskIt->second = nodeConverter->toNative(t);
			} else {
				// 无转换器：直接包装为 NativeTensor（持有 Tensor 所有权）
				auto* p = new Tensor(std::move(t));
				taskIt->second = NativeTensor(p, [](Tensor* ptr) { delete ptr; });
			}
			break;
		}
		case SlotDataType::NativeTensor: {
			auto nt = workSlot.take<NativeTensor>();
			taskIt->second = std::move(nt);
			break;
		}
		default:
			break;
		}
	}
}

} // namespace DC
