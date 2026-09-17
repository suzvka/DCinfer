#include "Node/internal/TaskBuffer.h"
#include "Node/internal/SlotWorkspace.h"
#include "Node.h"
#include "Tensor.hpp"

namespace DC {

// ── 输入 ──

void TaskBuffer::setInput(const TaskId& taskId, const std::string& portName, Value data,
						  const NodeSchema& schema) {
	if (!schema.findInput(portName)) {
		throw NodeException(NodeException::ErrorType::PortNotFound, "TaskBuffer::setInput",
							"port '" + portName + "' not found in schema");
	}

	std::unique_lock lk(_mutex);
	_ensureTaskExists(taskId, schema);
	_taskInputs[taskId].at(portName) = std::move(data);
}

bool TaskBuffer::setInputAndCheckReady(const TaskId& taskId, const std::string& portName,
									   Value data, const NodeSchema& schema) {
	if (!schema.findInput(portName)) {
		throw NodeException(NodeException::ErrorType::PortNotFound, "TaskBuffer::setInputAndCheckReady",
							"port '" + portName + "' not found in schema");
	}

	// 写入与就绪判定同一临界区：多上游并发传播时仅最后写入者观察到 ready
	std::unique_lock lk(_mutex);
	_ensureTaskExists(taskId, schema);
	_taskInputs[taskId].at(portName) = std::move(data);
	return _isReadyLocked(taskId, schema);
}

void TaskBuffer::setInputBatch(const TaskId& taskId,
							   std::unordered_map<std::string, TaskData> inputs,
							   const NodeSchema& schema) {
	// 预校验：所有端口名必须存在
	for (const auto& [name, data] : inputs) {
		if (!schema.findInput(name)) {
			throw NodeException(NodeException::ErrorType::PortNotFound, "TaskBuffer::setInputBatch",
								"port '" + name + "' not found in schema");
		}
	}

	std::unique_lock lk(_mutex);
	_ensureTaskExists(taskId, schema);

	for (auto& [name, data] : inputs) {
		_taskInputs[taskId].at(name) = std::move(data);
	}
}

// ── 就绪判断 ──

bool TaskBuffer::isReady(const TaskId& taskId, const NodeSchema& schema) const {
	std::shared_lock lk(_mutex);
	return _isReadyLocked(taskId, schema);
}

bool TaskBuffer::_isReadyLocked(const TaskId& taskId, const NodeSchema& schema) const {
	auto it = _taskInputs.find(taskId);
	if (it == _taskInputs.end())
		return false;

	for (const auto& port : schema.inputs) {
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

// ── 输出 ──

bool TaskBuffer::hasOutput(const TaskId& taskId, const std::string& name) const {
	std::shared_lock lk(_mutex);

	auto taskIt = _taskOutputs.find(taskId);
	if (taskIt == _taskOutputs.end())
		return false;
	auto slotIt = taskIt->second.find(name);
	if (slotIt == taskIt->second.end())
		return false;
	return slotIt->second.has_value();
}

Value TaskBuffer::takeOutput(const TaskId& taskId, const std::string& name) {
	std::unique_lock lk(_mutex);
	auto taskIt = _taskOutputs.find(taskId);
	if (taskIt == _taskOutputs.end()) {
		throw NodeException(NodeException::ErrorType::TaskNotFound, "TaskBuffer::takeOutput",
							"task '" + taskId + "' not found");
	}
	auto& optVal = taskIt->second.at(name);
	if (!optVal.has_value()) {
		throw NodeException(NodeException::ErrorType::OutputNotProduced, "TaskBuffer::takeOutput",
							"output '" + name + "' is empty");
	}
	Value result = std::move(optVal.value());
	optVal.reset();
	return result;
}

std::optional<Value> TaskBuffer::tryTakeOutput(const TaskId& taskId, const std::string& name) {
	std::unique_lock lk(_mutex);
	auto taskIt = _taskOutputs.find(taskId);
	if (taskIt == _taskOutputs.end())
		return std::nullopt;
	auto slotIt = taskIt->second.find(name);
	if (slotIt == taskIt->second.end() || !slotIt->second.has_value())
		return std::nullopt;
	Value result = std::move(slotIt->second.value());
	slotIt->second.reset();
	return result;
}

std::unordered_map<std::string, TaskBuffer::TaskData> TaskBuffer::collectOutputs(const TaskId& taskId) {
	std::unique_lock lk(_mutex);
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

// ── 生命周期 ──

bool TaskBuffer::hasTask(const TaskId& taskId) const {
	std::shared_lock lk(_mutex);
	return _taskInputs.contains(taskId) || _taskOutputs.contains(taskId);
}

void TaskBuffer::clearTask(const TaskId& taskId) {
	std::unique_lock lk(_mutex);
	_taskInputs.erase(taskId);
	_taskOutputs.erase(taskId);
}

size_t TaskBuffer::taskCount() const {
	std::shared_lock lk(_mutex);
	// 与 hasTask 同口径（#8-17）：输入表 ∪ 输出表——执行后输入已擦除
	// 但输出未取走的任务（eraseInputs 后）仍被计入
	size_t count = _taskInputs.size();
	for (const auto& [tid, entry] : _taskOutputs) {
		if (!_taskInputs.contains(tid))
			++count;
	}
	return count;
}

// ── 批量传输（供 ExecutionPipeline 使用）──

void TaskBuffer::drainInputsTo(const TaskId& taskId, SlotWorkspace& workspace,
							   const NodeSchema& schema) {
	std::unique_lock lk(_mutex);
	auto& taskInputs = _taskInputs[taskId];

	for (const auto& port : schema.inputs) {
		auto taskIt = taskInputs.find(port.name);
		auto& mutableInputSlots = workspace.mutableInputSlots();
		auto& workSlot = mutableInputSlots.at(port.name);

		if (taskIt != taskInputs.end() && taskIt->second.has_value()) {
			auto& nativeData = taskIt->second.value();

			// 对 DCTensor 类型的 Value 进行类型校验
			if (nativeData.innerType() == ensureSlotType<Tensor>() && port.type != Tensor::TensorType::Void) {
				const auto* t = static_cast<const Tensor*>(nativeData.get());
				if (!t || !t->valid()) {
					throw NodeException(NodeException::ErrorType::InternalError,
										"TaskBuffer::drainInputsTo",
										"invalid Tensor in Value for port '" + port.name + "'");
				}
				if (t->type() != port.type) {
					throw NodeException(NodeException::ErrorType::TypeMismatch,
										"TaskBuffer::drainInputsTo",
										"type mismatch for port '" + port.name + "'");
				}
			}

			workSlot.store(std::move(taskIt->second.value()));
			taskIt->second.reset();
		} else if (port.defaultValue.has_value()) {
			workSlot.store(Value(std::make_unique<Tensor>(port.defaultValue.value())));
		}
	}

	// 第二遍：懒求值 DefaultProvider（形状锚定等动态默认值）
	for (auto& [name, slot] : workspace.mutableInputSlots()) {
		slot.resolveDefaultIfNeeded(workspace.inputSlots());
	}
}

void TaskBuffer::fillOutputsFrom(const TaskId& taskId, SlotWorkspace& workspace,
								 const NodeSchema& schema) {
	std::unique_lock lk(_mutex);

	if (!_taskOutputs.contains(taskId)) {
		TaskBufferEntry outputs;
		for (const auto& port : schema.outputs) {
			outputs.emplace(port.name, std::nullopt);
		}
		_taskOutputs.emplace(taskId, std::move(outputs));
	}

	auto& taskOutputs = _taskOutputs[taskId];
	for (auto& [name, workSlot] : workspace.mutableOutputSlots()) {
		if (!workSlot.hasData())
			continue;
		auto taskIt = taskOutputs.find(name);
		if (taskIt == taskOutputs.end())
			continue;
		taskIt->second = workSlot.take<Value>();
	}
}

void TaskBuffer::eraseInputs(const TaskId& taskId) {
	std::unique_lock lk(_mutex);
	_taskInputs.erase(taskId);
}

bool TaskBuffer::validateOutputs(const TaskId& taskId, const NodeSchema& schema) const {
	std::shared_lock lk(_mutex);

	auto taskIt = _taskOutputs.find(taskId);
	if (taskIt == _taskOutputs.end())
		return false;

	for (const auto& port : schema.outputs) {
		auto outIt = taskIt->second.find(port.name);
		if (outIt == taskIt->second.end() || !outIt->second.has_value())
			return false;
	}
	return true;
}

// ── 内部方法 ──

void TaskBuffer::_ensureTaskExists(const TaskId& taskId, const NodeSchema& schema) {
	if (_taskInputs.contains(taskId))
		return;

	TaskBufferEntry inputs;
	for (const auto& port : schema.inputs) {
		inputs.emplace(port.name, std::nullopt);
	}
	_taskInputs.emplace(taskId, std::move(inputs));
}

} // namespace DC
