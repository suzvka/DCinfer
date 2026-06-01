#include "Node/internal/SlotWorkspace.h"
#include "Node.h"
#include "Tensor.hpp"

namespace DC {

SlotWorkspace::SlotWorkspace(const NodeSchema& schema) {
	for (const auto& port : schema.inputs) {
		TensorSlot::Config cfg;
		cfg.setPosition(TensorSlot::Config::Position::Input);
		_inputSlots.emplace(port.name, TensorSlot(port.name, port.type, port.typeSize, port.shape, cfg));
	}

	for (const auto& port : schema.outputs) {
		TensorSlot::Config cfg;
		cfg.setPosition(TensorSlot::Config::Position::Output);
		_outputSlots.emplace(port.name, TensorSlot(port.name, port.type, port.typeSize, port.shape, cfg));
	}

	// 为 shapeAnchor 端口安装懒求值 DefaultProvider
	for (const auto& port : schema.inputs) {
		if (port.shapeAnchor.has_value()) {
			auto& slot = _inputSlots.at(port.name);
			std::string anchorName = port.shapeAnchor.value();
			slot.setDefaultProvider([anchorName](const TensorSlot::SlotMap& peers) -> std::unique_ptr<Tensor> {
				auto it = peers.find(anchorName);
				if (it == peers.end())
					return nullptr;
				const auto* anchor = it->second.peek<Tensor>();
				if (!anchor || !anchor->valid())
					return nullptr;
				auto t = std::make_unique<Tensor>(anchor->type(), anchor->typeSize(), anchor->shape());
				t->fill<char>(0);
				return t;
			});
		}
	}
}

// ── RunContext 委托接口 ──

const Value& SlotWorkspace::peekInput(const std::string& name) const {
	auto it = _inputSlots.find(name);
	if (it == _inputSlots.end()) {
		throw NodeException(NodeException::ErrorType::PortNotFound, "SlotWorkspace::peekInput",
							"input '" + name + "' not found");
	}
	const auto* nt = it->second.peek<Value>();
	if (!nt) {
		throw NodeException(NodeException::ErrorType::TypeMismatch, "SlotWorkspace::peekInput",
							"input '" + name + "' is not a Value");
	}
	return *nt;
}

Value SlotWorkspace::popInput(const std::string& name) {
	auto it = _inputSlots.find(name);
	if (it == _inputSlots.end()) {
		throw NodeException(NodeException::ErrorType::PortNotFound, "SlotWorkspace::popInput",
							"input '" + name + "' not found");
	}
	if (!it->second.hasData()) {
		throw NodeException(NodeException::ErrorType::InternalError, "SlotWorkspace::popInput",
							"input '" + name + "' has no data to pop");
	}
	return it->second.take<Value>();
}

void SlotWorkspace::writeOutput(const std::string& name, Value tensor) {
	auto it = _outputSlots.find(name);
	if (it == _outputSlots.end()) {
		throw NodeException(NodeException::ErrorType::PortNotFound, "SlotWorkspace::writeOutput",
							"output '" + name + "' not found");
	}
	it->second.store(std::move(tensor));
}

const Value* SlotWorkspace::peekOutputRaw(const std::string& name) const {
	auto it = _outputSlots.find(name);
	if (it == _outputSlots.end())
		return nullptr;
	return it->second.peek<Value>();
}

// ── 执行保护 ──

bool SlotWorkspace::tryAcquire() {
	return !_executionGuard.test_and_set(std::memory_order_acquire);
}

void SlotWorkspace::release() {
	_executionGuard.clear(std::memory_order_release);
}

// ── 清空/跟踪 ──

void SlotWorkspace::clearOutputs() {
	for (auto& [name, slot] : _outputSlots) {
		slot.clear();
	}
}

void SlotWorkspace::setCurrentTask(const std::string& taskId) {
	_currentTaskId = taskId;
}

void SlotWorkspace::clearCurrentTask() {
	_currentTaskId.reset();
}

std::optional<std::string> SlotWorkspace::currentTask() const {
	return _currentTaskId;
}

} // namespace DC
