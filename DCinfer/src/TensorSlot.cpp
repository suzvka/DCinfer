#include "TensorSlot.h"

namespace DC {
using TensorType = TensorMeta::TensorType;

TensorSlot::TensorSlot(const std::string& name, TensorMeta::TensorType type, size_t typeSize, const Shape& shape,
					   const Config& config) {
	_rule.name = name;
	_rule.shape = shape;
	_rule.type = type;
	_rule.typeSize = typeSize;
	_config = config;
}

TensorSlot& TensorSlot::setDefaultTensor(const Tensor& data) {
	_defaultData = std::make_unique<Tensor>(data);
	return *this;
}

const std::string& TensorSlot::name() const {
	return _rule.name;
}
TensorType TensorSlot::type() const {
	return _rule.type;
}
size_t TensorSlot::typeSize() const {
	return _rule.typeSize;
}
TensorSlot::Shape TensorSlot::shape() const {
	return _rule.shape;
}

bool TensorSlot::isInput() const {
	return _config.position == Config::Position::Input;
}

bool TensorSlot::hasData() const {
	return _blob.has_value() || _defaultData != nullptr || static_cast<bool>(_defaultProvider);
}

bool TensorSlot::hasDefaultData() const {
	return _defaultData != nullptr;
}

TensorSlot& TensorSlot::setDefaultProvider(DefaultProvider fn) {
	_defaultProvider = std::move(fn);
	return *this;
}

void TensorSlot::resolveDefaultIfNeeded(const SlotMap& peers) {
	if (_blob.has_value() || !_defaultProvider)
		return;
	auto t = _defaultProvider(peers);
	if (t) {
		store(Value(std::move(t)));
	}
}

const Tensor& TensorSlot::view() const {
	// 优先返回运行时数据
	if (auto* t = peek<Tensor>()) {
		return *t;
	}
	if (_defaultData) {
		return *_defaultData;
	}
	abort(ErrorType::NotData, "Slot is empty");
}

SlotDataType TensorSlot::storedType() const {
	if (_blob.has_value()) {
		return _blob->type;
	}
	if (_defaultData) {
		return ensureSlotType<Tensor>();
	}
	return SlotDataTypeUnknown;
}

void TensorSlot::clear() {
	if (_blob.has_value() && _blob->deleter && _blob->ptr) {
		_blob->deleter(_blob->ptr);
	}
	_blob.reset();
	_defaultData.reset();
	_defaultProvider = nullptr;
}

TensorSlot::Config TensorSlot::CreateConfig() {
	return Config();
}

void TensorSlot::abort(ErrorType errorType, const std::string& message) const {
	std::string source = "TensorSlot";
	if (!_rule.name.empty()) {
		source += " (" + _rule.name + ")";
	}
	throw TensorException(errorType, source, message);
}

// ── Config ──
TensorSlot::Config& TensorSlot::Config::setPosition(Position p) {
	position = p;
	return *this;
}
} // namespace DC