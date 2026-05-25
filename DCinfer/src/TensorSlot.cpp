#include "TensorSlot.h"

namespace DC {
	using TensorType = TensorMeta::TensorType;

	TensorSlot::TensorSlot(
		const std::string& name,
		TensorMeta::TensorType type,
		size_t typeSize,
		const Shape& shape,
		const Config& config
	) {
		_rule.name     = name;
		_rule.shape    = shape;
		_rule.type     = type;
		_rule.typeSize = typeSize;
		_config        = config;
	}

	TensorSlot& TensorSlot::setDefaultTensor(const Tensor& data) {
		_defaultData = std::make_unique<Tensor>(data);
		return *this;
	}

	const std::string& TensorSlot::name()          const { return _rule.name; }
	TensorType          TensorSlot::type()          const { return _rule.type; }
	size_t              TensorSlot::typeSize()      const { return _rule.typeSize; }
	TensorSlot::Shape TensorSlot::shape()      const { return _rule.shape; }

	TensorSlot::Shape TensorSlot::dataShape() const {
		// 优先从运行时数据获取形状（仅 DCTensor 类型有意义）
		if (auto* t = peek<Tensor>()) {
			return t->shape();
		}
		if (_defaultData) {
			return _defaultData->shape();
		}
		return {};
	}

	bool TensorSlot::isInput()  const { return _config.position == Config::Position::Input; }
	bool TensorSlot::isOutput() const { return _config.position == Config::Position::Output; }

	bool TensorSlot::hasData() const {
		return _blob.has_value() || _defaultData != nullptr;
	}

	bool TensorSlot::hasDefaultData() const {
		return _defaultData != nullptr;
	}

	const Tensor& TensorSlot::defaultTensor() const {
		if (!_defaultData) {
			abort(ErrorType::NotData, "No default data");
		}
		return *_defaultData;
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
			return SlotDataType::DCTensor;
		}
		return SlotDataType::Unknown;
	}

	const void* TensorSlot::rawPtr() const {
		if (_blob.has_value()) {
			return _blob->ptr;
		}
		// 默认数据视为 DCTensor
		if (_defaultData) {
			return _defaultData.get();
		}
		return nullptr;
	}

	void TensorSlot::clear() {
		if (_blob.has_value() && _blob->deleter && _blob->ptr) {
			_blob->deleter(_blob->ptr);
		}
		_blob.reset();
		_defaultData.reset();
	}

	void TensorSlot::clearData() {
		if (_blob.has_value() && _blob->deleter && _blob->ptr) {
			_blob->deleter(_blob->ptr);
		}
		_blob.reset();
	}

	const TensorSlot::Config& TensorSlot::config() const { return _config; }

	TensorSlot::Config TensorSlot::CreateConfig() { return Config(); }

	void TensorSlot::abort(
		ErrorType errorType,
		const std::string& message
	) const {
		std::string source = "TensorSlotBase";
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
}