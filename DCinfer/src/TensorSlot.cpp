#include "TensorSlot.h"

namespace DC {
	using TensorType = TensorMeta::TensorType;

	TensorSlotBase::TensorSlotBase(
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

	TensorSlotBase& TensorSlotBase::setDefaultTensor(const Tensor& data) {
		_defaultData = std::make_unique<Tensor>(data);
		return *this;
	}

	const std::string& TensorSlotBase::name()          const { return _rule.name; }
	TensorType          TensorSlotBase::type()          const { return _rule.type; }
	size_t              TensorSlotBase::typeSize()      const { return _rule.typeSize; }
	TensorSlotBase::Shape TensorSlotBase::shape()      const { return _rule.shape; }

	TensorSlotBase::Shape TensorSlotBase::dataShape() const {
		// 优先从运行时数据获取形状（仅 DCTensor 类型有意义）
		if (auto* t = peek<Tensor>()) {
			return t->shape();
		}
		if (_defaultData) {
			return _defaultData->shape();
		}
		return {};
	}

	bool TensorSlotBase::isInput()  const { return _config.position == Config::Position::Input; }
	bool TensorSlotBase::isOutput() const { return _config.position == Config::Position::Output; }

	bool TensorSlotBase::hasData() const {
		return _blob.has_value() || _defaultData != nullptr;
	}

	bool TensorSlotBase::hasDefaultData() const {
		return _defaultData != nullptr;
	}

	const Tensor& TensorSlotBase::defaultTensor() const {
		if (!_defaultData) {
			abort(ErrorType::NotData, "No default data");
		}
		return *_defaultData;
	}

	const Tensor& TensorSlotBase::view() const {
		// 优先返回运行时数据
		if (auto* t = peek<Tensor>()) {
			return *t;
		}
		if (_defaultData) {
			return *_defaultData;
		}
		abort(ErrorType::NotData, "Slot is empty");
	}

	SlotDataType TensorSlotBase::storedType() const {
		if (_blob.has_value()) {
			return _blob->type;
		}
		if (_defaultData) {
			return SlotDataType::DCTensor;
		}
		return SlotDataType::Unknown;
	}

	const void* TensorSlotBase::rawPtr() const {
		if (_blob.has_value()) {
			return _blob->ptr;
		}
		// 默认数据视为 DCTensor
		if (_defaultData) {
			return _defaultData.get();
		}
		return nullptr;
	}

	void TensorSlotBase::clear() {
		if (_blob.has_value() && _blob->deleter && _blob->ptr) {
			_blob->deleter(_blob->ptr);
		}
		_blob.reset();
		_defaultData.reset();
	}

	void TensorSlotBase::clearData() {
		if (_blob.has_value() && _blob->deleter && _blob->ptr) {
			_blob->deleter(_blob->ptr);
		}
		_blob.reset();
	}

	const TensorSlotBase::Config& TensorSlotBase::config() const { return _config; }

	TensorSlotBase::Config TensorSlotBase::CreateConfig() { return Config(); }

	void TensorSlotBase::abort(
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
	TensorSlotBase::Config& TensorSlotBase::Config::setPosition(Position p) {
		position = p;
		return *this;
	}
}