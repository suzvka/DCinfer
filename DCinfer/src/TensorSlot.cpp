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
		_rule.name = name;
		_rule.shape = shape;
		_rule.type = type;
		_rule.typeSize = typeSize;
		_config = config;
	}

	TensorSlotBase& TensorSlotBase::setDefaultTensor(const Tensor& data) {
		_defaultData = std::make_unique<Tensor>(data);

		return *this;
	}

	const std::string& TensorSlotBase::name() const { return _rule.name; }

	TensorType TensorSlotBase::type() const { return _rule.type; }

	size_t TensorSlotBase::typeSize() const { return _rule.typeSize; }

	TensorSlotBase::Shape TensorSlotBase::shape() const { return _rule.shape; }

	TensorSlotBase::Shape TensorSlotBase::dataShape() const {
		if (_data) {
			return _data->shape();
		}
		else if (_defaultData) {
			return _defaultData->shape();
		}
		else {
			return {};
		}
	}

	bool TensorSlotBase::isInput() const { return _config.position == Config::Position::Input; }
	bool TensorSlotBase::isOutput() const { return _config.position == Config::Position::Output; }

	TensorSlotBase& TensorSlotBase::operator<<(const Tensor& data) {
		return write(Tensor(data));
	}

	TensorSlotBase& TensorSlotBase::operator<<(Tensor&& data) {
		return write(std::move(data));
	}

	TensorSlotBase& TensorSlotBase::write(Tensor&& data) {
		if (_config.position != Config::Position::Input) {
			abort(ErrorType::InvalidPath, "Cannot write to output slot");
		}
		return loadData(std::move(data));
	}

	bool TensorSlotBase::hasData() const { return _data != nullptr || _defaultData != nullptr; }

	bool TensorSlotBase::hasDefaultData() const { return _defaultData != nullptr; }

	bool TensorSlotBase::hasDynamicData() const { return _data != nullptr; }

	void TensorSlotBase::clear() { _data.reset(); _defaultData.reset(); }

	void TensorSlotBase::clearData() { _data.reset(); }

	const Tensor& TensorSlotBase::view() const {
		if (!_data && !_defaultData) {
			abort(ErrorType::NotData, "Slot is empty");
		}
		return _data ? *_data : *_defaultData;
	}

	const TensorSlotBase::Config& TensorSlotBase::config() const { return _config; }

	TensorSlotBase::Config TensorSlotBase::CreateConfig() { return Config(); }

	Tensor TensorSlotBase::takeTensor() {
		if (!hasData()) {
			abort(ErrorType::NotData, "TensorSlotBase has no data.");
		}

		if (!_data) {
			return Tensor(*_defaultData);
		}

		return std::move(*_data);
	}

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

	Tensor TensorSlotBase::align(
		const Shape& target,
		std::byte fillData
	) {
		auto tensor = takeTensor();
		auto targetShape = target;
		auto currentShape = dataShape();

		if (currentShape.size() != targetShape.size()) {
			abort(ErrorType::ShapeMismatch, "TensorSlotBase shape mismatch");
		}

		for (size_t i = 0; i < currentShape.size(); ++i) {
			// 维度过小时使用填充数据扩展到目标形状
			if (currentShape[i] < targetShape[i]) {
				tensor = tensor.expand(targetShape, fillData);
			}

			// 维度过大时裁剪到目标形状
			if (currentShape[i] > targetShape[i]) {
				tensor = tensor.crop(targetShape);
			}
		}

		return tensor;
	}

	TensorSlotBase& TensorSlotBase::read(Tensor& data) {
		if (_config.position != Config::Position::Output) {
			abort(ErrorType::InvalidPath, "Cannot read from input slot");
		}

		if (!hasData()) { 
			abort(ErrorType::InvalidPath, "no tensor data available"); 
		
		}

		if (_data) {
			data = std::move(*_data);
			_data.reset();
		}

		else {
			data = Tensor(*_defaultData);
		}

		return *this;
	}

	TensorSlotBase& TensorSlotBase::operator>>(Tensor& data) {
		return read(data);
	}

	TensorSlotBase::DataStatus TensorSlotBase::check() const {
		return check(view());
	}

	TensorSlotBase::DataStatus TensorSlotBase::check(const Tensor& data) const {
		DataStatus result;

		if (!data.valid()) { 
			result.invalid = true;
			return result;
		}

		if (_config.requiredcheckType()) {
			result.needConvert = type() != data.type();
		}

		result.needAlign = !_rule.checkShape(data.shape());

		return result;
	}

	TensorSlotBase& TensorSlotBase::loadData(Tensor&& data) {
		auto checkResult = check(data);
		if (!checkResult.ready()) {
			if (checkResult.invalid) {
				abort(ErrorType::InvalidShape, "Input tensor is invalid");
			}
			if (checkResult.needConvert) {
				abort(ErrorType::TypeMismatch, "Input tensor type mismatch and conversion is not allowed");
			}
			if (checkResult.needAlign) {
				abort(ErrorType::ShapeMismatch, "Input tensor shape mismatch and alignment is not allowed");
			}
		}

		_data = std::make_unique<Tensor>(std::move(data));
		return *this;
	}

	bool TensorSlotBase::Config::allowTypeConversion() const {
		return checkLevel == CheckLevel::Lenient;
	}

	bool TensorSlotBase::Config::requiredcheckType() const {
		if (type == Type::Value) {
			return true;
		}
		else {
			return checkLevel == CheckLevel::Strict;
		}
	}

	TensorSlotBase::Config& TensorSlotBase::Config::setType(Type t) {
		type = t;
		return *this;
	}

	TensorSlotBase::Config& TensorSlotBase::Config::setPosition(Position p) {
		position = p;
		return *this;
	}

	TensorSlotBase::Config& TensorSlotBase::Config::setCheckLevel(CheckLevel level) {
		checkLevel = level;
		return *this;
	}

	bool TensorSlotBase::DataStatus::ready() const {
		return !invalid && !needAlign && !needConvert;
	}
}