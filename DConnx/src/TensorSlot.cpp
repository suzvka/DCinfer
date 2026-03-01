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

	const std::string& TensorSlot::name() const { return _rule.name; }

	TensorType TensorSlot::type() const { return _rule.type; }

	size_t TensorSlot::typeSize() const { return _rule.typeSize; }

	TensorSlot::Shape TensorSlot::shape() const { return _rule.shape; }

	TensorSlot::Shape TensorSlot::dataShape() const {
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

	bool TensorSlot::isInput() const { return _config.position == Config::Position::Input; }
	bool TensorSlot::isOutput() const { return _config.position == Config::Position::Output; }

	TensorSlot& TensorSlot::operator<<(const Tensor& data) {
		return write(Tensor(data));
	}

	TensorSlot& TensorSlot::operator<<(Tensor&& data) {
		return write(std::move(data));
	}

	TensorSlot& TensorSlot::write(Tensor&& data) {
		if (_config.position != Config::Position::Input) {
			abort(ErrorType::InvalidPath, "Cannot write to output slot");
		}
		return loadData(std::move(data));
	}

	bool TensorSlot::hasData() const { return _data != nullptr || _defaultData != nullptr; }

	bool TensorSlot::hasDefaultData() const { return _defaultData != nullptr; }

	bool TensorSlot::hasDynamicData() const { return _data != nullptr; }

	void TensorSlot::clear() { _data.reset(); _defaultData.reset(); }

	void TensorSlot::clearData() { _data.reset(); }

	const Tensor& TensorSlot::view() const {
		if (!_data && !_defaultData) {
			abort(ErrorType::NotData, "Slot is empty");
		}
		return _data ? *_data : *_defaultData;
	}

	const TensorSlot::Config& TensorSlot::config() const { return _config; }

	TensorSlot::Config TensorSlot::CreateConfig() { return Config(); }

	Tensor TensorSlot::takeTensor() {
		if (!hasData()) {
			abort(ErrorType::NotData, "TensorSlot has no data.");
		}

		if (!_data) {
			return Tensor(*_defaultData);
		}

		return std::move(*_data);
	}

	void TensorSlot::abort(
		ErrorType errorType,
		const std::string& message
	) const {
		std::string source = "TensorSlot";
		if (!_rule.name.empty()) {
			source += " (" + _rule.name + ")";
		}
		throw TensorException(errorType, source, message);
	}

	Tensor TensorSlot::align(
		const Shape& target,
		std::byte fillData
	) {
		auto tensor = takeTensor();
		auto targetShape = target;
		auto currentShape = dataShape();

		if (currentShape.size() != targetShape.size()) {
			abort(ErrorType::ShapeMismatch, "TensorSlot shape mismatch");
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

	TensorSlot& TensorSlot::read(Tensor& data) {
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

	TensorSlot& TensorSlot::operator>>(Tensor& data) {
		return read(data);
	}

	TensorSlot::DataStatus TensorSlot::check() const {
		return check(view());
	}

	TensorSlot::DataStatus TensorSlot::check(const Tensor& data) const {
		DataStatus result;

		if (!data.valid()) result.invalid = true;

		if (_config.requiredcheckType()) {
			result.needConvert = type() != data.type();
		}

		bool shapeReady = _rule.checkShape(data.shape());
		if (_config.allowShapeAlignment()) {
			shapeReady ? (true) : (result.needAlign = true);
		}
		else {
			result.invalid = !shapeReady;
		}

		return result;
	}

	TensorSlot& TensorSlot::loadData(Tensor&& data) {
		auto checkResult = check(data);
		if (!checkResult.ready()) {
			if (checkResult.invalid) {
				abort(ErrorType::InvalidShape, "Input tensor is invalid");
			}
			if (checkResult.needConvert) {
				; // Todo: 类型转换，当前依赖 Tensor 按位强转
			}
			if (checkResult.needAlign) {
				abort(ErrorType::ShapeMismatch, "Input tensor shape mismatch and alignment is not allowed");
			}
		}

		_data = std::make_unique<Tensor>(std::move(data));
		return *this;
	}


	bool TensorSlot::Config::allowShapeAlignment() const {
		return checkLevel == CheckLevel::Lenient;
	}

	bool TensorSlot::Config::allowTypeConversion() const {
		return checkLevel == CheckLevel::Lenient;
	}

	bool TensorSlot::Config::requiredcheckType() const {
		if (type == Type::Value) {
			return true;
		}
		else {
			return checkLevel == CheckLevel::Strict;
		}
	}

	TensorSlot::Config& TensorSlot::Config::setType(Type t) {
		type = t;
		return *this;
	}

	TensorSlot::Config& TensorSlot::Config::setPosition(Position p) {
		position = p;
		return *this;
	}

	TensorSlot::Config& TensorSlot::Config::setCheckLevel(CheckLevel level) {
		checkLevel = level;
		return *this;
	}

	bool TensorSlot::DataStatus::ready() const {
		return !invalid && !needAlign && !needConvert;
	}
}