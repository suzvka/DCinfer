#include <algorithm>
#include "Tensor.hpp"

namespace DC {

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// Tensor implementation
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Tensor::Tensor() {
	TensorMeta::ensureTypeMap();
	_meta.type = TensorType::Void;
	_meta.typeSize = 1;
}

Tensor::Tensor(const TensorType& type, size_t typeSize, const Shape& shape, DataBlock&& data) {
	TensorMeta::ensureTypeMap();
	_meta.type = type;
	if (_meta.type == TensorType::Void) {
		_meta.typeSize = (typeSize > 0) ? typeSize : 1;
	} else {
		_meta.typeSize = (typeSize > 0) ? typeSize : DC::Type::getSize(type);
	}
	_meta.shape = shape;
	_data = TensorData(indexShape(shape, false), _meta.typeSize, std::move(data));
}

Tensor& Tensor::setName(const std::string& name) {
	_meta.name = name;
	return *this;
}

Tensor::View Tensor::operator[](int64_t index) const {
	Shape path = {index}; // Handle empty shape case
	return View(std::move(path), *this);
}

Tensor::Tensor(const Tensor& other) {
	_meta = other._meta;
	_data = other._data;
}

Tensor& Tensor::operator=(const Tensor& other) {
	if (this != &other) {
		_meta = other._meta;
		_data = other._data;
	}
	return *this;
}

Tensor::Tensor(Tensor&& other) noexcept {
	moveFrom(std::move(other));
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
	if (this != &other) {
		moveFrom(std::move(other));
	}
	return *this;
}

Tensor::TensorType Tensor::type() const {
	return _meta.type;
}

size_t Tensor::typeSize() const {
	return _meta.typeSize;
}

std::span<const std::byte> Tensor::bytes() const {
	// Provide a reliable byte view: build dense cache if needed
	return _data.data();
}

Tensor::Shape Tensor::shape() const {
	TensorData::Shape currentShape = _data.getCurrentShape();
	return Tensor::Shape(currentShape.begin(), currentShape.end());
}

Tensor& Tensor::loadData(DataBlock&& data, const Shape& shape) {
	_data.loadData(indexShape(shape, false), _meta.typeSize, std::move(data));
	return *this;
}

void Tensor::abort(ErrorType errorType, const std::string& message) const {
	std::string source = "Tensor";
	if (!_meta.name.empty()) {
		source += " (" + _meta.name + ")";
	}

	throw TensorException(errorType, source, message);
}

std::optional<Tensor::ErrorType> Tensor::checkTypeMatch(size_t size) const {
	if (_meta.typeSize == 0)
		return Tensor::ErrorType::TypeMismatch;
	if (size != _meta.typeSize)
		return Tensor::ErrorType::TypeMismatch;
	return std::nullopt;
}

std::optional<Tensor::ErrorType> Tensor::checkPathValid(const Shape& path, const TensorData::Shape& shape) const {
	if (path.size() > shape.size())
		return Tensor::ErrorType::InvalidPath;
	for (size_t i = 0; i < path.size(); ++i) {
		int64_t idx = path[i];
		int64_t bound = static_cast<int64_t>(shape[i]);
		if (idx < 0 || idx >= bound)
			return Tensor::ErrorType::InvalidPath;
	}
	return std::nullopt;
}

std::optional<Tensor::ErrorType> Tensor::checkSingleElementView(const Shape& path, const Shape& shape) const {
	size_t remaining = 1;
	for (size_t i = path.size(); i < shape.size(); ++i)
		remaining *= static_cast<size_t>(shape[i]);
	if (remaining != 1)
		return Tensor::ErrorType::NotAScalar;
	return std::nullopt;
}

void Tensor::moveFrom(Tensor&& other) noexcept {
	_meta = std::move(other._meta);
	_data = std::move(other._data);
}

std::vector<size_t> Tensor::indexShape(const Shape& shape, bool isRead) const {
	auto dataShape = _data.getCurrentShape();

	// For reads we must not request more dimensions than exist.
	if (shape.size() > dataShape.size() && isRead) {
		abort(ErrorType::InvalidPath, "index path has more dimensions than tensor shape");
	}

	std::vector<size_t> result;
	for (size_t i = 0; i < shape.size(); ++i) {
		int64_t idx = shape[i];
		int64_t dimSize = (i < dataShape.size()) ? static_cast<int64_t>(dataShape[i]) : 0;

		if (idx < 0) {
			if (i < dataShape.size()) {
				idx += dimSize;
			} else {
				abort(ErrorType::InvalidPath,
					  "negative index at dimension " + std::to_string(i) + " cannot be resolved");
			}
		}

		if (isRead) {
			if (idx < 0 || idx >= dimSize) {
				abort(ErrorType::InvalidPath, "index path dimension " + std::to_string(i) + " is out of bounds");
			}
		} else {
			if (idx < 0) {
				abort(ErrorType::InvalidPath, "index path dimension " + std::to_string(i) + " is out of bounds");
			}
		}

		result.push_back(static_cast<size_t>(idx));
	}

	return result;
}
} // namespace DC
