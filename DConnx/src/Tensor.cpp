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

    Tensor::Tensor(
        const TensorType& type,
        size_t typeSize,
        const Shape& shape,
        DataBlock&& data
    ) {
        TensorMeta::ensureTypeMap();
        _meta.type = type;
        if (_meta.type == TensorType::Void) {
            _meta.typeSize = (typeSize > 0) ? typeSize : 1;
        }
        else {
            _meta.typeSize = (typeSize > 0) ? typeSize : DC::Type::getSize(type);
        }
        _data = TensorData(indexShape(shape, false), _meta.typeSize, std::move(data));
    }

    Tensor& Tensor::setName(const std::string& name) {
        _meta.name = name;
        return *this;
	}


    Tensor::View Tensor::operator[](int64_t index) {
        return View(*this, { index });
    }

    Tensor::ConstView Tensor::operator[](int64_t index) const {
        return ConstView(*this, { index });
    }

    Tensor::View Tensor::view() {
        return View(*this);
    }

    Tensor::ConstView Tensor::view() const {
        return ConstView(*this);
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

    Expected<bool, Tensor::ErrorType> Tensor::checkTypeMatch(size_t size) const {
        if (_meta.typeSize == 0) return Expected<bool, Tensor::ErrorType>(Tensor::ErrorType::TypeMismatch);
        if (size != _meta.typeSize) return Expected<bool, Tensor::ErrorType>(Tensor::ErrorType::TypeMismatch);
        return Expected<bool, Tensor::ErrorType>(true);
    }

    Expected<bool, Tensor::ErrorType> Tensor::checkPathValid(const Shape& path, const TensorData::Shape& shape) const {
        if (path.size() > shape.size()) return Expected<bool, Tensor::ErrorType>(Tensor::ErrorType::InvalidPath);
        for (size_t i = 0; i < path.size(); ++i) {
            // path is signed (int64_t) while shape elements are size_t (unsigned).
            // Perform comparisons in a signed domain to avoid signed/unsigned mismatch warnings.
            int64_t idx = path[i];
            int64_t bound = static_cast<int64_t>(shape[i]);
            if (idx < 0 || idx >= bound) return Expected<bool, Tensor::ErrorType>(Tensor::ErrorType::InvalidPath);
        }
        return Expected<bool, Tensor::ErrorType>(true);
    }

    Expected<bool, Tensor::ErrorType> Tensor::checkSingleElementView(const Shape& path, const Shape& shape) const {
        size_t remaining = 1;
        for (size_t i = path.size(); i < shape.size(); ++i)
            remaining *= static_cast<size_t>(shape[i]);
        if (remaining != 1) return Expected<bool, Tensor::ErrorType>(Tensor::ErrorType::NotAScalar);
        return Expected<bool, Tensor::ErrorType>(true);
    }

    void Tensor::moveFrom(Tensor&& other) noexcept {
        _meta = std::move(other._meta);
        _data = std::move(other._data);
    }

    // 在这里支持负数索引
    // 先获取当前数据的实际形状，负数维度就可以解释为倒着数
    // 超出范围在这里就可以报错
    std::vector<size_t> Tensor::indexShape(const Shape& shape, bool isRead) const {
		auto dataShape = _data.getCurrentShape();

        // For reads we must not request more dimensions than exist.
        if (shape.size() > dataShape.size() && isRead) {
            abort(ErrorType::InvalidPath, "index path has more dimensions than tensor shape");
        }

		std::vector<size_t> result;
		for (size_t i = 0; i < shape.size(); ++i) {
			int64_t idx = shape[i];
			// If the requested dimension exists use its size; otherwise treat as not-yet-present.
			int64_t dimSize = (i < dataShape.size()) ? static_cast<int64_t>(dataShape[i]) : 0;

			// Negative indices can only be resolved when the dimension currently exists.
			if (idx < 0) {
				if (i < dataShape.size()) {
					idx += dimSize; // Convert negative index to positive
				}
				else {
					// Writing may create new dimensions, but a negative index cannot be
					// interpreted for a non-existent dimension.
					abort(ErrorType::InvalidPath, "negative index at dimension " + std::to_string(i) + " cannot be resolved");
				}
			}

			if (isRead) {
				// Reads must be fully in-range.
				if (idx < 0 || idx >= dimSize) {
					abort(ErrorType::InvalidPath, "index path dimension " + std::to_string(i) + " is out of bounds");
				}
			}
			else {
				// For writes we allow the index to be >= dimSize (will create/expand).
				if (idx < 0) {
					abort(ErrorType::InvalidPath, "index path dimension " + std::to_string(i) + " is out of bounds");
				}
			}

			result.push_back(static_cast<size_t>(idx));
		}

		return result;
    }

    TensorData::DataBlock TensorData::getData() {
        if (!hasCache()) {
            ensureCache();
        }

        DataBlock dataBlock = std::move(_dataCache);
        clearCache();
        return std::move(dataBlock);
    }

} // namespace DC
