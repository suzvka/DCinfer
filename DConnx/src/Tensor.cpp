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
        const std::vector<int64_t>& shape,
        std::vector<char>&& data
    ) {
        TensorMeta::ensureTypeMap();
        _meta.type = type;
        if (_meta.type == TensorType::Void) {
            _meta.typeSize = (typeSize > 0) ? typeSize : 1;
        }
        else {
            _meta.typeSize = (typeSize > 0) ? typeSize : DC::Type::getSize(type);
        }
        _data = TensorData(shape, _meta.typeSize, std::move(data));
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
        move_from(std::move(other));
    }

    Tensor& Tensor::operator=(Tensor&& other) noexcept {
        if (this != &other) {
            move_from(std::move(other));
        }
        return *this;
    }

    Tensor::TensorType Tensor::type() const {
        return _meta.type;
    }

    size_t Tensor::typeSize() const {
        return _meta.typeSize;
    }

    std::span<const char> Tensor::bytes() const {
        return _data.dataSpan();
    }

    std::vector<int64_t> Tensor::shape() const {
        return _data.getCurrentShape();
    }

    Tensor& Tensor::setDense(std::vector<char>&& bytes, const std::vector<int64_t>& shape) {
        _data.setDenseBytes(shape, _meta.typeSize, std::move(bytes));
        return *this;
    }

    Expected<bool, Tensor::TensorError> Tensor::checkTypeMatch(size_t size) const {
        if (_meta.typeSize == 0) return Expected<bool, Tensor::TensorError>(Tensor::TensorError::TypeMismatch);
        if (size != _meta.typeSize) return Expected<bool, Tensor::TensorError>(Tensor::TensorError::TypeMismatch);
        return Expected<bool, Tensor::TensorError>(true);
    }

    Expected<bool, Tensor::TensorError> Tensor::checkPathValid(const std::vector<int64_t>& path, const std::vector<int64_t>& shape) const {
        if (path.size() > shape.size()) return Expected<bool, Tensor::TensorError>(Tensor::TensorError::InvalidPath);
        for (size_t i = 0; i < path.size(); ++i) {
            if (path[i] < 0 || path[i] >= shape[i]) return Expected<bool, Tensor::TensorError>(Tensor::TensorError::InvalidPath);
        }
        return Expected<bool, Tensor::TensorError>(true);
    }

    Expected<bool, Tensor::TensorError> Tensor::checkShapeValid(const std::vector<int64_t>& shape) const {
        for (const auto d : shape) {
            if (d <= 0) return Expected<bool, Tensor::TensorError>(Tensor::TensorError::InvalidShape);
        }
        return Expected<bool, Tensor::TensorError>(true);
    }

    Expected<bool, Tensor::TensorError> Tensor::checkSingleElementView(const std::vector<int64_t>& path, const std::vector<int64_t>& shape) const {
        size_t remaining = 1;
        for (size_t i = path.size(); i < shape.size(); ++i)
            remaining *= static_cast<size_t>(shape[i]);
        if (remaining != 1) return Expected<bool, Tensor::TensorError>(Tensor::TensorError::NotAScalar);
        return Expected<bool, Tensor::TensorError>(true);
    }

    void Tensor::move_from(Tensor&& other) noexcept {
        _meta = std::move(other._meta);
        _data = std::move(other._data);
    }

} // namespace DC
