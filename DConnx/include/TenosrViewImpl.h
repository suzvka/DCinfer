#pragma once
#include "Tensor.h"

namespace DC::Tensor {
    template<bool IsConst>
    class ViewImpl {
    public:
        using TensorType = std::conditional_t<IsConst, const Tensor, Tensor>;

        // 构造函数：直接存储路径向量（移动或拷贝）
        ViewImpl(TensorType& tensor) : _tensor(&tensor) {}  // 空路径
        ViewImpl(TensorType& tensor, std::vector<int64_t> path)
            : _tensor(&tensor), _path(std::move(path)) {
        }

        // 索引操作：复制当前路径并追加索引，返回新视图
        ViewImpl operator[](int64_t index) const {
            ViewImpl next(*_tensor);
            next._path = _path;          // 拷贝当前路径（小向量，成本可控）
            next._path.push_back(index);  // 追加新索引
            return next;
        }

        // 赋值（仅非 const 版本）
        // Non-throwing try-set APIs (replace operator= for explicit error handling)
        template<typename T, bool C = IsConst, typename = std::enable_if_t<!C>>
        Expected<bool, Tensor::TensorError> trySet(const std::vector<T>& data) {
            try {
                _tensor->writeAt(path(), data);
                return Expected<bool, Tensor::TensorError>(true);
            }
            catch (const std::exception&) {
                return Expected<bool, Tensor::TensorError>(Tensor::TensorError::Other);
            }
        }

        template<typename T, bool C = IsConst, typename = std::enable_if_t<!C>>
        Expected<bool, Tensor::TensorError> trySet(const T& data) {
            return _tensor->tryWriteScalarAt<T>(path(), data);
        }

        // 读取数据
        template<typename T>
        std::vector<T> read() const {
            auto span = readSpan<T>();
            return std::vector<T>(span.begin(), span.end());
        }

        // Non-throwing readSpan variant that returns Expected
        template<typename T>
        Expected<std::vector<T>, Tensor::TensorError> tryRead() const {
            auto span = readSpan<T>();
            return Expected<std::vector<T>, Tensor::TensorError>(std::vector<T>(span.begin(), span.end()));
        }

        template<typename T, bool C = IsConst, typename = std::enable_if_t<!C>>
        std::vector<T> read() {
            auto span = readSpan<T>();
            return std::vector<T>(span.begin(), span.end());
        }

        template<typename T>
        std::span<const T> readSpan() const {
            return _tensor->template readSpanAt<T>(path());
        }

        template<typename T, bool C = IsConst, typename = std::enable_if_t<!C>>
        std::span<const T> readSpan() {
            return _tensor->template readSpanAt<T>(path());
        }

        // 获取视图对应的形状（从路径长度开始的剩余维度）
        std::vector<int64_t> shape() const {
            auto full_shape = _tensor->shape();          // 获取原始形状
            if (_path.size() > full_shape.size()) {
                throw std::runtime_error("View path exceeds tensor rank.");
            }
            // 返回从路径长度开始的剩余维度
            return std::vector<int64_t>(
                full_shape.begin() + _path.size(),
                full_shape.end()
            );
        }

        // 获取视图对应的秩（从路径长度开始的剩余维度数量）
        size_t rank() const {
            auto full_shape = _tensor->shape();
            if (_path.size() > full_shape.size()) {
                throw std::runtime_error("View path exceeds tensor rank.");
            }
            return full_shape.size() - _path.size();
        }

        template<typename T>
        T item() const {
            return _tensor->template readScalarAt<T>(path());
        }

        template<typename T, bool C = IsConst, typename = std::enable_if_t<!C>>
        T item() {
            return _tensor->template readScalarAt<T>(path());
        }

        // Non-throwing item
        template<typename T>
        Expected<T, Tensor::TensorError> tryItem() const {
            return _tensor->template tryReadScalarAt<T>(path());
        }

        template<typename T, bool C = IsConst, typename = std::enable_if_t<!C>>
        Expected<T, Tensor::TensorError> tryItem() {
            return _tensor->template tryReadScalarAt<T>(path());
        }

        // Non-throwing rank
        Expected<size_t, Tensor::TensorError> tryRank() const {
            auto full_shape = _tensor->shape();
            if (_path.size() > full_shape.size()) {
                return Expected<size_t, Tensor::TensorError>(Tensor::TensorError::InvalidPath);
            }
            return Expected<size_t, Tensor::TensorError>(full_shape.size() - _path.size());
        }

        // 获取路径（返回 const 引用，避免拷贝）
        const std::vector<int64_t>& path() const { return _path; }

    private:
        TensorType* _tensor = nullptr;
        std::vector<int64_t> _path;   // 连续存储路径
    };
}
