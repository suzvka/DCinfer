#pragma once
#include <algorithm>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "Expected.h"
#include "TensorMods.h"

namespace DC {
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// DC::Tensor
// 张量对象
// 栈占用：248 字节
// 
// 用于创建输入或输出张量的数据载体。
//
// 形状术语约定：
// - CurrentShape: 当前数据形状（TensorData::getCurrentShape）
// - View::path : 索引路径，不等于 shape
// 
// 	2026.2.16
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	
	class Tensor {
		using TensorType = TensorMeta::TensorType;

	public:
		virtual ~Tensor() = default;
		template<bool IsConst> class ViewImpl;
		enum class TensorError {
			TypeMismatch,
			ShapeMismatch,
			InvalidPath,
			InvalidShape,
			NotAScalar,
			Other
		};
		using View = ViewImpl<false>;
		using ConstView = ViewImpl<true>;

		Tensor();

		Tensor(
			const TensorType& type,					// - 张量类型
			size_t typeSize = 0,					// - 类型字节数（TypeSize）
			const std::vector<int64_t>& shape = {},	// - 数据形状（CurrentShape）
			std::vector<char>&& data = {}			// - 张量数据
		);

		template<typename T>
		static Tensor Create(
			const std::vector<int64_t>& shape = {},
			std::vector<char>&& data = {}
		);

		View operator[](int64_t index);
		ConstView operator[](int64_t index) const;

		View view();
		ConstView view() const;

		// Scalar item access for whole tensor (0-dim or 1-dim size 1)
		template<typename T>
		T item() const;

		// Non-throwing scalar access
		template<typename T>
		Expected<T, TensorError> tryItem() const;

		template<typename T>
		Expected<T, TensorError> tryItem();

		// 重载赋值
		Tensor& operator=(const Tensor& other);
		// 移动构造
		Tensor(Tensor&& other) noexcept;

		// 移动赋值运算符
		Tensor& operator=(Tensor&& other) noexcept;

		// 在 Tensor 类的 public 部分添加拷贝构造函数
		Tensor(const Tensor& other);

		// 重载赋值
		// 给当前设定的维度写入数据
		template<typename T>
		Tensor& operator=(const std::vector<T>& data);

		// 标量赋值：无索引直接赋值时，视为设置 0D 标量张量
		template<typename T>
		Tensor& operator=(const T& value);

		template<typename T>
		Tensor& fill(const T& value);

		TensorType type() const;
		size_t typeSize() const;

		std::span<const char> bytes() const;

		template<typename T>
		std::span<const T> data() const;

		template<typename T>
		std::span<T> data();

		// 获取当前的动态形状（CurrentShape）
		std::vector<int64_t> shape() const;

		// 直接设置为稠密（连续）数据。
		// bytes 为按 type 的字节序列（与 shape 对应）。
		Tensor& setDense(std::vector<char>&& bytes, const std::vector<int64_t>& shape);

	private:
		TensorMeta _meta;
		TensorData _data;

		Expected<bool, TensorError> checkTypeMatch(size_t size) const;

		Expected<bool, TensorError> checkPathValid(const std::vector<int64_t>& path, const std::vector<int64_t>& shape) const;

		Expected<bool, TensorError> checkShapeValid(const std::vector<int64_t>& shape) const;

		Expected<bool, TensorError> checkSingleElementView(const std::vector<int64_t>& path, const std::vector<int64_t>& shape) const;

		template<typename T>
		void writeAt(const std::vector<int64_t>& path, const std::vector<T>& data);

		template<typename T>
		void writeScalarAt(const std::vector<int64_t>& path, const T& data);

		// Expected-based variants (non-void success uses bool as success indicator)
		template<typename T>
		Expected<T, TensorError> tryReadScalarAt(const std::vector<int64_t>& path) const;

		template<typename T>
		Expected<T, TensorError> tryReadScalarAt(const std::vector<int64_t>& path);

		template<typename T>
		Expected<bool, TensorError> tryWriteScalarAt(const std::vector<int64_t>& path, const T& data);

		template<typename T>
		std::vector<T> readAt(const std::vector<int64_t>& path) const;

		template<typename T>
		std::span<const T> readSpanAt(const std::vector<int64_t>& path) const;

		template<typename T>
		std::span<const T> readSpanAt(const std::vector<int64_t>& path);

		template<typename T>
		T readScalarAt(const std::vector<int64_t>& path) const;

		template<typename T>
		T readScalarAt(const std::vector<int64_t>& path);

		void move_from(Tensor&& other) noexcept;
	};

template<bool IsConst>
class Tensor::ViewImpl {
public:
    using TensorType = std::conditional_t<IsConst, const Tensor, Tensor>;

    // 构造函数：直接存储路径向量（移动或拷贝）
    ViewImpl(TensorType& tensor) : _tensor(&tensor) {}  // 空路径
    ViewImpl(TensorType& tensor, std::vector<int64_t> path) 
        : _tensor(&tensor), _path(std::move(path)) {}

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

	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	// Template Implementations: Tensor
	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

	template<typename T>
	inline Tensor Tensor::Create(
		const std::vector<int64_t>& shape,
		std::vector<char>&& data
	) {
		TensorMeta::ensureTypeMap();
		Tensor tensor(
			DC::Type::getType<TensorType, T>(),
			sizeof(T),
			shape,
			std::move(data)
		);

		return tensor;
	}

	template<typename T>
	inline T Tensor::item() const {
    return readScalarAt<T>({});
	}

	template<typename T>
inline Expected<T, Tensor::TensorError> Tensor::tryItem() const {
    return tryReadScalarAt<T>({});
}

	template<typename T>
inline Expected<T, Tensor::TensorError> Tensor::tryItem() {
    return tryReadScalarAt<T>({});
}

	template<typename T>
	inline Tensor& Tensor::operator=(const std::vector<T>& data) {
		writeAt({}, data);
		return *this;
	}

	template<typename T>
	inline Tensor& Tensor::operator=(const T& value) {
		auto res = checkTypeMatch(sizeof(T));
		if (!res) {
			switch (res.getError()) {
			case TensorError::TypeMismatch: throw std::runtime_error("type mismatch");
			default: throw std::runtime_error("unknown tensor write error");
			}
		}

		std::vector<char> bytes(_meta.typeSize, 0);
		std::memcpy(bytes.data(), &value, _meta.typeSize);
		_data.setDenseBytes({}, _meta.typeSize, std::move(bytes));
		_data.setScalar(true);
		return *this;
	}

	template<typename T>
	Tensor& Tensor::fill(const T& data) {
		auto resType = checkTypeMatch(sizeof(T));
		if (!resType) {
			switch (resType.getError()) {
			case TensorError::TypeMismatch: throw std::runtime_error("type mismatch");
			default: throw std::runtime_error("unknown tensor write error");
			}
		}

		auto shape = _data.getCurrentShape();
		const bool isScalar0d = shape.empty();
		if (isScalar0d) {
			std::vector<char> bytes(_meta.typeSize, 0);
			std::memcpy(bytes.data(), &data, _meta.typeSize);
			_data.setDenseBytes({}, _meta.typeSize, std::move(bytes));
			return *this;
		}

		auto resShape = checkShapeValid(shape);
		if (!resShape) throw std::runtime_error("Tensor::fill: current shape does not match meta shape rules");
		size_t elementCount = 1;
		for (const auto d : shape) {
			elementCount *= static_cast<size_t>(d);
		}

		std::vector<char> bytes(elementCount * _meta.typeSize);
		std::vector<char> scalarBytes(_meta.typeSize, 0);
		std::memcpy(scalarBytes.data(), &data, std::min(sizeof(T), _meta.typeSize));
		for (size_t off = 0; off < bytes.size(); off += _meta.typeSize) {
			std::memcpy(bytes.data() + off, scalarBytes.data(), _meta.typeSize);
		}

		_data.setDenseBytes(shape, _meta.typeSize, std::move(bytes));
		return *this;
	}

	template<typename T>
	inline std::span<const T> Tensor::data() const {
		return _data.dataSpanAs<T>();
	}

	template<typename T>
	inline std::span<T> Tensor::data() {
		_data.ensureEditable();
		return _data.dataSpanAsMut<T>();
	}

	template<typename T>
	inline void Tensor::writeAt(const std::vector<int64_t>& path, const std::vector<T>& data) {
		_data.ensureEditable();
		_data.writeBitcast(path, _meta.typeSize, data);
	}

	template<typename T>
	void Tensor::writeScalarAt(const std::vector<int64_t>& path, const T& data) {
		auto res = tryWriteScalarAt<T>(path, data);
		if (!res) {
			switch (res.getError()) {
			case TensorError::TypeMismatch: throw std::runtime_error("type mismatch");
			case TensorError::InvalidPath: throw std::out_of_range("path out of range");
			default: throw std::runtime_error("unknown tensor write error");
			}
		}
		// success -> nothing to do
	}

	template<typename T>
	inline std::vector<T> Tensor::readAt(const std::vector<int64_t>& path) const {
		auto span = readSpanAt<T>(path);
		return std::vector<T>(span.begin(), span.end());
	}

	template<typename T>
	inline std::span<const T> Tensor::readSpanAt(const std::vector<int64_t>& path) const {
		return _data.readSpan<T>(path);
	}

	template<typename T>
	inline std::span<const T> Tensor::readSpanAt(const std::vector<int64_t>& path) {
		return _data.readSpan<T>(path);
	}

	// Expected-based implementations

	template<typename T>
	inline Expected<T, Tensor::TensorError> Tensor::tryReadScalarAt(const std::vector<int64_t>& path) const {
		auto shape = _data.getCurrentShape();
		auto resType = checkTypeMatch(sizeof(T));
		if (!resType) return Expected<T, Tensor::TensorError>(TensorError::TypeMismatch);
		auto resPath = checkPathValid(path, shape);
		if (!resPath) return Expected<T, Tensor::TensorError>(TensorError::InvalidPath);

		// 路径长度等于形状：直接读取单个元素
		if (path.size() == shape.size())
			return Expected<T, Tensor::TensorError>(_data.readElement<T>(path));

		// 路径长度小于形状：要求子视图恰好包含一个元素
		auto resSingle = checkSingleElementView(path, shape);
		if (!resSingle) return Expected<T, Tensor::TensorError>(TensorError::NotAScalar);

		// 构造完整路径（剩余维度索引全为 0）
		std::vector<int64_t> fullPath = path;
		fullPath.insert(fullPath.end(), shape.size() - path.size(), 0);
		return Expected<T, Tensor::TensorError>(_data.readElement<T>(fullPath));
	}

	template<typename T>
	inline Expected<T, Tensor::TensorError> Tensor::tryReadScalarAt(const std::vector<int64_t>& path) {
		auto shape = _data.getCurrentShape();
		auto resType = checkTypeMatch(sizeof(T));
		if (!resType) return Expected<T, Tensor::TensorError>(TensorError::TypeMismatch);
		auto resPath = checkPathValid(path, shape);
		if (!resPath) return Expected<T, Tensor::TensorError>(TensorError::InvalidPath);

		if (path.size() == shape.size()) {
			return Expected<T, Tensor::TensorError>(_data.readElement<T>(path));
		}

		auto resSingle = checkSingleElementView(path, shape);
		if (!resSingle) return Expected<T, Tensor::TensorError>(TensorError::NotAScalar);

		std::vector<int64_t> fullPath = path;
		fullPath.insert(fullPath.end(), shape.size() - path.size(), 0);
		return Expected<T, Tensor::TensorError>(_data.readElement<T>(fullPath));
	}

	template<typename T>
	inline Expected<bool, Tensor::TensorError> Tensor::tryWriteScalarAt(const std::vector<int64_t>& path, const T& data) {
		auto shape = _data.getCurrentShape();
		auto resType = checkTypeMatch(sizeof(T));
		if (!resType) return Expected<bool, Tensor::TensorError>(TensorError::TypeMismatch);
		auto resPath = checkPathValid(path, shape);
		if (!resPath) return Expected<bool, Tensor::TensorError>(TensorError::InvalidPath);

		// 标量张量特例
		if (shape.empty()) {
			_data.writeElement({}, data);
			return Expected<bool, Tensor::TensorError>(true);
		}

		// 路径长度等于形状：单元素写入
		if (path.size() == shape.size()) {
			_data.writeElement(path, data);
			return Expected<bool, Tensor::TensorError>(true);
		}

		// 路径长度小于形状：计算剩余元素个数
		size_t remaining = 1;
		for (size_t i = path.size(); i < shape.size(); ++i)
			remaining *= static_cast<size_t>(shape[i]);

		if (remaining == 1) {
			// 子视图只有一个元素：构造完整路径后写入
			std::vector<int64_t> fullPath = path;
			fullPath.insert(fullPath.end(), shape.size() - path.size(), 0);
			_data.writeElement(fullPath, data);
			return Expected<bool, Tensor::TensorError>(true);
		}
		else {
			// 广播：构造包含 remaining 个 data 的向量
			std::vector<T> block(remaining, data);
			_data.writeBitcast(path, _meta.typeSize, block);
			return Expected<bool, Tensor::TensorError>(true);
		}
	}

	template<typename T>
	T Tensor::readScalarAt(const std::vector<int64_t>& path) const {
		auto shape = _data.getCurrentShape();
		if (!checkTypeMatch(sizeof(T))) throw std::runtime_error("type mismatch");
		if (!checkPathValid(path, shape)) throw std::out_of_range("path out of range");

		// 路径长度等于形状：直接读取单个元素
		if (path.size() == shape.size())
			return _data.readElement<T>(path);

		// 路径长度小于形状：要求子视图恰好包含一个元素
		if (!checkSingleElementView(path, shape)) throw std::runtime_error("scalar read requires a single-element view");

		// 构造完整路径（剩余维度索引全为 0）
		std::vector<int64_t> fullPath = path;
		fullPath.insert(fullPath.end(), shape.size() - path.size(), 0);
		return _data.readElement<T>(fullPath);
	}

	template<typename T>
	T Tensor::readScalarAt(const std::vector<int64_t>& path) {
		auto res = tryReadScalarAt<T>(path);
		if (!res) {
			switch (res.getError()) {
			case TensorError::TypeMismatch: throw std::runtime_error("type mismatch");
			case TensorError::InvalidPath: throw std::out_of_range("path out of range");
			case TensorError::NotAScalar: throw std::runtime_error("scalar read requires a single-element view");
			default: throw std::runtime_error("unknown tensor read error");
			}
		}
		return res.get();
	}

}