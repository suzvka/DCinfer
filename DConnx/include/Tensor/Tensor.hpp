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
// 
// 	2026.2.16
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	
	class Tensor {
		using TensorType = TensorMeta::TensorType;
		using ErrorType = TensorException::ErrorType;
		using DataBlock = TensorData::DataBlock;
		using Shape = std::vector<int64_t>; // 支持负数

	public:
		virtual ~Tensor() = default;
		template<bool IsConst> class ViewImpl;
		
		using View = ViewImpl<false>;
		using ConstView = ViewImpl<true>;

		Tensor();

		Tensor(
			const TensorType& type,
			size_t typeSize = 0,
			const Shape& shape = {},
			DataBlock&& data = {}
		);

		template<typename T>
		static Tensor Create(
			const Shape& shape = {},
			DataBlock&& data = {}
		);

		View operator[](int64_t index);
		ConstView operator[](int64_t index) const;

		View view();
		ConstView view() const;

		template<typename T>
		T item() const;

		// 重载赋值
		Tensor& operator=(const Tensor& other);
		// 移动构造
		Tensor(Tensor&& other) noexcept;

		// 移动赋值运算符
		Tensor& operator=(Tensor&& other) noexcept;

		// 在 Tensor 类的 public 部分添加拷贝构造函数
		Tensor(const Tensor& other);

		// 标量赋值：无索引直接赋值时，视为设置 0D 标量张量
		template<typename T>
		Tensor& operator=(const T& value);

		template<typename T>
		Tensor& fill(const T& value);

		TensorType type() const;
		size_t typeSize() const;

		template<typename T>
		std::span<const T> data() const;

		// Return raw underlying bytes as a read-only span.
		std::span<const std::byte> bytes() const;

		// 获取当前的动态形状（CurrentShape）
		Shape shape() const;

		// 直接设置为稠密（连续）数据。
		Tensor& loadData(DataBlock&& data, const Shape& shape);

		// 异常中止
		void abort(
			ErrorType errorType = ErrorType::Other,
			const std::string& message = ""
		) const ;

		template<typename T>
		std::vector<T> getData();

		bool isScalar() const { return _data.isScalar(); }
		bool empty() const { return _data.empty(); }
		bool valid() const { return _data.valid(); }
		bool hasCache() const { return _data.hasCache(); }

	private:
		TensorMeta _meta;
		TensorData _data;

		Expected<bool, ErrorType> checkTypeMatch(size_t size) const;

		Expected<bool, ErrorType> checkPathValid(const Shape& path, const TensorData::Shape& shape) const;

		Expected<bool, ErrorType> checkSingleElementView(const Shape& path, const Shape& shape) const;

        // 向指定路径写数据块 (会在错误时抛出 via abort)
        template<typename T>
        void write(const Shape& path, const std::vector<T>& data);

        // 向指定路径写标量（路径可指向一个元素或一个单元素子视图，支持广播）(会在错误时抛出)
        template<typename T>
        void writeScalar(const Shape& path, const T& data);

		// 从指定路径读取数据块并按 typeSize 解释为 T 元素的只读 span。路径语义同上。
		template<typename T>
		std::span<const T> read(const Shape& path) const;

        // 读取标量，路径可指向一个元素或一个单元素子视图（支持广播）。在错误时抛出异常。
        template<typename T>
        T readScalar(const Shape& path) const;

		// 拷贝数据和元信息
		void moveFrom(Tensor&& other) noexcept;

		std::vector<size_t> indexShape(const Shape& shape, bool isRead) const;
	};

template<bool IsConst>
class Tensor::ViewImpl {
public:
    using TensorType = std::conditional_t<IsConst, const Tensor, Tensor>;

    // 构造函数：直接存储路径向量（移动或拷贝）
    ViewImpl(TensorType& tensor) : _tensor(&tensor) {}  // 空路径
    ViewImpl(TensorType& tensor, Shape path) 
        : _tensor(&tensor), _path(std::move(path)) {}

    // 索引操作：复制当前路径并追加索引，返回新视图
    ViewImpl operator[](int64_t index) const {
        ViewImpl next(*_tensor);
        next._path = _path;          // 拷贝当前路径（小向量，成本可控）
        next._path.push_back(index);  // 追加新索引
        return next;
    }

	template<typename T, bool C = IsConst, typename = std::enable_if_t<!C>>
	ViewImpl& operator=(const std::vector<T>& data) {
		set(data);
		return *this;
	}

    // 赋值（仅非 const 版本）
    // Non-throwing try-set APIs (replace operator= for explicit error handling)
    template<typename T, bool C = IsConst, typename = std::enable_if_t<!C>>
    void set(const std::vector<T>& data) {
        _tensor->write(path(), data);
    }

    template<typename T, bool C = IsConst, typename = std::enable_if_t<!C>>
    void set(const T& data) {
        _tensor->writeScalar(path(), data);
    }

    // 读取数据 (single implementation usable on const and non-const views)
    template<typename T>
    std::vector<T> read() const {
        auto span = _tensor->template read<T>(path());
        return std::vector<T>(span.begin(), span.end());
    }

	// 获取视图对应的形状（从路径长度开始的剩余维度）
	Shape shape() const {
		auto full_shape = _tensor->shape();          // 获取原始形状
		if (_path.size() > full_shape.size()) {
			_tensor->abort(ErrorType::InvalidPath, "View path exceeds tensor rank.");
		}
		// 返回从路径长度开始的剩余维度
		return Shape(
			full_shape.begin() + _path.size(),
			full_shape.end()
		);
	}

	// 获取视图对应的秩（从路径长度开始的剩余维度数量）
	size_t rank() const {
		// shape() already returns the remaining dimensions after the path;
		// its size is the rank of this view.
		return shape().size();
	}

    template<typename T>
    T item() const {
        return _tensor->template readScalar<T>(path());
    }

    // 获取路径（返回 const 引用，避免拷贝）
    const Shape& path() const { return _path; }

private:
    TensorType* _tensor = nullptr;
    Shape _path;   // 连续存储路径
};

	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	// Template Implementations: Tensor
	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

	template<typename T>
	Tensor Tensor::Create(
		const Shape& shape,
		DataBlock&& data
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
	T Tensor::item() const {
		return _data.readElement<T>({});  // 0-D scalar access via empty path
	}

	template<typename T>
	Tensor& Tensor::operator=(const T& value) {
		auto res = checkTypeMatch(sizeof(T));
		if (!res) {
			abort(ErrorType::TypeMismatch, "type mismatch in scalar assignment");
		}

        // Prefer dense-write fast-path to safely create 0-D scalar if no cache exists.
        DataBlock bytes(_meta.typeSize);
        std::fill(bytes.begin(), bytes.end(), std::byte(0));
		std::memcpy(bytes.data(), &value, std::min(sizeof(T), _meta.typeSize));
		if (_data.hasCache()) {
			// write into existing dense cache
			if (!_data.writeCacheElement({}, value)) {
				abort(ErrorType::Other, "failed to write scalar into dense cache");
			}
			_data.setScalar(true);
			return *this;
		}
		// No dense cache: set dense bytes to represent scalar (creates dense-pass-through mode)
		_data.loadData({}, _meta.typeSize, std::move(bytes));
		_data.setScalar(true);
		return *this;
	}

	template<typename T>
	Tensor& Tensor::fill(const T& data) {
		auto resType = checkTypeMatch(sizeof(T));
		if (!resType) {
			abort(ErrorType::TypeMismatch, "type mismatch in fill assignment");
		}

		auto shape = _data.getCurrentShape();
		const bool isScalar0d = shape.empty();
		if (isScalar0d) {
            DataBlock bytes(_meta.typeSize);
            std::fill(bytes.begin(), bytes.end(), std::byte(0));
            std::memcpy(bytes.data(), &data, _meta.typeSize);
			_data.loadData({}, _meta.typeSize, std::move(bytes));
			return *this;
		}

		size_t elementCount = 1;
		for (const auto d : shape) {
			elementCount *= static_cast<size_t>(d);
		}

        DataBlock bytes(elementCount * _meta.typeSize);
        std::fill(bytes.begin(), bytes.end(), std::byte(0));
        DataBlock scalarBytes(_meta.typeSize);
        std::fill(scalarBytes.begin(), scalarBytes.end(), std::byte(0));
        std::memcpy(scalarBytes.data(), &data, std::min(sizeof(T), _meta.typeSize));
        for (size_t off = 0; off < bytes.size(); off += _meta.typeSize) {
            std::memcpy(bytes.data() + off, scalarBytes.data(), _meta.typeSize);
        }

		_data.loadData(shape, _meta.typeSize, std::move(bytes));
		return *this;
	}

	template<typename T>
	inline std::span<const T> Tensor::data() const {
		return _data.data<T>();
	}

    template<typename T>
    void Tensor::write(const Shape& path, const std::vector<T>& data) {
        try {
            _data.editMode();
            _data.write(indexShape(path, false), data);
        }
		catch (const TensorException& e) {
			abort(e.getErrorType(), e.what());
		}
        catch (const std::exception& e) {
            abort(ErrorType::Other, e.what());
        }
    }

    template<typename T>
    void Tensor::writeScalar(const Shape& path, const T& data) {
        try {
            _data.editMode();
            _data.write(indexShape(path, false), data);
        }
		catch (const TensorException& e) {
			abort(e.getErrorType(), e.what());
		}
        catch(const std::exception& e) {
            abort(ErrorType::Other, e.what());
        }
    }

	template<typename T>
	std::span<const T> Tensor::read(const Shape& path) const {
		return _data.read<T>(indexShape(path, true));
	}

	// Expected-based implementations

    template<typename T>
    T Tensor::readScalar(const Shape& path) const {
        // dataShape is TensorData::Shape (vector<size_t>)
        auto dataShape = _data.getCurrentShape();
        auto resType = checkTypeMatch(sizeof(T));
        if (!resType) abort(ErrorType::TypeMismatch, "type mismatch in scalar read");
        auto resPath = checkPathValid(path, dataShape);
        if (!resPath) abort(ErrorType::InvalidPath, "invalid path in scalar read");

        // If path length equals rank, read the single element
        if (path.size() == dataShape.size()) {
            auto full = indexShape(path, true); // convert/validate
            return _data.readElement<T>(full);
        }

        // Path shorter than rank: ensure remaining dimensions multiply to 1
        // convert dataShape to Tensor::Shape for checkSingleElementView
        Shape asTensorShape(dataShape.begin(), dataShape.end());
        auto resSingle = checkSingleElementView(path, asTensorShape);
        if (!resSingle) abort(ErrorType::NotAScalar, "not a scalar view");

        // build full path with trailing zeros
        Shape fullPath = path;
        fullPath.insert(fullPath.end(), dataShape.size() - path.size(), 0);
        auto full = indexShape(fullPath, true);
        return _data.readElement<T>(full);
    }

	template<typename T>
	std::vector<T> Tensor::getData() {
		auto data = _data.getData();
		std::vector<T> result(data.size() / typeSize());
		std::memcpy(result.data(), data.data(), data.size());
		return result;
	}
}