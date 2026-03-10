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
	public:
		using TensorType = TensorMeta::TensorType;
		using ErrorType = TensorException::ErrorType;
		using DataBlock = TensorData::DataBlock;
		using Shape = std::vector<int64_t>; // 支持负数

		virtual ~Tensor() = default;
		class View;

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

		Tensor& setName(const std::string& name);

		View operator[](int64_t index) const;

		View view();

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

		template<typename T>
		Tensor& expand(const Shape& targetShape, const T& fillData = T());

		Tensor& crop(const Shape& targetShape);

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
        void write(const Shape& path, const T& data);

		// 从指定路径读取数据块并按 typeSize 解释为 T 元素的只读 span。路径语义同上。
		template<typename T>
		std::span<const T> read(const Shape& path) const;

        // 读取标量，路径可指向一个元素或一个单元素子视图（支持广播）。在错误时抛出异常。
        template<typename T>
        T readScalar(const Shape& path) const;

		// 拷贝数据和元信息
		void moveFrom(Tensor&& other) noexcept;

		TensorData::Shape indexShape(const Shape& shape, bool isRead) const;

		// 异常中止
		void abort(
			ErrorType errorType = ErrorType::Other,
			const std::string& message = ""
		) const;
	};

class Tensor::View {
public:
    View(Shape&& shape, Tensor& top):_shape(std::move(shape)), _top(top){}

	View(Shape&& shape, const Tensor& top) :_shape(std::move(shape)), _top(const_cast<Tensor&>(top)) {}

    // 索引操作：返回一个新的链节点
    View operator[](int64_t index) const {
		_shape.push_back(index);
        return View(std::move(_shape), _top);
    }

	// 赋值操作：代理调用顶层写入方法
	template<typename T>
	Tensor& operator=(const T& value) {
		set(value);
		return _top;
	}

	template<typename T>
	Tensor& set(const T& value) {
		_top.write(_shape, value);
		return _top;
	}

	template<typename T>
	Tensor& item(const T& value = T()) {
		_top.write(_shape, value);
		return _top;
	}

	template<typename T>
	T readScalar() const {
		return _top.readScalar<T>(_shape);
	}

	template<typename T>
	std::span<const T> read() const {
		return _top.read<T>(_shape);
	}

	mutable Shape _shape;
	Tensor& _top;
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

		Shape shape;
		if (_data.hasView()) {
			auto currentShape = _data.getCurrentShape();
			shape = Shape(currentShape.begin(), currentShape.end());
		}
		else {
			shape = _meta.shape;
		}

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

		_data.loadData(indexShape(shape, false), _meta.typeSize, std::move(bytes));
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
    void Tensor::write(const Shape& path, const T& data) {
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

	template<typename T>
	Tensor& Tensor::expand(const Tensor::Shape& targetShape, const T& fillData) {
		_data.expand(indexShape(targetShape, false), fillData);
		return *this;
	}

	inline Tensor& Tensor::crop(const Shape& targetShape) {
		_data.crop(indexShape(targetShape, false));
		return *this;
	}
	
}