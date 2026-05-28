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
// DC::Tensor 栈占用：248 字节。
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	
	/// @brief 推理框架的核心张量对象，作为输入/输出数据的统一载体。
	/// @brief 提供类型安全的创建、索引访问、形状变换和数据序列化能力。
	///
	/// @par 典型用法
	/// - 构造推理图的输入张量
	
	class Tensor {
	public:
		using TensorType = TensorMeta::TensorType;  ///< 张量逻辑类型枚举（Float/Int/Uint/Bool/Char/Data/Void）。
		using ErrorType = TensorException::ErrorType; ///< 错误类型枚举。
		using DataBlock = TensorData::DataBlock;     ///< 原始字节块，即 std::vector<std::byte>。
		using Shape = std::vector<int64_t>;          ///< 形状向量，支持负索引表示。

		virtual ~Tensor() = default;
		class View;

		/// @brief 默认构造一个 Void 类型的空张量。
		Tensor();

		/// @brief 指定类型、元素大小、形状和可选初始数据构造张量。
		/// @param type 张量逻辑类型。
		/// @param typeSize 单元素字节数（0 则自动推导）。
		/// @param shape 初始形状（空向量表示标量）。
		/// @param data 可选的初始稠密字节数据（移动语义）。
		Tensor(
			const TensorType& type,
			size_t typeSize = 0,
			const Shape& shape = {},
			DataBlock&& data = {}
		);

		/// @brief 工厂方法：从 C++ 类型 T 推导逻辑类型和元素大小创建张量。
		/// @tparam T 元素类型（如 float、int32_t），必须已在类型映射中注册。
		/// @param shape 初始形状（空向量表示标量）。
		/// @param data 可选的初始稠密字节数据（移动语义）。
		/// @return 构造完成的 Tensor 对象。
		/// @code
		/// auto t = Tensor::Create<float>({2, 3}, std::move(bytes));
		/// auto s = Tensor::Create<int32_t>(); // 标量
		/// @endcode
		template<typename T>
		static Tensor Create(
			const Shape& shape = {},
			DataBlock&& data = {}
		);

		/// @brief 设置张量名称（用于日志和错误定位）。
		/// @return 自身引用，支持链式调用。
		Tensor& setName(const std::string& name);

		/// @brief 索引访问，返回 View 代理对象支持链式索引和数据读写。
		/// @param index 维度索引，支持负数（从末尾倒序）。
		/// @return View 代理对象。
		View operator[](int64_t index) const;

		/// @brief 获取当前张量的顶层视图（空路径）。
		View view();

		/// @brief 将张量作为标量读取。要求张量为 0-D 标量。
		/// @tparam T 期望的 C++ 类型，其 sizeof(T) 必须等于 typeSize()。
		/// @return 标量值的副本。
		template<typename T>
		T item() const;

		/// @brief 拷贝构造：深拷贝元数据和内部数据。
		Tensor(const Tensor& other);

		/// @brief 移动构造：接管 other 的资源，other 变为可析构的空状态。
		Tensor(Tensor&& other) noexcept;

		/// @brief 拷贝赋值：深拷贝元数据和内部数据。
		Tensor& operator=(const Tensor& other);

		/// @brief 移动赋值：接管 other 的资源，other 变为可析构的空状态。
		Tensor& operator=(Tensor&& other) noexcept;

		/// @brief 标量赋值：无索引直接赋值时，将当前张量设置为 0-D 标量。
		/// @tparam T 要写入的标量类型。sizeof(T) 必须等于 typeSize()。
		/// @param value 要写入的标量值。
		/// @return 自身引用。
		/// @throws TensorException(TypeMismatch) 若类型大小不匹配。
		/// @code
		/// Tensor s = Tensor::Create<float>();
		/// s = 3.14f;  // 设置标量值为 3.14
		/// @endcode
		template<typename T>
		Tensor& operator=(const T& value);

		/// @brief 用指定的值填充整个张量的所有元素。
		/// @tparam T 填充值的类型。
		/// @param value 要填充的值。
		/// @return 自身引用。
		/// @throws TensorException(TypeMismatch) 若类型大小不匹配。
		/// @code
		/// auto t = Tensor::Create<int32_t>({2, 2});
		/// t.fill<int32_t>(7);  // 所有元素变为 7
		/// @endcode
		template<typename T>
		Tensor& fill(const T& value);

		/// @brief 获取张量的逻辑类型标签。
		TensorType type() const;

		/// @brief 获取单元素字节数。
		size_t typeSize() const;

		/// @brief 获取当前张量的动态形状（CurrentShape）。
		/// @details 根据当前实际数据的维度信息计算形状；与 RuleShape 可能不同。
		Shape shape() const;

		/// @brief 以类型 T 的只读视图访问底层稠密数据。
		/// @tparam T 期望的元素类型。sizeof(T) 必须是 typeSize() 的约数。
		/// @return 包含稠密连续数据的只读 span。若无数据则返回空 span。
		/// @throws std::invalid_argument 若类型大小不匹配。
		/// @note 若当前为稀疏视图模式，会自动触发稠密缓存构建。
		/// @code
		/// auto t = Tensor::Create<float>({2, 3});
		/// // ... 填充数据 ...
		/// auto span = t.data<float>();
		/// for (float v : span) { ... }
		/// @endcode
		template<typename T>
		std::span<const T> data() const;

		/// @brief 以原始字节的只读视图访问底层数据。
		/// @return 包含稠密连续字节的只读 span。
		std::span<const std::byte> bytes() const;

		/// @brief 直接加载外部稠密数据，避免逐块登记的开销。
		/// @param data 要移动进来的原始字节块。
		/// @param shape 对应的元素形状。
		/// @return 自身引用。
		/// @note 适用于推理输出等已是稠密数据的场景。
		Tensor& loadData(DataBlock&& data, const Shape& shape);

		/// @brief 将张量扩展至目标形状，新增区域用 fillData 填充。
		/// @tparam T 填充数据类型。
		/// @param targetShape 目标形状（每维必须 >= 当前维度大小）。
		/// @param fillData 新增区域的填充值（默认 T{}）。
		/// @return 自身引用。
		/// @throws std::invalid_argument 若 targetShape 的任一维小于当前维。
		/// @note 已有数据保持不变，仅对缺失的块写入填充值。
		template<typename T>
		Tensor& expand(const Shape& targetShape, const T& fillData = T());

		/// @brief 将张量裁剪至目标形状（沿每维截取前缀）。
		/// @param targetShape 目标形状（每维必须 <= 当前维度大小）。
		/// @return 自身引用。
		/// @throws std::invalid_argument 若 targetShape 的任一维大于当前维或秩不匹配。
		/// @note 裁剪会重新调整稠密缓存大小，丢弃超出部分的数据。
		Tensor& crop(const Shape& targetShape);

		/// @brief 消费式取出内部缓存数据（移动所有权后内部缓存清空）。
		/// @tparam T 期望的元素类型。
		/// @return 数据副本（std::vector<T>）。
		/// @note 取出后缓存被清空，hasCache() 将返回 false。
		template<typename T>
		std::vector<T> getData();

		/// @brief 查询是否为 0-D 标量张量。
		bool isScalar() const { return _data.isScalar(); }

		/// @brief 查询是否已初始化且无数据内容。
		bool empty() const { return _data.empty(); }

		/// @brief 查询是否包含有效数据（至少具备稀疏视图或稠密缓存之一）。
		bool valid() const { return _data.valid(); }

		/// @brief 查询稠密缓存是否已构建。
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

/// @brief 张量索引视图代理，支持链式索引和多维数据的读写。
///
/// View 是 [Tensor::operator[]](#) 的返回类型，通过链式调用累积索引路径，
/// 最终通过赋值或读取操作将数据写入张量或从张量读取。
/// 支持负索引（-1 表示最后一维）。
///
/// @par 典型用法
/// @code
/// t[0] = std::vector<float>{1.0f, 2.0f, 3.0f};  // 整行写入
/// t[1][2].set(99.0f);                              // 单元素写入
/// float v = t[1][2].readScalar<float>();            // 单元素读取
/// auto row = t[0].read<float>();                    // 整行读取
/// @endcode
class Tensor::View {
public:
    /// @brief 从可变张量构造视图。
    View(Shape&& shape, Tensor& top):_shape(std::move(shape)), _top(top){}

    /// @brief 从常量张量构造视图（内部通过 const_cast 支持 const View 读取）。
	View(Shape&& shape, const Tensor& top) :_shape(std::move(shape)), _top(const_cast<Tensor&>(top)) {}

    /// @brief 继续索引下一维，返回新的 View 节点以支持链式调用。
    /// @param index 维度索引，支持负数（从末尾倒序）。
    /// @return 包含扩展路径的新 View 对象。
    View operator[](int64_t index) const {
		_shape.push_back(index);
        return View(std::move(_shape), _top);
    }

	/// @brief 通过 View 向张量写入值（等价于 set(value)）。
	/// @tparam T 要写入的数据类型。
	/// @param value 要写入的值。
	/// @return 顶层张量引用。
	template<typename T>
	Tensor& operator=(const T& value) {
		set(value);
		return _top;
	}

	/// @brief 通过 View 向张量写入值。
	/// @tparam T 要写入的数据类型。
	/// @param value 要写入的值（标量或向量）。
	/// @return 顶层张量引用。
	template<typename T>
	Tensor& set(const T& value) {
		_top.write(_shape, value);
		return _top;
	}

	/// @brief 通过 View 向张量写入元素值。
	/// @tparam T 要写入的数据类型。
	/// @param value 要写入的值。
	/// @return 顶层张量引用。
	template<typename T>
	Tensor& item(const T& value = T()) {
		_top.write(_shape, value);
		return _top;
	}

	/// @brief 从 View 读取标量值。
	/// @tparam T 期望的 C++ 类型。
	/// @return 标量值的副本。
	template<typename T>
	T readScalar() const {
		return _top.readScalar<T>(_shape);
	}

	/// @brief 从 View 读取数据块（行或子张量）的只读 span。
	/// @tparam T 期望的元素类型。
	/// @return 数据的只读 span。
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