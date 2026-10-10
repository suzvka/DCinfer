#pragma once
#include <algorithm>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <optional>
#include "TensorMods.h"

namespace DC {

/// @brief 推理框架的核心张量对象：输入输出数据的统一载体，支持创建、索引、形状变换与数据读写。
class Tensor {
public:
	using TensorType = TensorMeta::TensorType;
	using ErrorType = TensorException::ErrorType;
	using DataBlock = TensorData::DataBlock;
	using Shape = std::vector<int64_t>;

	virtual ~Tensor() = default;
	class View;
	class ConstView;

	/// @brief 默认构造一个 Void 类型的空张量。
	Tensor();

	/// @brief 构造张量；typeSize=0 时自动推导，空 shape 表示标量。
	Tensor(const TensorType& type, size_t typeSize = 0, const Shape& shape = {}, DataBlock&& data = {});

	/// @brief 工厂方法：从 C++ 类型 T 推导逻辑类型与元素大小。
	template <typename T>
	static Tensor Create(const Shape& shape = {}, DataBlock&& data = {});

	/// @brief 设置张量名称，供日志与错误定位。
	Tensor& setName(const std::string& name);

	/// @brief 索引访问，非 const：返回可写 View 代理，负索引从末尾倒序。
	View operator[](int64_t index);

	/// @brief 索引访问，const：返回只读 ConstView，编译期禁止修改。
	ConstView operator[](int64_t index) const;

	/// @brief 获取顶层视图，即空路径。
	View view();

	/// @brief 以标量读取，要求 0-D 标量。
	template <typename T>
	T item() const;

	Tensor(const Tensor& other);

	Tensor(Tensor&& other) noexcept;

	Tensor& operator=(const Tensor& other);

	Tensor& operator=(Tensor&& other) noexcept;

	/// @brief 标量赋值：将当前张量设置为 0-D 标量；sizeof(T) 不匹配抛 TypeMismatch。
	template <typename T>
	Tensor& operator=(const T& value);

	/// @brief 用指定值填充整个张量；sizeof(T) 不匹配抛 TypeMismatch。
	template <typename T>
	Tensor& fill(const T& value);

	TensorType type() const;

	size_t typeSize() const;

	/// @brief 获取当前动态形状，由实际数据维度计算，与 RuleShape 可能不同。
	Shape shape() const;

	/// @brief 以类型 T 的只读视图访问底层稠密数据；稀疏模式自动物化缓存。
	template <typename T>
	std::span<const T> data() const;

	/// @brief 以原始字节只读视图访问底层数据。
	std::span<const std::byte> bytes() const;

	/// @brief 直接加载外部稠密数据，避免逐块登记开销。
	Tensor& loadData(DataBlock&& data, const Shape& shape);

	/// @brief 扩展至目标形状，每维不小于当前；已有数据不变，新增区域填 fillData。
	template <typename T>
	Tensor& expand(const Shape& targetShape, const T& fillData = T());

	/// @brief 裁剪至目标形状，沿每维截取前缀；每维不大于当前，秩须一致。
	Tensor& crop(const Shape& targetShape);

	/// @brief 消费式取出内部缓存，取出后清空；字节按 T 重解释，末元素零填充。
	template <typename T>
	std::vector<T> getData();

	bool isScalar() const {
		return _data.isScalar();
	}

	bool empty() const {
		return _data.empty();
	}

	bool valid() const {
		return _data.valid();
	}

	bool hasCache() const {
		return _data.hasCache();
	}

	/// @brief 冻结：预物化稠密缓存并置冻结位；冻结后写路径抛 TensorException(Frozen)，只读不再触发惰性物化。
	void freeze() {
		_data.freeze();
	}

	/// @brief 是否已冻结。
	bool isFrozen() const {
		return _data.isFrozen();
	}

private:
	TensorMeta _meta;
	TensorData _data;

	std::optional<ErrorType> checkTypeMatch(size_t size) const;

	std::optional<ErrorType> checkPathValid(const Shape& path, const TensorData::Shape& shape) const;

	std::optional<ErrorType> checkSingleElementView(const Shape& path, const Shape& shape) const;

	template <typename T>
	void write(const Shape& path, const std::vector<T>& data);

	/// @brief 写标量，支持广播到单元素子视图；错误经 abort 抛出。
	template <typename T>
	void write(const Shape& path, const T& data);

	template <typename T>
	std::span<const T> read(const Shape& path) const;

	/// @brief 读标量，支持单元素子视图广播；错误经 abort 抛出。
	template <typename T>
	T readScalar(const Shape& path) const;

	void moveFrom(Tensor&& other) noexcept;

	TensorData::Shape indexShape(const Shape& shape, bool isRead) const;

	void abort(ErrorType errorType = ErrorType::Other, const std::string& message = "") const;
};

/// @brief 张量索引视图代理：链式索引累积路径，赋值或读取时落到张量，-1 表示最后一维。
class Tensor::View {
public:
	View(Shape&& shape, Tensor& top) : _shape(std::move(shape)), _top(top) {}

	/// @brief 继续索引下一维，返回新 View，可与源视图分叉使用。
	View operator[](int64_t index) const {
		Shape next = _shape;
		next.push_back(index);
		return View(std::move(next), _top);
	}

	/// @brief 写入值，等价于 set。
	template <typename T>
	Tensor& operator=(const T& value) {
		set(value);
		return _top;
	}

	/// @brief 写入标量或向量。
	template <typename T>
	Tensor& set(const T& value) {
		_top.write(_shape, value);
		return _top;
	}

	/// @brief 读取标量值。
	template <typename T>
	T readScalar() const {
		return _top.readScalar<T>(_shape);
	}

	/// @brief 读取行或子张量数据块的只读 span。
	template <typename T>
	std::span<const T> read() const {
		return _top.read<T>(_shape);
	}

	Shape _shape;
	Tensor& _top;
};

/// @brief 张量只读索引视图代理：仅支持链式索引与读取，无任何写入口。
class Tensor::ConstView {
public:
	ConstView(Shape&& shape, const Tensor& top) : _shape(std::move(shape)), _top(top) {}

	/// @brief 继续索引下一维。
	ConstView operator[](int64_t index) const {
		Shape next = _shape;
		next.push_back(index);
		return ConstView(std::move(next), _top);
	}

	/// @brief 读取标量值。
	template <typename T>
	T readScalar() const {
		return _top.readScalar<T>(_shape);
	}

	/// @brief 读取行或子张量数据块的只读 span。
	template <typename T>
	std::span<const T> read() const {
		return _top.read<T>(_shape);
	}

	Shape _shape;
	const Tensor& _top;
};

template <typename T>
Tensor Tensor::Create(const Shape& shape, DataBlock&& data) {
	TensorMeta::ensureTypeMap();
	Tensor tensor(DC::Type::getType<TensorType, T>(), sizeof(T), shape, std::move(data));

	return tensor;
}

template <typename T>
T Tensor::item() const {
	return _data.readElement<T>({});
}

template <typename T>
Tensor& Tensor::operator=(const T& value) {
	if (auto err = checkTypeMatch(sizeof(T))) {
		abort(*err, "type mismatch in scalar assignment");
	}

	// 优先复用稠密缓存写入路径；无缓存则直接装入标量字节。
	DataBlock bytes(_meta.typeSize);
	std::fill(bytes.begin(), bytes.end(), std::byte(0));
	std::memcpy(bytes.data(), &value, std::min(sizeof(T), _meta.typeSize));
	if (_data.hasCache()) {
		if (!_data.writeCacheElement({}, value)) {
			abort(ErrorType::Other, "failed to write scalar into dense cache");
		}
		_data.setScalar(true);
		return *this;
	}
	_data.loadData({}, _meta.typeSize, std::move(bytes));
	_data.setScalar(true);
	return *this;
}

template <typename T>
Tensor& Tensor::fill(const T& data) {
	if (auto err = checkTypeMatch(sizeof(T))) {
		abort(*err, "type mismatch in fill assignment");
	}

	Shape shape;
	if (_data.hasView()) {
		auto currentShape = _data.getCurrentShape();
		shape = Shape(currentShape.begin(), currentShape.end());
	} else {
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

template <typename T>
inline std::span<const T> Tensor::data() const {
	return _data.data<T>();
}

template <typename T>
void Tensor::write(const Shape& path, const std::vector<T>& data) {
	try {
		_data.editMode();
		_data.write(indexShape(path, false), data);
	} catch (const TensorException& e) {
		abort(e.getErrorType(), e.what());
	} catch (const std::exception& e) {
		abort(ErrorType::Other, e.what());
	}
}

template <typename T>
void Tensor::write(const Shape& path, const T& data) {
	try {
		_data.editMode();
		_data.write(indexShape(path, false), data);
	} catch (const TensorException& e) {
		abort(e.getErrorType(), e.what());
	} catch (const std::exception& e) {
		abort(ErrorType::Other, e.what());
	}
}

template <typename T>
std::span<const T> Tensor::read(const Shape& path) const {
	return _data.read<T>(indexShape(path, true));
}

template <typename T>
T Tensor::readScalar(const Shape& path) const {
	auto dataShape = _data.getCurrentShape();
	if (auto err = checkTypeMatch(sizeof(T)))
		abort(*err, "type mismatch in scalar read");
	if (auto err = checkPathValid(path, dataShape))
		abort(*err, "invalid path in scalar read");

	if (path.size() == dataShape.size()) {
		auto full = indexShape(path, true);
		return _data.readElement<T>(full);
	}

	// 路径短于秩：校验剩余维度乘积为 1，即单元素视图
	Shape asTensorShape(dataShape.begin(), dataShape.end());
	if (auto err = checkSingleElementView(path, asTensorShape))
		abort(*err, "not a scalar view");

	Shape fullPath = path;
	fullPath.insert(fullPath.end(), dataShape.size() - path.size(), 0);
	auto full = indexShape(fullPath, true);
	return _data.readElement<T>(full);
}

template <typename T>
std::vector<T> Tensor::getData() {
	static_assert(std::is_trivially_copyable_v<T>, "Tensor::getData requires trivially copyable type");
	auto data = _data.getData();
	const size_t bytes = data.size();
	// 元素数 = ceil(总字节 / sizeof(T))；全部字节进入结果，末元素零填充。
	std::vector<T> result((bytes + sizeof(T) - 1) / sizeof(T));
	if (bytes > 0)
		std::memcpy(result.data(), data.data(), bytes);
	return result;
}

template <typename T>
Tensor& Tensor::expand(const Tensor::Shape& targetShape, const T& fillData) {
	_data.expand(indexShape(targetShape, false), fillData);
	return *this;
}

inline Tensor& Tensor::crop(const Shape& targetShape) {
	_data.crop(indexShape(targetShape, false));
	return *this;
}

} // namespace DC