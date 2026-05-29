#pragma once
#include <type_traits>
#include <stdexcept>
#include <optional>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "Tensor.hpp"
#include "Exception.h"
#include "DCtype.h"
#include "SlotType.h"
#include "Value.h"
#include <iostream>

namespace DC {

/// @brief 张量数据槽位：类型擦除的运行时数据容器，支持 store/take/peek。
///
/// TensorSlot 是 Node 输入/输出端口的基础存储单元。
/// 通过 ValidatorRegistry 在 store() 时自动完成类型校验。
class TensorSlot {
	using TensorType = TensorMeta::TensorType;
	using ErrorType = TensorException::ErrorType;

public:
	using Shape = Tensor::Shape;

	/// @brief 同节点内所有槽位的映射表，供 DefaultProvider 查询锚定张量。
	using SlotMap = std::unordered_map<std::string, TensorSlot>;

	/// @brief 懒求值默认值工厂：当槽位无显式数据、无静态默认值时调用。
	/// @param peers 同节点所有输入槽位（含自身），用于锚定形状等动态推导。
	/// @return 要写入槽位的 Tensor，返回 nullptr 表示无法解析。
	using DefaultProvider = std::function<std::unique_ptr<Tensor>(const SlotMap& peers)>;

	/// @brief 槽位配置：仅含位置标记（Input / Output / Auto）。
	class Config {
	public:
		/// @brief 槽位位置枚举。
		enum class Position {
			Input, ///< 输入槽位
			Output, ///< 输出槽位
			Auto ///< 自动推断
		};

		/// @brief  设置槽位位置。
		Config& setPosition(Position p);

		Position position = Position::Auto; ///< 当前位置标记。
	};

	// ── 生命周期 ──
	TensorSlot(const TensorSlot&) = delete;
	TensorSlot& operator=(const TensorSlot&) = delete;
	TensorSlot(TensorSlot&&) noexcept = default;
	TensorSlot& operator=(TensorSlot&&) noexcept = default;

	/// @brief 构造 TensorSlot。
	/// @param name   槽位名称（对应端口名）。
	/// @param type   期望的张量逻辑类型。
	/// @param size   单元素字节数。
	/// @param shape  期望的形状（空=不校验）。
	/// @param config 槽位配置（默认 Auto）。
	TensorSlot(const std::string& name, TensorMeta::TensorType type, size_t size, const Shape& shape,
			   const Config& config = Config());

	// ── 元数据 ──

	/// @brief  设置默认张量数据（输入槽位 fallback）。
	TensorSlot& setDefaultTensor(const Tensor& data);

	/// @brief  设置懒求值默认值工厂（输入槽位 fallback，优先级低于显式数据和静态默认值）。
	///         可用于运行时根据锚定端口动态确定形状的零张量等场景。
	TensorSlot& setDefaultProvider(DefaultProvider fn);

	/// @brief  若槽位无运行时数据且存在 DefaultProvider，调用之并将结果 store 到槽位。
	/// @param peers 同节点所有输入槽位（用于形状锚定等跨槽查询）。
	void resolveDefaultIfNeeded(const SlotMap& peers);

	/// @brief  槽位名称。
	const std::string& name() const;
	/// @brief  期望的张量逻辑类型。
	TensorType type() const;
	/// @brief  单元素字节数。
	size_t typeSize() const;
	/// @brief  期望形状。
	Shape shape() const;
	/// @brief  运行时数据的实际形状（若为 DCTensor）。
	Shape dataShape() const;

	/// @brief  是否为输入槽位。
	bool isInput() const;
	/// @brief  是否为输出槽位。
	bool isOutput() const;

	/// @brief  是否已设置默认数据。
	bool hasDefaultData() const;
	/// @brief  获取默认数据的只读引用。
	/// @throws TensorException(NotData) 若无默认数据。
	const Tensor& defaultTensor() const;

	/// @brief  槽位的张量逻辑类型是否与 T 匹配。
	template <typename T>
	bool isType() const;

	// ── 运行时存储：类型擦除 ──

	/// @brief  类型擦除存储：通过 DC::Type 推导 SlotDataType，经 ValidatorRegistry 校验后存储。
	/// @tparam T 要存储的数据类型（自动推导 SlotDataType 标签）。
	/// @throws TensorException(TypeMismatch)     若类型不匹配且不可转换。
	/// @throws TensorException(InvalidShape)      若数据无效。
	/// @throws TensorException(ShapeMismatch)     若形状不匹配且不可对齐。
	template <typename T>
	TensorSlot& store(T&& data);

	/// @brief  移动取出数据，运行时检查类型标签是否匹配。
	/// @throws TensorException(NotData)       若槽位为空。
	/// @throws TensorException(TypeMismatch)  若类型标签不匹配。
	template <typename T>
	T take();

	/// @brief  只读指针访问，类型不匹配返回 nullptr。
	template <typename T>
	const T* peek() const;

	/// @brief  以 const Tensor& 获取 DC::Tensor 数据（仅 DCTensor 类型有效）。
	/// @throws TensorException(NotData) 若无数据。
	const Tensor& view() const;

	/// @brief  槽位是否有数据（含默认数据）。
	bool hasData() const;
	/// @brief  当前存储数据的 SlotDataType 标签。
	SlotDataType storedType() const;
	/// @brief  原始数据指针（不担保类型）。
	const void* rawPtr() const;

	/// @brief  清空运行时数据与默认数据。
	void clear();
	/// @brief  仅清空运行时数据，保留默认数据。
	void clearData();

	/// @brief  获取槽位配置的只读引用。
	const Config& config() const;
	/// @brief  工厂：创建默认 Config。
	static Config CreateConfig();

private:
	// ── 类型擦除存储 ──
	struct TypedBlob {
		void* ptr = nullptr;
		std::function<void(void*)> deleter;
		SlotDataType type = SlotDataType::Unknown;
	};

	TensorMeta _rule;
	std::unique_ptr<Tensor> _defaultData; // 默认数据（永远是 DC::Tensor）
	DefaultProvider _defaultProvider; // 懒求值默认值工厂（优先级低于 _defaultData）
	std::optional<TypedBlob> _blob; // 运行时数据
	Config _config;

	[[noreturn]] void abort(ErrorType errorType = ErrorType::Other, const std::string& message = "") const;
};

// Template method definitions
template <typename T>
bool TensorSlot::isType() const {
	return type() == Type::getType<TensorMeta::TensorType, T>();
}

template <typename T>
TensorSlot& TensorSlot::store(T&& data) {
	ValidatorRegistry::ensureDefaults(); // 保证默认注册已执行（std::call_once）
	auto typeEnum = DC::Type::getType<SlotDataType, std::decay_t<T>>();

	// 校验：始终按存储的实际类型进行校验
	auto status = ValidatorRegistry::instance().validate(std::addressof(data), typeEnum, _rule);

	// Diagnostic log
	try {
		std::cerr << "TensorSlot::store name='" << _rule.name << "' type=" << static_cast<int>(typeEnum)
				  << " status.ready=" << status.ready() << " invalid=" << status.invalid
				  << " needConvert=" << status.needConvert << " needAlign=" << status.needAlign << std::endl;
	} catch (...) {}

	if (!status.ready()) {
		if (status.invalid) {
			abort(ErrorType::InvalidShape, "Input data is invalid");
		}
		if (status.needConvert) {
			abort(ErrorType::TypeMismatch, "Type mismatch and conversion not allowed");
		}
		if (status.needAlign) {
			abort(ErrorType::ShapeMismatch, "Shape mismatch and alignment not allowed");
		}
	}

	// 释放旧数据
	if (_blob.has_value() && _blob->deleter && _blob->ptr) {
		_blob->deleter(_blob->ptr);
	}

	// 类型擦除存储
	TypedBlob blob;
	blob.type = typeEnum;
	blob.ptr = new std::decay_t<T>(std::forward<T>(data));
	blob.deleter = [](void* p) { delete static_cast<std::decay_t<T>*>(p); };
	_blob = std::move(blob);

	return *this;
}

template <typename T>
T TensorSlot::take() {
	if (!_blob.has_value() || !_blob->ptr) {
		abort(ErrorType::NotData, "Slot is empty");
	}

	auto expectedType = DC::Type::getType<SlotDataType, T>();
	if (_blob->type != expectedType) {
		abort(ErrorType::TypeMismatch,
			  "take<T>: type mismatch, stored=" + std::to_string(static_cast<uint32_t>(_blob->type)) +
				  " expected=" + std::to_string(static_cast<uint32_t>(expectedType)));
	}

	auto* typed = static_cast<T*>(_blob->ptr);
	T result = std::move(*typed);

	// 释放存储（但不调用 deleter，因为已移动）
	typed->~T();
	operator delete(typed);
	_blob.reset();

	return result;
}

template <typename T>
const T* TensorSlot::peek() const {
	if (!_blob.has_value() || !_blob->ptr) {
		return nullptr;
	}

	auto expectedType = DC::Type::getType<SlotDataType, T>();
	if (_blob->type != expectedType) {
		return nullptr;
	}

	return static_cast<const T*>(_blob->ptr);
}
} // namespace DC