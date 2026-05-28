#pragma once

#include <functional>
#include <memory>
#include <utility>

#include "SlotType.h"
#include "DCtype.h"
#include "SlotType.h"

namespace DC {

/// @brief 原生张量包装类：move-only 类型擦除容器，管理引擎原生张量的所有权与析构。
///
/// 封装引擎原生张量（Ort::Value / nvinfer1::ITensor* / DC::Tensor / …）的
/// 所有权与析构逻辑。move-only 设计兼容 GPU 资源句柄。
///
/// 构造时自动从模板参数 T 推导 SlotDataType 标签，
/// 用于 TensorSlot::store() 中的校验路由。
///
/// 用法：
/// @code
///   DC::Value v(std::make_unique<Tensor>(TensorType::Float, sizeof(float)));
///   DC::Value v(std::unique_ptr<Ort::Value>(new Ort::Value(...)));  // 自定义 deleter
/// @endcode
class Value {
public:
	/// @brief 默认构造：空 Value。
	Value() = default;

	/// @brief 主构造函数：从 unique_ptr 接管所有权（推荐方式）。
	/// @tparam T       原生张量类型。
	/// @tparam Deleter unique_ptr 的删除器类型。
	/// @param ptr      独占所有权指针。
	template <typename T, typename Deleter>
	Value(std::unique_ptr<T, Deleter> ptr) {
		ValidatorRegistry::ensureDefaults();
		_innerType = DC::Type::getType<SlotDataType, T>();
		auto d = ptr.get_deleter(); // 在 release 前拷贝 deleter
		_ptr = ptr.release();
		_deleter = [d = std::move(d)](void* p) {
			if (p)
				d(static_cast<T*>(p));
		};
	}

	/// @brief 兼容构造函数：从原始指针 + 自定义删除器接管所有权（C API 场景）。
	/// @tparam T       原生张量类型。
	/// @tparam Deleter 删除器类型。
	/// @param ptr      原始指针。
	/// @param deleter  自定义删除器。
	template <typename T, typename Deleter>
	Value(T* ptr, Deleter&& deleter) : _ptr(ptr) {
		ValidatorRegistry::ensureDefaults();
		_innerType = DC::Type::getType<SlotDataType, T>();
		_deleter = [d = std::forward<Deleter>(deleter)](void* p) {
			if (p)
				d(static_cast<T*>(p));
		};
	}

	/// @brief 析构：若持有数据则调用自定义删除器。
	~Value() {
		if (_ptr && _deleter)
			_deleter(_ptr);
	}

	/// @brief 移动构造。
	Value(Value&& other) noexcept
		: _ptr(std::exchange(other._ptr, nullptr)), _innerType(other._innerType), _deleter(std::move(other._deleter)) {}

	/// @brief 移动赋值。
	Value& operator=(Value&& other) noexcept {
		if (this != &other) {
			if (_ptr && _deleter)
				_deleter(_ptr);
			_ptr = std::exchange(other._ptr, nullptr);
			_innerType = other._innerType;
			_deleter = std::move(other._deleter);
		}
		return *this;
	}

	/// @brief 禁止拷贝。
	Value(const Value&) = delete;
	Value& operator=(const Value&) = delete;

	/// @brief  返回内部原生张量的实际 SlotDataType 标签，用于校验路由。
	SlotDataType innerType() const {
		return _innerType;
	}

	/// @brief  转换为具体类型指针。
	/// @tparam T 目标原生张量类型。
	/// @return 类型化指针（调用者自行确保类型正确）。
	template <typename T>
	T* as() {
		return static_cast<T*>(_ptr);
	}
	template <typename T>
	const T* as() const {
		return static_cast<const T*>(_ptr);
	}

	/// @brief  获取原始 void* 指针。
	void* get() {
		return _ptr;
	}
	const void* get() const {
		return _ptr;
	}

	/// @brief  是否持有有效数据。
	explicit operator bool() const {
		return _ptr != nullptr;
	}

private:
	void* _ptr = nullptr; ///< 原始指针（类型擦除）。
	SlotDataType _innerType = SlotDataType::Unknown; ///< 内部数据类型的 SlotDataType 标签。
	std::function<void(void*)> _deleter; ///< 自定义删除器。
};

} // namespace DC
