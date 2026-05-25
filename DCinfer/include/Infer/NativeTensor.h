#pragma once

#include <functional>
#include <utility>

#include "SlotType.h"
#include "DCtype.h"
#include "SlotType.h"

namespace DC {

// ── 原生张量包装类 ──
// 封装引擎原生张量（Ort::Value / nvinfer1::ITensor* / …）的所有权与析构逻辑。
// move-only，避免对可拷贝类型的强制要求，兼容 move-only 的 GPU 资源句柄。
//
// 构造时自动从模板参数 T 推导 SlotDataType 标签，
// 用于 TensorSlot::store() 中的校验路由。
class Value {
public:
	Value() = default;

	// 通用构造函数：接受任意可调用对象作为 deleter，避免 std::function 导致模板推导失败
	template<typename T, typename Deleter>
	Value(T* ptr, Deleter&& deleter)
		: _ptr(ptr)
  {
		// Ensure default validators and type registrations exist before querying type map
		ValidatorRegistry::ensureDefaults();
		_innerType = DC::Type::getType<SlotDataType, T>();
		_deleter = [d = std::forward<Deleter>(deleter)](void* p) {
			if (p) d(static_cast<T*>(p));
		};
	}


	~Value() { if (_ptr && _deleter) _deleter(_ptr); }

	// move-only
	Value(Value&& other) noexcept
		: _ptr(std::exchange(other._ptr, nullptr))
		, _innerType(other._innerType)
		, _deleter(std::move(other._deleter)) {}

	Value& operator=(Value&& other) noexcept {
		if (this != &other) {
			if (_ptr && _deleter) _deleter(_ptr);
			_ptr = std::exchange(other._ptr, nullptr);
			_innerType = other._innerType;
			_deleter = std::move(other._deleter);
		}
		return *this;
	}

	Value(const Value&) = delete;
	Value& operator=(const Value&) = delete;

	/// @brief 返回内部原生张量的实际 SlotDataType 标签，用于校验路由
	SlotDataType innerType() const { return _innerType; }

	template<typename T> T*       as()       { return static_cast<T*>(_ptr); }
	template<typename T> const T* as() const { return static_cast<const T*>(_ptr); }

	void*       get()       { return _ptr; }
	const void* get() const { return _ptr; }

	explicit operator bool() const { return _ptr != nullptr; }

private:
	void* _ptr = nullptr;
	SlotDataType _innerType = SlotDataType::Unknown;
	std::function<void(void*)> _deleter;
};

} // namespace DC
