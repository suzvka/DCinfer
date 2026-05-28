#pragma once

#include <functional>
#include <memory>
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

	// ── 主构造函数：从 unique_ptr 接管所有权（推荐，无需手动 new）──
	// 用法：
	//   DC::Value v(std::make_unique<Tensor>(TensorType::Float, sizeof(float)));
	//   DC::Value v(std::unique_ptr<Ort::Value>(new Ort::Value(...)));  // 自定义 deleter
	template<typename T, typename Deleter>
	Value(std::unique_ptr<T, Deleter> ptr) {
		ValidatorRegistry::ensureDefaults();
		_innerType = DC::Type::getType<SlotDataType, T>();
		auto d = ptr.get_deleter();  // 在 release 前拷贝 deleter
		_ptr = ptr.release();
		_deleter = [d = std::move(d)](void* p) {
			if (p) d(static_cast<T*>(p));
		};
	}

	// ── 兼容构造函数（保留，用于 C API 原始指针场景）──
	template<typename T, typename Deleter>
	Value(T* ptr, Deleter&& deleter)
		: _ptr(ptr)
	{
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
