#pragma once

#include <memory>
#include <string>
#include <utility>

#include "NodeException.h"
#include "SlotType.h"

namespace DC {

/// @brief 原生张量包装类：move-only 类型擦除容器，管理引擎原生张量的所有权与析构。
///
/// 内部以 shared_ptr<void> 承载载荷并保留自定义 deleter：move 为唯一所有权转移；
/// share 产生共享只读别名，引用计数加一且零拷贝；构造时从模板参数 T 推导 SlotDataType
/// 标签，用于 TensorSlot::store 的校验路由。
class Value {
public:
	Value() = default;

	/// @brief 从 unique_ptr 接管所有权，推荐方式。
	template <typename T, typename Deleter>
	Value(std::unique_ptr<T, Deleter> ptr) {
		ValidatorRegistry::ensureDefaults();
		_innerType = ensureSlotType<T>();
		if (ptr) {
			auto d = ptr.get_deleter(); // 须在 release 前拷贝 deleter
			_ptr = std::shared_ptr<void>(ptr.release(), std::move(d));
		}
	}

	/// @brief 从原始指针与自定义删除器接管所有权，用于 C API 场景。
	template <typename T, typename Deleter>
	Value(T* ptr, Deleter&& deleter) {
		ValidatorRegistry::ensureDefaults();
		_innerType = ensureSlotType<T>();
		if (ptr)
			_ptr = std::shared_ptr<void>(ptr, std::forward<Deleter>(deleter));
	}

	~Value() = default;

	Value(Value&& other) noexcept = default;
	Value& operator=(Value&& other) noexcept = default;

	// 禁止拷贝：显式共享用 share。
	Value(const Value&) = delete;
	Value& operator=(const Value&) = delete;

	/// @brief 返回内部原生张量的 SlotDataType 标签。
	SlotDataType innerType() const {
		return _innerType;
	}

	/// @brief 转换为具体类型指针；调用者自行确保类型正确。
	template <typename T>
	T* as() {
		return static_cast<T*>(_ptr.get());
	}
	template <typename T>
	const T* as() const {
		return static_cast<const T*>(_ptr.get());
	}

	void* get() {
		return _ptr.get();
	}
	const void* get() const {
		return _ptr.get();
	}

	explicit operator bool() const {
		return _ptr != nullptr;
	}

	/// @brief 产生共享只读别名；零拷贝，发布标记粘性，见 isPublished。
	Value share() const {
		_published = true;
		Value v;
		v._innerType = _innerType;
		v._ptr = _ptr;
		v._published = true;
		return v;
	}

	/// @brief 载荷是否被多处引用。
	bool isShared() const {
		return _ptr && _ptr.use_count() > 1;
	}

	/// @brief 载荷处于或曾处于共享发布状态，具粘性；发布过的载荷可能为冻结只读态，
	///        需要可变所有权用 cloneOwned。
	bool isPublished() const {
		return _published || isShared();
	}

	/// @brief 当前引用计数；空 Value 返回 0。
	long useCount() const {
		return _ptr ? _ptr.use_count() : 0;
	}

	/// @brief 深度克隆：经 ValueCloneRegistry 复制载荷，产出独立可变副本。
	///        未注册深拷贝的类型如 GPU 句柄抛 NodeException，只能只读消费。
	Value cloneOwned() const {
		if (!_ptr)
			return {};
		const auto* fn = ValueCloneRegistry::instance().find(_innerType);
		if (!fn) {
			throw NodeException(NodeException::ErrorType::Other, "Value::cloneOwned",
								"no clone registered for slot type " + std::to_string(_innerType) +
									"; shared payload of this type is read-only");
		}
		Value v;
		v._innerType = _innerType;
		v._ptr = (*fn)(_ptr.get());
		return v;
	}

private:
	std::shared_ptr<void> _ptr;
	SlotDataType _innerType = SlotDataTypeUnknown;
	mutable bool _published = false; // 粘性发布标记，随 move 流转
};

} // namespace DC
