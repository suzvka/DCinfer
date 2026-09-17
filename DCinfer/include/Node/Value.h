#pragma once

#include <memory>
#include <string>
#include <utility>

#include "NodeException.h"
#include "SlotType.h"

namespace DC {

/// @brief 原生张量包装类：move-only 类型擦除容器，管理引擎原生张量的所有权与析构。
///
/// 封装引擎原生张量（Ort::Value / nvinfer1::ITensor* / DC::Tensor / …）的
/// 所有权与析构逻辑。move-only 设计兼容 GPU 资源句柄。
///
/// 内部采用 shared_ptr<void> 承载载荷（保留自定义 deleter）：move 为唯一
/// 所有权转移（零开销）；share() 产生共享只读别名（引用计数 +1，零拷贝）。
/// 引用计数仅在显式 share() 时产生，独占投递路径无任何开销。
///
/// 构造时自动从模板参数 T 推导 SlotDataType 标签，
/// 用于 TensorSlot::store() 中的校验路由。
///
/// 用法：
/// @code
///   DC::Value v(std::make_unique<Tensor>(TensorType::Float, sizeof(float)));
///   DC::Value v(std::unique_ptr<Ort::Value>(new Ort::Value(...)));  // 自定义 deleter
///   DC::Value alias = v.share();  // 共享只读别名（零拷贝）
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
		_innerType = ensureSlotType<T>();
		if (ptr) {
			auto d = ptr.get_deleter(); // 在 release 前拷贝 deleter
			_ptr = std::shared_ptr<void>(ptr.release(), std::move(d));
		}
	}

	/// @brief 接管构造函数：从原始指针 + 自定义删除器接管所有权（C API 场景）。
	/// @tparam T       原生张量类型。
	/// @tparam Deleter 删除器类型。
	/// @param ptr      原始指针。
	/// @param deleter  自定义删除器。
	template <typename T, typename Deleter>
	Value(T* ptr, Deleter&& deleter) {
		ValidatorRegistry::ensureDefaults();
		_innerType = ensureSlotType<T>();
		if (ptr)
			_ptr = std::shared_ptr<void>(ptr, std::forward<Deleter>(deleter));
	}

	/// @brief 析构：shared_ptr 自动调用删除器（引用计数归零时）。
	~Value() = default;

	/// @brief 移动构造：载荷所有权转移（零开销）。
	Value(Value&& other) noexcept = default;

	/// @brief 移动赋值：旧载荷按引用计数释放，接管新载荷。
	Value& operator=(Value&& other) noexcept = default;

	/// @brief 禁止拷贝（copy 会隐式共享载荷，与独占语义冲突；显式共享用 share()）。
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
		return static_cast<T*>(_ptr.get());
	}
	template <typename T>
	const T* as() const {
		return static_cast<const T*>(_ptr.get());
	}

	/// @brief  获取原始 void* 指针。
	void* get() {
		return _ptr.get();
	}
	const void* get() const {
		return _ptr.get();
	}

	/// @brief  是否持有有效数据。
	explicit operator bool() const {
		return _ptr != nullptr;
	}

	/// @brief  产生共享只读别名：与 *this 指向同一载荷（引用计数 +1，零拷贝）。
	///        别名生命周期独立延长载荷；不产生 deep copy；发布标记为粘性——
	///        源句柄与全部别名从此恒视为"已发布"（isPublished）。
	Value share() const {
		_published = true; // 粘性发布标记：源句柄（载荷曾被发布）也保持标记
		Value v;
		v._innerType = _innerType;
		v._ptr = _ptr;
		v._published = true;
		return v;
	}

	/// @brief  载荷是否被多处引用（引用计数 > 1，即经 share() 产生过别名）。
	bool isShared() const {
		return _ptr && _ptr.use_count() > 1;
	}

	/// @brief  载荷处于（或曾处于）共享发布状态：本句柄源自 share() 或
	///        曾调用过 share()（粘性标记，别名消亡后仍保持），或载荷当前
	///        仍被多处引用。发布过的载荷可能为冻结只读态——出口需要可变
	///        所有权时应先 cloneOwned()（take 语义）。
	bool isPublished() const {
		return _published || isShared();
	}

	/// @brief  当前引用计数（空 Value 返回 0；诊断/测试用）。
	long useCount() const {
		return _ptr ? _ptr.use_count() : 0;
	}

	/// @brief  深度克隆：经 ValueCloneRegistry 复制载荷，产出独立可变副本。
	///
	/// 对共享/冻结载荷需要独占可变所有权时使用（如用户 take 产物）。
	/// 未注册深拷贝函数（如 GPU 句柄类）则抛出 NodeException——
	/// 此类型共享载荷只能只读消费，无法产出独占副本。
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
	std::shared_ptr<void> _ptr; ///< 类型擦除载荷（引用计数共享；保留原始类型删除器）。
	SlotDataType _innerType = SlotDataTypeUnknown; ///< 内部数据类型的标签。
	mutable bool _published = false; ///< 粘性发布标记：句柄经 share() 发布过（随 move 流转）
};

} // namespace DC
