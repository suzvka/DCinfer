#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "TensorMeta.h"

namespace DC {

/// @brief 槽位数据类型标签，经 ensureSlotType<T>() 自动分配唯一值。
using SlotDataType = uint32_t;

/// @brief 未分配的类型标签。
inline constexpr SlotDataType SlotDataTypeUnknown = 0;

namespace detail {
inline std::atomic<SlotDataType>& slotTypeCounter() {
	static std::atomic<SlotDataType> counter{1};
	return counter;
}
} // namespace detail

/// @brief 获取类型 T 的唯一 SlotDataType 标签；首次调用分配，线程安全。
template <typename T>
SlotDataType ensureSlotType() {
	static SlotDataType id = detail::slotTypeCounter().fetch_add(1, std::memory_order_relaxed);
	return id;
}

/// @brief 槽位数据校验结果，即 store 时 ValidatorRegistry 的结论。
struct SlotDataStatus {
	bool needAlign = false;
	bool needConvert = false;
	bool invalid = false;

	/// @brief 数据是否可直接写入。
	bool ready() const {
		return !invalid && !needAlign && !needConvert;
	}
};

/// @brief 槽位校验函数签名。
using SlotCheckFn = std::function<SlotDataStatus(const void* data, SlotDataType type, const TensorMeta& rule)>;

/// @brief 校验器注册表：SlotDataType 到 SlotCheckFn 的映射，TensorSlot::store 经 validate 调用。
///
/// 并发契约：启动期注册、运行期并发读取，容器访问以 _mutex 保护。
/// 未注册类型直接放行。
class ValidatorRegistry {
public:
	static ValidatorRegistry& instance();

	/// @brief 注册默认类型映射与校验器，std::call_once 只执行一次。
	static void ensureDefaults();

	/// @brief 注册校验器，启动期调用。
	void registerValidator(SlotDataType type, SlotCheckFn fn);

	/// @brief 查找校验器；未注册返回 nullptr。
	const SlotCheckFn* find(SlotDataType type) const;

	/// @brief 执行校验；未注册类型直接放行。
	SlotDataStatus validate(const void* data, SlotDataType type, const TensorMeta& rule) const;

private:
	ValidatorRegistry() = default;
	mutable std::mutex _mutex;
	std::unordered_map<SlotDataType, SlotCheckFn> _validators;
};

/// @brief Value 载荷深拷贝函数签名：输入载荷指针，返回保留原始删除器的 shared_ptr<void>。
using ValueCloneFn = std::function<std::shared_ptr<void>(const void*)>;

/// @brief Value 克隆注册表：SlotDataType 到深拷贝函数的映射，供共享或冻结载荷产出独占副本。
///
/// 未注册类型的共享载荷只能只读消费。并发契约同 ValidatorRegistry：启动期注册、运行期并发读取。
class ValueCloneRegistry {
public:
	static ValueCloneRegistry& instance();

	/// @brief 注册克隆函数，启动期调用。
	void registerClone(SlotDataType type, ValueCloneFn fn);

	/// @brief 查找克隆函数；未注册返回 nullptr。
	const ValueCloneFn* find(SlotDataType type) const;

private:
	ValueCloneRegistry() = default;
	mutable std::mutex _mutex;
	std::unordered_map<SlotDataType, ValueCloneFn> _clones;
};

} // namespace DC
