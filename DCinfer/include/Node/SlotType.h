#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "TensorMeta.h"

namespace DC {

/// @brief 槽位数据类型标签。
///
/// 框架内部自动递增分配，用户不直接指定值。
/// 通过 ensureSlotType<T>() 获取类型 T 对应的唯一标签。
using SlotDataType = uint32_t;

/// @brief 未分配的类型标签。
inline constexpr SlotDataType SlotDataTypeUnknown = 0;

namespace detail {
inline std::atomic<SlotDataType>& slotTypeCounter() {
	static std::atomic<SlotDataType> counter{1};
	return counter;
}
} // namespace detail

/// @brief 获取类型 T 对应的 SlotDataType 标签（自动分配，线程安全）。
///
/// 首次调用时自动分配递增编号，后续调用返回同一编号。
/// 框架内部类型（DC::Tensor、DC::Value）在 ValidatorRegistry::ensureDefaults() 中自动分配。
/// 引擎适配器开发者直接调用此函数即可获取自定义类型的标签，无需手动指定编号。
template <typename T>
SlotDataType ensureSlotType() {
	static SlotDataType id = detail::slotTypeCounter().fetch_add(1, std::memory_order_relaxed);
	return id;
}

/// @brief 槽位数据校验结果。
///
/// 表示 store() 时 ValidatorRegistry 的校验结论。
/// ready() 返回 true 表示数据可直接写入槽位。
struct SlotDataStatus {
	bool needAlign = false; ///< 需要形状对齐
	bool needConvert = false; ///< 需要类型转换
	bool invalid = false; ///< 数据无效

	/// @brief 数据是否可直接写入（无需对齐/转换且非无效）。
	bool ready() const {
		return !invalid && !needAlign && !needConvert;
	}
};

/// @brief 槽位校验函数签名。
/// @param data 指向实际存储的 void*（调用方保证生命周期）。
/// @param type 数据类型标签。
/// @param rule 槽位的元规则（期望类型、形状等）。
using SlotCheckFn = std::function<SlotDataStatus(const void* data, SlotDataType type, const TensorMeta& rule)>;

/// @brief 校验器注册表：管理 SlotDataType → SlotCheckFn 映射。
///
/// 引擎注册时通过 registerValidator() 注册校验逻辑。
/// TensorSlot::store() 调用 validate() 执行运行时校验。
/// 未注册类型直接放行（返回 ready=true）。
///
/// 并发契约：registerValidator 与 find/validate 之间以 _mutex 保护容器访问，
/// 约定“启动期注册、运行期并发读取”：引擎注册均在启动期完成，
/// 运行期多方并发调用 validate() 安全。
///
/// 用法：
///   auto id = ensureSlotType<MyType>();
///   ValidatorRegistry::instance().registerValidator(id, myCheckFn);
class ValidatorRegistry {
public:
	/// @brief  获取全局单例。
	static ValidatorRegistry& instance();

	/// @brief  确保默认类型映射与校验器已注册（std::call_once 保证只执行一次）。
	static void ensureDefaults();

	/// @brief  注册校验器（通常在引擎注册时调用）。
	/// @param type 目标 SlotDataType。
	/// @param fn   校验函数。
	void registerValidator(SlotDataType type, SlotCheckFn fn);

	/// @brief  查找校验器。
	/// @return 校验函数指针，未注册返回 nullptr。
	const SlotCheckFn* find(SlotDataType type) const;

	/// @brief  执行校验：未注册类型直接放行（返回 ready=true）。
	/// @param data 数据指针。
	/// @param type 数据类型标签。
	/// @param rule 元数据规则。
	/// @return 校验结果。
	SlotDataStatus validate(const void* data, SlotDataType type, const TensorMeta& rule) const;

private:
	ValidatorRegistry() = default;
	mutable std::mutex _mutex;
	std::unordered_map<SlotDataType, SlotCheckFn> _validators;
};

/// @brief Value 载荷深拷贝函数签名：输入为载荷指针（类型由注册时的 SlotDataType 决定）。
/// @return 深拷贝结果（裸 shared_ptr<void>，保留原始类型的删除器）。
using ValueCloneFn = std::function<std::shared_ptr<void>(const void*)>;

/// @brief Value 克隆注册表：SlotDataType → 深拷贝函数。
///
/// 用途：共享/冻结载荷在 API 边界（用户 take 产物、显式 clone）需要产出
/// 独占可变副本时调用。未注册类型的共享载荷只能只读消费（clone 请求将报错）。
///
/// 并发契约与 ValidatorRegistry 相同：启动期注册、运行期并发读取
/// （unordered_map 节点地址稳定，find 返回的指针在运行期持续有效）。
class ValueCloneRegistry {
public:
	/// @brief  获取全局单例。
	static ValueCloneRegistry& instance();

	/// @brief  注册克隆函数（通常在引擎/类型注册时调用）。
	/// @param type 目标 SlotDataType。
	/// @param fn   深拷贝函数。
	void registerClone(SlotDataType type, ValueCloneFn fn);

	/// @brief  查找克隆函数。
	/// @return 克隆函数指针，未注册返回 nullptr。
	const ValueCloneFn* find(SlotDataType type) const;

private:
	ValueCloneRegistry() = default;
	mutable std::mutex _mutex;
	std::unordered_map<SlotDataType, ValueCloneFn> _clones;
};

} // namespace DC
