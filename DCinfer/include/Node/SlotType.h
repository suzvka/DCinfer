#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>

#include "TensorMeta.h"

namespace DC {

/// @brief 槽位数据类型枚举。
///
/// DC::Type 注册 T → SlotDataType 映射，store<T>() 时自动推导。
/// 引擎原生类型预留从 100 起始的编号段。
enum class SlotDataType : uint32_t {
	Unknown      = 0,    ///< 未知类型
	DCTensor     = 1,    ///< DC::Tensor
	Value = 2,           ///< DC::Value（原生张量包装）
	// 引擎原生类型预留从 100 开始
	ONNX_OrtValue   = 100,  ///< Ort::Value（ONNX Runtime）
	// TensorRT_ITensor = 200,
	UserDefined  = 1000, ///< 用户自定义类型起始编号
};

/// @brief 槽位数据校验结果。
///
/// 表示 store() 时 ValidatorRegistry 的校验结论。
/// ready() 返回 true 表示数据可直接写入槽位。
struct SlotDataStatus {
	bool needAlign   = false;   ///< 需要形状对齐
	bool needConvert = false;   ///< 需要类型转换
	bool invalid     = false;   ///< 数据无效

	/// @brief 数据是否可直接写入（无需对齐/转换且非无效）。
	bool ready() const {
		return !invalid && !needAlign && !needConvert;
	}
};

/// @brief 槽位校验函数签名。
/// @param data 指向实际存储的 void*（调用方保证生命周期）。
/// @param type 数据类型枚举。
/// @param rule 槽位的元规则（期望类型、形状等）。
using SlotCheckFn = std::function<SlotDataStatus(
	const void*       data,
	SlotDataType      type,
	const TensorMeta& rule
)>;

/// @brief 校验器注册表：管理 SlotDataType → SlotCheckFn 映射。
///
/// 引擎注册时通过 registerValidator() 注册校验逻辑。
/// TensorSlot::store() 调用 validate() 执行运行时校验。
/// 未注册类型直接放行（返回 ready=true）。
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
	SlotDataStatus validate(const void*       data,
	                        SlotDataType      type,
	                        const TensorMeta& rule) const;

private:
	ValidatorRegistry() = default;
	std::unordered_map<SlotDataType, SlotCheckFn> _validators;
};

} // namespace DC
