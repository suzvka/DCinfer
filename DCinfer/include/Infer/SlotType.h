#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>

#include "TensorMeta.h"

namespace DC {

// ── 槽位数据类型枚举 ──
// DC::Type 注册 T → SlotDataType 映射，store<T>() 时推导
enum class SlotDataType : uint32_t {
	Unknown      = 0,
	DCTensor     = 1,
	NativeTensor = 2,
	// 引擎原生类型预留从 100 开始
	ONNX_OrtValue   = 100,
	// TensorRT_ITensor = 200,
	UserDefined  = 1000,
};

// ── 槽位数据校验结果 ──
struct SlotDataStatus {
	bool needAlign   = false;   // 需要形状对齐
	bool needConvert = false;   // 需要类型转换
	bool invalid     = false;   // 数据无效

	bool ready() const {
		return !invalid && !needAlign && !needConvert;
	}
};

// ── 槽位校验函数签名 ──
// data: 指向实际存储的 void*（调用方保证生命周期）
// type: 数据类型枚举
// rule: 槽位的元规则（期望类型、形状等）
using SlotCheckFn = std::function<SlotDataStatus(
	const void*       data,
	SlotDataType      type,
	const TensorMeta& rule
)>;

// ── 校验器注册表 ──
class ValidatorRegistry {
public:
	static ValidatorRegistry& instance();

	// 确保默认类型映射与校验器已注册（std::call_once 保证只执行一次）
	static void ensureDefaults();

	// 注册校验器（通常在引擎注册时调用）
	void registerValidator(SlotDataType type, SlotCheckFn fn);

	// 查找校验器
	const SlotCheckFn* find(SlotDataType type) const;

	// 执行校验；未注册类型直接放行（返回 ready=true）
	SlotDataStatus validate(const void*       data,
	                        SlotDataType      type,
	                        const TensorMeta& rule) const;

private:
	ValidatorRegistry() = default;
	std::unordered_map<SlotDataType, SlotCheckFn> _validators;
};

} // namespace DC
