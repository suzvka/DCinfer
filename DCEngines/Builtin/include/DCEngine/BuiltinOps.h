#pragma once

#include "EngineRegistry.h"

namespace DC::Builtin {

/// @brief 注册所有内置 CPU 算子到引擎注册表。
///
/// 注册的算子：
/// - "Add"      两个 Float 标量相加（输入 a, b → 输出 sum）
/// - "Mul"      两个 Float 标量相乘（输入 a, b → 输出 product）
/// - "Identity" 恒等映射（输入 x → 输出 y）
///
/// 所有算子仅支持 DC::Tensor（Float 标量），不依赖任何外部推理引擎。
/// 注册后可通过 EngineRegistry::createOperator("Add", "node1") 创建节点。
///
/// @param reg 目标注册表，默认为全局单例 EngineRegistry::instance()
void registerBuiltinOperators(EngineRegistry& reg = EngineRegistry::instance());

} // namespace DC::Builtin
