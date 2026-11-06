#pragma once

#include "EngineRegistry.h"

namespace DC::Builtin {

/// @brief 注册内置 CPU 张量算子 Shape、Gather、Unsqueeze、Slice、Sub、Pad（ONNX-13 语义，参数即张量输入）。
void registerBuiltinTensorOps(EngineRegistry& reg = EngineRegistry::instance());

/// @brief 注册内置 CPU 算子：标量 Add、Mul、Identity，以及张量算子 Shape、Gather、Unsqueeze、Slice、Sub、Pad。
///        无外部引擎依赖。
void registerBuiltinOperators(EngineRegistry& reg = EngineRegistry::instance());

} // namespace DC::Builtin
