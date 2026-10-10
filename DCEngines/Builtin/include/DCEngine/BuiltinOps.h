#pragma once

#include "EngineRegistry.h"

namespace DC::Builtin {

/// @brief 注册内置 CPU 算子：Add / Mul / Identity（仅 Float 标量 Tensor，无外部引擎依赖）。
void registerBuiltinOperators(EngineRegistry& reg = EngineRegistry::instance());

} // namespace DC::Builtin
