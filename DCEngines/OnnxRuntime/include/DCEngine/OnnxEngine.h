#pragma once

#include "EngineRegistry.h"

namespace DC::Onnx {

/// @brief 注册 ONNX Runtime 引擎到引擎注册表。
///
/// 注册后可通过以下方式创建推理节点：
/// @code
///   auto node = EngineRegistry::instance().createNode("OnnxRuntime", "myNode", "model.onnx");
/// @endcode
///
/// 引擎特性：
/// - 从 .onnx 模型文件自动推导输入/输出端口 Schema
/// - 支持 CPU 和 GPU Execution Provider（通过 SessionOptions 配置）
/// - 算子内部单线程（SetIntraOpNumThreads(1)），与 DCinfer 图级并行模型一致
///
/// @param reg 目标注册表，默认为全局单例 EngineRegistry::instance()
void registerOnnxEngine(EngineRegistry& reg = EngineRegistry::instance());

} // namespace DC::Onnx
