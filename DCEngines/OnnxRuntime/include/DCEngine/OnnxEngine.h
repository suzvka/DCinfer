#pragma once

#include "EngineRegistry.h"

#include <functional>

namespace DC::Onnx {

/// @brief ONNX Runtime 引擎配置（注册级，注册时固化）。
struct OnnxOptions {
	/// 算子内并行线程数（默认 1，与 DCinfer 图级并行模型一致）
	int intraOpThreads = 1;

	/// 可选：Ort::SessionOptions 自定义器。
	/// 典型用途：追加 ExecutionProvider（CUDA / DirectML / TensorRT 等）、
	/// 设置图优化级别、内存模式。
	/// 参数为 Ort::SessionOptions*（调用方需自行 include onnxruntime_cxx_api.h
	/// 并 static_cast），并需使用包含对应 ExecutionProvider 的 onnxruntime 构建。
	std::function<void(void*)> sessionCustomizer;
};

/// @brief 注册 ONNX Runtime 引擎到引擎注册表。
///
/// 注册后可通过以下方式创建推理节点：
/// @code
///   auto node = EngineRegistry::instance().createNode("OnnxRuntime", "myNode", "model.onnx");
/// @endcode
///
/// 引擎特性：
/// - createNode(engineType, name, modelPath) 自动从模型创建并缓存引擎实例，
///   并从实例推导输入/输出端口 Schema（模型仅加载一次）
/// - 张量转换经 TensorConverter 契约：DC::Tensor ↔ Ort::Value
/// - 未知 ONNX 元素类型（FLOAT16/BFLOAT16/STRING 等）显式降级为 Void 并告警
/// - 算子内线程数与 SessionOptions 自定义（EP 选择）通过 OnnxOptions 配置
/// - 编译期默认 EP：构建选项 DCINFER_ORT_EP（CPU/CUDA/TENSORRT/OPENVINO）
///   决定默认追加的 ExecutionProvider，用户无需写代码；
///   sessionCustomizer 仍可在默认 EP 之后追加或覆盖
///
/// @param reg 目标注册表，默认为全局单例 EngineRegistry::instance()
/// @param opts 注册级配置（线程数 / SessionOptions 自定义器）
void registerOnnxEngine(EngineRegistry& reg = EngineRegistry::instance(), const OnnxOptions& opts = {});

} // namespace DC::Onnx
