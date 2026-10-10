#pragma once

#include "EngineRegistry.h"

#include <functional>

namespace DC::Onnx {

struct OnnxOptions {
	int intraOpThreads = 1;

	/// 自定义 SessionOptions 钩子；参数需 static_cast 为 Ort::SessionOptions*。
	std::function<void(void*)> sessionCustomizer;
};

/// @brief 注册 ONNX Runtime 引擎到引擎注册表。
///
/// 引擎核心（共享 Ort::Env）每 engineType 初始化一次，Session 按 modelPath 缓存；
/// createNode 自动确保核心就绪并加载模型，schema 由实例推导；
/// FP16 挂 Float 类型族，BFLOAT16/STRING 降级为 Void 并告警；
/// 默认 ExecutionProvider 由构建选项 DCINFER_ORT_EP 决定，sessionCustomizer 可覆盖。
void registerOnnxEngine(EngineRegistry& reg = EngineRegistry::instance(), const OnnxOptions& opts = {});

} // namespace DC::Onnx
