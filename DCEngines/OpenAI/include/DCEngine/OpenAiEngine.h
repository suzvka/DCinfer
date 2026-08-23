#pragma once

#include "EngineRegistry.h"

#include <string>

namespace DC::OpenAI {

/// @brief OpenAI 兼容远端引擎配置（注册级，注册时固化）。
struct OpenAiOptions {
	/// 请求体 model 字段（服务端标识模型）。
	std::string model = "default";

	/// 注册的引擎类型名。
	std::string engineType = "OpenAI";
};

/// @brief 注册 OpenAI 兼容远端服务引擎到引擎注册表。
///
/// 基于 DCNet 传输框架：HttpTransport（WinHTTP）+ OpenAI 兼容 chat codec
/// （NetCodec 契约实现），错误经 NetError 归一化出口上报。
///
/// 注册后可通过以下方式创建远端节点：
/// @code
///   auto node = EngineRegistry::instance().createNode("OpenAI", "llm", "http://host:port/v1");
/// @endcode
///
/// 引擎特性：
/// - createNode(engineType, name, modelPath)：modelPath 即远端端点（URL），
///   连接失败在配置期抛 NodeException（携带 NetError 归一化消息）
/// - 端口（本地形状规则）：in prompt（Data，必填）/ system（Data，可选）/
///   params（Data，可选，请求级采样参数 JSON，逐请求覆盖）→ out response（Data）
/// - 请求路径：{basePath}/chat/completions（OpenAI 兼容协议面）
/// - 节点归属 ThreadPoolAffinity::System（I/O 池）；失败经
///   Node::Result + NodeStatus + ErrorTracker 诊断，图级语义与本地引擎一致
///
/// @param reg 目标注册表，默认为全局单例 EngineRegistry::instance()
/// @param opts 注册级配置（model / engineType）
void registerOpenAiEngine(EngineRegistry& reg = EngineRegistry::instance(), const OpenAiOptions& opts = {});

} // namespace DC::OpenAI
