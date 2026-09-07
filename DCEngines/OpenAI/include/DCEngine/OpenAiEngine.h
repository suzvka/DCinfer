#pragma once

#include "EngineRegistry.h"

#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace DC::OpenAI {

/// @brief OpenAI 兼容远端引擎配置（注册级，注册时固化）。
struct OpenAiOptions {
	/// 请求体 model 字段（服务端标识模型）。
	std::string model = "default";

	/// 注册的引擎类型名。
	std::string engineType = "OpenAI";

	// ── 鉴权与附加头（注册级固化；敏感信息不写入日志与错误信息）──

	/// Bearer Token / API Key：注入 Authorization 头。裸 key 自动补 "Bearer "
	/// 前缀（已含 "Bearer " 前缀或需自定义头时，用 headers 显式给出）。
	std::string authToken;

	/// 动态取 token（注册时求值一次，结果同 authToken 语义；非空时优先于 authToken）。
	/// 注意：v1 不做运行时轮换，需轮换的场景请在应用层定期重新注册引擎。
	std::function<std::string()> tokenProvider;

	/// 附加请求头（"Name: value"；覆盖同名默认头）。
	std::vector<std::string> headers;

	// ── 超时与重试（0/负值 = 沿用 DCNet 默认：连接 5s / 请求 30s / 不重试）──
	std::chrono::milliseconds connectTimeout{0};
	std::chrono::milliseconds requestTimeout{0};
	int maxRetries = 0; ///< 传输级失败（超时/拒连/5xx/429）重试次数
};

/// @brief 注册 OpenAI 兼容远端服务引擎到引擎注册表。
///
/// 基于 DCNet 传输框架：HttpTransport（POCO，跨平台）+ OpenAI 兼容 chat codec
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
///   params（Data，可选，请求级采样参数 JSON，逐请求覆盖；非法 JSON 报 InvalidInput）
///   → out response（Data；响应缺 choices[0].message.content 报 RemoteMalformed）
/// - 请求路径：{basePath}/chat/completions（OpenAI 兼容协议面）
/// - 节点归属 ThreadPoolAffinity::System（I/O 池）；失败经
///   Node::Result + NodeStatus + ErrorTracker 诊断，图级语义与本地引擎一致
///
/// @param reg 目标注册表，默认为全局单例 EngineRegistry::instance()
/// @param opts 注册级配置（model / engineType / 鉴权 / 超时 / 重试）
void registerOpenAiEngine(EngineRegistry& reg = EngineRegistry::instance(), const OpenAiOptions& opts = {});

} // namespace DC::OpenAI
