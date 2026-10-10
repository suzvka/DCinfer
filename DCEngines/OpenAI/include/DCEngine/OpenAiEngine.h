#pragma once

#include "EngineRegistry.h"

#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace DC::OpenAI {

/// @brief OpenAI 兼容远端引擎配置（注册级固化）。
struct OpenAiOptions {
	/// 请求体 model 字段。
	std::string model = "default";

	std::string engineType = "OpenAI";

	// 鉴权与附加头（敏感信息不写入日志与错误信息）

	/// 注入 Authorization 头；裸 key 自动补 "Bearer " 前缀（自定义头用 headers）。
	std::string authToken;

	/// 动态取 token（注册时求值一次；非空时优先于 authToken）。
	/// v1 无运行时轮换：需轮换时由应用层定期重新注册引擎。
	std::function<std::string()> tokenProvider;

	/// 附加请求头（"Name: value"；覆盖同名默认头）。
	std::vector<std::string> headers;

	/// Development-only: permit credentials over HTTP. Keep false for production; use HTTPS.
	bool allowInsecureCredentials = false;

	// 超时与重试（0/负值 = 沿用 DCNet 默认：连接 5s / 请求 30s / 不重试）
	std::chrono::milliseconds connectTimeout{0};
	std::chrono::milliseconds requestTimeout{0};
	int maxRetries = 0; ///< 传输级失败（超时/拒连/5xx/429）重试次数
};

/// @brief 注册 OpenAI 兼容远端引擎到引擎注册表。
///
/// 基于 DCNet：HttpTransport（POCO）+ OpenAI 兼容 chat codec，错误经 NetError 归一化。
/// - createNode(engineType, name, modelPath)：modelPath 即远端端点（URL），
///   连接失败在配置期抛 NodeException
/// - 端口：in prompt（Data 必填）/ system / params（Data 可选，仅 temperature [0,2]、
///   top_p [0,1]、max_tokens [1,2147483647]、presence/frequency_penalty [-2,2]；
///   未知字段或越界报 InvalidInput）；out response（缺 choices[0].message.content
///   报 ExecutionFailed）
/// - 请求路径：{basePath}/chat/completions；节点归属 ResourceClass::System
void registerOpenAiEngine(EngineRegistry& reg = EngineRegistry::instance(), const OpenAiOptions& opts = {});

} // namespace DC::OpenAI
