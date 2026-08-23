#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace DC::Net {

/// @brief 远端端点配置（DCNet 契约的一部分）。
///
/// modelPath 语义重载：EngineRegistry::createNode(engineType, name, modelPath)
/// 的 modelPath 即端点描述串（URL 或 host[:port][/path]），经 parse() 填充本结构。
struct NetEndpoint {
	// ── 地址 ──
	std::string url;                   ///< 完整地址（优先）；为空时由 host/port/basePath 组合
	std::string host = "127.0.0.1";
	int port = 0;
	std::string basePath = "/v1";
	bool useTls = false;

	// ── 请求 ──
	std::string contentType = "application/json"; ///< 请求 Content-Type
	std::string requestPath;           ///< 协议子路径（如 "/infer" / "/chat/completions"），由 codec 经适配器注入

	// ── 超时与重试 ──
	std::chrono::milliseconds connectTimeout{5000};
	std::chrono::milliseconds requestTimeout{30000};
	int maxRetries = 0;

	// ── 鉴权与附加头 ──
	std::string authToken;             ///< "Bearer xxx" 或裸 key，按协议注入请求头
	std::vector<std::string> headers;  ///< "Name: value"

	/// @brief 从端点描述串解析（支持 "http(s)://host:port/path" 与 "host[:port][/path]"）。
	static NetEndpoint parse(const std::string& text);

	/// @brief 组合可达地址（http[s]://host:port/basePath）。
	std::string endpoint() const;
};

} // namespace DC::Net
