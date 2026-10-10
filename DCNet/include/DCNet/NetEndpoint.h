#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace DC::Net {

/// url 优先；为空时由 host/port/basePath 组合。
///
/// modelPath 语义重载：EngineRegistry::createNode(engineType, name, modelPath)
/// 的 modelPath 即端点描述串（URL 或 host[:port][/path]），经 parse() 填充本结构。
struct NetEndpoint {
	std::string url;
	std::string host = "127.0.0.1";
	int port = 0;
	std::string basePath = "/v1";
	bool useTls = false;

	std::string contentType = "application/json";
	std::string requestPath;

	std::chrono::milliseconds connectTimeout{5000};
	std::chrono::milliseconds requestTimeout{30000};
	int maxRetries = 0;

	/// 响应体大小上限（字节）；超限的成功响应按错误归一化。0 = 不限制。
	/// 默认 512 MiB：防恶意远端致客户端无界分配。
	size_t maxResponseBody = 512u * 1024u * 1024u;

	bool allowInsecureCredentials = false; ///< 仅限开发的明文凭据显式许可
	std::string authToken;
	std::vector<std::string> headers;

	/// 支持格式："http(s)://host:port/path" 或 "host[:port][/path]"。
	static NetEndpoint parse(const std::string& text);

	std::string endpoint() const;
};

} // namespace DC::Net
