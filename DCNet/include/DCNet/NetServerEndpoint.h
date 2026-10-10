#pragma once

#include <chrono>
#include <string>
#include <functional>
#include <cstddef>

namespace DC::Net {

/// 服务端监听端点配置：出站 NetEndpoint 的服务端镜像，独立成结构；
/// TLS 服务端证书（mTLS）暂不支持。
struct NetServerEndpoint {
	std::string listenHost = "127.0.0.1"; ///< 仅回环地址；远端部署须 TLS 代理回环上游
	int port = 0;                         ///< 0 = 随机端口，bind 后经 port() 回读
	std::string basePath = "/v1";

	std::string requestPath = "/infer";   ///< 由 server codec 注入

	std::string authToken;                ///< 非空时启用 Authorization 校验

	int backlog = 16;
	int maxConnections = 32;              ///< 并发连接上限（accept 期生效，含未过闸门连接）；0 = 不限制
	int maxInFlight = 8;                  ///< 已授权请求（含正文读取）上限；超出 wire 429

	std::size_t maxRequestBody = 8u * 1024u * 1024u; ///< 单请求正文硬上限（非零）
	std::size_t maxBufferedBodyBytes = 32u * 1024u * 1024u; ///< 聚合声明正文预留上限（非零）
	/// Controlled local sink receives correlation IDs and event labels, not exception/request text.
	std::function<void(const std::string&)> diagnosticSink;

	std::chrono::milliseconds requestTimeout{30000}; ///< 单请求处理预算（读超时 + 排水等待上限）
};

} // namespace DC::Net
