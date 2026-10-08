#pragma once

#include <chrono>
#include <string>
#include <functional>
#include <cstddef>

namespace DC::Net {

/// @brief 服务端监听端点配置（M-server；DESIGN.md §3.6）。
///
/// 出站 NetEndpoint 的服务端镜像：出站结构为请求导向，无服务端证书 /
/// backlog / 连接数配置，故独立成结构而非复用（ADR-7）。
/// TLS 服务端证书（mTLS）暂不支持。
struct NetServerEndpoint {
	// ── 监听地址 ──
	std::string listenHost = "127.0.0.1"; ///< 仅允许解析后为回环地址；远端部署须 TLS 代理回环上游
	int port = 0;                         ///< 监听端口（0 = 随机端口，bind 后经 port() 回读）
	std::string basePath = "/v1";         ///< 协议基路径（与出站 NetEndpoint::basePath 对称）

	// ── 协议子路径 ──
	std::string requestPath = "/infer";   ///< 由 server codec 注入（镜像出站 createEngine 装配）

	// ── 鉴权（可选；仅 Bearer token，mTLS 后置）──
	std::string authToken;                ///< 非空时启用 Authorization 校验（裸 key 或 "Bearer xxx"）

	// ── 并发与过载（两级闸门，DESIGN.md §6.1）──
	int backlog = 16;                     ///< listen backlog
	int maxConnections = 32;              ///< 并发连接（工作线程）上限；0 = 不限制。accept 期即生效，
	                                      ///< 约束"已接受但请求尚未读完"的连接（含慢速/半开连接），
	                                      ///< 未通过请求头闸门的连接不计入 maxInFlight，故需独立上限
	int maxInFlight = 8;                  ///< 已授权请求（含正文读取）上限；超出立即 wire 429

	std::size_t maxRequestBody = 8u * 1024u * 1024u; ///< 单请求正文硬上限（非零）
	std::size_t maxBufferedBodyBytes = 32u * 1024u * 1024u; ///< 聚合声明正文预留上限（非零）
	/// Controlled local sink receives correlation IDs and event labels, not exception/request text.
	std::function<void(const std::string&)> diagnosticSink;

	// ── 超时 ──
	std::chrono::milliseconds requestTimeout{30000}; ///< 单请求处理预算（读超时 + 排水等待上限）
};

} // namespace DC::Net
