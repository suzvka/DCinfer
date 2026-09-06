#pragma once

#include <chrono>
#include <string>

namespace DC::Net {

/// @brief 服务端监听端点配置（M-server；DESIGN.md §3.6）。
///
/// 出站 NetEndpoint 的服务端镜像：出站结构为请求导向，无服务端证书 /
/// backlog / 连接数配置，故独立成结构而非复用（ADR-7）。
/// TLS 服务端证书配置（mTLS）：占位，随 M-server 后续设计补齐。
struct NetServerEndpoint {
	// ── 监听地址 ──
	std::string listenHost = "127.0.0.1"; ///< 监听地址（"0.0.0.0" 对外开放）
	int port = 0;                         ///< 监听端口（0 = 随机端口，bind 后经 port() 回读）
	std::string basePath = "/v1";         ///< 协议基路径（与出站 NetEndpoint::basePath 对称）

	// ── 协议子路径 ──
	std::string requestPath = "/infer";   ///< 由 server codec 注入（镜像出站 createEngine 装配）

	// ── 鉴权（可选；仅 Bearer token，mTLS 后置）──
	std::string authToken;                ///< 非空时启用 Authorization 校验（裸 key 或 "Bearer xxx"）

	// ── 并发与过载 ──
	int backlog = 16;                     ///< listen backlog
	int maxInFlight = 8;                  ///< 在途请求上限；超出立即 wire 429（不排队、不静默丢弃）

	// ── 超时 ──
	std::chrono::milliseconds requestTimeout{30000}; ///< 单请求处理预算（读超时 + 排水等待上限）
};

} // namespace DC::Net
