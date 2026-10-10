#pragma once

#include "Node.h"

#include <string>

namespace DC::Net {

/// 归一化错误分类。
enum class NetErrorCategory {
	None,
	Timeout,
	Unreachable,
	RemoteRejected,
	RemoteAuth,
	RemoteRateLimited,
	RemoteServer,
	RemoteMalformed,
	Other,
};

/// 传输层错误原语；transport 填入，核心映射为 NetErrorCategory。
enum class NetTransportError {
	None,
	Timeout,
	ConnectionRefused,
	DnsFailed,
	Reset,
	TlsFailed,
	Other,
};

/// 归一化中间结构：适配器填 category、code、retryable 与 remoteDetail；
/// localStatus 与 localMessage 由 finalize 统一计算。
struct NetError {
	NetErrorCategory category = NetErrorCategory::None;
	std::string code;
	bool retryable = false;
	std::string remoteDetail;
	Node::Status localStatus = Node::Status::Ok;
	std::string localMessage;    ///< 形如 "net:timeout - <detail>"
	Diagnostic diagnostic;       ///< domain="dcnet"，code=NetErrorCategory 原值

	bool ok() const noexcept { return category == NetErrorCategory::None; }
};

/// 传输层错误归一化；detail 附加 errno 字符串或报文摘要等。
NetError normalizeTransportError(NetTransportError err, std::string detail = {});

/// HTTP 非 2xx 状态码归一化；2xx 由调用方先行判定。
NetError normalizeHttpStatus(int status, std::string body = {});

/// 远端错误报文归一化：支持 OpenAI 风格 {"error":{code,message}} 等；
/// 已知 code 精确映射，未知按 fallback 兜底。
NetError normalizeRemoteBody(const std::string& body,
							 NetErrorCategory fallback = NetErrorCategory::RemoteRejected);

/// HTTP 状态与报文组合归一化：报文中的已知 code 优先于状态码。
NetError normalizeHttpResponse(int status, const std::string& body);

/// 由已填字段计算 localStatus 与 localMessage；上述入口内部均已调用。
NetError finalize(NetError e);

/// 本地执行结果状态映射到 wire HTTP 状态码，保证对端归一化结果与本地 status 一致。
/// 非鉴权 InternalError 无忠实 wire 表示，按 5xx 应答；401/403/429/415
/// 不经本映射，由监听与装配层直接应答。
int wireHttpStatusFor(Node::Status status);

/// 服务端 wire 应答错误体 code，用于诊断细化；未知 code 仅使对端消息带 remote:<code> 前缀。
const char* wireCodeFor(Node::Status status);

} // namespace DC::Net
