#pragma once

#include "Node.h"

#include <string>

namespace DC::Net {

/// @brief 网络错误分类（核心统一维护的归一化分类，DESIGN.md §3.3 / §6）。
enum class NetErrorCategory {
	None,               ///< 无错误（成功）
	Timeout,            ///< 超时（可重试）
	Unreachable,        ///< 连接拒绝 / DNS / 重置 / TLS（可重试）
	RemoteRejected,     ///< 远端拒绝请求（4xx 输入类问题，不可重试）
	RemoteAuth,         ///< 鉴权失败（401/403，配置类问题）
	RemoteRateLimited,  ///< 远端限流（429，可重试）
	RemoteServer,       ///< 远端故障（5xx，可重试）
	RemoteMalformed,    ///< 远端报文不可解析
	Other,              ///< 其他未分类
};

/// @brief 传输层错误原语（transport 填入，核心映射为 NetErrorCategory）。
enum class NetTransportError {
	None,
	Timeout,
	ConnectionRefused,
	DnsFailed,
	Reset,
	TlsFailed,
	Other,
};

/// @brief 归一化中间结构 + 本地标准报错出口。
///
/// 图级语义（localStatus / localMessage）由核心统一计算（finalize），
/// 适配器只负责填 category / code / retryable / remoteDetail——
/// 归一化原则：只做"翻译"，不做"发明"（DESIGN.md §3.3）。
struct NetError {
	NetErrorCategory category = NetErrorCategory::None;
	std::string code;            ///< 对方原始错误码（如 "invalid_api_key"）
	bool retryable = false;      ///< 超时 / 5xx / 429 → true
	std::string remoteDetail;    ///< 对方原始报文摘要（保留回溯现场）
	Node::Status localStatus = Node::Status::Ok;   ///< 归一化出口（图级语义）
	std::string localMessage;    ///< 形如 "net:timeout - <detail>"

	bool ok() const noexcept { return category == NetErrorCategory::None; }
};

// ── 归一化入口（纯函数，无 I/O，可单测）──

/// @brief 传输层错误归一化。
/// @param err    transport 填入的传输错误原语
/// @param detail 附加细节（errno 字符串、报文摘要等）
NetError normalizeTransportError(NetTransportError err, std::string detail = {});

/// @brief HTTP 非 2xx 状态码归一化（2xx 由调用方先行判定成功）。
NetError normalizeHttpStatus(int status, std::string body = {});

/// @brief 远端错误报文归一化。
/// 支持 OpenAI 风格 {"error":{code,message}} / {"error":"..."} / {"detail":"..."}；
/// 已知 code 精确映射（如 invalid_api_key → Auth），未知 code 按 fallback 兜底。
/// @param fallback 报文解析不出已知 code 时使用的类别
NetError normalizeRemoteBody(const std::string& body,
							 NetErrorCategory fallback = NetErrorCategory::RemoteRejected);

/// @brief 组合入口：HTTP 状态 + 报文 → 归一化结果。
/// 先解析报文中的已知 code（优先于状态码），再按状态码兜底。
NetError normalizeHttpResponse(int status, const std::string& body);

/// @brief 由已填 category/code/retryable/remoteDetail 计算 localStatus/localMessage。
/// 核心统一出口；上述入口函数内部均已调用。
NetError finalize(NetError e);

} // namespace DC::Net
