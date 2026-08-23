#pragma once

#include "NetTransport.h"
#include "NetError.h"
#include "Node.h"

#include <string>

namespace DC::Net {

/// @brief 协议映射（DESIGN.md §3.2）：本地端口 ↔ 对方报文。
///
/// 错误提取/归一化**不属 codec 职责**：HTTP 场景由 transport 在 recv 时调用
/// 核心归一化函数（normalizeHttpResponse / normalizeRemoteBody）；
/// 协议特有错误码需要精确映射时，codec 内部调用同一组核心函数即可。
struct DcNetCodec {
	virtual ~DcNetCodec() = default;

	/// @brief 本地端口 → 对方请求报文（读 ctx.peek(port)，拼请求体）。
	virtual Payload encodeRequest(const Node::RunContext&) = 0;

	/// @brief 对方响应报文 → 本地端口（校验本地形状规则后 ctx.output）。
	virtual void decodeResponse(Payload&, Node::RunContext&) = 0;

	/// @brief 协议子路径（如 "/infer"），由适配器注入 NetEndpoint::requestPath。
	/// 默认空：POST 到端点 basePath 本身。
	virtual std::string requestPath() const { return {}; }

	/// @brief 本 codec 声明的本地形状规则（端口 Schema）。
	/// 默认空：适配器需在 DcNetAdapterDesc::schema 显式提供。
	virtual Node::Schema schema() const { return {}; }
};

} // namespace DC::Net
