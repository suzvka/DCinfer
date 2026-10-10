#pragma once

#include "NetTransport.h"
#include "NetError.h"
#include "Node.h"

#include <stdexcept>
#include <string>

namespace DC::Net {

/// encode 阶段抛出，由标准 RunFn 映射为 InvalidInput。
struct DcCodecInputError : std::runtime_error {
	using std::runtime_error::runtime_error;
};

/// decode 阶段抛出，由标准 RunFn 映射为 ExecutionFailed（RemoteMalformed）。
struct DcCodecRemoteError : std::runtime_error {
	using std::runtime_error::runtime_error;
};

/// 协议映射：本地端口 ↔ 对方报文。
///
/// 错误提取/归一化不属 codec 职责：由 transport 在 recv 时调用核心归一化函数。
struct DcNetCodec {
	virtual ~DcNetCodec() = default;

	/// 本地端口 → 对方请求报文。
	virtual Payload encodeRequest(const Node::RunContext&) = 0;

	/// 对方响应报文 → 本地端口。
	virtual void decodeResponse(Payload&, Node::RunContext&) = 0;

	/// 协议子路径（如 "/infer"）；默认空 = POST 到端点 basePath。
	virtual std::string requestPath() const { return {}; }

	/// 本地端口 Schema；默认空时需在 DcNetAdapterDesc::schema 显式提供。
	virtual Node::Schema schema() const { return {}; }
};

} // namespace DC::Net
