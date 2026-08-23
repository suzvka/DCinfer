#pragma once

#include "NetEndpoint.h"
#include "NetError.h"

#include <string>

namespace DC::Net {

/// @brief 传输载荷（v1：UTF-8 / JSON 文本；二进制帧由 DCNet.Native 另行约定）。
using Payload = std::string;

/// @brief 传输抽象（DESIGN.md §3.1）。
///
/// 契约约束（ADR-6）：对运行时永远呈现**同步接口**；I/O 线程 / 事件循环 /
/// 子进程全部是 transport 内部实现细节，换策略不动 RunFn / 图 / 契约。
/// 返回的 NetError 须已归一化（localStatus/localMessage 已填）。
struct DcNetTransport {
	virtual ~DcNetTransport() = default;

	/// @brief 连接建立 / 就绪探测（createEngine 时调用；失败抛 NodeException）。
	virtual NetError connect(const NetEndpoint&) = 0;

	/// @brief 发送请求载荷（阻塞，遵守超时；失败返回非 Ok 的 NetError）。
	virtual NetError send(const Payload&) = 0;

	/// @brief 接收响应载荷（阻塞，遵守超时；失败时 remoteDetail 保留原始错误报文）。
	virtual NetError recv(Payload&) = 0;

	/// @brief 健康判定（进程存活 / 心跳 / 连接可用）。
	virtual bool alive() const = 0;

	/// @brief 释放连接与资源（releaseEngine 调用）。
	virtual void close() = 0;
};

} // namespace DC::Net
