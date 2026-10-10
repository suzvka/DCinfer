#pragma once

#include "NetEndpoint.h"
#include "NetError.h"

#include <string>

namespace DC::Net {

/// 传输载荷，v1 为 UTF-8 或 JSON 文本；二进制帧另行约定。
using Payload = std::string;

/// 传输抽象：对运行时呈现同步接口；I/O 线程、事件循环与子进程为内部实现细节。
/// 返回的 NetError 须已归一化，localStatus 与 localMessage 已填。
struct DcNetTransport {
	virtual ~DcNetTransport() = default;

	/// 连接建立与就绪探测；loadModel 时调用，失败抛 NodeException。
	virtual NetError connect(const NetEndpoint&) = 0;

	/// 发送请求载荷，阻塞并遵守超时。
	virtual NetError send(const Payload&) = 0;

	/// 接收响应载荷，阻塞并遵守超时；失败时 remoteDetail 保留原文。
	virtual NetError recv(Payload&) = 0;

	/// 当前端点配置，connect 后有效；供标准 RunFn 读取编排参数。
	virtual const NetEndpoint& endpoint() const = 0;

	/// 健康判定：进程存活、心跳或连接可用。
	virtual bool alive() const = 0;

	/// 释放连接与资源，releaseModel 调用。
	virtual void close() = 0;
};

} // namespace DC::Net
