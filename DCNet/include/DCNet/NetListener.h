#pragma once

#include "NetServerEndpoint.h"

#include <functional>
#include <memory>
#include <string>

namespace DC::Net {

/// @brief 服务端 wire 应答（两段式归一化的第一段：状态码 + JSON 载荷/错误体）。
struct WireResponse {
	int status = 200;
	std::string body;
};

/// @brief 单请求业务处理钩子：已过方法 / 鉴权 / 路径 / 过载闸门，仅处理合法请求。
/// 在监听器自持线程上调用（ADR-6(3)），须可并发进入。
using RequestHandler = std::function<WireResponse(const std::string& path, const std::string& body)>;

/// @brief 监听端生命周期（DcNetTransport 的服务端镜像；DESIGN.md §3.6）。
///
/// bind 为配置期出口（ADR-7）：失败抛 NodeException（对齐 createEngine
/// 先例与 DESIGN.md §6「配置/编译期」约定）；start 后的运行期错误不抛出，
/// 以 wire 状态应答（5xx / 429）并保持监听存活——不得崩溃或静默丢弃请求。
/// ADR-6(3)：实现内部自持 I/O 线程 / accept 循环，对上层呈现同步契约。
struct DcNetListener {
	virtual ~DcNetListener() = default;

	/// @brief 绑定监听端点（配置期；失败抛 NodeException）。
	virtual void bind(const NetServerEndpoint& endpoint) = 0;

	/// @brief 开始 accept（未 bind 先调用 → NodeException；重复调用 → NodeException）。
	virtual void start(RequestHandler handler) = 0;

	/// @brief 优雅停止：停止 accept，等待在途请求完成（graceful drain，受
	/// requestTimeout 约束）。重复调用安全。
	virtual void stop() = 0;

	/// @brief 服务端健康镜像（对端 alive() 的镜像语义）。
	virtual bool alive() const = 0;

	/// @brief 实际绑定端口（endpoint.port = 0 时 bind 后回读）。
	virtual int port() const = 0;
};

/// @brief HTTP/1.1 监听器（POCO ServerSocket，跨平台；MockServer 的对外契约演进）。
/// v1 边界：仅 Content-Length 请求体（不支持 chunked）；逐请求应答后关闭连接；
/// 服务端证书 / TLS 监听随 mTLS 需求另行设计。
std::unique_ptr<DcNetListener> makeHttpListener();

} // namespace DC::Net
