#pragma once

#include "NetServerEndpoint.h"

#include <functional>
#include <memory>
#include <string>

namespace DC::Net {

/// 服务端 wire 应答：状态码与 JSON 载荷或错误体。
struct WireResponse {
	int status = 200;
	std::string body;
};

/// 仅处理通过方法、鉴权、路径与过载闸门的请求；在监听自持线程调用，须可并发进入。
using RequestHandler = std::function<WireResponse(const std::string& path, const std::string& body)>;

/// 监听端生命周期，transport 服务端镜像。
///
/// bind 为配置期出口，失败抛 NodeException；start 后运行期错误不抛出，
/// 以 wire 状态 5xx 或 429 应答并保持监听存活。内部自持 I/O 与 accept 线程。
struct DcNetListener {
	virtual ~DcNetListener() = default;

	/// 绑定监听端点，配置期调用；失败抛 NodeException。
	virtual void bind(const NetServerEndpoint& endpoint) = 0;

	/// 开始 accept；未 bind 或重复调用抛 NodeException。
	virtual void start(RequestHandler handler) = 0;

	/// 优雅停止，同步阻塞：停 accept 并 join 线程，以 requestTimeout 为 grace
	/// 等待已接受连接自然退出，到期强制关闭在册连接。返回即无工作线程再持有
	/// 监听器状态，析构安全；重复与并发调用安全，handler 内调用 stop 抛异常，
	/// 自 handler 析构监听器或服务被禁止。
	virtual void stop() = 0;

	/// 服务端健康镜像。
	virtual bool alive() const = 0;

	/// 实际绑定端口；endpoint.port 为 0 时 bind 后回读。
	virtual int port() const = 0;
};

/// HTTP/1.1 监听器，基于 POCO ServerSocket。v1 边界：仅 Content-Length 请求体；
/// 逐请求应答后关闭连接；不支持服务端 TLS；thread-per-connection，并发数受
/// NetServerEndpoint::maxConnections 约束，超限连接就地关闭。
std::unique_ptr<DcNetListener> makeHttpListener();

} // namespace DC::Net
