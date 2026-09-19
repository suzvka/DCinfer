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

	/// @brief 优雅停止（同步、阻塞）：停 accept 并 join accept 线程 → 以 requestTimeout
	/// 为 grace 等待已接受连接的工作线程自然退出（已完成请求正常应答）→ grace 到期
	/// 后强制关闭在册连接（把阻塞在读/写上的线程立即放倒）→ 等其全部退出。
	/// 因此本函数返回即蕴含：无任何工作线程再持有监听器状态（不存在"线程存活而
	/// 对象已析构"的窗口），紧随其后的析构安全。重复调用安全。
	virtual void stop() = 0;

	/// @brief 服务端健康镜像（对端 alive() 的镜像语义）。
	virtual bool alive() const = 0;

	/// @brief 实际绑定端口（endpoint.port = 0 时 bind 后回读）。
	virtual int port() const = 0;
};

/// @brief HTTP/1.1 监听器（POCO ServerSocket，跨平台）。
/// v1 边界：仅 Content-Length 请求体（不支持 chunked）；逐请求应答后关闭连接；
/// 服务端 TLS 监听暂不支持；thread-per-connection，并发连接数受
/// NetServerEndpoint::maxConnections 约束（超限连接就地关闭，不起线程）。
std::unique_ptr<DcNetListener> makeHttpListener();

} // namespace DC::Net
