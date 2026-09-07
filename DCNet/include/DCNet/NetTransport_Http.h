#pragma once

#include "NetTransport.h"

#include <iosfwd>
#include <memory>
#include <string>

namespace Poco::Net {
class HTTPClientSession;
}

namespace DC::Net {

/// @brief 内置 HTTP transport（POCO 实现，跨平台；DESIGN.md §9 方案 C）。
///
/// 语义（对运行时同步接口，ADR-6）：
/// - connect()：解析端点 URL（Poco::URI）→ TCP 就绪探测 → 建立会话
///   （HTTPClientSession / HTTPSClientSession，keep-alive 复用）；
///   探测失败即返回归一化错误（createEngine 配置期报告）
/// - send()：POST {basePath}{requestPath}，2xx → None（响应体留待 recv 读取）；
///   非 2xx → 读取错误体并归一化（normalizeHttpResponse）
/// - recv()：读取 2xx 响应体
///
/// 实现策略：transport 内部为同步阻塞调用，无 I/O 线程——简单 HTTP/JSON 场景
/// 直接跑在 RunFn 所在 System 池线程（ADR-6 判定矩阵第一行）。
/// POCO 细节不进契约：本头文件仅前置声明，TLS 会话等实现见 .cpp。
class HttpTransport : public DcNetTransport {
public:
	HttpTransport();
	~HttpTransport() override;

	NetError connect(const NetEndpoint&) override;
	NetError send(const Payload&) override;
	NetError recv(Payload&) override;
	const NetEndpoint& endpoint() const override { return _ep; }
	bool alive() const override;
	void close() override;

private:
	Payload readBody();
	void abortResponse();
	void dropSession();

	NetEndpoint _ep;
	std::unique_ptr<Poco::Net::HTTPClientSession> _session; ///< HTTPS 时指向 HTTPSClientSession
	std::istream* _response = nullptr;                      ///< 挂起的响应流（send → recv 之间有效）
	std::string _basePath;                                  ///< 端点 basePath（不含 requestPath）
	bool _useTls = false;
	bool _failed = false;
	NetError _connectError; ///< connect 失败的归一化错误（send 未连接时复现）
};

} // namespace DC::Net
