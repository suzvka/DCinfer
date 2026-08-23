#pragma once

#include "NetTransport.h"

namespace DC::Net {

/// @brief 内置 HTTP transport（WinHTTP，Windows 零新增依赖；DESIGN.md §9 方案 B）。
///
/// 语义（对运行时同步接口，ADR-6）：
/// - connect()：解析端点（WinHttpCrackUrl）→ 建立会话/连接句柄（keep-alive 复用）
/// - send()：POST {basePath}{requestPath}，2xx → None（响应体留待 recv 读取）；
///   非 2xx → 读取错误体并归一化（normalizeHttpResponse）
/// - recv()：读取 2xx 响应体
///
/// 实现策略：transport 内部为同步阻塞调用，无 I/O 线程——简单 HTTP/JSON 场景
/// 直接跑在 RunFn 所在 System 池线程（ADR-6 判定矩阵第一行）。
class HttpTransport : public DcNetTransport {
public:
	HttpTransport() = default;
	~HttpTransport() override;

	NetError connect(const NetEndpoint&) override;
	NetError send(const Payload&) override;
	NetError recv(Payload&) override;
	bool alive() const override;
	void close() override;

private:
	Payload readBody();
	void closeRequest();

	NetEndpoint _ep;
	void* _session = nullptr;   // HINTERNET (WinHTTP session)
	void* _connect = nullptr;   // HINTERNET (connection handle)
	void* _request = nullptr;   // HINTERNET (request handle)
	std::wstring _basePath;     // 端点 basePath（不含 requestPath）
	bool _useTls = false;
	bool _failed = false;
};

} // namespace DC::Net
