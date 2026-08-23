#include "DCNet/NetTransport_Http.h"

#ifdef _WIN32

#include <windows.h>
#include <winhttp.h>

#include <string>

namespace DC::Net {

namespace {

NetTransportError mapWinHttpError(DWORD err) {
	switch (err) {
	case ERROR_WINHTTP_TIMEOUT:
		return NetTransportError::Timeout;
	case ERROR_WINHTTP_NAME_NOT_RESOLVED:
		return NetTransportError::DnsFailed;
	case ERROR_WINHTTP_CANNOT_CONNECT:
	case ERROR_WINHTTP_CONNECTION_ERROR:
		return NetTransportError::ConnectionRefused;
	case ERROR_WINHTTP_SECURE_FAILURE:
		return NetTransportError::TlsFailed;
	default:
		return NetTransportError::Other;
	}
}

std::wstring toWide(const std::string& s) {
	return std::wstring(s.begin(), s.end());
}

} // namespace

HttpTransport::~HttpTransport() {
	close();
}

NetError HttpTransport::connect(const NetEndpoint& ep) {
	close();
	_ep = ep;

	// 解析端点 URL → host/port/basePath（WinHttpCrackUrl 处理默认端口与 IPv6）
	const std::wstring url = toWide(ep.endpoint());
	URL_COMPONENTS comp{};
	comp.dwStructSize = sizeof(comp);
	comp.dwHostNameLength = (DWORD)-1;
	comp.dwUrlPathLength = (DWORD)-1;
	comp.dwExtraInfoLength = (DWORD)-1;
	if (!WinHttpCrackUrl(url.c_str(), 0, 0, &comp)) {
		_failed = true;
		return normalizeTransportError(NetTransportError::Other,
									   "WinHttpCrackUrl failed on '" + ep.endpoint() + "'");
	}

	const std::wstring host(comp.lpszHostName, comp.dwHostNameLength);
	const std::wstring path(comp.lpszUrlPath, comp.dwUrlPathLength);
	const bool useTls = (comp.nScheme == INTERNET_SCHEME_HTTPS);

	// 会话：本地地址绕过系统代理（测试与局域网直连）
	_session = WinHttpOpen(L"DCinfer-DCNet/0.1", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
						   WINHTTP_NO_PROXY_NAME, L"<local>", 0);
	if (!_session) {
		_failed = true;
		return normalizeTransportError(mapWinHttpError(GetLastError()), "WinHttpOpen failed");
	}

	WinHttpSetTimeouts(_session,                       // 会话句柄
					   (int)ep.connectTimeout.count(), // 解析
					   (int)ep.connectTimeout.count(), // 连接
					   (int)ep.requestTimeout.count(), // 发送
					   (int)ep.requestTimeout.count());// 接收

	_connect = WinHttpConnect(_session, host.c_str(), comp.nPort, 0);
	if (!_connect) {
		_failed = true;
		return normalizeTransportError(mapWinHttpError(GetLastError()), "WinHttpConnect failed");
	}

	// 记忆 basePath（不含 requestPath，后者由 send 时拼接）
	const size_t q = path.find_first_of(L"?");
	_basePath = path.substr(0, q);
	_useTls = useTls;
	_failed = false;
	return {};
}

NetError HttpTransport::send(const Payload& payload) {
	if (_failed || !_session || !_connect)
		return normalizeTransportError(NetTransportError::Other, "not connected");

	// 完整路径 = basePath + requestPath（requestPath 自带前导 '/' 时避免双斜杠）
	std::wstring path = _basePath;
	if (!_ep.requestPath.empty()) {
		std::wstring rp = toWide(_ep.requestPath);
		if (!rp.empty() && rp[0] == L'/') {
			if (!path.empty() && path.back() == L'/')
				path.pop_back();
			path += rp;
		} else {
			if (path.empty() || path.back() != L'/')
				path += L'/';
			path += rp;
		}
	}

	_request = WinHttpOpenRequest(_connect, L"POST", path.c_str(), NULL, WINHTTP_NO_REFERER,
								  WINHTTP_DEFAULT_ACCEPT_TYPES, _useTls ? WINHTTP_FLAG_SECURE : 0);
	if (!_request) {
		_failed = true;
		return normalizeTransportError(mapWinHttpError(GetLastError()), "WinHttpOpenRequest failed");
	}

	// 请求头：Content-Type / Authorization（Bearer 由调用方填全）/ 附加头
	std::wstring headers = L"Content-Type: " + toWide(_ep.contentType) + L"\r\n";
	if (!_ep.authToken.empty())
		headers += L"Authorization: " + toWide(_ep.authToken) + L"\r\n";
	for (const auto& h : _ep.headers)
		headers += toWide(h) + L"\r\n";

	const BOOL sent = WinHttpSendRequest(
		_request, headers.c_str(), (DWORD)headers.size(),
		payload.empty() ? WINHTTP_NO_REQUEST_DATA : const_cast<char*>(payload.data()),
		(DWORD)payload.size(), (DWORD)payload.size(), 0);
	if (!sent) {
		const DWORD err = GetLastError();
		closeRequest();
		if (err == ERROR_WINHTTP_OPERATION_CANCELLED || err == ERROR_WINHTTP_CONNECTION_ERROR)
			_failed = true;
		return normalizeTransportError(mapWinHttpError(err), "WinHttpSendRequest failed");
	}

	if (!WinHttpReceiveResponse(_request, NULL)) {
		const DWORD err = GetLastError();
		closeRequest();
		_failed = true;
		return normalizeTransportError(mapWinHttpError(err), "WinHttpReceiveResponse failed");
	}

	// 状态码：2xx → None（体由 recv 读取）；非 2xx → 读错误体并归一化
	DWORD status = 0;
	DWORD statusSize = sizeof(status);
	if (!WinHttpQueryHeaders(_request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
							 WINHTTP_HEADER_NAME_BY_INDEX, &status, &statusSize, WINHTTP_NO_HEADER_INDEX)) {
		const DWORD err = GetLastError();
		closeRequest();
		return normalizeTransportError(mapWinHttpError(err), "WinHttpQueryHeaders failed");
	}
	if (status >= 400) {
		Payload body = readBody();
		closeRequest();
		return normalizeHttpResponse((int)status, body);
	}
	return {};
}

NetError HttpTransport::recv(Payload& out) {
	if (_failed || !_request)
		return normalizeTransportError(NetTransportError::Other, "no pending response");
	out = readBody();
	closeRequest();
	return {};
}

bool HttpTransport::alive() const {
	return !_failed && _session && _connect;
}

void HttpTransport::close() {
	closeRequest();
	if (_connect) {
		WinHttpCloseHandle(_connect);
		_connect = nullptr;
	}
	if (_session) {
		WinHttpCloseHandle(_session);
		_session = nullptr;
	}
	_failed = false;
}

Payload HttpTransport::readBody() {
	Payload out;
	if (!_request)
		return out;
	for (;;) {
		DWORD available = 0;
		if (!WinHttpQueryDataAvailable(_request, &available) || available == 0)
			break;
		const size_t old = out.size();
		out.resize(old + available);
		DWORD read = 0;
		if (!WinHttpReadData(_request, out.data() + old, available, &read))
			break;
		out.resize(old + read);
		if (read < available)
			break;
	}
	return out;
}

void HttpTransport::closeRequest() {
	if (_request) {
		WinHttpCloseHandle(_request);
		_request = nullptr;
	}
}

} // namespace DC::Net

#else // 非 Windows：空实现（POSIX 后端见 DESIGN.md §9 展望）

namespace DC::Net {

HttpTransport::~HttpTransport() = default;
NetError HttpTransport::connect(const NetEndpoint&) { return normalizeTransportError(NetTransportError::Other, "HttpTransport: not implemented on this platform"); }
NetError HttpTransport::send(const Payload&) { return normalizeTransportError(NetTransportError::Other, "HttpTransport: not implemented on this platform"); }
NetError HttpTransport::recv(Payload&) { return normalizeTransportError(NetTransportError::Other, "HttpTransport: not implemented on this platform"); }
bool HttpTransport::alive() const { return false; }
void HttpTransport::close() {}

} // namespace DC::Net

#endif
