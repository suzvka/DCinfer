#include "DCNet/NetTransport_Http.h"

#include <Poco/Exception.h>
#include <Poco/Net/HTTPClientSession.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/Net/HTTPSClientSession.h>
#include <Poco/Net/NetException.h>
#include <Poco/Net/SSLException.h>
#include <Poco/Net/SocketAddress.h>
#include <Poco/Net/StreamSocket.h>
#include <Poco/StreamCopier.h>
#include <Poco/Timespan.h>
#include <Poco/URI.h>

#include <chrono>
#include <istream>
#include <memory>
#include <ostream>
#include <utility>

namespace DC::Net {

namespace {

/// POCO 异常 → 传输层错误原语（子类优先，SSLException 覆盖全部 TLS 系异常）。
NetTransportError classifyPocoException(const Poco::Exception& e) {
	if (dynamic_cast<const Poco::TimeoutException*>(&e))
		return NetTransportError::Timeout;
	if (dynamic_cast<const Poco::Net::DNSException*>(&e))
		return NetTransportError::DnsFailed;
	if (dynamic_cast<const Poco::Net::ConnectionRefusedException*>(&e))
		return NetTransportError::ConnectionRefused;
	if (dynamic_cast<const Poco::Net::ConnectionResetException*>(&e))
		return NetTransportError::Reset;
	if (dynamic_cast<const Poco::Net::SSLException*>(&e))
		return NetTransportError::TlsFailed;
	return NetTransportError::Other;
}

Poco::Timespan toTimespan(const std::chrono::milliseconds& ms) {
	return Poco::Timespan(0, std::chrono::duration_cast<std::chrono::microseconds>(ms).count());
}

} // namespace

HttpTransport::HttpTransport() = default;

HttpTransport::~HttpTransport() {
	close();
}

NetError HttpTransport::connect(const NetEndpoint& ep) {
	close();
	_ep = ep;

	// 解析端点 URL → scheme/host/port/basePath（Poco::URI 处理默认端口与 IPv6）
	Poco::URI uri;
	try {
		uri = Poco::URI(ep.endpoint());
	} catch (const Poco::Exception& e) {
		_failed = true;
		_connectError = normalizeTransportError(NetTransportError::Other,
												"bad endpoint '" + ep.endpoint() + "': " + e.displayText());
		return _connectError;
	}
	_useTls = (uri.getScheme() == "https");
	_basePath = uri.getPath();

	// TCP 就绪探测（契约 §3.1：connect 即就绪探测，拒连/DNS 失败在 createEngine 配置期报告）
	try {
		Poco::Net::SocketAddress addr(uri.getHost(), static_cast<Poco::UInt16>(uri.getPort()));
		Poco::Net::StreamSocket probe;
		probe.connect(addr, toTimespan(ep.connectTimeout));
		probe.close();
	} catch (const Poco::Exception& e) {
		_failed = true;
		_connectError = normalizeTransportError(classifyPocoException(e),
												"probe " + ep.endpoint() + " failed: " + e.displayText());
		return _connectError;
	}

	// 会话：HTTP 或 HTTPS（HTTPSClientSession 使用默认客户端 TLS 上下文）
	try {
		const std::string host = uri.getHost();
		const Poco::UInt16 port = static_cast<Poco::UInt16>(uri.getPort());
		std::unique_ptr<Poco::Net::HTTPClientSession> session;
		if (_useTls)
			session = std::make_unique<Poco::Net::HTTPSClientSession>(host, port);
		else
			session = std::make_unique<Poco::Net::HTTPClientSession>(host, port);
		session->setKeepAlive(true);
		session->setConnectTimeout(toTimespan(ep.connectTimeout));
		session->setSendTimeout(toTimespan(ep.requestTimeout));
		session->setReceiveTimeout(toTimespan(ep.requestTimeout));
		_session = std::move(session);
	} catch (const Poco::Exception& e) {
		_failed = true;
		_connectError = normalizeTransportError(classifyPocoException(e),
												"session " + ep.endpoint() + " failed: " + e.displayText());
		return _connectError;
	}

	_failed = false;
	return {};
}

NetError HttpTransport::send(const Payload& payload) {
	if (!_session || _failed)
		return _connectError.ok() ? normalizeTransportError(NetTransportError::Other, "not connected")
								  : _connectError;

	try {
		// 完整路径 = basePath + requestPath（requestPath 自带前导 '/' 时避免双斜杠）
		std::string path = _basePath;
		if (!_ep.requestPath.empty()) {
			const std::string& rp = _ep.requestPath;
			if (rp[0] == '/') {
				if (!path.empty() && path.back() == '/')
					path.pop_back();
				path += rp;
			} else {
				if (path.empty() || path.back() != '/')
					path += '/';
				path += rp;
			}
		}
		if (path.empty())
			path = "/";

		// 请求头：Content-Type / Authorization（Bearer 由调用方填全）/ 附加头
		Poco::Net::HTTPRequest req(Poco::Net::HTTPRequest::HTTP_POST, path,
								   Poco::Net::HTTPMessage::HTTP_1_1);
		req.setContentType(_ep.contentType);
		if (!_ep.authToken.empty())
			req.set("Authorization", _ep.authToken);
		for (const auto& h : _ep.headers) {
			const auto colon = h.find(':');
			if (colon == std::string::npos)
				continue;
			req.set(h.substr(0, colon), h.substr(colon + 1));
		}
		req.setContentLength(static_cast<int>(payload.size()));

		std::ostream& os = _session->sendRequest(req);
		if (!payload.empty())
			os.write(payload.data(), static_cast<std::streamsize>(payload.size()));
		if (!os) {
			abortResponse();
			_failed = true;
			return normalizeTransportError(NetTransportError::Reset, "send request body failed");
		}

		// 状态码：2xx → None（体由 recv 读取）；非 2xx → 读错误体并归一化
		Poco::Net::HTTPResponse res;
		std::istream& rs = _session->receiveResponse(res);
		_response = &rs;
		const int status = res.getStatus();
		if (status >= 400) {
			const Payload body = readBody();
			_response = nullptr;
			return normalizeHttpResponse(status, body);
		}
		return {};
	} catch (const Poco::Exception& e) {
		abortResponse();
		_failed = true;
		return normalizeTransportError(classifyPocoException(e), e.displayText());
	} catch (const std::exception& e) {
		abortResponse();
		_failed = true;
		return normalizeTransportError(NetTransportError::Other, e.what());
	}
}

NetError HttpTransport::recv(Payload& out) {
	if (!_response)
		return normalizeTransportError(NetTransportError::Other, "no pending response");
	out = readBody();
	_response = nullptr;
	return {};
}

bool HttpTransport::alive() const {
	return !_failed && _session != nullptr;
}

void HttpTransport::close() {
	dropSession();
	_failed = false;
	_connectError = {};
}

Payload HttpTransport::readBody() {
	Payload out;
	if (!_response)
		return out;
	Poco::StreamCopier::copyToString(*_response, out);
	return out;
}

void HttpTransport::abortResponse() {
	// 响应状态未知（异常路径）：丢弃挂起响应并关闭底层连接
	_response = nullptr;
	if (_session) {
		try {
			_session->abort(); // 关闭底层 socket，下次 send 重连（残留报文不致污染后续响应）
		} catch (...) {
		}
	}
}

void HttpTransport::dropSession() {
	_response = nullptr;
	if (_session) {
		try {
			_session->abort();
		} catch (...) {
		}
		_session = nullptr;
	}
}

} // namespace DC::Net
