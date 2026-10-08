#include "DCNet/NetTransport_Http.h"

#include <Poco/Exception.h>
#include <Poco/Net/HTTPClientSession.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/Net/HTTPSClientSession.h>
#include <Poco/Net/NetException.h>
#include <Poco/Net/SSLException.h>
#include <Poco/Net/SSLManager.h>
#include <Poco/Net/SocketAddress.h>
#include <Poco/Net/StreamSocket.h>
#include <Poco/StreamCopier.h>
#include <Poco/Timespan.h>
#include <Poco/URI.h>
#if defined(_WIN32)
#include <Poco/Net/SecureStreamSocket.h>
#include <Poco/UnicodeConverter.h>
#include <wininet.h>
#pragma comment(lib, "crypt32.lib")
#endif

#include <chrono>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <condition_variable>
#include <istream>
#include <memory>
#include <mutex>
#include <ostream>
#include <thread>
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

#if defined(_WIN32)
// NetSSLWin X509Certificate::verify resolves DNS SANs when the endpoint is
// an IP literal, accepting localhost certificates for 127.0.0.1. Validate the
// native SSL policy instead, before HTTPSClientSession serializes any headers.
class VerifiedHttpsClientSession final : public Poco::Net::HTTPSClientSession {
public:
	VerifiedHttpsClientSession(const std::string& host, Poco::UInt16 port,
		Poco::Net::Context::Ptr context)
		: Poco::Net::HTTPSClientSession(host, port, context), _context(context) {
		if (context->verificationMode() == Poco::Net::Context::VERIFY_NONE ||
			!context->extendedCertificateVerificationEnabled())
			throw Poco::Net::SSLException("HTTPS requires certificate verification");
	}
protected:
	void connect(const Poco::Net::SocketAddress& address) override {
		Poco::Net::HTTPSClientSession::connect(address);
		Poco::Net::SecureStreamSocket secure(socket());
		secure.completeHandshake(); // chain is checked against the POCO Context
		const auto cert = secure.peerCertificate();
		CERT_CHAIN_PARA parameters{};
		parameters.cbSize = sizeof(parameters);
		PCCERT_CHAIN_CONTEXT chain = nullptr;
		if (!CertGetCertificateChain(nullptr, cert.system(), nullptr,
			_context->certificateStore(), &parameters, 0, nullptr, &chain))
			throw Poco::Net::SSLException("HTTPS certificate chain unavailable");
		const std::unique_ptr<const CERT_CHAIN_CONTEXT, decltype(&CertFreeCertificateChain)>
			chainOwner(chain, &CertFreeCertificateChain);
		// Do not allow host certificate-error handlers to bypass trust. Require
		// every completed chain to terminate at an issuer in this Context store.
		bool trusted = chain->cChain > 0;
		for (DWORD i = 0; i < chain->cChain && trusted; ++i) {
			const auto* simple = chain->rgpChain[i];
			if (!simple->cElement) { trusted = false; break; }
			const auto root = simple->rgpElement[simple->cElement - 1]->pCertContext;
			const auto issuer = CertFindCertificateInStore(_context->certificateStore(),
				root->dwCertEncodingType, 0, CERT_FIND_ISSUER_OF, root, nullptr);
			trusted = issuer != nullptr;
			if (issuer) CertFreeCertificateContext(issuer);
		}
		if (!trusted)
			throw Poco::Net::SSLException("HTTPS certificate authority not trusted");
		std::wstring host;
		Poco::UnicodeConverter::convert(getHost(), host);
		SSL_EXTRA_CERT_CHAIN_POLICY_PARA ssl{};
		ssl.cbSize = sizeof(ssl);
		ssl.dwAuthType = AUTHTYPE_SERVER;
		// Root trust was already checked by the handshake against Context's
		// private memory store; do not require installing its CA in Windows ROOT.
		ssl.fdwChecks = SECURITY_FLAG_IGNORE_UNKNOWN_CA;
		ssl.pwszServerName = const_cast<wchar_t*>(host.c_str());
		CERT_CHAIN_POLICY_PARA policy{};
		policy.cbSize = sizeof(policy);
		policy.pvExtraPolicyPara = &ssl;
		CERT_CHAIN_POLICY_STATUS status{};
		status.cbSize = sizeof(status);
		const BOOL ok = CertVerifyCertificateChainPolicy(CERT_CHAIN_POLICY_SSL, chain, &policy, &status);
		if (!ok || status.dwError)
			throw Poco::Net::SSLException(status.dwError == CERT_E_CN_NO_MATCH
				? "HTTPS certificate hostname mismatch" : "HTTPS certificate policy rejected");
	}
private:
	Poco::Net::Context::Ptr _context;
};
#endif

Poco::Timespan toTimespan(const std::chrono::milliseconds& ms) {
	return Poco::Timespan(0, std::chrono::duration_cast<std::chrono::microseconds>(ms).count());
}

/// header 值安全检查（P1）：拒绝 CR/LF、内嵌 NUL 与其余控制字符——
/// 这些字节会经请求序列化注入伪造报文行（header 注入/请求走私）。
std::string requestTarget(std::string base, const std::string& suffix) {
	if (!suffix.empty()) {
		if (suffix.front() == '/') {
			if (!base.empty() && base.back() == '/') base.pop_back();
			base += suffix;
		} else {
			if (base.empty() || base.back() != '/') base += '/';
			base += suffix;
		}
	}
	return base.empty() ? "/" : base;
}

bool validRequestTarget(const std::string& target) {
	if (target.empty() || target.front() != '/' || target.rfind("//", 0) == 0) return false;
	for (unsigned char c : target) if (c <= 0x20 || c == 0x7f || c == '#') return false;
	return true;
}

bool isHeaderName(const std::string& name) {
	if (name.empty()) return false;
	for (unsigned char c : name)
		if (!std::isalnum(c) && std::string("!#$%&'*+-.^_`|~").find(static_cast<char>(c)) == std::string::npos) return false;
	return true;
}

bool isPrintableHeaderBytes(const std::string& v) {
	for (unsigned char c : v) {
		if (c == '\t')
			continue; // 字段值内的空白折叠是合法形式
		if (c < 0x20 || c == 0x7f)
			return false;
	}
	return true;
}

/// 客户端 TLS 上下文（P1）：宿主已初始化 SSLManager（defaultClientContext
/// 可用）时按宿主管辖；否则框架兜底初始化 VERIFY_STRICT（证书链 + 主机名
/// 校验 + 默认 CA）——不再依赖宿主全局配置，HTTPS 开箱即安全可用。
/// 宿主自定义信任锚：在首次 connect 前调用
///   Poco::Net::SSLManager::instance().initializeClient(context)
/// 即可接管（本函数检测到已初始化时不覆盖）。
Poco::Net::Context::Ptr ensureClientTlsContext() {
	try {
		return Poco::Net::SSLManager::instance().defaultClientContext();
	} catch (const Poco::Exception&) {
#if defined(_WIN32)
		// NetSSLWin can itself supply a rejecting VERIFY_RELAXED default when
		// no application configuration exists; client RELAXED still checks CA
		// trust. If POCO cannot supply one, use system-root VERIFY_STRICT.
		// VerifiedHttpsClientSession independently enforces trust and native
		// hostname policy for either mode before any HTTP bytes are written.
		Poco::Net::Context::Ptr ctx(new Poco::Net::Context(
			Poco::Net::Context::CLIENT_USE, "", Poco::Net::Context::VERIFY_STRICT));
		Poco::Net::SSLManager::instance().initializeClient(nullptr, nullptr, ctx);
#else
		// OpenSSL 后端：显式加载系统默认 CA 并禁用废弃协议版本
		Poco::Net::Context::Ptr ctx(new Poco::Net::Context(
			Poco::Net::Context::CLIENT_USE, "", "", "",
			Poco::Net::Context::VERIFY_STRICT, 9, true,
			"ALL:!SSLv2:!SSLv3:!TLSv1:!TLSv1.1"));
		Poco::Net::SSLManager::instance().initializeClient(nullptr, nullptr, ctx);
#endif
		return Poco::Net::SSLManager::instance().defaultClientContext();
	}
}

} // namespace

HttpTransport::HttpTransport() = default;

HttpTransport::~HttpTransport() {
	// 析构期不等交换权（持有者仍在跑即调用方违约）：强制清占用后丢会话。
	// 交换方持有的 session 副本保活对象，dropSession 的 abort 仅中断其阻塞读。
	{
		std::lock_guard lk(_ioMutex);
		_claimed = false;
		_claimOwner = std::thread::id{};
	}
	_ioCv.notify_all();
	dropSession();
}

void HttpTransport::acquireClaim() {
	std::unique_lock lk(_ioMutex);
	const auto self = std::this_thread::get_id();
	// 同线程遗留占用（上次交换未由 recv 收尾）直接回收：不自锁
	_ioCv.wait(lk, [this, self] { return !_closing && (!_claimed || _claimOwner == self); });
	_callActive = true;
	_claimed = true;
	_claimOwner = self;
}

void HttpTransport::finishCall(bool releaseClaim) {
	std::lock_guard lk(_ioMutex);
	if (releaseClaim && _claimed && _claimOwner == std::this_thread::get_id()) {
		_claimed = false;
		_claimOwner = {};
	}
	_callActive = false;
	// Notify while locked; no member access remains after inactive state is observable.
	_ioCv.notify_all();
}

void HttpTransport::resetLocked() {
	dropSession(); // 调用者已持交换权：无并发交换可误伤
	_failed = false;
	_connectError = {};
}

NetError HttpTransport::connect(const NetEndpoint& ep) {
	acquireClaim();
	const ClaimScope scope{this}; // 所有出口回收占用（_ep 写入也受占用保护）
	resetLocked();

	// 配置期校验（P1）：headers/authToken/contentType 中的 CR/LF/控制字符
	// 会在请求序列化时注入伪造报文行——connect 即就绪探测 + 配置报错点，
	// 此处拒绝不发起任何网络 I/O
	for (const auto& h : ep.headers) {
		const auto colon = h.find(':');
		const std::string name = (colon == std::string::npos) ? h : h.substr(0, colon);
		const std::string value =
			(colon == std::string::npos) ? std::string() : h.substr(colon + 1);
		std::string lowerName = name;
		std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		if (lowerName == "transfer-encoding") {
			_failed = true;
			_connectError = normalizeTransportError(NetTransportError::Other, "transport-owned framing header is not configurable");
			return _connectError;
		}
		if (colon == std::string::npos || !isHeaderName(name) || !isPrintableHeaderBytes(value)) {
			_failed = true;
			_connectError = normalizeTransportError(
				NetTransportError::Other,
				"invalid header in endpoint config (header " + std::to_string(&h - ep.headers.data() + 1) + ")");
			return _connectError;
		}
	}
	if (!isPrintableHeaderBytes(ep.authToken)) {
		_failed = true;
		_connectError = normalizeTransportError(NetTransportError::Other,
												"invalid authToken (CR/LF/control chars not allowed)");
		return _connectError;
	}
	if (!isPrintableHeaderBytes(ep.contentType)) {
		_failed = true;
		_connectError = normalizeTransportError(NetTransportError::Other,
												"invalid contentType (CR/LF/control chars not allowed)");
		return _connectError;
	}

	_ep = ep;

	// 解析端点 URL → scheme/host/port/basePath（Poco::URI 处理默认端口与 IPv6）
	Poco::URI uri;
	try {
		uri = Poco::URI(ep.endpoint());
	} catch (const Poco::Exception& e) {
		_failed = true;
		_connectError = normalizeTransportError(NetTransportError::Other,
												"invalid endpoint URI");
		return _connectError;
	}
	_useTls = (uri.getScheme() == "https");
	bool credentials = !ep.authToken.empty() || !uri.getUserInfo().empty();
	for (const auto& header : ep.headers) {
		auto name = header.substr(0, header.find(':'));
		std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		credentials = credentials || name == "authorization" || name == "proxy-authorization" || name == "cookie" || name == "x-api-key" || name == "api-key" || name == "apikey" || name == "x-auth-token";
	}
	if ((uri.getScheme() != "http" && !_useTls) || (!_useTls && credentials && !ep.allowInsecureCredentials)) {
		_failed = true;
		_connectError = normalizeTransportError(NetTransportError::Other, "unsupported scheme or plaintext credentials require explicit allowInsecureCredentials");
		return _connectError;
	}
	_basePath = uri.getPath();
	if (!validRequestTarget(requestTarget(_basePath, ep.requestPath))) {
		_failed = true;
		_connectError = normalizeTransportError(NetTransportError::Other, "invalid HTTP request target");
		return _connectError;
	}

	// TCP 就绪探测（契约 §3.1：connect 即就绪探测，拒连/DNS 失败在 createEngine 配置期报告）
	try {
		Poco::Net::SocketAddress addr(uri.getHost(), static_cast<Poco::UInt16>(uri.getPort()));
		Poco::Net::StreamSocket probe;
		probe.connect(addr, toTimespan(ep.connectTimeout));
		probe.close();
	} catch (const Poco::Exception& e) {
		_failed = true;
		_connectError = normalizeTransportError(classifyPocoException(e),
												"endpoint TCP probe failed");
		return _connectError;
	}

	// 会话：HTTP 或 HTTPS（HTTPSClientSession 使用显式客户端 TLS 上下文——
	// 宿主已初始化则用宿主的，否则兜底 VERIFY_STRICT，不依赖全局默认）
	try {
		const std::string host = uri.getHost();
		const Poco::UInt16 port = static_cast<Poco::UInt16>(uri.getPort());
		std::shared_ptr<Poco::Net::HTTPClientSession> session;
		if (_useTls) {
#if defined(_WIN32)
			session = std::make_shared<VerifiedHttpsClientSession>(host, port, ensureClientTlsContext());
#else
			session = std::make_shared<Poco::Net::HTTPSClientSession>(host, port,
																  ensureClientTlsContext());
#endif
		} else
			session = std::make_shared<Poco::Net::HTTPClientSession>(host, port);
		session->setKeepAlive(true);
		session->setConnectTimeout(toTimespan(ep.connectTimeout));
		session->setSendTimeout(toTimespan(ep.requestTimeout));
		session->setReceiveTimeout(toTimespan(ep.requestTimeout));
		{
			std::lock_guard lk(_ioMutex);
			_session = std::move(session);
		}
	} catch (const Poco::Exception& e) {
		_failed = true;
		_connectError = normalizeTransportError(classifyPocoException(e),
												"HTTP session initialization failed");
		return _connectError;
	}

	_failed = false;
	return {};
}

NetError HttpTransport::send(const Payload& payload) {
	// 一次交换的起点：领取占用，连同后续 recv 整体串行化（多节点共享同一实例时
	// 不得出现 A 的 send 接上 B 的 recv）；maxRetries 重试路径由 acquireClaim 自动回收
	acquireClaim();
	ClaimScope scope{this};
	// 会话快照（与 close/析构强收的 _ioMutex 临界区互斥）：close 超时强收
	// 仅 abort 并释放其侧引用，本交换继续使用的 session 由副本保活，
	// 不因强收期间的 _session 置空而悬垂
	std::shared_ptr<Poco::Net::HTTPClientSession> session;
	{
		std::lock_guard lk(_ioMutex);
		session = _session;
	}
	if (!session || _failed)
		return _connectError.ok() ? normalizeTransportError(NetTransportError::Other, "not connected")
								  : _connectError;

	try {
		const std::string path = requestTarget(_basePath, _ep.requestPath);

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

		std::ostream& os = session->sendRequest(req);
		if (!payload.empty())
			os.write(payload.data(), static_cast<std::streamsize>(payload.size()));
		if (!os) {
			abortResponse();
			_failed = true;
			return normalizeTransportError(NetTransportError::Reset, "send request body failed");
		}

		// 状态码：2xx → None（体由 recv 读取）；非 2xx（含 3xx：不自动
		// 跟随重定向）→ 读错误体并归一化
		Poco::Net::HTTPResponse res;
		std::istream& rs = session->receiveResponse(res);
		{
			std::lock_guard lk(_ioMutex);
			_response = &rs;
			_responseLength = res.hasContentLength() ? res.getContentLength64() : -1;
		}
		const int status = res.getStatus();
		if (status < 200 || status >= 300) {
			// 错误体仅作诊断：允许截断到 maxResponseBody，不再读取
			bool diagnosticTruncated = false;
			const Payload body = readBody(rs, _ep.maxResponseBody == 0 ? 4096 : std::min<std::size_t>(_ep.maxResponseBody, 4096), &diagnosticTruncated);
			if (diagnosticTruncated || (_responseLength >= 0 && static_cast<unsigned long long>(_responseLength) != body.size())) abortResponse();
			{
				std::lock_guard lk(_ioMutex);
				_response = nullptr;
			}
			return normalizeHttpResponse(status, body);
		}
		// 2xx：交换未结束——占用延续给 recv（否则响应流会被另一交换接管）
		scope.dismiss();
		return {};
	} catch (const Poco::Exception& e) {
		abortResponse();
		_failed = true;
		return normalizeTransportError(classifyPocoException(e), "HTTP exchange failed");
	} catch (const std::exception& e) {
		abortResponse();
		_failed = true;
		return normalizeTransportError(NetTransportError::Other, "HTTP exchange failed");
	}
}

NetError HttpTransport::recv(Payload& out) {
	out.clear();
	std::shared_ptr<Poco::Net::HTTPClientSession> session;
	std::istream* rs = nullptr;
	long long expected = -1;
	{
		std::lock_guard lk(_ioMutex);
		if (_closing || !_claimed || _claimOwner != std::this_thread::get_id())
			return normalizeTransportError(NetTransportError::Other, "recv requires send ownership on the same thread");
		session = _session;
		rs = _response;
		expected = _responseLength;
		_callActive = true;
	}
	const ClaimScope scope{this};
	if (!rs || !session) return normalizeTransportError(NetTransportError::Other, "no pending response");
	try {
		bool truncated = false;
		out = readBody(*rs, _ep.maxResponseBody, &truncated);
		bool preempted;
		{
			std::lock_guard lk(_ioMutex);
			_response = nullptr;
			preempted = (_session != session);
		}
		if (truncated || preempted || (expected >= 0 && static_cast<unsigned long long>(expected) != out.size())) {
			abortResponse();
			out.clear();
			return normalizeTransportError(NetTransportError::Reset, "incomplete or oversized response body");
		}
		return {};
	} catch (const Poco::Exception& e) {
		abortResponse(); out.clear();
		return normalizeTransportError(classifyPocoException(e), "response read failed");
	} catch (...) {
		abortResponse(); out.clear();
		return normalizeTransportError(NetTransportError::Other, "response read failed");
	}
}

bool HttpTransport::alive() const {
	std::lock_guard lk(_ioMutex);
	return !_failed && _session != nullptr;
}

void HttpTransport::close() {
	std::unique_lock lk(_ioMutex);
	_ioCv.wait(lk, [this] { return !_closing; });
	_closing = true;
	const auto self = std::this_thread::get_id();
	const bool ready = _ioCv.wait_for(lk, std::chrono::seconds(5),
		[this, self] { return !_claimed || (_claimOwner == self && !_callActive); });
	if (!ready) {
		// Abort the old session, but do not publish a free lease while old I/O still runs.
		lk.unlock();
		dropSession();
		lk.lock();
		_ioCv.wait(lk, [this] { return !_callActive; });
	}
	_claimed = false;
	_claimOwner = {};
	lk.unlock();
	dropSession();
	lk.lock();
	_failed = false;
	_connectError = {};
	_closing = false;
	lk.unlock();
	_ioCv.notify_all();
}

Payload HttpTransport::readBody(std::istream& rs, size_t limit, bool* truncated) {
	Payload out;
	// 分块读取替代无界 copyToString：达到 limit 即截断返回（是否视为错误
	// 由调用方按语义决定——成功体超限报错，错误体仅作诊断）
	char buffer[64 * 1024];
	while (true) {
		rs.read(buffer, sizeof(buffer));
		const size_t got = static_cast<size_t>(rs.gcount());
		if (got > 0) {
			if (limit != 0 && out.size() + got > limit) {
				const size_t room = limit - out.size();
				out.append(buffer, room);
				if (truncated)
					*truncated = true;
				return out;
			}
			out.append(buffer, got);
		}
		if (rs.bad() || (rs.fail() && !rs.eof())) throw std::runtime_error("response stream failure");
		if (rs.eof()) break;
	}
	return out;
}

void HttpTransport::abortResponse() {
	// 响应状态未知（异常路径）：丢弃挂起响应并关闭底层连接
	std::shared_ptr<Poco::Net::HTTPClientSession> session;
	{
		std::lock_guard lk(_ioMutex);
		_response = nullptr;
		session = _session;
	}
	if (session) {
		try {
			session->abort(); // 关闭底层 socket，下次 send 重连（残留报文不致污染后续响应）
		} catch (...) {
		}
	}
}

void HttpTransport::dropSession() {
	// 自持锁置空成员；session 在锁外释放——交换方持有的本地副本保活对象
	// （abort 中断其阻塞读，对象延迟析构至读取结束）
	std::shared_ptr<Poco::Net::HTTPClientSession> session;
	{
		std::lock_guard lk(_ioMutex);
		_response = nullptr;
		session = std::move(_session);
	}
	if (session) {
		try {
			session->abort();
		} catch (...) {
		}
	}
}

} // namespace DC::Net
