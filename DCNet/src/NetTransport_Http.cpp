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

#include <chrono>
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

Poco::Timespan toTimespan(const std::chrono::milliseconds& ms) {
	return Poco::Timespan(0, std::chrono::duration_cast<std::chrono::microseconds>(ms).count());
}

/// header 值安全检查（P1）：拒绝 CR/LF、内嵌 NUL 与其余控制字符——
/// 这些字节会经请求序列化注入伪造报文行（header 注入/请求走私）。
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
/// 可用）时按宿主管辖；否则框架兕底初始化 VERIFY_STRICT（证书链 + 主机名
/// 校验 + 默认 CA）——不再依赖宿主全局配置，HTTPS 开箱即安全可用。
/// 宿主自定义信任锚：在首次 connect 前调用
///   Poco::Net::SSLManager::instance().initializeClient(context)
/// 即可接管（本函数检测到已初始化时不覆盖）。
Poco::Net::Context::Ptr ensureClientTlsContext() {
	try {
		return Poco::Net::SSLManager::instance().defaultClientContext();
	} catch (const Poco::Exception&) {
#if defined(_WIN32)
		// SChannel 后端（NetSSL_Win）：Context(usage, certPath, verMode, options,
		// storeName)，信任链来自系统证书库。注：本分支未经 CI 验证（DCNet CI
		// 仅 Ubuntu/OpenSSL），发布前需在 Windows 上人工验证。
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
	_ioCv.wait(lk, [this, self] { return !_claimed || _claimOwner == self; });
	_claimed = true;
	_claimOwner = self;
}

void HttpTransport::releaseClaimIfOwned() {
	{
		std::lock_guard lk(_ioMutex);
		if (!_claimed || _claimOwner != std::this_thread::get_id())
			return; // 非本线程占用：不动他人租约
		_claimed = false;
		_claimOwner = std::thread::id{};
	}
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
		if (name.empty() || !isPrintableHeaderBytes(name) || !isPrintableHeaderBytes(value)
			|| name.find(' ') != std::string::npos || name.find(':') != std::string::npos) {
			_failed = true;
			_connectError = normalizeTransportError(
				NetTransportError::Other,
				"invalid header in endpoint config (name/value must be printable, no CR/LF): '" + h + "'");
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

	// 会话：HTTP 或 HTTPS（HTTPSClientSession 使用显式客户端 TLS 上下文——
	// 宿主已初始化则用宿主的，否则兕底 VERIFY_STRICT，不依赖全局默认）
	try {
		const std::string host = uri.getHost();
		const Poco::UInt16 port = static_cast<Poco::UInt16>(uri.getPort());
		std::shared_ptr<Poco::Net::HTTPClientSession> session;
		if (_useTls)
			session = std::make_shared<Poco::Net::HTTPSClientSession>(host, port,
																  ensureClientTlsContext());
		else
			session = std::make_shared<Poco::Net::HTTPClientSession>(host, port);
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
		}
		const int status = res.getStatus();
		if (status < 200 || status >= 300) {
			// 错误体仅作诊断：允许截断到 maxResponseBody，不再读取
			const Payload body = readBody(rs, _ep.maxResponseBody, nullptr);
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
		return normalizeTransportError(classifyPocoException(e), e.displayText());
	} catch (const std::exception& e) {
		abortResponse();
		_failed = true;
		return normalizeTransportError(NetTransportError::Other, e.what());
	}
}

NetError HttpTransport::recv(Payload& out) {
	const auto self = std::this_thread::get_id();
	std::shared_ptr<Poco::Net::HTTPClientSession> session;
	std::istream* rs = nullptr;
	{
		std::lock_guard lk(_ioMutex);
		if (_response) {
			// claim 内快照交换状态（与 close/析构的强制回收同锁互斥）：
			// 本地副本保活 session——close 超时强收仅 abort 中断读并释放
			// close 侧引用，本函数继续使用的 session 对象由副本持有至
			// 读取结束，不悬垂
			session = _session;
			rs = _response;
		} else if (_claimed && _claimOwner == self) {
			_claimed = false; // 无挂起响应：回收本线程遗留占用
			_claimOwner = std::thread::id{};
		}
	}
	if (!rs) {
		_ioCv.notify_all();
		return normalizeTransportError(NetTransportError::Other, "no pending response");
	}
	bool truncated = false;
	out = readBody(*rs, _ep.maxResponseBody, &truncated);
	bool preempted = false;
	{
		std::lock_guard lk(_ioMutex);
		_response = nullptr;
		// 交换期间会话被 close/析构强收（其侧引用已置空/替换）：本次读取被
		// abort 中断，返回错误而非静默短读（部分 body 不是有效交换结果）
		preempted = (_session != session);
	}
	if (session && (truncated || preempted)) {
		try {
			session->abort(); // 丢弃未读完的挂起响应（残留报文不污染下次交换）
		} catch (...) {
		}
	}
	if (truncated) {
		// 成功体超限：按错误归一化，session 由副本存活至本函数结束
		releaseClaimIfOwned(); // 交换异常结束：释放占用并唤醒等待者
		return normalizeTransportError(NetTransportError::Other,
									   "response body exceeds maxResponseBody ("
										   + std::to_string(_ep.maxResponseBody) + " bytes)");
	}
	if (preempted) {
		releaseClaimIfOwned(); // 交换异常结束：释放占用并唤醒等待者
		return normalizeTransportError(NetTransportError::Reset, "transport closed while reading response");
	}
	releaseClaimIfOwned(); // 交换结束：释放占用并唤醒等待者
	return {};
}

bool HttpTransport::alive() const {
	std::lock_guard lk(_ioMutex);
	return !_failed && _session != nullptr;
}

void HttpTransport::close() {
	std::unique_lock lk(_ioMutex);
	// 等他人交换收尾至多 5s（超时即占用泄漏——强制回收，close 不得挂死）
	const auto self = std::this_thread::get_id();
	const bool freeOrMine = !_claimed || _claimOwner == self;
	const bool acquired = freeOrMine || _ioCv.wait_for(lk, std::chrono::seconds(5),
													   [this, self] { return !_claimed || _claimOwner == self; });
	if (!acquired) {
		_claimed = false;
		_claimOwner = std::thread::id{};
	}
	lk.unlock();
	dropSession(); // 自持锁：abort 中断仍在进行的读（读方副本保活 session）
	{
		std::lock_guard lk2(_ioMutex);
		_failed = false;
		_connectError = {};
	}
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
		if (!rs)
			break; // eof/fail：读取结束
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
