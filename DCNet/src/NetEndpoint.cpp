#include "DCNet/NetEndpoint.h"

#include "NodeException.h"

#include <cctype>
#include <string>

namespace DC::Net {

namespace {

/// 大小写不敏感识别已知 scheme：0 = 无 scheme（默认 http）、1 = http、
/// 2 = https、-1 = 未知 scheme（调用方拒绝——静默当作明文 HTTP 会把
/// 凭据经明文通道发往非预期协议端点）。
int classifyScheme(const std::string& scheme) {
	std::string lower;
	lower.reserve(scheme.size());
	for (char c : scheme)
		lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
	if (lower == "http")
		return 1;
	if (lower == "https")
		return 2;
	return -1;
}

} // namespace

NetEndpoint NetEndpoint::parse(const std::string& text) {
	NetEndpoint ep;
	std::string s = text;

	// 1. 提取 scheme（http/https 大小写不敏感）与 TLS 标记；显式未知
	//    scheme（如 ftp://）在入口拒绝（P1），不再静默降级为明文 HTTP
	auto schemePos = s.find("://");
	if (schemePos != std::string::npos) {
		const int known = classifyScheme(s.substr(0, schemePos));
		if (known < 0)
			throw NodeException(NodeException::ErrorType::InternalError, "NetEndpoint::parse",
								"unsupported URL scheme in endpoint '" + text
									+ "' (only http/https are supported)");
		ep.useTls = (known == 2);
		s = s.substr(schemePos + 3);
	}

	// 2. 分离 authority 与 path
	auto slash = s.find('/');
	std::string authority = (slash == std::string::npos) ? s : s.substr(0, slash);
	std::string path = (slash == std::string::npos) ? std::string() : s.substr(slash);

	// 3. 剥离 userinfo（http://user:pass@host —— v1 忽略）
	auto at = authority.rfind('@');
	if (at != std::string::npos)
		authority = authority.substr(at + 1);

	// 4. host[:port]（v1 不处理 IPv6 字面量）。端口严格解析（P1）：仅十进制
	//    数字、全串消费、≤65535——atoi 时代非法/负数/超范围值会被静默接受
	//    （归零后回退默认端口，或溢出为未定义值）
	auto colon = authority.rfind(':');
	if (colon != std::string::npos) {
		ep.host = authority.substr(0, colon);
		const std::string portStr = authority.substr(colon + 1);
		auto reject = [&](const std::string& why) {
			throw NodeException(NodeException::ErrorType::InternalError, "NetEndpoint::parse",
								why + " in endpoint '" + text + "'");
		};
		if (portStr.empty())
			reject("empty port");
		if (portStr.find_first_not_of("0123456789") != std::string::npos)
			reject("invalid port '" + portStr + "'");
		if (portStr.size() > 5)
			reject("port '" + portStr + "' out of range");
		const int port = std::stoi(portStr);
		if (port > 65535)
			reject("port '" + portStr + "' out of range");
		ep.port = port;
	} else {
		ep.host = authority;
	}

	// 5. basePath（默认 /v1；URL 中显式给出则以显式值为准）
	if (!path.empty())
		ep.basePath = path;
	if (ep.basePath.empty() || ep.basePath[0] != '/')
		ep.basePath = "/" + ep.basePath;

	ep.url = ep.endpoint();
	return ep;
}

std::string NetEndpoint::endpoint() const {
	std::string s = useTls ? "https://" : "http://";
	s += host;
	if (port > 0)
		s += ":" + std::to_string(port);
	if (!basePath.empty()) {
		if (basePath[0] != '/')
			s += '/';
		s += basePath;
	}
	return s;
}

} // namespace DC::Net
