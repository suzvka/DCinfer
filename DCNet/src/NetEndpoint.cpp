#include "DCNet/NetEndpoint.h"

#include "NodeException.h"

#include <cctype>
#include <string>

namespace DC::Net {

namespace {

/// 大小写不敏感识别 scheme（1 = http、2 = https、-1 = 未知）：
/// 未知 scheme 由调用方拒绝——静默降级为明文 HTTP 会把凭据发往非预期端点。
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

	// 提取 scheme 与 TLS 标记；未知 scheme（如 ftp://）在入口拒绝
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

	auto slash = s.find('/');
	std::string authority = (slash == std::string::npos) ? s : s.substr(0, slash);
	std::string path = (slash == std::string::npos) ? std::string() : s.substr(slash);

	// userinfo 不支持：拒绝而非静默忽略
	auto at = authority.rfind('@');
	if (at != std::string::npos)
		throw NodeException(NodeException::ErrorType::InternalError, "NetEndpoint::parse", "URI userinfo is not supported; configure authToken instead");

	// host[:port]（v1 不支持 IPv6 字面量）；端口严格解析：仅十进制数字、
	// 全串消费、≤65535
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

	// basePath：URL 显式给出则覆盖默认
	if (!path.empty())
		ep.basePath = path;
	if (ep.basePath.empty() || ep.basePath[0] != '/')
		ep.basePath = "/" + ep.basePath;

	ep.url = ep.endpoint();
	return ep;
}

std::string NetEndpoint::endpoint() const {
	if (!url.empty()) return url;
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
