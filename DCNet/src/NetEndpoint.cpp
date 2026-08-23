#include "DCNet/NetEndpoint.h"

#include <cstdlib>
#include <string>

namespace DC::Net {

NetEndpoint NetEndpoint::parse(const std::string& text) {
	NetEndpoint ep;
	std::string s = text;

	// 1. 提取 scheme（http/https）与 TLS 标记
	auto schemePos = s.find("://");
	if (schemePos != std::string::npos) {
		const std::string scheme = s.substr(0, schemePos);
		ep.useTls = (scheme == "https");
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

	// 4. host[:port]（v1 不处理 IPv6 字面量）
	auto colon = authority.rfind(':');
	if (colon != std::string::npos) {
		ep.host = authority.substr(0, colon);
		ep.port = std::atoi(authority.substr(colon + 1).c_str());
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
