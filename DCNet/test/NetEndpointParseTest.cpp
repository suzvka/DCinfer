// NetEndpoint::parse 严格校验回归测试：scheme 大小写 / 未知 scheme 拒绝 /
// 端口严格解析 / 无 scheme 简写兼容。
#include "DCNet/NetEndpoint.h"
#include "NodeException.h"

#include <cstdio>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond, msg)                                                                                               \
	do {                                                                                                               \
		++g_checks;                                                                                                    \
		if (!(cond)) {                                                                                                 \
			++g_failures;                                                                                              \
			std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);                                                  \
		}                                                                                                              \
	} while (0)

#define CHECK_THROWS(stmt, msg)                                                                                        \
	do {                                                                                                               \
		++g_checks;                                                                                                    \
		bool thrown = false;                                                                                           \
		try {                                                                                                          \
			stmt;                                                                                                      \
		} catch (const DC::NodeException&) {                                                                           \
			thrown = true;                                                                                             \
		} catch (...) {                                                                                                \
			thrown = true;                                                              \
		}                                                                                                              \
		if (!thrown) {                                                                                                 \
			++g_failures;                                                                                              \
			std::printf("FAIL %s:%d  %s (no NodeException)\n", __FILE__, __LINE__, msg);                               \
		}                                                                                                              \
	} while (0)

using DC::Net::NetEndpoint;

static void testKnownSchemesCaseInsensitive() {
	{
		auto ep = NetEndpoint::parse("HTTPS://example.com/v1");
		CHECK(ep.useTls, "HTTPS:// must be recognized as TLS");
		CHECK(ep.host == "example.com", "host must be preserved");
	}
	{
		auto ep = NetEndpoint::parse("Http://example.com/v1");
		CHECK(!ep.useTls, "Http:// must be recognized as plain HTTP");
	}
	{
		auto ep = NetEndpoint::parse("hTtPs://example.com");
		CHECK(ep.useTls, "mixed-case scheme must be recognized");
	}
}

static void testUnknownSchemeRejected() {
	CHECK_THROWS(NetEndpoint::parse("ftp://example.com/v1"), "ftp:// must be rejected");
	CHECK_THROWS(NetEndpoint::parse("gopher://example.com"), "gopher:// must be rejected");
	CHECK_THROWS(NetEndpoint::parse("javascript://example.com"), "javascript:// must be rejected");
}

static void testPortValidation() {
	{
		auto ep = NetEndpoint::parse("http://example.com:8080/v1");
		CHECK(ep.port == 8080, "valid port must be accepted");
	}
	{
		auto ep = NetEndpoint::parse("http://example.com:65535");
		CHECK(ep.port == 65535, "max port must be accepted");
	}
	CHECK_THROWS(NetEndpoint::parse("http://example.com:abc/v1"), "non-numeric port must be rejected");
	CHECK_THROWS(NetEndpoint::parse("http://example.com:-1/v1"), "negative port must be rejected");
	CHECK_THROWS(NetEndpoint::parse("http://example.com:99999/v1"), "out-of-range port must be rejected");
	CHECK_THROWS(NetEndpoint::parse("http://example.com:/v1"), "empty port must be rejected");
	CHECK_THROWS(NetEndpoint::parse("http://example.com: 80/v1"), "port with whitespace must be rejected");
}

static void testNoSchemeShorthandUnchanged() {
	{
		auto ep = NetEndpoint::parse("example.com");
		CHECK(!ep.useTls, "no-scheme shorthand defaults to HTTP");
		CHECK(ep.host == "example.com", "shorthand host");
		CHECK(ep.port == 0, "shorthand without port keeps 0 (protocol default via Poco::URI)");
		CHECK(ep.basePath == "/v1", "shorthand without path keeps default basePath");
	}
	{
		auto ep = NetEndpoint::parse("host:8080/path");
		CHECK(ep.host == "host", "shorthand host with port");
		CHECK(ep.port == 8080, "shorthand port parsed strictly");
		CHECK(ep.basePath == "/path", "shorthand explicit path");
	}
	{
		// port=0 则不附加端口；下游 Poco::URI 回退协议默认端口，属兼容行为
		auto ep = NetEndpoint::parse("http://example.com");
		CHECK(ep.port == 0 && !ep.useTls, "scheme without port defaults to protocol default port");
		CHECK(ep.endpoint() == "http://example.com/v1", "endpoint composition unchanged");
	}
}

int main() {
	testKnownSchemesCaseInsensitive();
	testUnknownSchemeRejected();
	testPortValidation();
	testNoSchemeShorthandUnchanged();

	std::printf("%d checks, %d failures\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}
