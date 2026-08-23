// NetError 归一化映射表纯单测（DESIGN.md §6）
// 验证：网络错误 / HTTP 状态 / 远端错误体 → category / retryable / localStatus / 消息前缀

#include "DCNet/NetError.h"

#include <cstdio>
#include <string>

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

#define CHECK_MSG_PREFIX(msg, prefix)                                                                                  \
	do {                                                                                                               \
		++g_checks;                                                                                                    \
		if ((msg).rfind(prefix, 0) != 0) {                                                                             \
			++g_failures;                                                                                              \
			std::printf("FAIL %s:%d  message '%s' should start with '%s'\n", __FILE__, __LINE__, (msg).c_str(),        \
						prefix);                                                                                       \
		}                                                                                                              \
	} while (0)

#define TEST(name) static void test_##name()

using DC::Net::NetError;
using DC::Net::NetErrorCategory;
using DC::Net::NetTransportError;
using Status = DC::Node::Status; // MSVC 不支持非类作用域 using-declaration 引入嵌套别名

TEST(transportTimeout) {
	auto e = DC::Net::normalizeTransportError(NetTransportError::Timeout, "connect timed out");
	CHECK(e.category == NetErrorCategory::Timeout, "category should be Timeout");
	CHECK(e.retryable, "timeout should be retryable");
	CHECK(e.localStatus == Status::ExecutionFailed, "timeout → ExecutionFailed");
	CHECK_MSG_PREFIX(e.localMessage, "net:timeout");
	CHECK(e.localMessage.find("connect timed out") != std::string::npos, "detail should be appended");
}

TEST(transportUnreachable) {
	auto e = DC::Net::normalizeTransportError(NetTransportError::ConnectionRefused, "refused");
	CHECK(e.category == NetErrorCategory::Unreachable, "refused → Unreachable");
	CHECK(e.retryable, "unreachable should be retryable");
	CHECK(e.localStatus == Status::ExecutionFailed, "unreachable → ExecutionFailed");
	CHECK_MSG_PREFIX(e.localMessage, "net:unreachable");
	CHECK(e.localMessage.find("refused") != std::string::npos, "detail should be appended");
}

TEST(httpStatus4xx) {
	auto e = DC::Net::normalizeHttpStatus(400);
	CHECK(e.category == NetErrorCategory::RemoteRejected, "400 → RemoteRejected");
	CHECK(!e.retryable, "400 not retryable");
	CHECK(e.localStatus == Status::InvalidInput, "400 → InvalidInput");
	CHECK_MSG_PREFIX(e.localMessage, "remote:invalid_request");
}

TEST(httpStatusAuth) {
	auto e = DC::Net::normalizeHttpStatus(401);
	CHECK(e.category == NetErrorCategory::RemoteAuth, "401 → RemoteAuth");
	CHECK(!e.retryable, "401 not retryable");
	CHECK(e.localStatus == Status::InternalError, "401 → InternalError");
	CHECK_MSG_PREFIX(e.localMessage, "remote:auth");

	e = DC::Net::normalizeHttpStatus(403);
	CHECK(e.category == NetErrorCategory::RemoteAuth, "403 → RemoteAuth");
}

TEST(httpStatusNotFound) {
	auto e = DC::Net::normalizeHttpStatus(404);
	CHECK(e.category == NetErrorCategory::RemoteRejected, "404 → RemoteRejected");
	CHECK(e.localStatus == Status::InvalidInput, "404 → InvalidInput");
	CHECK_MSG_PREFIX(e.localMessage, "remote:not_found");
}

TEST(httpStatusTimeoutAndRateLimit) {
	auto e = DC::Net::normalizeHttpStatus(408);
	CHECK(e.category == NetErrorCategory::Timeout, "408 → Timeout");
	CHECK(e.retryable, "408 retryable");
	CHECK_MSG_PREFIX(e.localMessage, "net:timeout");

	e = DC::Net::normalizeHttpStatus(429);
	CHECK(e.category == NetErrorCategory::RemoteRateLimited, "429 → RemoteRateLimited");
	CHECK(e.retryable, "429 retryable");
	CHECK(e.localStatus == Status::ExecutionFailed, "429 → ExecutionFailed");
	CHECK_MSG_PREFIX(e.localMessage, "remote:rate_limited");
}

TEST(httpStatus5xx) {
	auto e = DC::Net::normalizeHttpStatus(503, "temporarily down");
	CHECK(e.category == NetErrorCategory::RemoteServer, "503 → RemoteServer");
	CHECK(e.retryable, "5xx retryable");
	CHECK(e.localStatus == Status::ExecutionFailed, "503 → ExecutionFailed");
	CHECK_MSG_PREFIX(e.localMessage, "remote:server_error");
	CHECK(e.localMessage.find("temporarily down") != std::string::npos, "body should be preserved");
}

TEST(remoteBodyKnownCode) {
	auto e = DC::Net::normalizeRemoteBody(R"({"error":{"code":"invalid_api_key","message":"bad key"}})");
	CHECK(e.category == NetErrorCategory::RemoteAuth, "invalid_api_key → RemoteAuth");
	CHECK(e.code == "invalid_api_key", "code should be extracted");
	CHECK(e.remoteDetail == "bad key", "message should be extracted");
	CHECK(e.localStatus == Status::InternalError, "auth → InternalError");
	CHECK_MSG_PREFIX(e.localMessage, "remote:auth");

	e = DC::Net::normalizeRemoteBody(R"({"error":{"code":"rate_limit_exceeded","message":"slow down"}})");
	CHECK(e.category == NetErrorCategory::RemoteRateLimited, "rate_limit_exceeded → RemoteRateLimited");
	CHECK(e.retryable, "rate limited retryable");

	e = DC::Net::normalizeRemoteBody(R"({"error":{"code":"model_not_found","message":"no such model"}})");
	CHECK(e.category == NetErrorCategory::RemoteRejected, "model_not_found → RemoteRejected");
	CHECK(e.localStatus == Status::InvalidInput, "not found → InvalidInput");
	CHECK_MSG_PREFIX(e.localMessage, "remote:model_not_found");
}

TEST(remoteBodyStringAndDetail) {
	auto e = DC::Net::normalizeRemoteBody(R"({"error":"boom"})");
	CHECK(e.category == NetErrorCategory::RemoteRejected, "string error → fallback RemoteRejected");
	CHECK(e.remoteDetail == "boom", "string error message extracted");
	CHECK(e.localStatus == Status::InvalidInput, "fallback rejected → InvalidInput");

	e = DC::Net::normalizeRemoteBody(R"({"detail":"no such model"})");
	CHECK(e.category == NetErrorCategory::RemoteRejected, "detail → fallback RemoteRejected");
	CHECK(e.remoteDetail == "no such model", "detail extracted");
}

TEST(remoteBodyUnknownCodeUsesFallback) {
	auto e = DC::Net::normalizeRemoteBody(R"({"error":{"code":"mystery_code","message":"x"}})",
										  DC::Net::NetErrorCategory::RemoteServer);
	CHECK(e.category == NetErrorCategory::RemoteServer, "unknown code → caller fallback");
	CHECK(e.retryable, "server fallback retryable");
	CHECK_MSG_PREFIX(e.localMessage, "remote:server_error");
}

TEST(remoteBodyMalformed) {
	auto e = DC::Net::normalizeRemoteBody("this is not json at all");
	CHECK(e.category == NetErrorCategory::RemoteMalformed, "non-json → RemoteMalformed");
	CHECK(!e.retryable, "malformed not retryable");
	CHECK(e.localStatus == Status::InternalError, "malformed → InternalError");
	CHECK_MSG_PREFIX(e.localMessage, "remote:malformed");
}

TEST(httpResponseCombo) {
	// 2xx → 成功
	CHECK(DC::Net::normalizeHttpResponse(200, "ok").ok(), "2xx should be success");

	// 已知 code 优先于状态码
	auto e = DC::Net::normalizeHttpResponse(400, R"({"error":{"code":"invalid_api_key","message":"bad"}})");
	CHECK(e.category == NetErrorCategory::RemoteAuth, "known code should override 400 status");
	CHECK(e.localStatus == Status::InternalError, "auth → InternalError");

	// 未知 code → 按状态码兜底
	e = DC::Net::normalizeHttpResponse(500, R"({"error":{"code":"weird","message":"x"}})");
	CHECK(e.category == NetErrorCategory::RemoteServer, "unknown code on 500 → RemoteServer");
	CHECK(e.retryable, "5xx retryable");

	// 无报文 → 状态码兜底
	e = DC::Net::normalizeHttpResponse(404, "");
	CHECK(e.category == NetErrorCategory::RemoteRejected, "404 without body → RemoteRejected");
	CHECK_MSG_PREFIX(e.localMessage, "remote:not_found");

	// 报文不可解析 → 状态码兜底
	e = DC::Net::normalizeHttpResponse(503, "oops");
	CHECK(e.category == NetErrorCategory::RemoteServer, "unparseable body on 503 → RemoteServer");
	CHECK_MSG_PREFIX(e.localMessage, "remote:server_error");
}

TEST(defaultIsSuccess) {
	NetError e;
	CHECK(e.ok(), "default NetError should be success");
}

int main() {
	test_transportTimeout();
	test_transportUnreachable();
	test_httpStatus4xx();
	test_httpStatusAuth();
	test_httpStatusNotFound();
	test_httpStatusTimeoutAndRateLimit();
	test_httpStatus5xx();
	test_remoteBodyKnownCode();
	test_remoteBodyStringAndDetail();
	test_remoteBodyUnknownCodeUsesFallback();
	test_remoteBodyMalformed();
	test_httpResponseCombo();
	test_defaultIsSuccess();
	std::printf("NetErrorTest: %d checks, %d failures\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}
