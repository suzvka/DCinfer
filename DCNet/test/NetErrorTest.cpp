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
	CHECK(e.localStatus == Status::RemoteMalformed, "malformed → RemoteMalformed");
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

// ── 入站类：服务端 wire 逆向映射（M-server / DESIGN.md §6.1）──

TEST(wireStatusMapping) {
	CHECK(DC::Net::wireHttpStatusFor(Status::Ok) == 200, "Ok → 200");
	CHECK(DC::Net::wireHttpStatusFor(Status::InvalidInput) == 400, "InvalidInput → 400");
	CHECK(DC::Net::wireHttpStatusFor(Status::SchemaMismatch) == 422, "SchemaMismatch → 422");
	CHECK(DC::Net::wireHttpStatusFor(Status::ExecutionFailed) == 500, "ExecutionFailed → 500");
	CHECK(DC::Net::wireHttpStatusFor(Status::InternalError) == 500, "InternalError → 500");
}

TEST(wireCodeFor) {
	CHECK(std::string(DC::Net::wireCodeFor(Status::Ok)) == "ok", "code ok");
	CHECK(std::string(DC::Net::wireCodeFor(Status::InvalidInput)) == "invalid_input", "code invalid_input");
	CHECK(std::string(DC::Net::wireCodeFor(Status::SchemaMismatch)) == "schema_mismatch", "code schema_mismatch");
	CHECK(std::string(DC::Net::wireCodeFor(Status::ExecutionFailed)) == "execution_failed", "code execution_failed");
	CHECK(std::string(DC::Net::wireCodeFor(Status::InternalError)) == "internal_error", "code internal_error");
}

TEST(wireRoundTripParity) {
	// 语义一致性（DESIGN.md §6.1）：对端 normalizeHttpResponse 归一化结果 == 本地 status。
	// 请求体携带推荐 code（未知 code 不影响归类，按状态码兜底）。
	using DC::Net::normalizeHttpResponse;
	using DC::Net::wireHttpStatusFor;

	CHECK(normalizeHttpResponse(wireHttpStatusFor(Status::Ok), "{}").ok(), "Ok → 2xx 直接成功");

	{
		const auto e = normalizeHttpResponse(wireHttpStatusFor(Status::InvalidInput),
											 R"({"error":{"code":"invalid_input","message":"x"}})");
		CHECK(e.localStatus == Status::InvalidInput, "400+invalid_input → InvalidInput（与本地一致）");
	}
	{
		// SchemaMismatch 预留行：本地当前不产出该值（Node.h L125-131 预留），
		// 对端按 422 归一化为 InvalidInput —— 与本地形状违例的现行行为一致；
		// 本地改产后按需扩表维持一致（DESIGN.md §6.1 备注）。
		const auto e = normalizeHttpResponse(wireHttpStatusFor(Status::SchemaMismatch),
											 R"({"error":{"code":"schema_mismatch","message":"x"}})");
		CHECK(e.localStatus == Status::InvalidInput, "422+schema_mismatch → InvalidInput（预留行）");
	}
	{
		const auto e = normalizeHttpResponse(wireHttpStatusFor(Status::ExecutionFailed),
											 R"({"error":{"code":"execution_failed","message":"x"}})");
		CHECK(e.localStatus == Status::ExecutionFailed, "500+execution_failed → ExecutionFailed（与本地一致）");
	}
	{
		// 解析限度（DESIGN.md §6.1 备注）：非鉴权 InternalError 无忠实 wire 表示，
		// 按「本地执行失败 → 5xx」应答，对端归一化为 ExecutionFailed。
		const auto e = normalizeHttpResponse(wireHttpStatusFor(Status::InternalError),
											 R"({"error":{"code":"internal_error","message":"x"}})");
		CHECK(e.localStatus == Status::ExecutionFailed, "500+internal_error → ExecutionFailed（解析限度）");
	}
	// 无本地对应物的闸门类（DESIGN.md §6.1，由监听/装配层直接应答）：
	CHECK(normalizeHttpResponse(401, R"({"error":{"code":"unauthorized"}})").localStatus == Status::InternalError,
		  "401 → RemoteAuth → InternalError（remote:auth）");
	CHECK(normalizeHttpResponse(429, R"({"error":{"code":"overloaded"}})").localStatus == Status::ExecutionFailed,
		  "429 → RemoteRateLimited → ExecutionFailed（retryable）");
	CHECK(normalizeHttpResponse(415, R"({"error":{"code":"malformed_frame"}})").localStatus == Status::RemoteMalformed,
		  "415（未列举状态） → RemoteMalformed → RemoteMalformed（remote:malformed）");
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
	test_wireStatusMapping();
	test_wireCodeFor();
	test_wireRoundTripParity();
	std::printf("NetErrorTest: %d checks, %d failures\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}
