#include "DCNet/NetError.h"

#include <nlohmann/json.hpp>

#include <exception>
#include <string>
#include <unordered_map>

namespace DC::Net {

namespace {

/// HTTP 状态码 → 默认分类（报文解析不出已知 code 时的兜底）。
NetErrorCategory categoryForHttpStatus(int status) {
	if (status == 400 || status == 422)
		return NetErrorCategory::RemoteRejected;
	if (status == 401 || status == 403)
		return NetErrorCategory::RemoteAuth;
	if (status == 404)
		return NetErrorCategory::RemoteRejected;
	if (status == 408)
		return NetErrorCategory::Timeout;
	if (status == 429)
		return NetErrorCategory::RemoteRateLimited;
	if (status >= 500 && status < 600)
		return NetErrorCategory::RemoteServer;
	return NetErrorCategory::RemoteMalformed;
}

/// 已知远端错误码 → 精确分类（OpenAI 兼容服务常用错误码）。
const std::unordered_map<std::string, NetErrorCategory>& knownRemoteCodes() {
	static const std::unordered_map<std::string, NetErrorCategory> k = {
		{"invalid_api_key", NetErrorCategory::RemoteAuth},
		{"authentication_error", NetErrorCategory::RemoteAuth},
		{"permission_error", NetErrorCategory::RemoteAuth},
		{"model_not_found", NetErrorCategory::RemoteRejected},
		{"invalid_request_error", NetErrorCategory::RemoteRejected},
		{"bad_request", NetErrorCategory::RemoteRejected},
		{"rate_limit_exceeded", NetErrorCategory::RemoteRateLimited},
		{"rate_limited", NetErrorCategory::RemoteRateLimited},
		{"server_error", NetErrorCategory::RemoteServer},
		{"internal_server_error", NetErrorCategory::RemoteServer},
		{"timeout", NetErrorCategory::Timeout},
		{"request_timeout", NetErrorCategory::Timeout},
	};
	return k;
}

} // namespace

NetError finalize(NetError e) {
	switch (e.category) {
	case NetErrorCategory::None:
		e.localStatus = Node::Status::Ok;
		e.localMessage.clear();
		return e;
	case NetErrorCategory::Timeout:
		e.localStatus = Node::Status::ExecutionFailed;
		e.localMessage = "net:timeout";
		break;
	case NetErrorCategory::Unreachable:
		e.localStatus = Node::Status::ExecutionFailed;
		e.localMessage = "net:unreachable";
		break;
	case NetErrorCategory::RemoteRejected:
		e.localStatus = Node::Status::InvalidInput;
		e.localMessage = e.code.empty() ? "remote:invalid_request" : "remote:" + e.code;
		break;
	case NetErrorCategory::RemoteAuth:
		e.localStatus = Node::Status::InternalError;
		e.localMessage = "remote:auth";
		break;
	case NetErrorCategory::RemoteRateLimited:
		e.localStatus = Node::Status::ExecutionFailed;
		e.localMessage = "remote:rate_limited";
		break;
	case NetErrorCategory::RemoteServer:
		e.localStatus = Node::Status::ExecutionFailed;
		e.localMessage = "remote:server_error";
		break;
	case NetErrorCategory::RemoteMalformed:
		e.localStatus = Node::Status::InternalError;
		e.localMessage = "remote:malformed";
		break;
	case NetErrorCategory::Other:
		e.localStatus = Node::Status::ExecutionFailed;
		e.localMessage = "net:error";
		break;
	}
	if (!e.remoteDetail.empty())
		e.localMessage += " - " + e.remoteDetail;
	return e;
}

// ── 入站 wire 逆向映射（M-server；DESIGN.md §6.1）──
// 逆向表与 categoryForHttpStatus / finalize 正向表逐行对偶：
//   Ok              → 200（2xx 直接成功）
//   InvalidInput    → 400（RemoteRejected → InvalidInput）
//   SchemaMismatch  → 422（预留行：本地当前不产出该值，对端归一化仍为
//                     InvalidInput，与本地形状违例现行行为一致；本地改产后
//                     按需扩表，DESIGN.md §6.1 备注）
//   ExecutionFailed → 500（RemoteServer → ExecutionFailed）
//   InternalError   → 500（解析限度：对端归一化为 ExecutionFailed）
int wireHttpStatusFor(Node::Status status) {
	switch (status) {
	case Node::Status::Ok:
		return 200;
	case Node::Status::InvalidInput:
		return 400;
	case Node::Status::SchemaMismatch:
		return 422;
	case Node::Status::ExecutionFailed:
	case Node::Status::InternalError:
		return 500;
	}
	return 500;
}

const char* wireCodeFor(Node::Status status) {
	switch (status) {
	case Node::Status::Ok:
		return "ok";
	case Node::Status::InvalidInput:
		return "invalid_input";
	case Node::Status::SchemaMismatch:
		return "schema_mismatch";
	case Node::Status::ExecutionFailed:
		return "execution_failed";
	case Node::Status::InternalError:
		return "internal_error";
	}
	return "internal_error";
}

NetError normalizeTransportError(NetTransportError err, std::string detail) {
	NetError e;
	switch (err) {
	case NetTransportError::Timeout:
		e.category = NetErrorCategory::Timeout;
		e.retryable = true;
		break;
	case NetTransportError::ConnectionRefused:
	case NetTransportError::DnsFailed:
	case NetTransportError::Reset:
	case NetTransportError::TlsFailed:
		e.category = NetErrorCategory::Unreachable;
		e.retryable = true;
		break;
	default:
		e.category = NetErrorCategory::Other;
		break;
	}
	e.remoteDetail = std::move(detail);
	return finalize(std::move(e));
}

NetError normalizeHttpStatus(int status, std::string body) {
	NetError e;
	e.category = categoryForHttpStatus(status);
	// 404 细化：remote:not_found（区别于 generic invalid_request）
	if (status == 404)
		e.code = "not_found";
	if (e.category == NetErrorCategory::Timeout || e.category == NetErrorCategory::RemoteServer ||
		e.category == NetErrorCategory::RemoteRateLimited)
		e.retryable = true;
	e.remoteDetail = std::move(body);
	return finalize(std::move(e));
}

NetError normalizeRemoteBody(const std::string& body, NetErrorCategory fallback) {
	NetError e;
	e.remoteDetail = body;
	try {
		const auto j = nlohmann::json::parse(body);
		if (j.contains("error")) {
			const auto& err = j["error"];
			if (err.is_string()) {
				e.remoteDetail = err.get<std::string>();
			} else if (err.is_object()) {
				if (err.contains("code") && err["code"].is_string())
					e.code = err["code"].get<std::string>();
				if (err.contains("message") && err["message"].is_string())
					e.remoteDetail = err["message"].get<std::string>();
			}
		} else if (j.contains("detail") && j["detail"].is_string()) {
			e.remoteDetail = j["detail"].get<std::string>();
		} else if (j.contains("message") && j["message"].is_string()) {
			e.remoteDetail = j["message"].get<std::string>();
		}
	} catch (const std::exception&) {
		e.category = NetErrorCategory::RemoteMalformed;
		return finalize(std::move(e));
	}

	if (!e.code.empty()) {
		const auto& known = knownRemoteCodes();
		auto it = known.find(e.code);
		e.category = (it != known.end()) ? it->second : fallback;
	} else {
		e.category = fallback;
	}
	e.retryable = (e.category == NetErrorCategory::Timeout || e.category == NetErrorCategory::RemoteServer ||
				   e.category == NetErrorCategory::RemoteRateLimited);
	return finalize(std::move(e));
}

NetError normalizeHttpResponse(int status, const std::string& body) {
	if (status >= 200 && status < 300)
		return {};
	const NetErrorCategory fallback = categoryForHttpStatus(status);
	if (!body.empty()) {
		NetError e = normalizeRemoteBody(body, fallback);
		if (e.category != NetErrorCategory::RemoteMalformed) {
			// 报文可解析：已知 code 优先；无 code 时叠加状态码细化（如 404 → not_found）
			if (e.code.empty() && status == 404)
				e.code = "not_found";
			return finalize(std::move(e));
		}
	}
	return normalizeHttpStatus(status, body);
}

} // namespace DC::Net
