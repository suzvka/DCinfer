#pragma once

// 内部共享（仅 src/）：入站 wire 应答的错误体组装与状态码文本。
// 监听器（NetListener_Http）与装配层（NetServerAdapter）共用，
// 保证闸门类应答（401/415/429/5xx）的错误体形态一致。

#include <nlohmann/json.hpp>

#include <string>

namespace DC::Net::detail {

/// @brief wire 错误体：{"error":{"code":...,"message":...}}（对端可解析出
/// message 作为 remoteDetail；code 未知不影响按状态码兜底归类）。
inline std::string wireErrorBody(const char* code, const std::string& message) {
	nlohmann::json j;
	j["error"] = {{"code", code}, {"message", message}};
	return j.dump();
}

/// @brief HTTP 状态码标准短语（Response 行用；未列举状态回落通用短语）。
inline const char* wireStatusText(int status) {
	switch (status) {
	case 200: return "OK";
	case 400: return "Bad Request";
	case 401: return "Unauthorized";
	case 403: return "Forbidden";
	case 404: return "Not Found";
	case 405: return "Method Not Allowed";
	case 413: return "Payload Too Large";
	case 415: return "Unsupported Media Type";
	case 429: return "Too Many Requests";
	case 500: return "Internal Server Error";
	case 503: return "Service Unavailable";
	default: return "Server Error";
	}
}

} // namespace DC::Net::detail
