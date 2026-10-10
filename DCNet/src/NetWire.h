#pragma once

// 入站 wire 错误体组装与状态码文本：监听器与装配层共用，保证错误体形态一致。

#include <nlohmann/json.hpp>

#include <string>

namespace DC::Net::detail {

/// wire 错误体：{"error":{"code":...,"message":...}}，对端可解析出 message。
inline std::string wireErrorBody(const char* code, const std::string& message) {
	nlohmann::json j;
	j["error"] = {{"code", code}, {"message", message}};
	return j.dump();
}

/// HTTP 状态码标准短语；未列举状态回落通用短语。
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
