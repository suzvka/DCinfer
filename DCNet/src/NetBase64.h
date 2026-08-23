#pragma once

#include <cstdint>
#include <string>

// DCNet 内部工具：极简 base64（RFC 4648，无填充依赖的外部实现）。
// 仅用于 DCNet v1 张量 JSON 线上格式（DESIGN.md §4）与测试。

namespace DC::Net::detail {

inline std::string base64Encode(const std::uint8_t* data, size_t len) {
	static constexpr char kTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	std::string out;
	out.reserve(((len + 2) / 3) * 4);
	size_t i = 0;
	while (i + 3 <= len) {
		uint32_t v = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) | uint32_t(data[i + 2]);
		out.push_back(kTable[(v >> 18) & 63]);
		out.push_back(kTable[(v >> 12) & 63]);
		out.push_back(kTable[(v >> 6) & 63]);
		out.push_back(kTable[v & 63]);
		i += 3;
	}
	const size_t rem = len - i;
	if (rem == 1) {
		uint32_t v = uint32_t(data[i]) << 16;
		out.push_back(kTable[(v >> 18) & 63]);
		out.push_back(kTable[(v >> 12) & 63]);
		out.push_back('=');
		out.push_back('=');
	} else if (rem == 2) {
		uint32_t v = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8);
		out.push_back(kTable[(v >> 18) & 63]);
		out.push_back(kTable[(v >> 12) & 63]);
		out.push_back(kTable[(v >> 6) & 63]);
		out.push_back('=');
	}
	return out;
}

inline std::string base64Decode(const std::string& in) {
	auto val = [](char c) -> int {
		if (c >= 'A' && c <= 'Z') return c - 'A';
		if (c >= 'a' && c <= 'z') return c - 'a' + 26;
		if (c >= '0' && c <= '9') return c - '0' + 52;
		if (c == '+') return 62;
		if (c == '/') return 63;
		return -1;
	};
	std::string out;
	out.reserve((in.size() / 4) * 3);
	uint32_t acc = 0;
	int bits = 0;
	for (char c : in) {
		if (c == '=')
			break; // 填充结束
		const int v = val(c);
		if (v < 0)
			continue; // 容忍空白/换行
		acc = (acc << 6) | uint32_t(v);
		bits += 6;
		if (bits >= 8) {
			bits -= 8;
			out.push_back(char((acc >> bits) & 0xFF));
		}
	}
	return out;
}

} // namespace DC::Net::detail
