#pragma once

#include <cctype>
#include <filesystem>
#include <string>
#include <string_view>

namespace DC::Ir::detail {

namespace fs = std::filesystem;

/// @brief 校验归档条目名的纯词法安全性，与解压目标目录无关。
///
/// 拒绝规则跨平台一致，不依赖宿主平台路径语义：空路径或内嵌 NUL；
/// 前导 '/' 即 Unix 绝对或 "//" 即 UNC；盘符前缀，跨平台移动后会获得新语义；
/// 任意 ".." 父目录跳转组件；Windows 罪名字形，如设备名、ADS 冒号、尾点尾空格。
/// 反斜杠统一归一为 '/' 即 ZIP 规范分隔符；本函数只做词法校验，不触碰文件系统。
///
/// @param rel      归档内条目路径，如 "models/resnet.onnx"
/// @param normPath 可选输出：反斜杠归一为 '/' 后的条目名
/// @param reason   可选输出：拒绝原因描述
/// @return true 安全；false 拒绝
inline bool isSafeArchiveEntryName(std::string_view rel, std::string* normPath = nullptr,
								   std::string* reason = nullptr) {
	auto fail = [reason](const char* why) {
		if (reason)
			*reason = why;
		return false;
	};

	if (rel.empty())
		return fail("empty archive path");
	if (rel.find('\0') != std::string_view::npos)
		return fail("archive path contains embedded NUL");

	// 反斜杠统一折为 '/' 即 ZIP 规范分隔符，保证跨平台一致判定
	std::string normalized(rel);
	for (auto& c : normalized) {
		if (c == '\\')
			c = '/';
	}

	// 显式拒绝歧义前缀，不依赖 std::filesystem 平台分支语义
	if (normalized.front() == '/')
		return fail("absolute archive path is not allowed");
	if (normalized.size() >= 2 && normalized[1] == ':'
		&& ((normalized[0] >= 'A' && normalized[0] <= 'Z') || (normalized[0] >= 'a' && normalized[0] <= 'z')))
		return fail("drive-letter archive path is not allowed");

	fs::path p(normalized);
	if (p.is_absolute() || p.has_root_name())
		return fail("absolute or rooted archive path is not allowed");

	// Windows 罪名字形：设备名命中 DOS 设备、冒号写入 ADS、尾点尾空格被
	// Win32 静默裁剪，三者都会使落盘位置与声明路径不一致，跨平台一致拒绝。
	auto isReservedDeviceName = [](const std::string& comp) {
		std::string stem = comp;
		if (auto dot = stem.find('.'); dot != std::string::npos)
			stem = stem.substr(0, dot); // "CON.txt" 同样是设备名
		std::string upper;
		upper.reserve(stem.size());
		for (char c : stem)
			upper.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
		if (upper == "CON" || upper == "PRN" || upper == "AUX" || upper == "NUL")
			return true;
		if (upper.size() == 4 && (upper.rfind("COM", 0) == 0 || upper.rfind("LPT", 0) == 0)
			&& upper[3] >= '1' && upper[3] <= '9')
			return true;
		return false;
	};

	for (const auto& comp : p) {
		if (comp == "..")
			return fail("parent directory traversal ('..') is not allowed");
		const std::string c = comp.string();
		if (c.find(':') != std::string::npos)
			return fail("colon (alternate data stream) in archive path is not allowed");
		if (c != "." && !c.empty() && (c.back() == '.' || c.back() == ' '))
			return fail("archive path component with trailing dot or space is not allowed");
		if (isReservedDeviceName(c))
			return fail("Windows reserved device name in archive path is not allowed");
	}

	if (normPath)
		*normPath = normalized;
	return true;
}

/// @brief 校验归档条目相对路径可解压到 baseDir 之内。
///
/// 两层：先 isSafeArchiveEntryName 纯词法校验，再归一化组件级包含检查以防 base
/// 前缀误判；符号链接逃逸由调用侧另行防御。
///
/// @param rel     归档内条目路径，如 "models/resnet.onnx"
/// @param baseDir 解包目标基目录，即临时目录
/// @param reason  可选输出：拒绝原因描述
/// @return true 安全；false 拒绝
inline bool isSafeArchiveRelPath(std::string_view rel, const std::filesystem::path& baseDir,
								 std::string* reason = nullptr) {
	auto fail = [reason](const char* why) {
		if (reason)
			*reason = why;
		return false;
	};

	std::string normalized;
	if (!isSafeArchiveEntryName(rel, &normalized, reason))
		return false;

	// 组件级包含校验：归一化后 (baseDir / rel) 必须以 baseDir 的组件序列为前缀；
	// lexically_normal 为纯词法运算，不触碰文件系统。
	const auto base = baseDir.lexically_normal();
	const auto full = (base / fs::path(normalized)).lexically_normal();
	auto baseIt = base.begin();
	auto fullIt = full.begin();
	for (; baseIt != base.end(); ++baseIt, ++fullIt) {
		if (fullIt == full.end() || *fullIt != *baseIt)
			return fail("archive path escapes the extraction directory after normalization");
	}
	return true;
}

} // namespace DC::Ir::detail
