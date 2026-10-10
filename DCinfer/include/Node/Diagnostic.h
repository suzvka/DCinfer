#pragma once

#include <string>

namespace DC {

/// @brief 领域结构化诊断：子系统细分错误经 {domain, code, message} 附带上报，核心只透传。
struct Diagnostic {
	std::string domain;
	int code = 0;
	std::string message;
};

} // namespace DC
