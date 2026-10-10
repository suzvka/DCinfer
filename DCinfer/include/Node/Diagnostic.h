#pragma once

#include <string>

namespace DC {

/// @brief 领域结构化诊断：子系统细分错误经 {domain, code, message} 附带上报，核心只透传。
struct Diagnostic {
	std::string domain;  ///< 诊断域（子系统定义；空串表示无领域诊断）
	int code = 0;        ///< 领域内错误码
	std::string message; ///< 人类可读详情
};

} // namespace DC
