#pragma once

#include <string>

namespace DC {

/// @brief 领域结构化诊断：子系统错误分类不进入核心状态枚举的逃生通道。
///
/// 核心执行状态（NodeStatus）保持最小通用词表；后端 / 协议 / 子系统的
/// 细分错误经 Diagnostic{domain, code, message} 附带上报——
/// 分类法保留在产生它的子系统（如 DCNet 的 NetErrorCategory），核心只透传。
/// 典型用例：远端报文结构异常在 DCNet 内归类为 RemoteMalformed，
/// 对核心仅表现为 ExecutionFailed + Diagnostic{domain="dcnet", code=...}。
struct Diagnostic {
	std::string domain;  ///< 诊断域（如 "dcnet"、"onnx"）；空串表示无领域诊断
	int code = 0;        ///< 领域内错误码（语义由 domain 定义）
	std::string message; ///< 人类可读详情（与所在 NodeResult/TaskError 的 message 可重复）
};

} // namespace DC
