#pragma once

#include "NetTransport.h"
#include "Tensor.hpp"

#include <memory>
#include <string>
#include <unordered_map>

namespace DC::Net {

/// @brief 服务端协议映射（DcNetCodec 的镜像；DESIGN.md §3.6）。
///
/// 载荷格式复用出站 codec（v1：张量 JSON，数值 base64 / Data 文本 UTF-8），
/// 不另造格式。错误归一化不属 codec 职责：
/// decodeRequest 抛异常视为 wire 级垃圾报文（监听器回 415，对端归一化
/// ExecutionFailed + dcnet 诊断 code=RemoteMalformed，见 DESIGN.md §6.1）；
/// 本地执行失败经 wireStatusFor 逆向映射。
///
/// 不暴露 Node::RunContext（ADR-7）：服务端一请求一节点实例，codec 以
/// 「端口名 ↔ 张量」为界，实例级隔离由装配层保证。
struct DcNetServerCodec {
	virtual ~DcNetServerCodec() = default;

	/// @brief 对方请求报文 → 本地输入端口张量映射（端口名 → 张量）。
	/// 抛异常 = 报文不可解析（wire 级损坏，非图级输入错误；见 DESIGN.md §6.1）。
	virtual std::unordered_map<std::string, Tensor> decodeRequest(const Payload& request) = 0;

	/// @brief 本地输出端口张量映射 → 对方响应报文。
	/// 输出缺端口 / 编码失败抛异常（装配层按本地执行失败语义应答 5xx）。
	virtual Payload encodeResponse(const std::unordered_map<std::string, Tensor>& outputs) = 0;

	/// @brief 协议子路径（如 "/infer"），与出站 DcNetCodec::requestPath 对称；
	/// 监听完整路径 = basePath + requestPath。
	virtual std::string requestPath() const { return "/infer"; }
};

// 服务端 codec 工厂（makeTensorJsonServerCodec）声明于 NetCodec_Tensor.h，
// 与出站 tensor/text codec 工厂并列——载荷格式同源维护。

} // namespace DC::Net
