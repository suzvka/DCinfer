#pragma once

#include "NetTransport.h"
#include "Tensor.hpp"

#include <memory>
#include <string>
#include <unordered_map>

namespace DC::Net {

/// 服务端协议映射，出站 DcNetCodec 镜像：载荷格式复用出站 codec，不另造。
///
/// decodeRequest 抛异常即 wire 级垃圾报文，监听器回 415；本地执行失败
/// 经 wireStatusFor 逆向映射。codec 以端口名与张量为界，不暴露 RunContext。
struct DcNetServerCodec {
	virtual ~DcNetServerCodec() = default;

	/// 对方请求报文映射到输入端口张量；抛异常即报文不可解析，属 wire 级损坏。
	virtual std::unordered_map<std::string, Tensor> decodeRequest(const Payload& request) = 0;

	/// 输出端口张量映射到响应报文；缺端口或编码失败抛异常，按执行失败应答 5xx。
	virtual Payload encodeResponse(const std::unordered_map<std::string, Tensor>& outputs) = 0;

	/// 协议子路径，与出站 codec 对称；监听路径为 basePath 加 requestPath。
	virtual std::string requestPath() const { return "/infer"; }
};

// 服务端 codec 工厂 makeTensorJsonServerCodec 声明于 NetCodec_Tensor.h。

} // namespace DC::Net
