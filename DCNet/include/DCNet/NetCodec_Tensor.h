#pragma once

#include "NetCodec.h"
#include "NetServerCodec.h"

#include <memory>

namespace DC::Net {

/// @brief 张量 JSON codec（DCNet v1 线上格式）——数值张量端口。
///
/// 端口（本地形状规则）：in "data"（Float，任意形状）→ out "result"（Float）。
/// 报文格式：
/// @code
///   {"dtype":"float32","shape":[1,1,28,28],"data":"<base64>"}
/// @endcode
/// 请求路径：{basePath}/infer
std::shared_ptr<DcNetCodec> makeTensorJsonCodec();

/// @brief 张量 JSON codec——Data 文本端口。
///
/// 端口（本地形状规则）：in "text"（Data）→ out "result"（Data）。
/// 报文格式（文本 UTF-8 直传，非 base64）：
/// @code
///   {"dtype":"text","shape":[5],"data":"hello"}
/// @endcode
/// 请求路径：{basePath}/infer
std::shared_ptr<DcNetCodec> makeTextJsonCodec();

/// @brief 张量 JSON 服务端 codec（v1 线上格式的服务端镜像；M-server 变体 A）。
/// 单张量进出，dtype 自适应（数值 base64 / Data 文本 UTF-8 直传），
/// 与出站 tensor/text codec 同一报文格式（提案 FR-2：载荷复用、不另造格式）。
/// @param inputPort  请求张量注入的本地输入端口名（默认 "data"）
/// @param outputPort 响应张量取自的本地输出端口名（默认 "result"）
/// 请求路径：{basePath}/infer
std::shared_ptr<DcNetServerCodec> makeTensorJsonServerCodec(std::string inputPort = "data",
                                                            std::string outputPort = "result");

} // namespace DC::Net
