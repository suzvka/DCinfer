#pragma once

#include "NetCodec.h"
#include "NetServerCodec.h"

#include <memory>

namespace DC::Net {

/// 张量 JSON codec，v1 线上格式：in "data" 为 Float 任意形状，out "result" 为 Float。
/// 报文：{"dtype":"float32","shape":[1,1,28,28],"data":"<base64>"}；路径 {basePath}/infer。
std::shared_ptr<DcNetCodec> makeTensorJsonCodec();

/// Data 文本端口 codec：in "text" 为 Data，out "result" 为 Data。
/// 报文：{"dtype":"text","shape":[5],"data":"hello"}，UTF-8 直传非 base64；路径 {basePath}/infer。
std::shared_ptr<DcNetCodec> makeTextJsonCodec();

/// 服务端镜像，单张量进出且 dtype 自适应：与出站 codec 同一报文格式；路径 {basePath}/infer。
std::shared_ptr<DcNetServerCodec> makeTensorJsonServerCodec(std::string inputPort = "data",
                                                            std::string outputPort = "result");

} // namespace DC::Net
