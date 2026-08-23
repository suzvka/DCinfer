#pragma once

#include "NetCodec.h"

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

} // namespace DC::Net
