#pragma once

#include "EngineRegistry.h"
#include "NetCodec.h"
#include "Node.h"

#include <memory>
#include <string>

namespace DC::Net {

/// @brief 张量 JSON codec（DCNet v1 线上格式，DESIGN.md §4 内置适配器）。
///
/// 端口（本地形状规则）：in "data"（Float，任意形状）→ out "result"（Float）。
/// 报文格式：
/// @code
///   {"dtype":"float32","shape":[1,1,28,28],"data":"<base64>"}
/// @endcode
/// 请求路径：{basePath}/infer
std::shared_ptr<DcNetCodec> makeTensorJsonCodec();

/// @brief OpenAI 兼容 chat codec（POST {basePath}/chat/completions）。
///
/// 端口：in prompt（Data，必填）/ system（Data，可选）/ params（Data，可选，
/// 请求级采样参数 JSON，逐请求覆盖）→ out response（Data）。
/// @param model 请求体 model 字段（服务端标识模型）
std::shared_ptr<DcNetCodec> makeChatCodec(std::string model = "default");

/// @brief 便捷注册：HttpTransport + 指定 codec（默认 engineType "DCNet.Http"）。
///
/// 注册后：
/// @code
///   registerDcNetHttp(EngineRegistry::instance(), makeTensorJsonCodec());
///   auto node = EngineRegistry::instance().createNode(
///       "DCNet.Http", "remote", "http://192.168.1.10:8080/v1");
/// @endcode
///
/// 注意：同一 engineType 重复注册会被注册表静默拒绝（保留首次）；
/// 需要同一端点协议族下的多个变体时请使用不同 engineType。
///
/// @param reg        目标注册表
/// @param codec      协议映射（makeTensorJsonCodec / makeChatCodec 或自定义）
/// @param schema     本地形状规则；为空时取 codec->schema()
/// @param engineType 注册的引擎类型名（默认 "DCNet.Http"）
void registerDcNetHttp(EngineRegistry& reg, std::shared_ptr<DcNetCodec> codec,
					   Node::Schema schema = {}, std::string engineType = "DCNet.Http");

} // namespace DC::Net
