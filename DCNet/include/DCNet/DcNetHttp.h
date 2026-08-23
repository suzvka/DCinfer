#pragma once

#include "EngineRegistry.h"
#include "NetCodec.h"
#include "Node.h"

#include <memory>
#include <string>

namespace DC::Net {

/// @brief 便捷注册：HttpTransport + 指定 codec（默认 engineType "DCNet.Tensor"）。
///
/// HTTP 是通用传输栈；codec 决定数据格式与协议语义：
/// - makeTensorJsonCodec() → 数值张量格式（engineType "DCNet.Tensor"）
/// - makeTextJsonCodec()   → Data 文本张量格式
/// - 自定义 codec → 协议级适配器（如 DCEngines/OpenAI，另见其独立注册入口）
///
/// 注册后：
/// @code
///   registerDcNetHttp(EngineRegistry::instance(), makeTensorJsonCodec());
///   auto node = EngineRegistry::instance().createNode(
///       "DCNet.Tensor", "remote", "http://192.168.1.10:8080/v1");
/// @endcode
///
/// 注意：同一 engineType 重复注册会被注册表静默拒绝（保留首次）；
/// 需要同一端点协议族下的多个变体时请使用不同 engineType。
///
/// @param reg        目标注册表
/// @param codec      协议映射（makeTensorJsonCodec / makeTextJsonCodec 或自定义）
/// @param schema     本地形状规则；为空时取 codec->schema()
/// @param engineType 注册的引擎类型名（默认 "DCNet.Tensor"）
void registerDcNetHttp(EngineRegistry& reg, std::shared_ptr<DcNetCodec> codec,
					   Node::Schema schema = {}, std::string engineType = "DCNet.Tensor");

} // namespace DC::Net
