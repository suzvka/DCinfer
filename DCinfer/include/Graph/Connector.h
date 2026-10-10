#pragma once

#include "Node.h"
#include "EngineRegistry.h"

#include <cstddef>
#include <memory>
#include <string>

namespace DC::Connector {

// 广播连接器：1 上游 → N 下游，注册名 "Connector.Broadcast"。
// N=1 零拷贝 move 直通；N>1 发布时冻结并共享 N 份只读副本。

/// @brief 生成 N 个下游输出口的广播 Schema
Node::Schema broadcastSchema(size_t downstreamCount);

/// @brief 广播 RunFn：N=1 直通；N>1 冻结后共享
Node::RunFn broadcastRunFn();

/// @brief 注册 Connector 到 EngineRegistry（1→1 退化模板；实际经 broadcastSchema(n) 构造）。
void registerBuiltinConnectors(EngineRegistry& reg);

} // namespace DC::Connector
