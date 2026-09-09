#pragma once

#include "Node.h"
#include "EngineRegistry.h"

#include <cstddef>
#include <memory>
#include <string>

namespace DC::Connector {

// ── 广播连接器 ──
// 1 个上游 → N 个下游。
//   - N=1：零拷贝 move 直通，等效导线
//   - N>1：拷贝 N-1 份 + move 最后一份
//
// Schema:  inputs  = [{"in",  Void, 0, {}}]
//          outputs = [{"out_0", Void, 0, {}}, ..., {"out_{N-1}", Void, 0, {}}]
//
// 注册名："Connector.Broadcast"

/// @brief 生成 N 个下游输出口的广播 Schema
Node::Schema broadcastSchema(size_t downstreamCount);

/// @brief 广播 RunFn：N=1 零拷贝 move，N>1 拷贝 N-1 份 + move 最后一份
Node::RunFn broadcastRunFn();

// ── 便捷：将 Connector 注册到 EngineRegistry ──
// 由于下游数量在创建节点时才知道，Schema 和 RunFn 是参数化的，
// 此处注册的是 1→1 的退化版本作为占位模板。
// 实际使用时通过 broadcastSchema(n) 构造。
void registerBuiltinConnectors(EngineRegistry& reg);

} // namespace DC::Connector
