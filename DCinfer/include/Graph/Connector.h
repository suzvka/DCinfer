#pragma once

#include "Node.h"
#include "EngineRegistry.h"

#include <cstddef>
#include <memory>
#include <string>

namespace DC::Connector {

// ── 广播连接器 ──
// 1 个上游 → N 个下游，拷贝 N 份张量，每份 move 到一个输出口
//
// Schema:  inputs  = [{"in",  Void, 0, {}}]
//          outputs = [{"out_0", Void, 0, {}}, ..., {"out_{N-1}", Void, 0, {}}]
//
// 注册名："Connector.Broadcast"

/// @brief 生成 N 个下游输出口的广播 Schema
Node::Schema broadcastSchema(size_t downstreamCount);

/// @brief 广播 RunFn：从 "in" 读入 → 拷贝 N-1 次 → 写入所有 out_i
Node::RunFn broadcastRunFn();

// ── 路由连接器 ──
// 1 个上游 → N 个下游，轮询选取一个输出口，只拷贝 1 份（不拷贝其余）
// 阻塞语义由 InferGraph 在分发时处理
//
// Schema:  同广播（1 输入 + N 输出）
// 注册名："Connector.Routing"

/// @brief 生成 N 个下游输出口的路由 Schema（与广播共用 Schema 格式）
Node::Schema routingSchema(size_t downstreamCount);

/// @brief 路由 RunFn：从 "in" 读入 → 轮询选取 out_i → 写入一份拷贝
Node::RunFn routingRunFn();

// ── 导线连接器 ──
// 1 进 1 出直通，实现"边即节点"语义。
// 由 InferGraph::wire() 自动插入，用户无需手动创建。
//
// Schema:  inputs  = [{"in",  Void, 0, {}}]
//          outputs = [{"out", Void, 0, {}}]
//
// 注册名："Connector.Wire"

/// @brief 导线 Schema（1 输入 + 1 输出）
Node::Schema wireSchema();

/// @brief 导线 RunFn：从 "in" 读入 → 写入 "out"
Node::RunFn wireRunFn();

// ── 便捷：将三种 Connector 注册到 EngineRegistry ──
// 由于下游数量在创建节点时才知道，Schema 和 RunFn 是参数化的，
// 此处注册的是 1→1 的退化版本作为占位模板。
// 实际使用时通过 broadcastSchema(n) / routingSchema(n) 构造。
void registerBuiltinConnectors(EngineRegistry& reg);

} // namespace DC::Connector
