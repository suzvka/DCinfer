#pragma once

#include "GraphStore.h"
#include "GraphSignature.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace DC {

/// @brief 编译期 lowering pass：把语义等价于"边"的连接器从运行时视图擦除。
///
/// Phase 2 规则（刻意窄化）：
///
///   Broadcast(N=1)（零拷贝 move 直通，等效导线） → 直接运行时边
///
/// 识别条件（全部满足才擦除）：
/// - isConnector() 且 type == "Connector.Broadcast" 且 schema.outputs.size() == 1；
/// - 恰有 1 条入边（in）与 1 条出边（out_0）——多出边意味着实际是 1→多
///   分发（takeOutput 消费式下第二条边拿不到数据，语义不完整），不擦除；
/// - wire 的输出端口未被 GraphSignature 绑定为图级输出（否则声明永不满足）；
/// - wire 的输入端口未被 GraphSignature 绑定为图级输入（否则 feedInput 直喂
///   wire，必须保留其可执行性）。
///
/// 语义说明（两个显式决策）：
/// - TTL（maxHops）：hop 计数只统计运行时顶点，被擦除的 wire 不再消耗 TTL
///   ——成环图 TTL 触发时机后移（方向安全：更不易误杀深图）；
/// - 池亲和：直连后传播握手在上游节点的完成线程执行（原为 System 池）；
///   N=1 wire 本就是零拷贝 move 直通、无数据搬运，不违背三层池隔离初衷
///   （Broadcast(N>1) 等重型连接器不受影响，仍在 System 池）。
///
/// 源图不受影响：DCIr 序列化、exportNode、nodeCount/edges 内省均反映源图。
struct GraphLoweringStats {
	size_t erasedConnectors = 0; ///< 被擦除的退化连接器数量
};

/// @brief  从源图构建 lowering 后的运行时视图。
/// @param  source    源图拓扑（只读）
/// @param  signature 图级签名（输入/输出绑定，参与擦除防护判定）
/// @param  outNodes  [out] 运行时节点表（借用源图节点指针，源图存活期由快照保证）
/// @param  outEdges  [out] 运行时边表（1:1 wire 的入边已改写为直连边）
/// @param  stats     [out] lowering 统计
void buildRuntimeView(const GraphStore& source, const GraphSignature& signature,
					  std::unordered_map<std::string, const Node*>& outNodes,
					  std::vector<GraphStore::Edge>& outEdges, GraphLoweringStats& stats);

} // namespace DC
