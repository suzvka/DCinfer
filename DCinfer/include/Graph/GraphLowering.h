#pragma once

#include "GraphStore.h"
#include "GraphSignature.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace DC {

/// @brief 编译期 lowering pass：把语义等价于导线的连接器从运行时视图擦除。
///
/// 仅擦除 Connector.Broadcast、N=1、恰一入一出且端口未被 GraphSignature 绑定的 wire；
/// 融合沿唯一出边链追踪至首个保留节点，纯 wire 环上的融合边丢弃；
/// 收尾校验运行边端点必须存在于运行节点集合，违反抛 GraphException。
/// 被擦除的 wire 不再消耗 TTL，maxHops 只统计运行时顶点；源图不受影响。
struct GraphLoweringStats {
	size_t erasedConnectors = 0;
};

/// @brief 从源图构建 lowering 后的运行时视图；1:1 wire 的入边已改写为直连边。
void buildRuntimeView(const GraphStore& source, const GraphSignature& signature,
					  std::unordered_map<std::string, const Node*>& outNodes,
					  std::vector<GraphStore::Edge>& outEdges, GraphLoweringStats& stats);

} // namespace DC
