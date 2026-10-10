#pragma once

#include "GraphStore.h"
#include "GraphSignature.h"
#include "GraphLowering.h"

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

namespace DC {

/// @brief 运行时视图：lowering 后参与调度的节点与边集合，执行引擎只遍历本视图。
class GraphRuntimeView {
public:
	std::unordered_map<std::string, const Node*> nodes;
	std::vector<GraphStore::Edge> edges;

	/// @brief 按名查找节点；不存在返回 nullptr。
	const Node* node(const std::string& name) const {
		auto it = nodes.find(name);
		return it != nodes.end() ? it->second : nullptr;
	}
	size_t nodeCount() const { return nodes.size(); }
	size_t edgeCount() const { return edges.size(); }
};

/// @brief 冻结后的不可变图快照，即 Build、Freeze、Execute 中的 Freeze 产物。
///
/// 拓扑与图级签名自此不可增删改：构建面、拓扑面与节点面三级封闭。
/// 源图视角保留供序列化与内省，执行引擎经 runtimeView 读取 lowering 后形态。
class CompiledGraph {
public:
	/// @brief 冻结的源图拓扑；结构与节点状态均不可经此修改。
	const GraphStore& store() const { return *_store; }

	/// @brief 图级签名，不可变，执行期读取无锁。
	const GraphSignature& signature() const { return _signature; }

	/// @brief 运行时视图；Broadcast(1) wire 已擦除，入边改写为直连。
	const GraphRuntimeView& runtimeView() const { return _runtimeView; }

	size_t runtimeNodeCount() const { return _runtimeView.nodeCount(); }
	size_t runtimeEdgeCount() const { return _runtimeView.edgeCount(); }
	const GraphLoweringStats& loweringStats() const { return _loweringStats; }

private:
	friend class GraphBuilder;
	CompiledGraph(std::shared_ptr<GraphStore> store, GraphSignature signature,
				  GraphRuntimeView view, GraphLoweringStats stats)
		: _store(std::move(store)), _signature(std::move(signature)),
		  _runtimeView(std::move(view)), _loweringStats(stats) {}

	std::shared_ptr<GraphStore> _store;
	GraphSignature _signature;
	GraphRuntimeView _runtimeView;
	GraphLoweringStats _loweringStats;
};

} // namespace DC
