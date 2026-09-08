#pragma once

#include "GraphStore.h"
#include "GraphSignature.h"
#include "GraphLowering.h"

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

namespace DC {

/// @brief 运行时视图：lowering 后参与调度的节点/边集合。
///
/// 与源图的关系：节点指针借用源图（源图存活期由快照保证），
/// 边为 lowering 改写后的直连边。执行引擎只遍历本视图——
/// 被擦除的退化连接器（Broadcast(1) wire）不再参与就绪扫描、
/// 调度、传播与耗尽检查。
class GraphRuntimeView {
public:
	std::unordered_map<std::string, const Node*> nodes; ///< 参与调度的节点表（借用指针）
	std::vector<GraphStore::Edge> edges;          ///< lowering 后的运行时边

	/// @brief  按名查找节点（借用指针；不存在返回 nullptr）
	const Node* node(const std::string& name) const {
		auto it = nodes.find(name);
		return it != nodes.end() ? it->second : nullptr;
	}
	size_t nodeCount() const { return nodes.size(); }
	size_t edgeCount() const { return edges.size(); }
};

/// @brief 冻结后的不可变图快照（Build → Freeze → Execute 中的 Freeze 产物）。
///
/// 由 GraphBuilder::compile() 一次性产出，确立核心不变量：
///
/// > 图在执行期间不可变。
///
/// - 拓扑（节点/边）与图级签名（输入/输出绑定）自此不可增删改——
///   增删改 API 仅存在于构建期的 GraphBuilder，冻结后无入口；
/// - 执行引擎与飞行任务经本快照读取拓扑与签名（GraphRuntimeState::graph）；
/// - 保留源图视角：DCIr 序列化、exportNode、nodeCount/edges 等内省
///   均反映源图；运行时视图（runtimeView）为 lowering 后形态。
///
/// 拓扑演进官方路径：回到 GraphBuilder 重新 compile 产生新快照，
/// 旧图任务排空后由调用方替换。
class CompiledGraph {
public:
	/// @brief  冻结的源图拓扑。
	///
	/// 返回 const：结构不可变，节点 task 级状态已归 task 执行域
	/// （GraphRuntimeState::exec）——运行期不再经 Node 写入任何状态，
	/// 拓扑增删改（addNode/connect/...）在冻结后无公开入口，属不可达 API。
	const GraphStore& store() const { return *_store; }

	/// @brief  图级签名（不可变；执行期别名/绑定解析无锁）
	const GraphSignature& signature() const { return _signature; }

	// ── 运行时视图（lowering 后；执行引擎的调度/传播/清理面）──

	/// @brief  运行时视图（Broadcast(1) wire 已擦除，入边改写为直连）
	const GraphRuntimeView& runtimeView() const { return _runtimeView; }

	/// @brief  运行时节点数（源图 nodeCount − 被擦除连接器数）
	size_t runtimeNodeCount() const { return _runtimeView.nodeCount(); }

	/// @brief  运行时边数（源图 edgeCount − 被擦除连接器数）
	size_t runtimeEdgeCount() const { return _runtimeView.edgeCount(); }

	/// @brief  lowering 统计（被擦除的退化连接器数量）
	const GraphLoweringStats& loweringStats() const { return _loweringStats; }

private:
	friend class GraphBuilder;
	CompiledGraph(std::shared_ptr<GraphStore> store, GraphSignature signature,
				  GraphRuntimeView view, GraphLoweringStats stats)
		: _store(std::move(store)), _signature(std::move(signature)),
		  _runtimeView(std::move(view)), _loweringStats(stats) {}

	std::shared_ptr<GraphStore> _store; ///< 冻结拓扑（所有权随快照移交）
	GraphSignature _signature;
	GraphRuntimeView _runtimeView;      ///< lowering 后的运行时视图
	GraphLoweringStats _loweringStats;
};

} // namespace DC
