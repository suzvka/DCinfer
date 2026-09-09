#include "GraphLowering.h"
#include "GraphException.h"

namespace DC {

void buildRuntimeView(const GraphStore& source, const GraphSignature& signature,
					  std::unordered_map<std::string, const Node*>& outNodes,
					  std::vector<GraphStore::Edge>& outEdges, GraphLoweringStats& stats) {
	// ── 1. 识别退化连接器候选：Broadcast(N=1) wire ──
	// 前置防护：wire 的输出口被绑定为图级输出 / 输入口被绑定为图级输入时，
	// wire 承担图对外契约（声明满足判定 / feedInput 直喂），必须保留可执行性。
	std::unordered_set<std::string> candidates;
	for (const auto& [name, nodePtr] : source.nodes()) {
		const Node* n = nodePtr.get();
		if (!n->isConnector() || n->type() != "Connector.Broadcast")
			continue;
		if (n->schema().outputs.size() != 1)
			continue;
		if (signature.isOutputBound(name, n->schema().outputs[0].name))
			continue; // wire 的 out_0 被绑定为图级输出 → 保留（声明满足依赖 wire 搬运）
		candidates.insert(name);
	}
	// 图级输入绑定面检查（bindInput(wire, ...)）：wire 的任意口被绑定为
	// 图级输入 → feedInput 直喂 wire，必须保留其可执行性。
	for (const auto& ib : signature.inputs) {
		if (candidates.contains(ib.nodeName))
			candidates.erase(ib.nodeName); // wire 的任意口被绑定为图级输入 → 保留
	}

	// ── 2. 出边计数：候选 wire 必须恰有 1 条出边（多出边 = 1→多分发，不擦除）──
	std::unordered_map<std::string, const GraphStore::Edge*> wireOutEdge;
	std::unordered_map<std::string, size_t> wireOutCount;
	for (const auto& e : source.edges()) {
		if (!candidates.contains(e.srcNode))
			continue;
		++wireOutCount[e.srcNode];
		wireOutEdge[e.srcNode] = &e;
	}
	std::unordered_set<std::string> erased;
	for (const auto& name : candidates) {
		auto it = wireOutCount.find(name);
		if (it != wireOutCount.end() && it->second == 1) {
			erased.insert(name);
			++stats.erasedConnectors;
		}
	}

	// ── 3. 运行时节点表 = 源图 − 被擦除 wire ──
	outNodes.reserve(source.nodeCount());
	for (const auto& [name, nodePtr] : source.nodes()) {
		if (!erased.contains(name))
			outNodes.emplace(name, nodePtr.get());
	}

	// ── 4. 边改写：wire 入边沿唯一出边链追踪到最终保留节点后融合为直连边 ──
	outEdges.reserve(source.edges().size());
	for (const auto& e : source.edges()) {
		if (erased.contains(e.srcNode))
			continue; // wire 出边：已被入边融合吸收
		if (erased.contains(e.dstNode)) {
			// wire 入边 → 融合边。链上后继也可能已被擦除（wire→wire 链，
			// connectRaw 允许连接器与连接器相连），必须追到首个保留节点；
			// visited 防纯 wire 环——环上无保留端点，融合边丢弃（数据在源
			// 语义中同样永远无法到达任何业务节点）。
			std::unordered_set<std::string> visited;
			const std::string* dstNode = &e.dstNode;
			const std::string* dstPort = &e.dstPort;
			bool resolved = true;
			while (erased.contains(*dstNode)) {
				if (!visited.insert(*dstNode).second) {
					resolved = false; // wire 环：无保留端点
					break;
				}
				const auto* out = wireOutEdge.at(*dstNode); // 擦除条件保证恰有 1 条出边
				dstNode = &out->dstNode;
				dstPort = &out->dstPort;
			}
			if (resolved)
				outEdges.push_back({e.srcNode, e.srcPort, *dstNode, *dstPort});
			continue;
		}
		outEdges.push_back(e);
	}

	// ── 5. 不变量校验：运行边的端点必须存在于运行节点集合 ──
	// 防悬空边回归（悬空边在运行期表现为数据静默滞留 + 任务无法完成）。
	for (const auto& e : outEdges) {
		if (!outNodes.contains(e.srcNode) || !outNodes.contains(e.dstNode))
			throw GraphException(GraphException::ErrorType::Other, "buildRuntimeView",
								 "lowering invariant violated: edge '" + e.srcNode + ":"
									 + e.srcPort + "' → '" + e.dstNode + ":" + e.dstPort
									 + "' references a node outside the runtime view");
	}
}

} // namespace DC
