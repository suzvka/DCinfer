#include "GraphLowering.h"

namespace DC {

void buildRuntimeView(const GraphStore& source, const GraphSignature& signature,
					  std::unordered_map<std::string, Node*>& outNodes,
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

	// ── 4. 边改写：wire 入边与其唯一出边融合为直连边 ──
	outEdges.reserve(source.edges().size());
	for (const auto& e : source.edges()) {
		if (erased.contains(e.srcNode))
			continue; // wire 出边：已被入边融合吸收
		if (erased.contains(e.dstNode)) {
			// wire 入边 → 直连边（源 → wire 的唯一出边目标）
			const auto* out = wireOutEdge[e.dstNode];
			outEdges.push_back({e.srcNode, e.srcPort, out->dstNode, out->dstPort});
			continue;
		}
		outEdges.push_back(e);
	}
}

} // namespace DC
