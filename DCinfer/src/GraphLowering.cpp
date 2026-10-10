#include "GraphLowering.h"
#include "GraphException.h"

namespace DC {

void buildRuntimeView(const GraphStore& source, const GraphSignature& signature,
					  std::unordered_map<std::string, const Node*>& outNodes,
					  std::vector<GraphStore::Edge>& outEdges, GraphLoweringStats& stats) {
	// 被绑定为图级输出/输入的 wire 承担对外契约，必须保留可执行性。
	std::unordered_set<std::string> candidates;
	for (const auto& [name, nodePtr] : source.nodes()) {
		const Node* n = nodePtr.get();
		if (!n->isConnector() || n->type() != "Connector.Broadcast")
			continue;
		if (n->schema().outputs.size() != 1)
			continue;
		if (signature.isOutputBound(name, n->schema().outputs[0].name))
			continue;
		candidates.insert(name);
	}
	for (const auto& ib : signature.inputs) {
		if (candidates.contains(ib.nodeName))
			candidates.erase(ib.nodeName);
	}

	// 仅恰有 1 条出边的 wire 可擦除，多出边无法融合为单条直连边。
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

	outNodes.reserve(source.nodeCount());
	for (const auto& [name, nodePtr] : source.nodes()) {
		if (!erased.contains(name))
			outNodes.emplace(name, nodePtr.get());
	}

	outEdges.reserve(source.edges().size());
	for (const auto& e : source.edges()) {
		if (erased.contains(e.srcNode))
			continue;
		if (erased.contains(e.dstNode)) {
			// 入边融合：沿唯一出边链追至首个保留节点，wire 可串 wire；
			// visited 防纯 wire 环，环上无保留端点，融合边丢弃。
			std::unordered_set<std::string> visited;
			const std::string* dstNode = &e.dstNode;
			const std::string* dstPort = &e.dstPort;
			bool resolved = true;
			while (erased.contains(*dstNode)) {
				if (!visited.insert(*dstNode).second) {
					resolved = false;
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

	// 不变量校验：运行边端点必须存在于运行节点集合，悬空边会导致数据静默滞留、任务无法完成。
	for (const auto& e : outEdges) {
		if (!outNodes.contains(e.srcNode) || !outNodes.contains(e.dstNode))
			throw GraphException(GraphException::ErrorType::Other, "buildRuntimeView",
								 "lowering invariant violated: edge '" + e.srcNode + ":"
									 + e.srcPort + "' → '" + e.dstNode + ":" + e.dstPort
									 + "' references a node outside the runtime view");
	}
}

} // namespace DC
