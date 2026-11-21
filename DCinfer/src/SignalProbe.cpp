#include "SignalProbe.h"
#include "CompiledGraph.h"

#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace DC {

bool canSatisfyTopologically(const GraphRuntimeView& view,
							 const std::vector<std::string>& startNodes,
							 const std::unordered_set<std::string>& targetNodes) {
	// 无从判定时放行，由宿主护栏兜底。
	if (targetNodes.empty() || startNodes.empty())
		return true;

	for (const auto& s : startNodes)
		if (targetNodes.contains(s))
			return true;

	std::unordered_map<std::string, std::vector<const GraphStore::Edge*>> adjacency;
	adjacency.reserve(view.nodeCount());
	for (const auto& edge : view.edges)
		adjacency[edge.srcNode].push_back(&edge);

	// BFS 纯拓扑遍历，不检查信号/阻塞
	std::queue<std::string> frontier;
	std::unordered_set<std::string> visited{startNodes.begin(), startNodes.end()};
	for (const auto& s : startNodes)
		frontier.push(s);

	while (!frontier.empty()) {
		std::string cur = std::move(frontier.front());
		frontier.pop();

		auto it = adjacency.find(cur);
		if (it == adjacency.end())
			continue;
		for (const auto* edge : it->second) {
			if (targetNodes.contains(edge->dstNode))
				return true;
			if (visited.insert(edge->dstNode).second)
				frontier.push(edge->dstNode);
		}
	}
	return false;
}

std::vector<const Node*> killCone(const GraphRuntimeView& view, const std::string& targetNode) {
	// 反向邻接：仅“必需且无默认值”输入口传递死区。
	// 可选口与带默认值口不参与就绪 AND（见 TaskBuffer::_isReadyLocked），其生产者被关不致下游于死地。
	std::unordered_map<std::string, std::vector<std::string>> producers;
	producers.reserve(view.nodeCount());
	for (const auto& edge : view.edges) {
		const Node* dst = view.node(edge.dstNode);
		if (!dst)
			continue;
		const NodePort* port = dst->schema().findInput(edge.dstPort);
		if (!port || !port->required || port->defaultValue.has_value())
			continue;
		producers[edge.dstNode].push_back(edge.srcNode);
	}

	// 从目标反向 BFS，visited 兜底环；锥含目标自身。
	std::vector<const Node*> cone;
	std::unordered_set<std::string> visited;
	std::queue<std::string> frontier;
	frontier.push(targetNode);
	visited.insert(targetNode);
	while (!frontier.empty()) {
		std::string cur = std::move(frontier.front());
		frontier.pop();
		if (const Node* n = view.node(cur))
			cone.push_back(n);
		auto it = producers.find(cur);
		if (it == producers.end())
			continue;
		for (const auto& p : it->second)
			if (visited.insert(p).second)
				frontier.push(p);
	}
	return cone;
}

} // namespace DC
