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

} // namespace DC
