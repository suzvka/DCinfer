#include "SignalProbe.h"
#include "GraphStore.h"
#include "OutputZone.h"

#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace DC {

bool canSatisfyDeclarations(const GraphStore& store, const OutputZone& output,
							const Node::TaskId& taskId) {
	// 目标集：输出绑定端口所在节点（图级静态签名）。
	// exportNode 的 RunFn 每次执行声明全部输出绑定，故与内部实际声明一致；
	// 且父级查询 isBlocked 时内部尚未 submit，task 级声明不可用。
	const auto& outputBindings = output.bindings();
	if (outputBindings.empty())
		return true; // 防御：无输出绑定，放行（运行时由看门狗兜底）

	std::unordered_set<std::string> targets;
	targets.reserve(outputBindings.size());
	for (const auto& b : outputBindings)
		targets.insert(b.nodeName);

	// 起点：输入绑定中未被阻塞的节点
	const auto& bindings = store.inputBindings();
	if (bindings.empty())
		return true; // 防御：无输入绑定，放行

	// 按源节点分组的邻接表（每次查询构建，O(E)）
	const auto& edges = store.edges();
	std::unordered_map<std::string, std::vector<const GraphStore::Edge*>> adjacency;
	adjacency.reserve(store.nodeCount());
	for (const auto& edge : edges)
		adjacency[edge.srcNode].push_back(&edge);

	std::queue<const Node*> frontier;
	std::unordered_set<std::string> visited;
	visited.reserve(store.nodeCount());

	for (const auto& b : bindings) {
		const auto* n = store.node(b.nodeName);
		if (!n || n->isBlocked(taskId))
			continue;
		if (targets.contains(n->name()))
			return true; // 入口即目标
		frontier.push(n);
		visited.insert(n->name());
	}

	// 正向 BFS：沿边遍历，跳过被信号阻塞的节点（断路）
	while (!frontier.empty()) {
		const auto* cur = frontier.front();
		frontier.pop();

		auto it = adjacency.find(cur->name());
		if (it == adjacency.end())
			continue;
		for (const auto* edge : it->second) {
			if (visited.contains(edge->dstNode))
				continue;
			const auto* dst = store.node(edge->dstNode);
			if (!dst || dst->isBlocked(taskId)) {
				visited.insert(edge->dstNode); // 断路标记，避免重复检查
				continue;
			}
			if (targets.contains(dst->name()))
				return true;
			visited.insert(dst->name());
			frontier.push(dst);
		}
	}
	return false;
}

} // namespace DC
