#pragma once

#include "GraphStore.h"
#include "Node/NodeExecutionGate.h"
#include "Node/internal/NodeExecState.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace DC {

using TaskId = std::string; ///< 与 Node::TaskId 同义

/// @brief 一次 task 的执行态：按节点名惰性容纳 NodeExecState（未触及的节点无条目）。
class TaskExecutionState {
public:
	/// @brief 取指定节点的执行态，不存在则按 schema 构建（线程安全）。
	NodeExecState& ensure(const std::string& nodeName, const NodeSchema& schema) {
		std::lock_guard lk(_mutex);
		auto& slot = _nodes[nodeName];
		if (!slot)
			slot = std::make_unique<NodeExecState>(schema);
		return *slot;
	}

	/// @brief 查找指定节点的执行态（不创建）；不存在返回 nullptr。
	NodeExecState* find(const std::string& nodeName) {
		std::lock_guard lk(_mutex);
		auto it = _nodes.find(nodeName);
		return it != _nodes.end() ? it->second.get() : nullptr;
	}

	/// @brief 已创建执行态的节点名列表（已注入 input 或传播触及）。
	std::vector<std::string> nodeNames() {
		std::lock_guard lk(_mutex);
		std::vector<std::string> names;
		names.reserve(_nodes.size());
		for (const auto& [name, _] : _nodes)
			names.push_back(name);
		return names;
	}

private:
	std::mutex _mutex;
	// unique_ptr：NodeExecState 不可移动（内含 shared_mutex），需地址稳定。
	std::unordered_map<std::string, std::unique_ptr<NodeExecState>> _nodes;
};

/// @brief 图级 task 执行域：task → TaskExecutionState + 节点执行闸表。
///
/// 闸表在 attachGraph 冻结事务中按源图节点集合预建（发布前完成，此后结构不可变，
/// 运行期只翻标志位）；task 态条目按需惰性创建，终止/复用时整体清除。
class TaskExecutionDomain {
public:
	/// @brief 冻结时调用：按源图节点集合预建每节点执行闸（一次性，单线程）。
	void attachGraph(const GraphStore& store) {
		for (const auto& [name, nodePtr] : store.nodes())
			_gates[name] = std::make_unique<NodeExecutionGate>();
	}

	/// @brief 节点执行闸（attachGraph 后结构不可变，并发只读安全）。
	/// @throws NodeException(InternalError) 节点名不在冻结集合
	NodeExecutionGate& gateFor(const std::string& nodeName) const {
		auto it = _gates.find(nodeName);
		if (it == _gates.end()) {
			throw NodeException(NodeException::ErrorType::InternalError,
								"TaskExecutionDomain::gateFor",
								"node '" + nodeName + "' has no execution gate (not in frozen graph)");
		}
		return *it->second;
	}

	/// @brief 取指定 task 的执行态，不存在则创建。
	/// 返回 shared_ptr：终止清理与在飞 lambda 并发时，lambda 的副本保证执行态存活到流水线结束。
	std::shared_ptr<TaskExecutionState> taskState(const TaskId& taskId) {
		std::lock_guard lk(_mutex);
		auto& slot = _tasks[taskId];
		if (!slot)
			slot = std::make_shared<TaskExecutionState>();
		return slot;
	}

	/// @brief 查找指定 task 的执行态（不创建）；不存在返回 nullptr。
	std::shared_ptr<TaskExecutionState> findTaskState(const TaskId& taskId) {
		std::lock_guard lk(_mutex);
		auto it = _tasks.find(taskId);
		return it != _tasks.end() ? it->second : nullptr;
	}

	/// @brief 清除指定 task 的全部节点执行态。
	void clearTaskState(const TaskId& taskId) {
		std::lock_guard lk(_mutex);
		_tasks.erase(taskId);
	}

private:
	std::mutex _mutex; ///< 保护 _tasks 结构变更（_gates attach 后只读）
	std::unordered_map<TaskId, std::shared_ptr<TaskExecutionState>> _tasks;
	std::unordered_map<std::string, std::unique_ptr<NodeExecutionGate>> _gates;
};

} // namespace DC
