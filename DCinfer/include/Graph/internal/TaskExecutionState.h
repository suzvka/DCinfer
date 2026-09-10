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

using TaskId = std::string; ///< 与 Node::TaskId / ExecutionEngine::TaskId 同义（task 标识）

/// @brief 一次 task 的执行态：按节点名惰性容纳 NodeExecState。
///
/// 未被 feedInput/传播触及的节点不产生条目。
class TaskExecutionState {
public:
	/// @brief  取指定节点的执行态，不存在则按 schema 构建（线程安全）
	NodeExecState& ensure(const std::string& nodeName, const NodeSchema& schema) {
		std::lock_guard lk(_mutex);
		auto& slot = _nodes[nodeName];
		if (!slot)
			slot = std::make_unique<NodeExecState>(schema);
		return *slot;
	}

	/// @brief  查找指定节点的执行态（不创建）；不存在返回 nullptr
	NodeExecState* find(const std::string& nodeName) {
		std::lock_guard lk(_mutex);
		auto it = _nodes.find(nodeName);
		return it != _nodes.end() ? it->second.get() : nullptr;
	}

	/// @brief  返回已创建执行态的节点名列表（已注入 input/传播触及的节点）
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
	/// unique_ptr 保证 NodeExecState 地址稳定（TaskBuffer 内含不可移动的
	/// shared_mutex），并允许惰性构建
	std::unordered_map<std::string, std::unique_ptr<NodeExecState>> _nodes;
};

/// @brief 图级 task 执行域：task → TaskExecutionState + 节点执行闸表。
///
/// 生命周期与 GraphRuntimeState 一致（飞行任务经共享句柄保活）。
/// - 闸表在 attachGraph（惰性冻结）时按源图节点集合一次性预建，此后
///   结构不可变（与 CompiledGraph 同为冻结产物），运行期只翻标志位；
/// - task 态条目按需惰性创建，终止/复用时整体清除。
class TaskExecutionDomain {
public:
	/// @brief  冻结时调用：按源图节点集合预建每节点执行闸（一次性，单线程）
	void attachGraph(const GraphStore& store) {
		for (const auto& [name, nodePtr] : store.nodes())
			_gates[name] = std::make_unique<NodeExecutionGate>();
	}

	/// @brief  节点执行闸（attachGraph 后结构不可变，并发只读安全）
	/// @throws NodeException(InternalError) 节点名不在冻结集合（不变量破坏）
	NodeExecutionGate& gateFor(const std::string& nodeName) const {
		auto it = _gates.find(nodeName);
		if (it == _gates.end()) {
			throw NodeException(NodeException::ErrorType::InternalError,
								"TaskExecutionDomain::gateFor",
								"node '" + nodeName + "' has no execution gate (not in frozen graph)");
		}
		return *it->second;
	}

	/// @brief  取指定 task 的执行态，不存在则创建（线程安全）。
	///
	/// 返回 shared_ptr 而非裸引用：终止清理（clearTaskState）可能与在飞
	/// 节点 lambda 并发——lambda 持有的副本保证其引用的执行态在流水线
	/// 结束前不被销毁（与“取消不中断在飞执行”的协作式语义一致）。
	std::shared_ptr<TaskExecutionState> taskState(const TaskId& taskId) {
		std::lock_guard lk(_mutex);
		auto& slot = _tasks[taskId];
		if (!slot)
			slot = std::make_shared<TaskExecutionState>();
		return slot;
	}

	/// @brief  查找指定 task 的执行态（不创建）；不存在返回 nullptr
	std::shared_ptr<TaskExecutionState> findTaskState(const TaskId& taskId) {
		std::lock_guard lk(_mutex);
		auto it = _tasks.find(taskId);
		return it != _tasks.end() ? it->second : nullptr;
	}

	/// @brief  清除指定 task 的全部节点执行态
	void clearTaskState(const TaskId& taskId) {
		std::lock_guard lk(_mutex);
		_tasks.erase(taskId);
	}

private:
	std::mutex _mutex; ///< 保护 _tasks 的结构变更（闸表 attach 后只读，无需此锁）
	/// shared_ptr：_terminate 清理仅从表中摘除；在飞 lambda 的副本
	/// 保证其引用的执行态存活到流水线结束
	std::unordered_map<TaskId, std::shared_ptr<TaskExecutionState>> _tasks;
	std::unordered_map<std::string, std::unique_ptr<NodeExecutionGate>> _gates;
};

} // namespace DC
