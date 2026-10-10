#pragma once

#include "Node.h"
#include "NodeExecutionGate.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace DC {

using TaskId = std::string; ///< 与 Node::TaskId 同义

/// @brief 单节点执行器：显式持有执行一个 Node 所需的全部 task 态（node 为借用，须更长寿）。
class NodeExecutor {
public:
	explicit NodeExecutor(const Node& node);
	~NodeExecutor();

	NodeExecutor(const NodeExecutor&) = delete;
	NodeExecutor& operator=(const NodeExecutor&) = delete;

	void setInput(const TaskId& taskId, const std::string& portName, Value data);
	void setInput(const TaskId& taskId, const std::string& portName, Tensor data);
	void setInput(const TaskId& taskId, std::unordered_map<std::string, Value> inputs);
	void setInput(const TaskId& taskId, std::unordered_map<std::string, Tensor> inputs);

	bool isReady(const TaskId& taskId) const;
	bool hasOutput(const TaskId& taskId, const std::string& name) const;
	Value takeOutput(const TaskId& taskId, const std::string& name);
	Tensor takeOutputTensor(const TaskId& taskId, const std::string& name);
	std::unordered_map<std::string, Value> collectOutputs(const TaskId& taskId);
	std::unordered_map<std::string, Tensor> collectOutputTensors(const TaskId& taskId);

	bool hasTask(const TaskId& taskId) const;
	void clearTask(const TaskId& taskId);
	size_t taskCount() const;

	/// @brief 执行节点流水线；失败经 NodeResult 返回，未就绪/重入抛 NodeException。
	NodeResult tryExecute(const TaskId& taskId);

private:
	struct Impl;
	std::unique_ptr<Impl> _impl;
	const Node* _node;
};

} // namespace DC
