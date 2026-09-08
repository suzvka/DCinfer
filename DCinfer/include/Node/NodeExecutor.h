#pragma once

#include "Node.h"
#include "NodeExecutionGate.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace DC {

using TaskId = std::string; ///< 与 Node::TaskId 同义（task 标识）

/// @brief 单节点执行器：显式持有执行一个 Node 所需的全部 task 态。
///
/// 原 Node 内嵌 task 态（TaskBuffer / SlotWorkspace / 执行闸）迁出后的
/// 单节点归宿——Node 因此成为纯 plan 对象。载体生命周期由使用者决定：
/// - 服务化（NetServerAdapter）：每请求一个实例（实例级隔离）；
/// - 测试 / 工具：按需创建，接口与原 Node task 级 API 一一对应。
///
/// 执行语义与原 Node::tryExecute 完全一致（就绪预检 → 节点闸租约 →
/// 7 步流水线 → 释放），异常语义不变（NotReady / Reentrant）。
///
/// @note  node 为借用引用，须比本执行器存活更久。
class NodeExecutor {
public:
	/// @param node 执行目标（借用）
	explicit NodeExecutor(const Node& node);
	~NodeExecutor();

	NodeExecutor(const NodeExecutor&) = delete;
	NodeExecutor& operator=(const NodeExecutor&) = delete;

	// ── task 级输入（原 Node::setInput 语义）──

	void setInput(const TaskId& taskId, const std::string& portName, Value data);
	void setInput(const TaskId& taskId, const std::string& portName, Tensor data);
	void setInput(const TaskId& taskId, std::unordered_map<std::string, Value> inputs);
	void setInput(const TaskId& taskId, std::unordered_map<std::string, Tensor> inputs);

	// ── task 级输出（原 Node 输出 API 语义）──

	bool isReady(const TaskId& taskId) const;
	bool hasOutput(const TaskId& taskId, const std::string& name) const;
	Value takeOutput(const TaskId& taskId, const std::string& name);
	Tensor takeOutputTensor(const TaskId& taskId, const std::string& name);
	const Value& peekOutput(const TaskId& taskId, const std::string& name) const;
	std::unordered_map<std::string, Value> collectOutputs(const TaskId& taskId);
	std::unordered_map<std::string, Tensor> collectOutputTensors(const TaskId& taskId);

	// ── task 生命周期 ──

	bool hasTask(const TaskId& taskId) const;
	void clearTask(const TaskId& taskId);
	size_t taskCount() const;

	// ── 执行 ──

	/// @brief  执行节点流水线（就绪判定 + 节点闸租约 + 7 步执行）
	/// @return 执行结果（失败不抛出，通过 NodeResult 返回）
	/// @throws NodeException(NotReady)     任务未就绪
	/// @throws NodeException(Reentrant)    节点正被另一任务占用
	NodeResult tryExecute(const TaskId& taskId);

private:
	struct Impl;
	std::unique_ptr<Impl> _impl; ///< task 态（TaskBuffer/SlotWorkspace 为模块内私有组件）
	const Node* _node;
};

} // namespace DC
