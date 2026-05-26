#pragma once

#include "Node.h"
#include "EngineRegistry.h"

#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DC {

// ── 推理图：DC 电路图语义 ──
//
// 持有所有顶点（Node/Connector），管理端口级拓扑连接。
// 数据流由 Graph 驱动：执行节点 → 读取输出 → 分发到下游输入端 → 调度下游。
//
// Node 不知下游，Connector 即 Node。Graph 对一切顶点统一处理。

class InferGraph {
public:
	using TaskId = Node::TaskId;

	// ── 端口级边 ──
	struct Edge {
		std::string srcNode;
		std::string srcPort;
		std::string dstNode;
		std::string dstPort;
	};

	// ── 输出区绑定 ──
	struct OutputBinding {
		std::string nodeName;
		std::string portName;
	};

	InferGraph() = default;
	~InferGraph() = default;

	InferGraph(const InferGraph&)            = delete;
	InferGraph& operator=(const InferGraph&) = delete;
	InferGraph(InferGraph&&)                 = default;
	InferGraph& operator=(InferGraph&&)      = default;

	// ── 图构建 ──

	/// @brief  添加节点（转移所有权），返回裸指针供后续接线引用
	/// @return 指向已存入图内节点的非拥有指针；若节点名为空或重名则返回 nullptr
	Node* addNode(std::unique_ptr<Node> node);

	/// @brief  端口级接线：上游输出口 → 下游输入口
	/// @return true 表示两端节点和端口均存在
	bool connect(
		const std::string& srcNode, const std::string& srcPort,
		const std::string& dstNode, const std::string& dstPort);

	/// @brief  快捷接线：自动匹配上游所有输出口到下游同名的输入口
	///         典型用途：Connector 的 out_0→in_0, out_1→in_1, ...
	///         多个下游时需为每个下游各调用一次
	/// @return 成功匹配的端口对数
	size_t connectAll(
		const std::string& srcNode,
		const std::string& dstNode);

	/// @brief  标记输出：该节点的该端口结果输出到图外
	void bindOutput(const std::string& nodeName, const std::string& portName);

	// ── 数据注入 ──

	/// @brief  从图外注入数据到指定节点的输入端口（写入缓冲，不触发执行）
	bool feedInput(
		const TaskId& taskId,
		const std::string& nodeName,
		const std::string& portName,
		Value data);

	/// @brief  便捷接口：直接传入 DC::Tensor
	bool feedInput(
		const TaskId& taskId,
		const std::string& nodeName,
		const std::string& portName,
		Tensor data);

	// ── 执行驱动 ──

	/// @brief  驱动一轮：弹出就绪队列中所有任务，执行 → 分发 → 递归调度下游
	///         若初始队列为空，扫描全图查找已就绪节点并入队
	void run();

	/// @brief  将指定节点的指定任务排入就绪队列（外部可在 feedInput 后调用）
	void schedule(const std::string& nodeName, const TaskId& taskId);

	// ── 结果获取 ──

	/// @brief  获取输出区中指定端口的结果（消费式取出）
	Value getOutput(const TaskId& taskId, const std::string& nodeName, const std::string& portName);

	/// @brief  便捷接口：取出 DC::Tensor
	Tensor getOutputTensor(const TaskId& taskId, const std::string& nodeName, const std::string& portName);

	/// @brief  检查输出区中是否有结果
	bool hasOutput(const TaskId& taskId, const std::string& nodeName, const std::string& portName) const;

	// ── 查询 ──

	/// @brief  获取节点指针（非拥有），不存在返回 nullptr
	Node* node(const std::string& name);

	/// @brief  获取节点指针（只读）
	const Node* node(const std::string& name) const;

	/// @brief  节点数量
	size_t nodeCount() const { return _nodes.size(); }

	/// @brief  边数量
	size_t edgeCount() const { return _edges.size(); }

private:
	// 查找节点：存在返回指针，不存在返回 nullptr
	Node* _findNode(const std::string& name);
	const Node* _findNode(const std::string& name) const;

	// 分发指定节点的指定 task 的输出到所有下游
	void _dispatchOutputs(const std::string& nodeName, const TaskId& taskId);

	// 扫描全图，将就绪节点入队
	void _scanReady(const TaskId& taskId);

	// ── 成员 ──
	std::unordered_map<std::string, std::unique_ptr<Node>> _nodes;
	std::vector<Edge>                                      _edges;
	std::vector<OutputBinding>                             _outputBindings;

	// 就绪队列：(nodeName, taskId)
	std::queue<std::pair<std::string, TaskId>> _readyQueue;
};

} // namespace DC
