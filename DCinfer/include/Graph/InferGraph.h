#pragma once

#include "Node.h"
#include "EngineRegistry.h"
#include "CoroScheduler.h"
#include "ThreadPool.h"

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace DC {

// ── 推理图：DC 电路图语义 ──
//
// 持有所有顶点（Node/Connector），管理端口级拓扑连接。
// 数据流由协程驱动：节点完成 → 消费输出 → 分发到下游输入端 → 调度下游。
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

	/// @brief  默认构造（向后兼容同步 API）
	///         自动创建内部协程调度器和默认线程池
	InferGraph();

	/// @brief  构造推理图（传入外部协程调度器，用于异步 submit 模式）
	/// @param  scheduler     协程调度器（非拥有引用，外部管理生命周期）
	/// @param  computeCfg    计算线程池配置
	/// @param  operatorCfg   算子线程池配置
	/// @param  systemCfg     系统线程池配置（连接器、数据搬运等基础设施）
	InferGraph(CoroScheduler& scheduler,
	           const PoolConfig& computeCfg = {},
	           const PoolConfig& operatorCfg = {},
	           const PoolConfig& systemCfg = {});
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

	/// @brief  接线：在两个节点间自动插入导线连接器（Wire Connector）
	///         等价于 connect(src→wire.in) + connect(wire.out→dst)
	///         适用于两个业务节点之间的 1→1 直连场景
	/// @return 指向自动创建的导线连接器的非拥有指针；若节点或端口不存在则返回 nullptr
	Node* wire(
		const std::string& srcNode, const std::string& srcPort,
		const std::string& dstNode, const std::string& dstPort);

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

	/// @brief  同步驱动：扫描全图查找已就绪节点，BFS 级联执行直到无更多可执行节点
	void run();

	/// @brief  异步启动整张图的计算
	///         为指定 taskId 扫描入口节点，创建协程链驱动数据流
	void submit(const TaskId& taskId);

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

	// ── 协程数据传播 ──
	//
	// 核心：从指定节点出发，沿边传播数据到所有下游
	// co_await 节点完成 → 消费输出 → 写入下游 → 按 affinity 提交线程池 → spawn 下游传播协程
	Task<void> _propagateFrom(const std::string& nodeName, const TaskId& taskId);

	// ── 成员 ──
	std::unordered_map<std::string, std::unique_ptr<Node>> _nodes;
	std::vector<Edge>                                      _edges;
	std::vector<OutputBinding>                             _outputBindings;

	// 活跃任务集：feedInput 时自动记录，供 run() 扫描
	std::unordered_set<TaskId> _activeTasks;

	// 协程调度器（非拥有指针，指向外部或 _ownedScheduler）
	CoroScheduler* _scheduler;
	std::unique_ptr<CoroScheduler> _ownedScheduler;  // 默认构造时内部持有

	// 线程池
	ThreadPool     _computePool;
	ThreadPool     _operatorPool;
	ThreadPool     _systemPool;

	// 导线连接器自动命名计数器
	std::atomic<size_t> _nextWireId{0};
};

} // namespace DC
