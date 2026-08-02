#pragma once

#include "Node.h"
#include "GraphStore.h"
#include "ExecutionEngine.h"
#include "OutputZone.h"
#include "SignalStore.h"
#include "ErrorTracker.h"
#include "GraphException.h"

#include <chrono>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace DC {

// ── 推理图：DC 电路图语义 ──
//
// InferGraph 是用户唯一接触的 Facade，内部委托给：
//   - GraphStore      图拓扑存储
//   - ExecutionEngine 执行调度与协程传播
//   - OutputZone      输出聚合
//   - ErrorTracker    错误收集
//   - SignalStore     信号仓库
//
// Node 不知下游，Connector 即 Node。Graph 对一切顶点统一处理。

class InferGraph {
public:
	using TaskId = Node::TaskId;

	/// @brief  端口级边（类型别名，定义见 GraphStore::Edge）
	using Edge = GraphStore::Edge;

	/// @brief  默认最大跳数（TTL），防止循环无限传播
	static constexpr uint32_t kDefaultMaxHops = ExecutionEngine::kDefaultMaxHops;

	/// @brief  默认构造
	///         自动创建内部协程调度器和默认线程池
	InferGraph();

	/// @brief  构造推理图（传入外部协程调度器，用于异步 submit 模式）
	/// @param  scheduler     协程调度器（非拥有引用，外部管理生命周期）
	/// @param  computeCfg    计算线程池配置
	/// @param  operatorCfg   算子线程池配置
	/// @param  systemCfg     系统线程池配置（连接器、数据搬运等基础设施）
	InferGraph(CoroScheduler& scheduler, const PoolConfig& computeCfg = {}, const PoolConfig& operatorCfg = {},
			   const PoolConfig& systemCfg = {});
	~InferGraph() = default;

	InferGraph(const InferGraph&) = delete;
	InferGraph& operator=(const InferGraph&) = delete;

	// 移动语义禁止：ExecutionEngine 持有活跃线程和协程状态，
	// 移动会导致 TaskGate 中裸指针（graph/output/signals）悬空。
	InferGraph(InferGraph&&) = delete;
	InferGraph& operator=(InferGraph&&) = delete;

	// ── 图构建 ──

	/// @brief  添加节点（转移所有权），返回引用供后续接线引用
	/// @throws GraphException(DuplicateNode) 若节点名为空或重名
	Node& addNode(std::unique_ptr<Node> node) { return _store.addNode(std::move(node)); }

	/// @brief  端口级接线：上游输出口 → 下游输入口
	/// @throws GraphException(NodeNotFound/PortNotFound/DirectConnect) 若接线不合法
	void connect(const std::string& srcNode, const std::string& srcPort,
				 const std::string& dstNode, const std::string& dstPort) {
		_store.connect(srcNode, srcPort, dstNode, dstPort);
	}

	/// @brief  快捷接线：自动匹配上游所有输出口到下游同名的输入口
	/// @return 成功匹配的端口对数
	size_t connectAll(const std::string& srcNode, const std::string& dstNode) {
		return _store.connectAll(srcNode, dstNode);
	}

	/// @brief  接线：在两个节点间自动插入广播连接器（Broadcast Connector, N=1）
	/// @throws GraphException(NodeNotFound/PortNotFound) 若节点或端口不存在
	/// @return 指向自动创建的广播连接器的引用
	Node& wire(const std::string& srcNode, const std::string& srcPort,
			   const std::string& dstNode, const std::string& dstPort) {
		return _store.wire(srcNode, srcPort, dstNode, dstPort);
	}

	/// @brief  标记输入：该节点的该端口为图级输入口，外部通过此口注入数据
	void bindInput(const std::string& nodeName, const std::string& portName) {
		_store.bindInput(nodeName, portName);
	}

	/// @brief  标记输出：该节点的该端口产出进入 OutputZone（与边目的地互斥）
	void bindOutput(const std::string& nodeName, const std::string& portName) {
		_outputZone.bind(nodeName, portName);
	}

	// ── 子图声明 ──

	/// @brief  声明子图：将一组节点编入同名分组，在执行时互斥（同时只有一个执行）。
	///
	/// 子图不改变图拓扑，仅通过跨池共享的线程池分组信号量实现串行约束。
	/// 组内节点可属于不同线程池（混合 affinity）：同一 tag 的分组信号量
	/// 由 Compute / Operator / System 三个线程池共享，全局互斥。
	///
	/// @param  name       子图名（作为线程池分组 tag）
	/// @param  nodeNames  属于该子图的节点名列表
	/// @throws GraphException(NodeNotFound) 若任何节点不存在
	void declareSubgraph(const std::string& name, std::initializer_list<std::string> nodeNames);

	// ── 数据注入 ──

	/// @brief  从图外注入数据到指定节点的输入端口（写入缓冲，不触发执行）
	/// @throws GraphException(NodeNotFound) 若节点不存在
	/// @throws GraphException(FeedFailed) 若 setInput 失败
	void feedInput(const TaskId& taskId, const std::string& nodeName, const std::string& portName, Value data);

	/// @brief  便捷接口：直接传入 DC::Tensor
	void feedInput(const TaskId& taskId, const std::string& nodeName, const std::string& portName, Tensor data);

	// ── 执行驱动 ──

	/// @brief  异步启动整张图的计算。输出声明直接作为 submit 参数，消除 temporal coupling。
	/// @param declarations  期望产出：{nodeName, portName, count} 列表
	void submit(const TaskId& taskId, std::vector<OutputDeclaration> declarations,
				std::chrono::milliseconds timeout = std::chrono::milliseconds(0),
				uint32_t maxHops = kDefaultMaxHops) {
		_outputZone.declare(taskId, std::move(declarations));
		_engine.submit(taskId, timeout, maxHops, _store, _outputZone, *_signalStore, _errors);
	}

	/// @brief  单输出便捷重载
	void submit(const TaskId& taskId, const std::string& nodeName, const std::string& portName,
				size_t count = 1,
				std::chrono::milliseconds timeout = std::chrono::milliseconds(0),
				uint32_t maxHops = kDefaultMaxHops) {
		_outputZone.declare(taskId, nodeName, portName, count);
		_engine.submit(taskId, timeout, maxHops, _store, _outputZone, *_signalStore, _errors);
	}

	// ── 结果获取 ──

	/// @brief  获取输出区中指定端口的结果（消费式取出）
	/// @throws GraphException(NodeNotFound) 若节点不存在
	Value getOutput(const TaskId& taskId, const std::string& nodeName, const std::string& portName);

	/// @brief  便捷接口：取出 DC::Tensor
	/// @throws GraphException(NodeNotFound) 若节点不存在
	Tensor getOutputTensor(const TaskId& taskId, const std::string& nodeName, const std::string& portName);

	/// @brief  检查输出区中是否有结果
	bool hasOutput(const TaskId& taskId, const std::string& nodeName, const std::string& portName) const;

	// ── 查询 ──

	/// @brief  获取节点指针（非拥有），不存在返回 nullptr
	Node* node(const std::string& name) { return _store.node(name); }

	/// @brief  获取节点指针（只读）
	const Node* node(const std::string& name) const { return _store.node(name); }

	/// @brief  节点数量
	size_t nodeCount() const { return _store.nodeCount(); }

	/// @brief  边数量
	size_t edgeCount() const { return _store.edgeCount(); }

	/// @brief  获取所有节点名的列表
	std::vector<std::string> nodeNames() const { return _store.nodeNames(); }

	/// @brief  获取所有边的只读引用
	const std::vector<Edge>& edges() const { return _store.edges(); }

	/// @brief  获取所有输入绑定的只读引用
	const std::vector<InputBinding>& inputBindings() const { return _store.inputBindings(); }

	/// @brief  获取所有输出绑定的只读引用
	const std::vector<OutputBinding>& outputBindings() const { return _outputZone.bindings(); }

	// ── 错误诊断 ──

	/// @brief  查询指定 task 在整条传播链上的所有错误记录
	std::vector<TaskError> taskErrors(const TaskId& taskId) const { return _errors.taskErrors(taskId); }

	/// @brief  清除所有 task 级错误记录（通常在重新 submit 前调用）
	void clearErrors() { _errors.clearErrors(); }

	/// @brief  是否有任何 task 发生过错误
	bool hasErrors() const { return _errors.hasErrors(); }

	// ── task 完成回调 ──

	using TaskCompleteCallback = std::function<void(const TaskId&)>;

	/// @brief  设置 task 完成回调（每次 submit 前设置；_terminate 末尾触发）
	void setTaskCompleteCallback(TaskCompleteCallback cb) { _engine.setTaskCompleteCallback(std::move(cb)); }

	// ── 信号系统 ──

	/// @brief  写入图级信号值（广播，所有 task 生效）。
	void setSignal(const std::string& name, bool value) { _signalStore->set(name, value); }

	/// @brief  写入 task 级信号值（仅对指定 taskId 生效，覆盖同名的广播信号）。
	void setSignal(const std::string& name, const TaskId& taskId, bool value) { _signalStore->set(name, taskId, value); }

	/// @brief  读取全局信号值。
	bool getSignal(const std::string& name) const { return _signalStore->get(name); }

	/// @brief  读取信号值（task 级优先 → 全局回退）。
	bool getSignal(const std::string& name, const TaskId& taskId) const { return _signalStore->get(name, taskId); }

	/// @brief  获取信号仓库指针，供 Node::bindSignal 使用。
	std::shared_ptr<SignalStore> signalStore() { return _signalStore; }

	// ── 同步等待与图导出 ──

	/// @brief  同步等待 task 完成（内部阻塞，供 exportNode / 外部同步使用）
	bool wait(const TaskId& taskId,
			  std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
		return _engine.wait(taskId, timeout);
	}

	/// @brief  导出为可嵌入父图的包装 Node
	///         子图内部用独立 CoroScheduler 执行，与父图隔离
	/// @note   前提：已调用 bindInput + bindOutput 定义了图接口
	///         调用者必须保证 InferGraph 在返回的 Node 使用期间存活
	///
	/// @note   性能提示：若仅需将一组节点约束为串行执行（共享单线程），
	///         优先使用 declareSubgraph()——零额外线程开销、零调度器开销。
	///         exportNode 适用于需要独立 InferGraph 实例的部署边界（如跨设备/跨进程）。
	std::unique_ptr<Node> exportNode(const std::string& nodeName,
									uint32_t maxHops = kDefaultMaxHops);

private:
	// ── 内部组件（声明顺序决定析构顺序）──
	// ExecutionEngine 必须最后声明 → 最先析构：
	//   其 CoroScheduler shutdown 期间 TaskGate 析构函数需访问下方成员。
	std::shared_ptr<SignalStore> _signalStore;
	ErrorTracker    _errors;
	OutputZone      _outputZone;
	GraphStore      _store;
	ExecutionEngine _engine;
};

} // namespace DC
