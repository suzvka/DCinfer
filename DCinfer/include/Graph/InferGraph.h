#pragma once

#include "Node.h"
#include "EngineRegistry.h"
#include "CoroScheduler.h"
#include "ThreadPool.h"
#include "GraphException.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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

/// @brief 输出声明：声明某 task 期望哪个节点的哪个端口产出多少次
struct OutputDeclaration {
	std::string nodeName; ///< 目标节点名
	std::string portName; ///< 目标端口名
	size_t count = 1; ///< 预期产出次数（>=1）
};

/// @brief 单条 task 级错误记录，包含节点名和异常信息
struct TaskError {
	std::string nodeName;  ///< 发生错误的节点名
	std::string source;    ///< 异常来源（如 "InferGraph::_propagateFrom"）
	std::string message;   ///< 错误详情
};

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
	InferGraph(InferGraph&&) = default;
	InferGraph& operator=(InferGraph&&) = default;

	// ── 图构建 ──

	/// @brief  添加节点（转移所有权），返回裸指针供后续接线引用
	/// @return 指向已存入图内节点的非拥有指针；若节点名为空或重名则返回 nullptr
	Node* addNode(std::unique_ptr<Node> node);

	/// @brief  端口级接线：上游输出口 → 下游输入口
	/// @return true 表示两端节点和端口均存在
	bool connect(const std::string& srcNode, const std::string& srcPort, const std::string& dstNode,
				 const std::string& dstPort);

	/// @brief  快捷接线：自动匹配上游所有输出口到下游同名的输入口
	///         典型用途：Connector 的 out_0→in_0, out_1→in_1, ...
	///         多个下游时需为每个下游各调用一次
	/// @return 成功匹配的端口对数
	size_t connectAll(const std::string& srcNode, const std::string& dstNode);

	/// @brief  接线：在两个节点间自动插入广播连接器（Broadcast Connector, N=1）
	///         等价于 connect(src→bc.in) + connect(bc.out_0→dst)
	///         N=1 广播走 takeInput 零拷贝 move 路径，等效直通导线
	///         适用于两个业务节点之间的 1→1 直连场景
	/// @return 指向自动创建的广播连接器的非拥有指针；若节点或端口不存在则返回 nullptr
	Node* wire(const std::string& srcNode, const std::string& srcPort, const std::string& dstNode,
			   const std::string& dstPort);

	/// @brief  标记输出：该节点的该端口结果输出到图外
	void bindOutput(const std::string& nodeName, const std::string& portName);

	// ── 数据注入 ──

	/// @brief  从图外注入数据到指定节点的输入端口（写入缓冲，不触发执行）
	bool feedInput(const TaskId& taskId, const std::string& nodeName, const std::string& portName, Value data);

	/// @brief  便捷接口：直接传入 DC::Tensor
	bool feedInput(const TaskId& taskId, const std::string& nodeName, const std::string& portName, Tensor data);

	// ── 执行驱动 ──

	/// @brief  异步启动整张图的计算
	///         为指定 taskId 扫描入口节点，创建协程链驱动数据流
	/// @param  timeout  超时时间（毫秒），0 表示不启用超时，默认 0
	/// @throws GraphException(NoDeclaration) 若未事先调用 declareOutput
	void submit(const TaskId& taskId, std::chrono::milliseconds timeout = std::chrono::milliseconds(0));

	// ── 结果获取 ──

	/// @brief  获取输出区中指定端口的结果（消费式取出）
	/// @throws GraphException(NodeNotFound) 若节点不存在
	Value getOutput(const TaskId& taskId, const std::string& nodeName, const std::string& portName);

	/// @brief  便捷接口：取出 DC::Tensor
	/// @throws GraphException(NodeNotFound) 若节点不存在
	Tensor getOutputTensor(const TaskId& taskId, const std::string& nodeName, const std::string& portName);

	/// @brief  检查输出区中是否有结果
	bool hasOutput(const TaskId& taskId, const std::string& nodeName, const std::string& portName) const;

	// ── 输出声明（submit 前必须调用）──

	/// @brief  声明某 task 的输出期望：指定节点端口需产出 N 次才停止
	///         声明条件满足时自动终止；数据耗尽而未满足则报错
	/// @note   必须在 submit() 前调用，否则 submit 会抛出 NoDeclaration
	void declareOutput(const TaskId& taskId, std::vector<OutputDeclaration> declarations);

	/// @brief  单条声明的便捷重载
	void declareOutput(const TaskId& taskId, const std::string& nodeName,
					   const std::string& portName, size_t count = 1);

	// ── 查询 ──

	/// @brief  获取节点指针（非拥有），不存在返回 nullptr
	Node* node(const std::string& name);

	/// @brief  获取节点指针（只读）
	const Node* node(const std::string& name) const;

	/// @brief  节点数量
	size_t nodeCount() const {
		return _nodes.size();
	}

	/// @brief  边数量
	size_t edgeCount() const {
		return _edges.size();
	}

	/// @brief  获取所有节点名的列表
	std::vector<std::string> nodeNames() const;

	/// @brief  获取所有边的只读引用
	const std::vector<Edge>& edges() const {
		return _edges;
	}

	/// @brief  获取所有输出绑定的只读引用
	const std::vector<OutputBinding>& outputBindings() const {
		return _outputBindings;
	}

	// ── 错误诊断 ──

	/// @brief  查询指定 task 在整条传播链上的所有错误记录
	///         包含 feedInput、tryExecute、setInput 过程中捕获的异常信息
	/// @return 按发生顺序排列的错误列表；若无错误则返回空向量
	std::vector<TaskError> taskErrors(const TaskId& taskId) const;

	/// @brief  清除所有 task 级错误记录（通常在重新 submit 前调用）
	void clearErrors();

	/// @brief  是否有任何 task 发生过错误
	bool hasErrors() const;

	// ── task 完成回调 ──

	/// @brief  task 完成回调类型：在 _terminate（task 彻底结束）时触发
	///        此时所有节点已执行完毕、传播链已走完、缓冲区已清理
	using TaskCompleteCallback = std::function<void(const TaskId&)>;

	/// @brief  设置 task 完成回调（每次 submit 前设置；_terminate 末尾触发）
	void setTaskCompleteCallback(TaskCompleteCallback cb) { _taskCompleteCb = std::move(cb); }

private:
	// ── 任务门控：shared_ptr 生命周期驱动耗尽检测 ──
	//
	// 每个活跃协程 + 超时看门狗各持有一份 shared_ptr<TaskGate>。
	// 当最后一个持有者析构时，若 task 未被终止，则触发 _onExhausted。
	struct TaskGate {
		std::atomic<bool> terminated{false};
		InferGraph* graph = nullptr;
		TaskId taskId;

		~TaskGate() {
			if (!terminated.load(std::memory_order_acquire) && graph) {
				graph->_onExhausted(taskId);
			}
		}
	};

	// 查找节点：存在返回指针，不存在返回 nullptr
	Node* _findNode(const std::string& name);
	const Node* _findNode(const std::string& name) const;

	// ── 协程数据传播 ──
	//
	// 核心：从指定节点出发，沿边传播数据到所有下游
	// co_await 节点完成 → 消费输出 → 写入下游 → 按 affinity 提交线程池 → spawn 下游传播协程
	// @param gate  共享任务门控，最后一个持有者析构时触发耗尽检测
	Task<void> _propagateFrom(std::string nodeName, TaskId taskId,
							  std::shared_ptr<TaskGate> gate);

	// ── 输出声明辅助 ──

	/// @brief  累积输出计数，返回 true 表示所有声明均已满足
	bool _accumulateAndCheck(const std::string& nodeName, const std::string& portName, const TaskId& taskId);

	/// @brief  终止指定 task：标记 + 遍历所有节点清理缓冲区
	void _terminate(const TaskId& taskId);

	/// @brief  查询 task 是否已被终止
	bool _isTerminated(const TaskId& taskId) const;

	/// @brief  数据耗尽时的处理：检查声明是否满足，满足则正常终止，否则报错并终止
	/// @note   由 TaskGate 析构函数调用，或超时看门狗触发
	void _onExhausted(const TaskId& taskId);

	// ── 错误记录 ──

	/// @brief  记录一条 task 级错误（线程安全，可在协程/线程池中调用）
	void _recordError(const TaskId& taskId, std::string nodeName, std::string source, std::string message);

	// ── 成员 ──
	std::unordered_map<std::string, std::unique_ptr<Node>> _nodes;
	std::vector<Edge> _edges;
	std::vector<OutputBinding> _outputBindings;

	// _ownedScheduler 必须在 _scheduler 之前声明
	std::unique_ptr<CoroScheduler> _ownedScheduler;
	CoroScheduler* _scheduler;

	// 线程池
	ThreadPool _computePool;
	ThreadPool _operatorPool;
	ThreadPool _systemPool;

	// 导线连接器自动命名计数器
	std::atomic<size_t> _nextWireId{0};

	// 输出声明：每 task 的期望输出列表（_declarationMutex 保护）
	std::unordered_map<TaskId, std::vector<OutputDeclaration>> _outputDeclarations;
	mutable std::mutex _declarationMutex;

	// 运行时累加器：每 task 的每端口产出次数（_declarationMutex 保护）
	// key = "nodeName:portName"
	std::unordered_map<TaskId, std::unordered_map<std::string, size_t>> _accumulatedCounts;

	// 已终止的 task 集合（_terminationMutex 保护）
	std::unordered_set<TaskId> _terminatedTasks;
	mutable std::mutex _terminationMutex;

	// task 级错误收集：每 taskId 对应按发生顺序排列的错误列表（_errorMutex 保护）
	std::unordered_map<TaskId, std::vector<TaskError>> _taskErrors;
	mutable std::mutex _errorMutex;

	// task 完成回调：在 _terminate 末尾触发（无需互斥锁，仅在 _terminate 中调用）
	TaskCompleteCallback _taskCompleteCb;
};

} // namespace DC
