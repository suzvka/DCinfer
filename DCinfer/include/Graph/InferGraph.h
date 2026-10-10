#pragma once

#include "Node.h"
#include "GraphRuntimeState.h"
#include "GraphBuilder.h"
#include "GraphInterface.h"
#include "ExecutionEngine.h"
#include "ResourceScheduler.h"
#include "GraphException.h"
#include "TaskStatus.h"

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace DC {

// 推理图：用户唯一接触的 Facade。生命周期：构建期（GraphBuilder 可变构建 API）
// 经 compile() 惰性冻结（首次运行期 API 自动触发，或显式 freeze()）进入执行期——
// 构建面关闭，运行期只读 CompiledGraph 冻结快照。
class InferGraph {
public:
	using TaskId = Node::TaskId;

	using Edge = GraphStore::Edge;

	/// @brief 默认最大跳数（TTL），防止循环无限传播
	static constexpr uint32_t kDefaultMaxHops = ExecutionEngine::kDefaultMaxHops;

	/// @brief 构造推理图（scheduler 为空时共享进程级默认实例）。
	explicit InferGraph(std::shared_ptr<ResourceScheduler> scheduler = nullptr);
	/// 必须由外部持有者销毁：从自身 RunFn/完成回调中销毁本图会打印诊断并 terminate。
	~InferGraph() = default;

	InferGraph(const InferGraph&) = delete;
	InferGraph& operator=(const InferGraph&) = delete;

	// 禁止移动：在飞任务持有 this 捕获，移动会悬空。
	InferGraph(InferGraph&&) = delete;
	InferGraph& operator=(InferGraph&&) = delete;

	// 构建期 API：首次运行期 API（submit/feedInput）或 freeze() 后抛 GraphException(Frozen)。
	// 拓扑演进：重建 GraphBuilder 并 compile 新快照。

	/// @brief 添加节点（转移所有权）；空名或重名抛 DuplicateNode。
	Node& addNode(std::unique_ptr<Node> node) {
		_ensureNotFrozen("InferGraph::addNode");
		return _builder->addNode(std::move(node));
	}

	/// @brief 端口级接线（自动插入广播连接器 N=1；同源口再次 connect 扩容扇出）。
	Node& connect(const std::string& srcNode, const std::string& srcPort,
				  const std::string& dstNode, const std::string& dstPort) {
		_ensureNotFrozen("InferGraph::connect");
		return _builder->connect(srcNode, srcPort, dstNode, dstPort);
	}

	/// @brief 图级输入绑定（alias 必填且唯一；不参与运行时寻址，见 interface()）。
	void bindInput(const std::string& alias, const std::string& nodeName,
				   const std::string& portName) {
		_ensureNotFrozen("InferGraph::bindInput");
		_builder->bindInput(nodeName, portName, alias);
	}

	/// @brief 图级输出绑定（alias 必填且唯一；submitBound 以此为声明来源）。
	/// @throws GraphException(NonTerminalPort) 绑定端口有出边（取数与出边传播共享消费槽）。
	void bindOutput(const std::string& alias, const std::string& nodeName,
					const std::string& portName) {
		_ensureNotFrozen("InferGraph::bindOutput");
		_builder->bindOutput(nodeName, portName, alias);
	}

	/// @brief 取图公开接口：冻结图并一次性解析绑定（坐标存在性创建时校验）。
	GraphInterface interface();

	/// @brief 从图外注入数据到指定节点的输入端口（写入缓冲，不触发执行）。
	/// @throws GraphException(DuplicateTask) task 处于收尾窗口（waitForResult 返回后再 feed）。
	void feedInput(const TaskId& taskId, const std::string& nodeName, const std::string& portName, Value data);

	/// @brief 便捷：直接传入 DC::Tensor。
	void feedInput(const TaskId& taskId, const std::string& nodeName, const std::string& portName, Tensor data);

	/// @brief 异步启动整张图的计算；声明是完成条件：全部声明满足即终止。
	/// @param declarations 期望产出 {nodeName, portName, count} 列表
	/// @throws GraphException DuplicateTask（同 taskId 在飞或收尾窗口内）/
	///         UnreachableDeclaration（拓扑不可达，提交期即暴露）。
	void submit(const TaskId& taskId, std::vector<OutputDeclaration> declarations,
				uint32_t maxHops = kDefaultMaxHops) {
		_ensureFrozen(); // 惰性冻结：首次提交即编译（此后拓扑不可变）
		_engine->submit(taskId, maxHops, _state, std::move(declarations));
	}

	/// @brief 单输出便捷重载。
	void submit(const TaskId& taskId, const std::string& nodeName, const std::string& portName,
				size_t count = 1,
				uint32_t maxHops = kDefaultMaxHops) {
		_ensureFrozen(); // 惰性冻结：首次提交即编译（此后拓扑不可变）
		std::vector<OutputDeclaration> declarations{{nodeName, portName, count}};
		_engine->submit(taskId, maxHops, _state, std::move(declarations));
	}

	/// @brief 消费式取出输出（取出即清空；结果存活至下一次同 ID submit 或 releaseTask）。
	Value takeOutput(const TaskId& taskId, const std::string& nodeName, const std::string& portName);

	/// @brief 便捷：消费式取出 DC::Tensor。
	Tensor takeOutputTensor(const TaskId& taskId, const std::string& nodeName, const std::string& portName);

	/// @brief 检查输出区中是否有结果。
	bool hasOutput(const TaskId& taskId, const std::string& nodeName, const std::string& portName) const;

	/// @brief 以全部 bindOutput 绑定作为输出声明提交（各 count=1）；无绑定抛 NoDeclaration。
	void submitBound(const TaskId& taskId,
					 uint32_t maxHops = kDefaultMaxHops);

	/// @brief 查询 task 当前状态（Succeeded 但存在 Error 级诊断时归一化为 Failed）。
	TaskStatus taskStatus(const TaskId& taskId) const;

	/// @brief 请求取消活动中的 task（幂等；协作式：在飞节点不中断，状态置 Cancelled）。
	bool cancel(const TaskId& taskId) { return _engine->cancel(taskId); }

	/// @brief 同步等待 task 终止并返回结果（无限等待；长阻塞场景用超时重载）。
	TaskResult waitForResult(const TaskId& taskId);

	/// @brief 同步等待 task 终止并返回结果（显式超时）。
	/// @param timeout 等待超时；未满足时按 {status=Running} 返回，超时只放弃等待不取消任务
	TaskResult waitForResult(const TaskId& taskId, std::chrono::milliseconds timeout);

	/// @brief 释放已终止 task 的全部资源（状态/结果/诊断；活动或收尾中不可释放）。
	void releaseTask(const TaskId& taskId);

	/// @brief 弃置"已喂数据但从未提交"的任务输入（句柄析构路径调用）。
	void discardUnsubmitted(const TaskId& taskId);

	/// @brief 弃置在飞任务的托管句柄：不取消任务，终态收尾时自动回收状态/结果/诊断；
	///        已终止 → 等价 releaseTask；未知 → no-op。
	void detachTask(const TaskId& taskId);

	/// @brief 获取节点可写指针（构建期专用；冻结后抛 Frozen）。
	Node* node(const std::string& name) {
		_ensureNotFrozen("InferGraph::node");
		return _builder->node(name);
	}

	/// @brief 获取节点指针（只读）。
	const Node* node(const std::string& name) const { return _topology().node(name); }

	size_t nodeCount() const { return _topology().nodeCount(); }
	size_t edgeCount() const { return _topology().edgeCount(); }
	std::vector<std::string> nodeNames() const { return _topology().nodeNames(); }

	/// @brief 获取所有边（值副本）。
	std::vector<Edge> edges() const { return _topology().edges(); }

	std::vector<InputBinding> inputBindings() const { return _inputBindingsView(); }
	const std::vector<OutputBinding>& outputBindings() const { return _outputBindingsView(); }

	/// @brief 资源调度器共享句柄。
	const std::shared_ptr<ResourceScheduler>& scheduler() const { return _scheduler; }

	/// @brief 显式冻结：立即编译为不可变快照（幂等；此后构建 API 抛 Frozen）。
	std::shared_ptr<const CompiledGraph> freeze() { return _ensureFrozen(); }

	/// @brief 查询指定 task 的所有错误记录。
	std::vector<TaskError> taskErrors(const TaskId& taskId) const { return _state->errors.taskErrors(taskId); }

	/// @brief 清除所有 task 级错误记录。
	void clearErrors() { _state->errors.clearErrors(); }

	/// @brief 是否有任何 task 发生过错误。
	bool hasErrors() const { return _state->errors.hasErrors(); }

	using TaskCompleteCallback = std::function<void(const TaskId&)>;

	/// @brief 设置 task 完成回调（_terminate 末尾触发）。
	void setTaskCompleteCallback(TaskCompleteCallback cb) { _engine->setTaskCompleteCallback(std::move(cb)); }

	/// @brief 写入图级信号值（广播）。
	void setSignal(const std::string& name, bool value) { _state->signals->set(name, value); }

	/// @brief 写入 task 级信号值（覆盖同名广播信号）。
	void setSignal(const std::string& name, const TaskId& taskId, bool value) { _state->signals->set(name, taskId, value); }

	/// @brief 读取全局信号值。
	bool getSignal(const std::string& name) const { return _state->signals->get(name); }

	/// @brief 读取信号值（task 级优先 → 全局回退）。
	bool getSignal(const std::string& name, const TaskId& taskId) const { return _state->signals->get(name, taskId); }

	/// @brief 获取信号仓库指针。
	std::shared_ptr<SignalStore> signalStore() { return _state->signals; }

private:
	/// @brief 惰性冻结：首次运行期调用时编译为快照（快路径无锁读；首次编译由 _freezeMutex 串行化）。
	std::shared_ptr<const CompiledGraph> _ensureFrozen() const {
		if (auto snap = _state->snapshot())
			return snap;
		std::lock_guard lk(_freezeMutex);
		if (auto snap = _state->snapshot())
			return snap;
		_state->attachGraph(_builder->compile()); // 快照 + 节点执行闸表一并就位
		return _state->snapshot();
	}

	/// @brief 构建期守卫：冻结后调用构建 API 抛 Frozen。
	void _ensureNotFrozen(const char* api) const {
		if (_state->snapshot())
			throw GraphException(GraphException::ErrorType::Frozen, api,
								 "graph is frozen; topology is immutable after first submit/feedInput"
								 " (rebuild a GraphBuilder and compile a new snapshot to evolve)"
								 );
	}

	/// @brief 拓扑访问：冻结后读快照，构建期读 builder。
	const GraphStore& _topology() const {
		if (auto snap = _state->snapshot())
			return snap->store();
		return _builder->store();
	}

	/// @brief 输入绑定视图：冻结后读签名，构建期读 builder。
	std::vector<InputBinding> _inputBindingsView() const {
		if (auto snap = _state->snapshot())
			return snap->signature().inputs;
		return _builder->inputBindings();
	}

	/// @brief 输出绑定视图：冻结后读签名，构建期读 builder。
	const std::vector<OutputBinding>& _outputBindingsView() const {
		if (auto snap = _state->snapshot())
			return snap->signature().outputs;
		return _builder->outputBindings();
	}

	// engine 最后声明 → 最先析构：自排水完成后才释放图组件引用。
	std::shared_ptr<GraphRuntimeState> _state;
	std::shared_ptr<ResourceScheduler> _scheduler;
	std::unique_ptr<GraphBuilder> _builder = std::make_unique<GraphBuilder>();
	std::unique_ptr<ExecutionEngine> _engine;
	mutable std::mutex _freezeMutex;
};

} // namespace DC
