#pragma once

#include "Node.h"
#include "GraphRuntimeState.h"
#include "GraphBuilder.h"
#include "ExecutionEngine.h"
#include "GraphException.h"
#include "TaskStatus.h"

#include <chrono>
#include <functional>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace DC {

// ── 推理图：DC 电路图语义 ──
//
// InferGraph 是用户唯一接触的 Facade，生命周期分两个阶段：
//
//   Build（构建期）── GraphBuilder 承接 addNode/connect/bind... 等可变构建 API
//        │ compile()（惰性：首次运行期 API 自动触发；或显式 freeze()）
//        v
//   Freeze/Execute（执行期）── 构建面关闭（Frozen），运行期只读冻结快照：
//   - CompiledGraph  不可变拓扑 + GraphSignature（图级绑定契约）
//   - ExecutionEngine 执行调度与事件驱动数据传播
//   - OutputZone      输出聚合（纯任务态）
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

	/// @brief  构造推理图（默认线程池配置；可通过参数自定义三层线程池）
	/// @param  computeCfg    计算线程池配置
	/// @param  operatorCfg   算子线程池配置
	/// @param  systemCfg     系统线程池配置（连接器、数据搬运等基础设施）
	explicit InferGraph(const PoolConfig& computeCfg = {}, const PoolConfig& operatorCfg = {},
						const PoolConfig& systemCfg = {});
	~InferGraph() = default;

	InferGraph(const InferGraph&) = delete;
	InferGraph& operator=(const InferGraph&) = delete;

	// 移动语义禁止：ExecutionEngine 持有活跃线程池状态与任务状态表，
	// 移动后线程池内飞行任务的 this 捕获会悬空。
	InferGraph(InferGraph&&) = delete;
	InferGraph& operator=(InferGraph&&) = delete;

	// ── 图构建（构建期 API：惰性冻结后抛 GraphException(Frozen)）──
	//
	// Build → Freeze → Execute：构建方法仅可在首次运行期 API（submit/feedInput）
	// 之前使用；届时构建面自动编译为不可变 CompiledGraph 快照（惰性冻结），
	// 也可经 freeze() 显式提前冻结。执行期拓扑不可变——
	// "Can topology change while tasks are active?" 的答案恒为 No。
	// 拓扑演进路径：重新构建 GraphBuilder → compile 产生新快照，旧图任务排空后替换。

	/// @brief  添加节点（转移所有权），返回引用供后续接线引用
	/// @throws GraphException(DuplicateNode) 若节点名为空或重名
	/// @throws GraphException(Frozen) 若图已冻结
	Node& addNode(std::unique_ptr<Node> node) {
		_ensureNotFrozen("InferGraph::addNode");
		return _builder->addNode(std::move(node));
	}

	/// @brief  端口级接线（默认方式）：上游输出口 → 下游输入口，
	///         自动插入广播连接器（Broadcast Connector, N=1）
	/// @throws GraphException(NodeNotFound/PortNotFound) 若节点或端口不存在
	/// @throws GraphException(Frozen) 若图已冻结
	/// @return 指向自动创建的广播连接器的引用
	Node& connect(const std::string& srcNode, const std::string& srcPort,
				  const std::string& dstNode, const std::string& dstPort) {
		_ensureNotFrozen("InferGraph::connect");
		return _builder->connect(srcNode, srcPort, dstNode, dstPort);
	}

	/// @brief  端口级接线原语（低层）：上游输出口 → 下游输入口，直接建边
	///         约束：至少有一端是连接器（两个业务节点禁止直连）
	/// @throws GraphException(NodeNotFound/PortNotFound/DirectConnect) 若接线不合法
	/// @throws GraphException(Frozen) 若图已冻结
	void connectRaw(const std::string& srcNode, const std::string& srcPort,
					const std::string& dstNode, const std::string& dstPort) {
		_ensureNotFrozen("InferGraph::connectRaw");
		_builder->connectRaw(srcNode, srcPort, dstNode, dstPort);
	}

	/// @brief  快捷批量接线（低层）：自动匹配上游所有输出口到下游同名的输入口，
	///         直接建边，不插入连接器
	/// @return 成功匹配的端口对数
	/// @throws GraphException(Frozen) 若图已冻结
	size_t connectAll(const std::string& srcNode, const std::string& dstNode) {
		_ensureNotFrozen("InferGraph::connectAll");
		return _builder->connectAll(srcNode, dstNode);
	}

	/// @brief  标记输入：该节点的该端口为图级输入口，外部通过此口注入数据
	/// @throws GraphException(Frozen) 若图已冻结
	void bindInput(const std::string& nodeName, const std::string& portName) {
		_ensureNotFrozen("InferGraph::bindInput");
		_builder->bindInput(nodeName, portName);
	}

	/// @brief  带公共别名的图级输入绑定
	///
	/// 别名是图对外契约的一部分：内部节点/端口重构后，只要别名映射不变，
	/// 调用方代码无需改动。feedBoundInput 优先按别名解析，可消除
	/// 跨节点同名端口的注入歧义。
	/// @param  alias  公共别名（须在全部输入绑定中唯一）
	/// @throws GraphException(DuplicateBinding) 若别名已被其他输入绑定使用
	/// @throws GraphException(Frozen) 若图已冻结
	void bindInput(const std::string& alias, const std::string& nodeName,
				   const std::string& portName) {
		_ensureNotFrozen("InferGraph::bindInput");
		_builder->bindInput(nodeName, portName, alias);
	}

	/// @brief  标记输出：该节点的该端口产出进入输出区（与边目的地互斥）
	/// @throws GraphException(Frozen) 若图已冻结
	void bindOutput(const std::string& nodeName, const std::string& portName) {
		_ensureNotFrozen("InferGraph::bindOutput");
		_builder->bindOutput(nodeName, portName);
	}

	/// @brief  带公共别名的图级输出绑定
	///
	/// 绑定后即可用 takeOutput(taskId, alias) / takeOutputTensor(taskId, alias)
	/// 按公共名取结果，无需向调用方暴露内部节点名与端口名。
	/// @param  alias  公共别名（须在全部输出绑定中唯一）
	/// @throws GraphException(DuplicateBinding) 若别名已被其他输出绑定使用
	/// @throws GraphException(Frozen) 若图已冻结
	void bindOutput(const std::string& alias, const std::string& nodeName,
					const std::string& portName) {
		_ensureNotFrozen("InferGraph::bindOutput");
		_builder->bindOutput(nodeName, portName, alias);
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

	/// @brief  便捷注入：按 bindInput 声明的端口名定位，无需重复提供节点名。
	/// @throws GraphException(NodeNotFound) 无此绑定端口（需先 bindInput）
	/// @throws GraphException(FeedFailed)   绑定名跨节点歧义，或底层注入失败
	void feedBoundInput(const TaskId& taskId, const std::string& portName, Value data);

	/// @brief  便捷注入：DC::Tensor 重载
	void feedBoundInput(const TaskId& taskId, const std::string& portName, Tensor data);

	// ── 执行驱动 ──

	/// @brief  异步启动整张图的计算。输出声明直接作为 submit 参数，消除 temporal coupling。
	/// @param  declarations  期望产出：{nodeName, portName, count} 列表
	/// @throws GraphException(DuplicateTask) 若同 taskId 任务仍在执行
	/// @note   复用已终止的 taskId 合法：上一轮的声明/结果/诊断随之清理。
	///         输出在 task 终止后仍保留，供 wait → takeOutput 取用。
	void submit(const TaskId& taskId, std::vector<OutputDeclaration> declarations,
				std::chrono::milliseconds timeout = std::chrono::milliseconds(0),
				uint32_t maxHops = kDefaultMaxHops) {
		_ensureFrozen();                     // 惰性冻结：首次提交即编译（此后拓扑不可变）
		_ensureSubmittable(taskId);
		_state->errors.clearTask(taskId);    // 上一轮诊断不残留（影响 taskStatus 归一化）
		_state->output.clearTask(taskId);    // 复用同 ID：清掉上一轮声明/累加/结果
		_state->output.declare(taskId, std::move(declarations));
		_engine->submit(taskId, timeout, maxHops, _state);
	}

	/// @brief  单输出便捷重载（生命周期语义同上）
	void submit(const TaskId& taskId, const std::string& nodeName, const std::string& portName,
				size_t count = 1,
				std::chrono::milliseconds timeout = std::chrono::milliseconds(0),
				uint32_t maxHops = kDefaultMaxHops) {
		_ensureFrozen();                     // 惰性冻结：首次提交即编译（此后拓扑不可变）
		_ensureSubmittable(taskId);
		_state->errors.clearTask(taskId);
		_state->output.clearTask(taskId);
		_state->output.declare(taskId, nodeName, portName, count);
		_engine->submit(taskId, timeout, maxHops, _state);
	}

	// ── 结果获取（消费式：取出即消耗）──

	/// @brief  消费式取出输出区中指定端口的结果（取出后内部清空，不可重复读取）
	/// @note   结果在 task 终止（wait 返回）后仍然有效，直至下一次同 ID submit
	///         或 releaseTask()——支持 submit → wait → takeOutput 的同步用法；
	///         非破坏式预览见 NodeExecutor::peekOutput（底层接口）
	/// @throws GraphException(NodeNotFound) 若节点不存在
	Value takeOutput(const TaskId& taskId, const std::string& nodeName, const std::string& portName);

	/// @brief  便捷接口：消费式取出 DC::Tensor
	/// @throws GraphException(NodeNotFound) 若节点不存在
	Tensor takeOutputTensor(const TaskId& taskId, const std::string& nodeName, const std::string& portName);

	/// @brief  消费式取出：按公共别名或唯一绑定端口名定位，无需内部节点名
	/// @throws GraphException(NodeNotFound) 无此别名/绑定端口
	/// @throws GraphException(FeedFailed)   名称跨绑定歧义（用唯一别名消除）
	Value takeOutput(const TaskId& taskId, const std::string& name);

	/// @brief  消费式取出 Tensor：按公共别名或唯一绑定端口名定位
	/// @throws 同 2 参 takeOutput
	Tensor takeOutputTensor(const TaskId& taskId, const std::string& name);

	/// @brief  检查输出区中是否有结果
	bool hasOutput(const TaskId& taskId, const std::string& nodeName, const std::string& portName) const;

	/// @brief  检查结果是否存在（按公共别名或唯一绑定端口名）
	bool hasOutput(const TaskId& taskId, const std::string& name) const;

	/// @brief  便捷提交：以全部 bindOutput 绑定作为输出声明（各 count=1）。
	///         已 bindOutput 的端口无需在 submit 时重复声明。
	/// @throws GraphException(NoDeclaration) 未 bindOutput 任何端口
	void submitBound(const TaskId& taskId,
					 std::chrono::milliseconds timeout = std::chrono::milliseconds(0),
					 uint32_t maxHops = kDefaultMaxHops);

	// ── task 生命周期（状态 / 取消 / 结构化等待 / 资源回收）──

	/// @brief  查询 task 当前状态
	/// @return Unknown=从未提交；Running=执行中；Succeeded/TimedOut/Cancelled=已终止；
	///         Succeeded 但存在 Error 级诊断时归一化为 Failed（部分节点执行失败）
	TaskStatus taskStatus(const TaskId& taskId) const;

	/// @brief  请求取消活动中的 task（幂等；未知或已终止返回 false）。
	///         协作式取消：在飞节点执行不被中断，传播链即刻停止，
	///         wait()/waitForResult() 被唤醒，状态置 Cancelled。
	bool cancel(const TaskId& taskId) { return _engine->cancel(taskId); }

	/// @brief  同步等待 task 终止并返回结构化结果（无限等待直至终止）
	/// @note   可能长时间阻塞（远端/慢引擎/信号阻塞）的场景应改用
	///         显式超时重载，或从其他线程调用 cancel() 唤醒等待
	TaskResult waitForResult(const TaskId& taskId);

	/// @brief  同步等待 task 终止并返回结构化结果（显式超时）
	/// @param  timeout 等待超时（超时未终止时 status 为 Running，调用方可据此区分
	///         "仍在运行"与各类终止态）；超时只放弃等待，不取消任务
	/// @note   输出数据在终止后仍由 OutputZone 持有，经 takeOutput 取出
	TaskResult waitForResult(const TaskId& taskId, std::chrono::milliseconds timeout);

	/// @brief  释放已终止 task 的全部资源（状态表条目、OutputZone 结果、诊断记录）
	/// @note   此后 taskStatus 返回 Unknown、hasOutput 返回 false；
	///         活动 task 不可释放；"大量短任务"场景建议在消费结果后调用以防内存增长
	void releaseTask(const TaskId& taskId);

	// ── 查询（源图视角：冻结前后均反映源图拓扑/绑定，供内省与序列化）──

	/// @brief  获取节点指针（非拥有），不存在返回 nullptr。
	///         构建期专用（冻结后抛 Frozen）：运行期节点不可变，
	///         只读访问用 const 重载。
	Node* node(const std::string& name) {
		_ensureNotFrozen("InferGraph::node");
		return _builder->store().node(name);
	}

	/// @brief  获取节点指针（只读）
	const Node* node(const std::string& name) const { return _topology().node(name); }

	/// @brief  节点数量
	size_t nodeCount() const { return _topology().nodeCount(); }

	/// @brief  边数量
	size_t edgeCount() const { return _topology().edgeCount(); }

	/// @brief  获取所有节点名的列表
	std::vector<std::string> nodeNames() const { return _topology().nodeNames(); }

	/// @brief  获取所有边的只读引用
	const std::vector<Edge>& edges() const { return _topology().edges(); }

	/// @brief  获取所有输入绑定的只读引用
	const std::vector<InputBinding>& inputBindings() const { return _inputBindingsView(); }

	/// @brief  获取所有输出绑定的只读引用
	const std::vector<OutputBinding>& outputBindings() const { return _outputBindingsView(); }

	// ── 冻结 ──

	/// @brief  显式冻结（高级用法）：立即编译构建面为不可变快照。
	///         此后所有构建 API 抛 GraphException(Frozen)；运行期 API 照常。
	/// @return 冻结快照（与运行时共享同一份；幂等：重复调用返回同一快照）
	std::shared_ptr<const CompiledGraph> freeze() { return _ensureFrozen(); }

	// ── 错误诊断 ──

	/// @brief  查询指定 task 在整条传播链上的所有错误记录
	std::vector<TaskError> taskErrors(const TaskId& taskId) const { return _state->errors.taskErrors(taskId); }

	/// @brief  清除所有 task 级错误记录（通常在重新 submit 前调用）
	void clearErrors() { _state->errors.clearErrors(); }

	/// @brief  是否有任何 task 发生过错误
	bool hasErrors() const { return _state->errors.hasErrors(); }

	// ── task 完成回调 ──

	using TaskCompleteCallback = std::function<void(const TaskId&)>;

	/// @brief  设置 task 完成回调（每次 submit 前设置；_terminate 末尾触发）
	void setTaskCompleteCallback(TaskCompleteCallback cb) { _engine->setTaskCompleteCallback(std::move(cb)); }

	// ── 信号系统 ──

	/// @brief  写入图级信号值（广播，所有 task 生效）。
	void setSignal(const std::string& name, bool value) { _state->signals->set(name, value); }

	/// @brief  写入 task 级信号值（仅对指定 taskId 生效，覆盖同名的广播信号）。
	void setSignal(const std::string& name, const TaskId& taskId, bool value) { _state->signals->set(name, taskId, value); }

	/// @brief  读取全局信号值。
	bool getSignal(const std::string& name) const { return _state->signals->get(name); }

	/// @brief  读取信号值（task 级优先 → 全局回退）。
	bool getSignal(const std::string& name, const TaskId& taskId) const { return _state->signals->get(name, taskId); }

	/// @brief  获取信号仓库指针，供 Node::bindSignal 使用。
	std::shared_ptr<SignalStore> signalStore() { return _state->signals; }

	// ── 同步等待与图导出 ──

	/// @brief  同步等待 task 终止（无限等待；返回后可经 takeOutput 读取结果）
	/// @return true 已终止；false taskId 未知（从未提交或已 releaseTask）
	bool wait(const TaskId& taskId) {
		return _engine->wait(taskId, std::chrono::milliseconds(0));
	}

	/// @brief  同步等待 task 终止（显式超时；timeout <= 0 视为无限等待）
	/// @return true 在超时内终止，false 超时或 taskId 未知（任务仍在运行，未被取消）
	bool wait(const TaskId& taskId, std::chrono::milliseconds timeout) {
		return _engine->wait(taskId, timeout);
	}

	/// @brief  导出为可嵌入父图的包装 Node
	///         子图复用本图的 ExecutionEngine（三层线程池）执行，与父图隔离
	/// @note   前提：已调用 bindInput + bindOutput 定义了图接口
	///         调用者必须保证 InferGraph 在返回的 Node 使用期间存活
	///
	/// @note   性能提示：若仅需将一组节点约束为串行执行（共享单线程），
	///         优先使用 declareSubgraph()——零额外线程开销、零调度开销。
	///         exportNode 适用于需要独立 InferGraph 实例的部署边界（如跨设备/跨进程）。
	std::unique_ptr<Node> exportNode(const std::string& nodeName,
									uint32_t maxHops = kDefaultMaxHops);

private:
	/// @brief  提交前置校验：活动 task 拒绝重复提交（engine.submit 内部还有权威校验）
	void _ensureSubmittable(const TaskId& taskId) const {
		if (_engine->status(taskId) == TaskStatus::Running)
			throw GraphException(GraphException::ErrorType::DuplicateTask, "InferGraph::submit",
								 "task '" + taskId + "' is still running; duplicate submit rejected");
	}

	/// @brief  惰性冻结：首次运行期调用时把构建面编译为不可变快照。
	/// @return 冻结快照（幂等：已冻结时直接返回现有快照）
	/// @note   快指针无锁（快照非空即已冻结）；竞态由 _freezeMutex 串行化，
	///         compile 仅在首个 submit/feedInput 时执行一次
	std::shared_ptr<const CompiledGraph> _ensureFrozen() const {
		if (_state->graph)
			return _state->graph;
		std::lock_guard lk(_freezeMutex);
		if (!_state->graph)
			_state->attachGraph(_builder->compile()); // 快照 + 节点执行闸表一并就位
		return _state->graph;
	}

	/// @brief  构建期守卫：冻结后调用构建 API 抛 GraphException(Frozen)
	void _ensureNotFrozen(const char* api) const {
		if (_state->graph)
			throw GraphException(GraphException::ErrorType::Frozen, api,
								 "graph is frozen; topology is immutable after first submit/feedInput"
								 " (rebuild a GraphBuilder and compile a new snapshot to evolve)"
								 );
	}

	/// @brief  拓扑访问（源图视角）：冻结后读快照，构建期读 builder
	const GraphStore& _topology() const {
		return _state->graph ? _state->graph->store() : _builder->store();
	}

	/// @brief  输入绑定视图：冻结后读 GraphSignature（无锁），构建期读 builder
	const std::vector<InputBinding>& _inputBindingsView() const {
		return _state->graph ? _state->graph->signature().inputs : _builder->inputBindings();
	}

	/// @brief  输出绑定视图：冻结后读 GraphSignature（无锁），构建期读 builder
	const std::vector<OutputBinding>& _outputBindingsView() const {
		return _state->graph ? _state->graph->signature().outputs : _builder->outputBindings();
	}

	/// @brief  解析图级输出名：公共别名优先，其次唯一绑定的端口名
	/// @return (nodeName, portName)
	std::pair<std::string, std::string>
	_resolveOutputName(const std::string& name, const char* api) const;

	/// @brief  解析图级输入名：公共别名优先，其次唯一绑定的端口名
	std::pair<std::string, std::string>
	_resolveInputName(const std::string& name, const char* api) const;

	// ── 内部组件 ──
	// 图状态（graph/output/signals/errors）聚合为共享的 GraphRuntimeState：
	// 飞行任务经 TaskGate/任务 lambda/看门狗持有同一 shared_ptr，图组件的
	// 存活期由引用计数保证，不再依赖成员声明顺序约定。
	// _builder 为构建期唯一可变面（compile 时拓扑所有权移交快照）；
	// ExecutionEngine 保持与图同生命周期：engine 最后声明 → 最先析构，
	// 线程池 shutdown（join 全部 worker）先于 state 释放发生。
	std::shared_ptr<GraphRuntimeState> _state;
	std::unique_ptr<GraphBuilder> _builder = std::make_unique<GraphBuilder>();
	std::unique_ptr<ExecutionEngine> _engine;
	mutable std::mutex _freezeMutex; ///< 惰性冻结串行化（快照填充一次性）
};

} // namespace DC
