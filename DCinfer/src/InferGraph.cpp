#include "InferGraph.h"
#include "NodeException.h"
#include "GraphException.h"
#include "Graph/internal/TaskExecutionState.h"
#include "SignalProbe.h"

namespace DC {

// ════════════════════════════════════════════
// 构造
// ════════════════════════════════════════════

InferGraph::InferGraph(const PoolConfig& computeCfg,
					   const PoolConfig& operatorCfg, const PoolConfig& systemCfg)
	: _state(std::make_shared<GraphRuntimeState>()),
	  _engine(std::make_unique<ExecutionEngine>(computeCfg, operatorCfg, systemCfg)) {}

// ════════════════════════════════════════════
// 子图声明
// ════════════════════════════════════════════

void InferGraph::declareSubgraph(const std::string& name,
								  std::initializer_list<std::string> nodeNames) {
	_ensureNotFrozen("InferGraph::declareSubgraph");

	// 1. 验证所有节点存在（affinity 可混合：组信号量跨池共享，全局互斥）
	for (const auto& nname : nodeNames) {
		auto* n = _topology().node(nname);
		if (!n)
			throw GraphException(GraphException::ErrorType::NodeNotFound, "InferGraph::declareSubgraph",
								 "node '" + nname + "' not found");
	}

	// 2. 设置所有节点的 tag 为子图名（构建期专用：_ensureNotFrozen 已校验未冻结）
	for (const auto& nname : nodeNames)
		_builder->store().node(nname)->setTag(name);

	// 3. 注册跨池分组限流（共享信号量，对三个线程池同时生效）
	_engine->registerGroupLimit(name, 1);
}

// ════════════════════════════════════════════
// 数据注入
// ════════════════════════════════════════════

void InferGraph::feedInput(const TaskId& taskId, const std::string& nodeName,
						   const std::string& portName, Value data) {
	_ensureFrozen(); // 惰性冻结：运行期 API 入口统一编译（此后拓扑不可变）
	auto* n = _topology().node(nodeName);
	if (!n)
		throw GraphException(GraphException::ErrorType::NodeNotFound, "InferGraph::feedInput",
							 "node '" + nodeName + "' not found");
	try {
		// 输入写入 task 执行域的 per-node 缓冲（原 Node 内嵌 TaskBuffer）；
		// shared_ptr 先落局部量，防止临时量析构导致引用悬垂
		auto ts = _state->exec->taskState(taskId);
		auto& ns = ts->ensure(nodeName, n->schema());
		ns.buffer.setInput(taskId, portName, std::move(data), n->schema());
	} catch (const NodeException& e) {
		_state->errors.recordError(taskId, nodeName, "InferGraph::feedInput",
							"NodeException in setInput for port '" + portName
								+ "': " + std::string(e.what()));
		throw GraphException(GraphException::ErrorType::FeedFailed, "InferGraph::feedInput",
							"failed to feed input '" + portName + "' on node '" + nodeName
								+ "': " + std::string(e.what()));
	}
}

void InferGraph::feedInput(const TaskId& taskId, const std::string& nodeName,
						   const std::string& portName, Tensor data) {
	feedInput(taskId, nodeName, portName, Value(std::make_unique<Tensor>(std::move(data))));
}

void InferGraph::feedBoundInput(const TaskId& taskId, const std::string& portName, Value data) {
	auto [nodeName, resolvedPort] = _resolveInputName(portName, "InferGraph::feedBoundInput");
	feedInput(taskId, nodeName, resolvedPort, std::move(data));
}

void InferGraph::feedBoundInput(const TaskId& taskId, const std::string& portName, Tensor data) {
	feedBoundInput(taskId, portName, Value(std::make_unique<Tensor>(std::move(data))));
}

void InferGraph::submitBound(const TaskId& taskId, std::chrono::milliseconds timeout,
							 uint32_t maxHops) {
	std::vector<OutputDeclaration> declarations;
	for (const auto& ob : _outputBindingsView())
		declarations.push_back({ob.nodeName, ob.portName, 1});
	if (declarations.empty())
		throw GraphException(GraphException::ErrorType::NoDeclaration, "InferGraph::submitBound",
							 "no bound output ports; call bindOutput(nodeName, portName) "
								 "or bindOutput(alias, nodeName, portName) first");
	submit(taskId, std::move(declarations), timeout, maxHops);
}

// ════════════════════════════════════════════
// 结果获取
// ════════════════════════════════════════════

Value InferGraph::takeOutput(const TaskId& taskId, const std::string& nodeName,
							const std::string& portName) {
	// 优先查 OutputZone（OutputZone 绑定端口的数据在 _propagateFrom 第二步已搬运至此）
	auto ozVal = _state->output.take(taskId, nodeName, portName);
	if (ozVal)
		return std::move(*ozVal);

	auto* n = _topology().node(nodeName);
	if (!n) {
		throw GraphException(GraphException::ErrorType::NodeNotFound, "InferGraph::takeOutput",
							 "node '" + nodeName + "' not found");
	}
	// 回退查 task 执行域的 per-node 缓冲（消息/异常语义与原 Node::takeOutput 一致）
	auto taskExec = _state->exec->findTaskState(taskId);
	auto* ns = taskExec ? taskExec->find(nodeName) : nullptr;
	if (!ns)
		throw NodeException(NodeException::ErrorType::TaskNotFound, "TaskBuffer::takeOutput",
							"task '" + taskId + "' not found");
	return ns->buffer.takeOutput(taskId, portName);
}

Tensor InferGraph::takeOutputTensor(const TaskId& taskId, const std::string& nodeName,
								   const std::string& portName) {
	// 优先查 OutputZone
	auto ozVal = _state->output.take(taskId, nodeName, portName);
	if (ozVal) {
		auto* t = ozVal->as<Tensor>();
		if (t)
			return std::move(*t);
		throw GraphException(GraphException::ErrorType::Other, "InferGraph::takeOutputTensor",
							 "OutputZone artifact for '" + nodeName + "." + portName
								 + "' is not a DC::Tensor");
	}

	auto* n = _topology().node(nodeName);
	if (!n) {
		throw GraphException(GraphException::ErrorType::NodeNotFound, "InferGraph::takeOutputTensor",
							 "node '" + nodeName + "' not found");
	}
	auto taskExec = _state->exec->findTaskState(taskId);
	auto* ns = taskExec ? taskExec->find(nodeName) : nullptr;
	if (!ns)
		throw NodeException(NodeException::ErrorType::TaskNotFound, "TaskBuffer::takeOutput",
							"task '" + taskId + "' not found");
	auto nt = ns->buffer.takeOutput(taskId, portName);
	auto* t = nt.as<Tensor>();
	if (!t) {
		throw NodeException(NodeException::ErrorType::TypeMismatch, "Node::takeOutputTensor",
							"output '" + portName + "' is not a DC::Tensor (innerType=" +
								std::to_string(static_cast<uint32_t>(nt.innerType())) + ")");
	}
	return std::move(*t);
}

bool InferGraph::hasOutput(const TaskId& taskId, const std::string& nodeName,
						   const std::string& portName) const {
	// 优先查 OutputZone
	if (_state->output.hasOutput(taskId, nodeName, portName))
		return true;

	auto* n = _topology().node(nodeName);
	if (!n)
		return false;
	auto taskExec = _state->exec->findTaskState(taskId);
	auto* ns = taskExec ? taskExec->find(nodeName) : nullptr;
	return ns && ns->buffer.hasOutput(taskId, portName);
}

// ── 按公共别名 / 唯一绑定端口名的图级取用 ──

Value InferGraph::takeOutput(const TaskId& taskId, const std::string& name) {
	auto [nodeName, portName] = _resolveOutputName(name, "InferGraph::takeOutput");
	return takeOutput(taskId, nodeName, portName);
}

Tensor InferGraph::takeOutputTensor(const TaskId& taskId, const std::string& name) {
	auto [nodeName, portName] = _resolveOutputName(name, "InferGraph::takeOutputTensor");
	return takeOutputTensor(taskId, nodeName, portName);
}

bool InferGraph::hasOutput(const TaskId& taskId, const std::string& name) const {
	auto [nodeName, portName] = _resolveOutputName(name, "InferGraph::hasOutput");
	return hasOutput(taskId, nodeName, portName);
}

// ── 名称解析：公共别名优先，其次唯一绑定的端口名 ──

std::pair<std::string, std::string>
InferGraph::_resolveOutputName(const std::string& name, const char* api) const {
	const auto& bindings = _outputBindingsView();
	size_t aliasMatches = 0;
	size_t portMatches = 0;
	std::pair<std::string, std::string> resolved;
	for (const auto& b : bindings) {
		if (!b.alias.empty() && b.alias == name) {
			if (++aliasMatches == 1)
				resolved = {b.nodeName, b.portName};
		}
	}
	if (aliasMatches == 1)
		return resolved;
	for (const auto& b : bindings) {
		if (b.portName == name) {
			if (++portMatches == 1)
				resolved = {b.nodeName, b.portName};
		}
	}
	if (portMatches == 1)
		return resolved;
	if (aliasMatches > 1 || portMatches > 1)
		throw GraphException(GraphException::ErrorType::FeedFailed, api,
							 "'" + name + "' is ambiguous across output bindings; "
								 "disambiguate with a unique alias or the 3-argument overload");
	throw GraphException(GraphException::ErrorType::NodeNotFound, api,
							 "no bound output port or alias named '" + name
								 + "' (call bindOutput(nodeName, portName) first)");
}

std::pair<std::string, std::string>
InferGraph::_resolveInputName(const std::string& name, const char* api) const {
	const auto& bindings = _inputBindingsView();
	size_t aliasMatches = 0;
	size_t portMatches = 0;
	std::pair<std::string, std::string> resolved;
	for (const auto& b : bindings) {
		if (!b.alias.empty() && b.alias == name) {
			if (++aliasMatches == 1)
				resolved = {b.nodeName, b.portName};
		}
	}
	if (aliasMatches == 1)
		return resolved;
	for (const auto& b : bindings) {
		if (b.portName == name) {
			if (++portMatches == 1)
				resolved = {b.nodeName, b.portName};
		}
	}
	if (portMatches == 1)
		return resolved;
	if (aliasMatches > 1 || portMatches > 1)
		throw GraphException(GraphException::ErrorType::FeedFailed, api,
							 "bound input '" + name + "' is ambiguous across nodes; "
								 "use feedInput(taskId, nodeName, ...) or a unique alias instead");
	throw GraphException(GraphException::ErrorType::NodeNotFound, api,
							 "no bound input port or alias named '" + name
								 + "' (call bindInput(nodeName, portName) first)");
}

// ════════════════════════════════════════════
// task 生命周期：状态 / 结构化等待 / 资源回收
// ════════════════════════════════════════════

TaskStatus InferGraph::taskStatus(const TaskId& taskId) const {
	auto st = _engine->status(taskId);
	if (st == TaskStatus::Succeeded) {
		// 正常终止但存在 Error 级诊断 → 归一化为 Failed（部分节点执行失败）
		for (const auto& e : _state->errors.taskErrors(taskId)) {
			if (e.level == DiagnosticLevel::Error)
				return TaskStatus::Failed;
		}
	}
	return st;
}

TaskResult InferGraph::waitForResult(const TaskId& taskId) {
	return waitForResult(taskId, std::chrono::milliseconds(0)); // 0 = 无限等待
}

TaskResult InferGraph::waitForResult(const TaskId& taskId, std::chrono::milliseconds timeout) {
	_engine->wait(taskId, timeout);   // timeout <= 0 视为无限等待
	TaskResult result;
	result.status = taskStatus(taskId);
	result.errors = _state->errors.taskErrors(taskId);
	return result;
}

void InferGraph::releaseTask(const TaskId& taskId) {
	_engine->releaseTask(taskId);     // 仅终止态可释放（活动任务拒绝）
	_state->output.clearTask(taskId);   // 释放结果 artifact
	_state->errors.clearTask(taskId);       // 释放诊断记录
}

// ════════════════════════════════════════════
// 图导出：将完整推理图包装为可嵌入父图的 Node
// ════════════════════════════════════════════

std::unique_ptr<Node> InferGraph::exportNode(const std::string& nodeName, uint32_t maxHops) {
	// ① 从 InputZone 推导输入 Schema
	Node::Schema inSchema;
	for (auto& b : _inputBindingsView()) {
		auto* n = _topology().node(b.nodeName);
		if (!n) continue;
		auto* port = n->schema().findInput(b.portName);
		if (port) inSchema.inputs.push_back(*port);
	}

	// ② 从 OutputZone 推导输出 Schema（跳过连接器）
	Node::Schema outSchema;
	for (auto& b : _outputBindingsView()) {
		auto* n = _topology().node(b.nodeName);
		if (!n || n->isConnector()) continue;
		auto* port = n->schema().findOutput(b.portName);
		if (port) outSchema.outputs.push_back(*port);
	}

	Node::Schema fullSchema;
	fullSchema.inputs = std::move(inSchema.inputs);
	fullSchema.outputs = std::move(outSchema.outputs);

	// ③ 构造 RunFn：捕获 this + maxHops
	//    调用者必须保证 this 在 Node 生命周期内有效
	auto runFn = [this, maxHops](Node::RunContext& ctx) -> Node::Result {
		const std::string tid = ctx.name();

		// 将 RunContext 的输入注入子图
		int fedCount = 0;
		for (auto& ib : _inputBindingsView()) {
			const auto& inVal = ctx.peek(ib.portName);
			if (!inVal.as<Tensor>()) {
				continue;
			}
			auto val = ctx.pop(ib.portName);
			feedInput(tid, ib.nodeName, ib.portName, std::move(val));
			++fedCount;
		}

		// 收集输出声明
		std::vector<OutputDeclaration> declarations;
		for (auto& ob : _outputBindingsView()) {
			declarations.push_back({ob.nodeName, ob.portName, 1});
		}

		// 通过回调在 _terminate 清理数据前捕获输出
		auto mtx = std::make_shared<std::mutex>();
		auto cv = std::make_shared<std::condition_variable>();
		auto done = std::make_shared<bool>(false);
		auto capturedOutputs = std::make_shared<std::unordered_map<std::string, Value>>();

		setTaskCompleteCallback([this, tid, mtx, cv, done, capturedOutputs](const TaskId& task) {
			if (task != tid) {
				return;
			}
			for (auto& ob : _outputBindingsView()) {
				if (!hasOutput(tid, ob.nodeName, ob.portName)) continue;
				(*capturedOutputs)[ob.portName] = takeOutput(tid, ob.nodeName, ob.portName);
			}
			{
				std::lock_guard lk(*mtx);
				*done = true;
			}
			cv->notify_one();
		});

		// 驱动子图（不启用内部超时，由父图控制）
		submit(tid, std::move(declarations), std::chrono::milliseconds(0), maxHops);

		// 等待回调完成
		{
			std::unique_lock lk(*mtx);
			cv->wait(lk, [&] { return *done; });
		}
		setTaskCompleteCallback(nullptr);

		// 检查是否有错误
		if (hasErrors()) {
			auto errors = taskErrors(tid);
			std::string msg = errors.empty() ? "unknown error" : errors[0].message;
			clearErrors();
			return ctx.failure(Node::Status::ExecutionFailed, msg);
		}

		// 收集输出到 RunContext
		for (auto& ob : _outputBindingsView()) {
			auto it = capturedOutputs->find(ob.portName);
			if (it != capturedOutputs->end()) {
				ctx.output(ob.portName, std::move(it->second));
			}
		}

		return ctx.success();
	};

	// 构造 GraphNode：注册状态委托（内部声明通路检测 → 资源中介语义）
	auto graphNode = std::make_unique<Node>(
		"GraphNode", nodeName, fullSchema,
		std::move(runFn),
		ThreadPoolAffinity::Compute);

	// blockedOverride：内部无通路满足输出声明时，子图节点向父级应答阻塞。
	// isReady 保持边界缓冲语义（父级数据齐即可进入执行）。
	graphNode->setBlockedOverride([this](const Node::TaskId& tid) {
		_ensureFrozen(); // 通路检测读图级签名（冻结快照）
		return !canSatisfyDeclarations(_state->graph->store(), _state->graph->signature(), tid);
	});

	return graphNode;
}

} // namespace DC
