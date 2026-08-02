#include "InferGraph.h"
#include "NodeException.h"
#include "GraphException.h"
#include "SignalProbe.h"

namespace DC {

// ════════════════════════════════════════════
// 构造
// ════════════════════════════════════════════

InferGraph::InferGraph()
	: _signalStore(std::make_shared<SignalStore>()), _engine(2) {}

InferGraph::InferGraph(CoroScheduler& scheduler, const PoolConfig& computeCfg,
					   const PoolConfig& operatorCfg, const PoolConfig& systemCfg)
	: _signalStore(std::make_shared<SignalStore>()),
	  _engine(scheduler, computeCfg, operatorCfg, systemCfg) {}

// ════════════════════════════════════════════
// 子图声明
// ════════════════════════════════════════════

void InferGraph::declareSubgraph(const std::string& name,
								  std::initializer_list<std::string> nodeNames) {
	// 1. 验证所有节点存在（affinity 可混合：组信号量跨池共享，全局互斥）
	for (const auto& nname : nodeNames) {
		auto* n = _store.node(nname);
		if (!n)
			throw GraphException(GraphException::ErrorType::NodeNotFound, "InferGraph::declareSubgraph",
								 "node '" + nname + "' not found");
	}

	// 2. 设置所有节点的 tag 为子图名
	for (const auto& nname : nodeNames)
		_store.node(nname)->setTag(name);

	// 3. 注册跨池分组限流（共享信号量，对三个线程池同时生效）
	_engine.registerGroupLimit(ThreadPoolAffinity::Compute, name, 1);
}

// ════════════════════════════════════════════
// 数据注入
// ════════════════════════════════════════════

void InferGraph::feedInput(const TaskId& taskId, const std::string& nodeName,
						   const std::string& portName, Value data) {
	auto* n = _store.node(nodeName);
	if (!n)
		throw GraphException(GraphException::ErrorType::NodeNotFound, "InferGraph::feedInput",
							 "node '" + nodeName + "' not found");
	try {
		n->setInput(taskId, portName, std::move(data));
	} catch (const NodeException& e) {
		_errors.recordError(taskId, nodeName, "InferGraph::feedInput",
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

// ════════════════════════════════════════════
// 结果获取
// ════════════════════════════════════════════

Value InferGraph::getOutput(const TaskId& taskId, const std::string& nodeName,
							const std::string& portName) {
	// 优先查 OutputZone（OutputZone 绑定端口的数据在 _propagateFrom 第二步已搬运至此）
	auto ozVal = _outputZone.take(taskId, nodeName, portName);
	if (ozVal)
		return std::move(*ozVal);

	auto* n = _store.node(nodeName);
	if (!n) {
		throw GraphException(GraphException::ErrorType::NodeNotFound, "InferGraph::getOutput",
							 "node '" + nodeName + "' not found");
	}
	return n->getOutput(taskId, portName);
}

Tensor InferGraph::getOutputTensor(const TaskId& taskId, const std::string& nodeName,
								   const std::string& portName) {
	// 优先查 OutputZone
	auto ozVal = _outputZone.take(taskId, nodeName, portName);
	if (ozVal) {
		auto* t = ozVal->as<Tensor>();
		if (t)
			return std::move(*t);
		throw GraphException(GraphException::ErrorType::Other, "InferGraph::getOutputTensor",
							 "OutputZone artifact for '" + nodeName + "." + portName
								 + "' is not a DC::Tensor");
	}

	auto* n = _store.node(nodeName);
	if (!n) {
		throw GraphException(GraphException::ErrorType::NodeNotFound, "InferGraph::getOutputTensor",
							 "node '" + nodeName + "' not found");
	}
	return n->getOutputTensor(taskId, portName);
}

bool InferGraph::hasOutput(const TaskId& taskId, const std::string& nodeName,
						   const std::string& portName) const {
	// 优先查 OutputZone
	if (_outputZone.hasOutput(taskId, nodeName, portName))
		return true;

	auto* n = _store.node(nodeName);
	if (!n)
		return false;
	return n->hasOutput(taskId, portName);
}

// ════════════════════════════════════════════
// 图导出：将完整推理图包装为可嵌入父图的 Node
// ════════════════════════════════════════════

std::unique_ptr<Node> InferGraph::exportNode(const std::string& nodeName, uint32_t maxHops) {
	// ① 从 InputZone 推导输入 Schema
	Node::Schema inSchema;
	for (auto& b : _store.inputBindings()) {
		auto* n = _store.node(b.nodeName);
		if (!n) continue;
		auto* port = n->schema().findInput(b.portName);
		if (port) inSchema.inputs.push_back(*port);
	}

	// ② 从 OutputZone 推导输出 Schema（跳过连接器）
	Node::Schema outSchema;
	for (auto& b : _outputZone.bindings()) {
		auto* n = _store.node(b.nodeName);
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
		for (auto& ib : _store.inputBindings()) {
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
		for (auto& ob : _outputZone.bindings()) {
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
			for (auto& ob : _outputZone.bindings()) {
				if (!hasOutput(tid, ob.nodeName, ob.portName)) continue;
				(*capturedOutputs)[ob.portName] = getOutput(tid, ob.nodeName, ob.portName);
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
		for (auto& ob : _outputZone.bindings()) {
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
		return !canSatisfyDeclarations(_store, _outputZone, tid);
	});

	return graphNode;
}

} // namespace DC
