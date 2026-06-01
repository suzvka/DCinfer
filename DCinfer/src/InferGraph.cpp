#include "InferGraph.h"
#include "NodeException.h"
#include "GraphException.h"

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
// 数据注入
// ════════════════════════════════════════════

bool InferGraph::feedInput(const TaskId& taskId, const std::string& nodeName,
						   const std::string& portName, Value data) {
	auto* n = _store.node(nodeName);
	if (!n)
		return false;
	try {
		n->setInput(taskId, portName, std::move(data));
		return true;
	} catch (const NodeException& e) {
		_errors.recordError(taskId, nodeName, "InferGraph::feedInput",
							"NodeException in setInput for port '" + portName
								+ "': " + std::string(e.what()));
		return false;
	}
}

bool InferGraph::feedInput(const TaskId& taskId, const std::string& nodeName,
						   const std::string& portName, Tensor data) {
	return feedInput(taskId, nodeName, portName, Value(std::make_unique<Tensor>(std::move(data))));
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

		// 声明输出
		for (auto& ob : _outputZone.bindings()) {
			declareOutput(tid, ob.nodeName, ob.portName, 1);
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
		submit(tid, std::chrono::milliseconds(0), maxHops);

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

	return std::make_unique<Node>(
		"GraphNode", nodeName, fullSchema,
		std::move(runFn), nullptr,
		ThreadPoolAffinity::Compute);
}

} // namespace DC
