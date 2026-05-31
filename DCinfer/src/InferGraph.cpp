#include "InferGraph.h"
#include "Connector.h"
#include "NodeException.h"

#include <algorithm>

namespace DC {

// ════════════════════════════════════════════
// 构造
// ════════════════════════════════════════════

InferGraph::InferGraph()
	: _ownedScheduler(std::make_unique<CoroScheduler>(2)), _scheduler(_ownedScheduler.get()), _computePool({}),
	  _operatorPool({}), _systemPool({}) {}

InferGraph::InferGraph(CoroScheduler& scheduler, const PoolConfig& computeCfg, const PoolConfig& operatorCfg,
					   const PoolConfig& systemCfg)
	: _scheduler(&scheduler), _computePool(computeCfg), _operatorPool(operatorCfg), _systemPool(systemCfg) {}

// ════════════════════════════════════════════
// _findNode
// ════════════════════════════════════════════

Node* InferGraph::_findNode(const std::string& name) {
	auto it = _nodes.find(name);
	return it != _nodes.end() ? it->second.get() : nullptr;
}

const Node* InferGraph::_findNode(const std::string& name) const {
	auto it = _nodes.find(name);
	return it != _nodes.end() ? it->second.get() : nullptr;
}

// ════════════════════════════════════════════
// 图构建
// ════════════════════════════════════════════

Node* InferGraph::addNode(std::unique_ptr<Node> node) {
	if (!node || node->name().empty())
		return nullptr;

	const auto& name = node->name();
	if (_nodes.contains(name))
		return nullptr;

	auto* raw = node.get();
	_nodes.emplace(name, std::move(node));
	return raw;
}

bool InferGraph::connect(const std::string& srcNode, const std::string& srcPort, const std::string& dstNode,
						 const std::string& dstPort) {
	auto* src = _findNode(srcNode);
	auto* dst = _findNode(dstNode);
	if (!src || !dst)
		return false;

	// 约束：至少有一端是连接器（两个业务节点禁止直连，必须通过 Connector 中转）
	if (!src->isConnector() && !dst->isConnector())
		return false;

	// 验证端口存在
	if (!src->schema().findOutput(srcPort))
		return false;
	if (!dst->schema().findInput(dstPort))
		return false;

	_edges.push_back({srcNode, srcPort, dstNode, dstPort});
	return true;
}

size_t InferGraph::connectAll(const std::string& srcNode, const std::string& dstNode) {
	auto* src = _findNode(srcNode);
	auto* dst = _findNode(dstNode);
	if (!src || !dst)
		return 0;

	size_t matched = 0;
	for (const auto& outPort : src->schema().outputs) {
		if (dst->schema().findInput(outPort.name)) {
			_edges.push_back({srcNode, outPort.name, dstNode, outPort.name});
			++matched;
		}
	}
	return matched;
}

void InferGraph::bindOutput(const std::string& nodeName, const std::string& portName) {
	_outputBindings.push_back({nodeName, portName});
}

// ════════════════════════════════════════════
// wire：自动插入广播连接器（1→1，零拷贝 move 直通）
// ════════════════════════════════════════════

Node* InferGraph::wire(const std::string& srcNode, const std::string& srcPort, const std::string& dstNode,
					   const std::string& dstPort) {
	auto* src = _findNode(srcNode);
	auto* dst = _findNode(dstNode);
	if (!src || !dst)
		return nullptr;
	if (!src->schema().findOutput(srcPort))
		return nullptr;
	if (!dst->schema().findInput(dstPort))
		return nullptr;

	// 自动创建广播连接器（1 下游 → 零拷贝 move 直通，等效导线）
	auto wireName = "__wire_" + std::to_string(_nextWireId.fetch_add(1));
	auto wireNode = std::make_unique<Node>("Connector.Broadcast", wireName, Connector::broadcastSchema(1),
										   Connector::broadcastRunFn(), nullptr, ThreadPoolAffinity::System);
	wireNode->setConnector(true);
	auto* wirePtr = addNode(std::move(wireNode));
	if (!wirePtr)
		return nullptr;

	// 上游 → 广播
	_edges.push_back({srcNode, srcPort, wireName, "in"});
	// 广播(out_0) → 下游
	_edges.push_back({wireName, "out_0", dstNode, dstPort});

	return wirePtr;
}

// ════════════════════════════════════════════
// 数据注入
// ════════════════════════════════════════════

bool InferGraph::feedInput(const TaskId& taskId, const std::string& nodeName, const std::string& portName, Value data) {
	auto* n = _findNode(nodeName);
	if (!n)
		return false;
	try {
		n->setInput(taskId, portName, std::move(data));
		return true;
	} catch (const NodeException& e) {
		_recordError(taskId, nodeName, "InferGraph::feedInput",
					 "NodeException in setInput for port '" + portName + "': " + std::string(e.what()));
		return false;
	}
}

bool InferGraph::feedInput(const TaskId& taskId, const std::string& nodeName, const std::string& portName,
						   Tensor data) {
	return feedInput(taskId, nodeName, portName, Value(std::make_unique<Tensor>(std::move(data))));
}

// ════════════════════════════════════════════
// 输出声明
// ════════════════════════════════════════════

void InferGraph::declareOutput(const TaskId& taskId, std::vector<OutputDeclaration> declarations) {
	std::lock_guard lk(_declarationMutex);
	auto& existing = _outputDeclarations[taskId];
	// 追加式写入（与单条重载语义一致）
	existing.reserve(existing.size() + declarations.size());
	for (auto& decl : declarations) {
		existing.push_back(std::move(decl));
	}
}

void InferGraph::declareOutput(const TaskId& taskId, const std::string& nodeName, const std::string& portName,
							   size_t count) {
	std::lock_guard lk(_declarationMutex);
	_outputDeclarations[taskId].push_back({nodeName, portName, count});
}

// ════════════════════════════════════════════
// 异步提交：协程驱动的数据传播
// ════════════════════════════════════════════

void InferGraph::submit(const TaskId& taskId, std::chrono::milliseconds timeout) {
	// 校验：必须已声明输出
	{
		std::lock_guard lk(_declarationMutex);
		if (!_outputDeclarations.contains(taskId)) {
			throw GraphException(GraphException::ErrorType::NoDeclaration, "InferGraph::submit",
								 "no output declarations for task '" + taskId
									 + "'. Call declareOutput() before submit().");
		}
	}

	// 创建任务门控：协程链与看门狗共享，最后一个持有者析构时触发耗尽检测
	auto gate = std::make_shared<TaskGate>();
	gate->graph = this;
	gate->taskId = taskId;

	// 超时看门狗（独立线程）
	if (timeout.count() > 0) {
		auto deadline = std::chrono::steady_clock::now() + timeout;
		std::thread([this, taskId, timeout, deadline, gate]() {
			std::this_thread::sleep_until(deadline);
			if (!gate->terminated.exchange(true, std::memory_order_acq_rel)) {
				_recordError(taskId, "<watchdog>", "InferGraph::submit",
							 "task timed out (" + std::to_string(timeout.count()) + "ms) without meeting output declarations");
				_terminate(taskId);
			}
		}).detach();
	}

	// 扫描全图，对所有已就绪的节点创建传播协程
	for (const auto& [nodeName, nodePtr] : _nodes) {
		if (!nodePtr->isReady(taskId))
			continue;

		// 根据 affinity 将入口节点提交到对应线程池
		switch (nodePtr->affinity()) {
		case ThreadPoolAffinity::Compute:
			_computePool.submit(nodePtr->tag(), [nodePtr = nodePtr.get(), taskId] {
				try {
					nodePtr->tryExecute(taskId);
				} catch (const NodeException& e) {
					// 线程池中无法直接调用 _recordError，错误通过协程链中 whenComplete 的 result 传播
				}
			});
			break;
		case ThreadPoolAffinity::Operator:
			_operatorPool.submit(nodePtr->tag(), [nodePtr = nodePtr.get(), taskId] {
				try {
					nodePtr->tryExecute(taskId);
				} catch (const NodeException& e) {
				}
			});
			break;
		case ThreadPoolAffinity::System:
			_systemPool.submit(nodePtr->tag(), [nodePtr = nodePtr.get(), taskId] {
				try {
					nodePtr->tryExecute(taskId);
				} catch (const NodeException& e) {
				}
			});
			break;
		}

		// 创建协程：等待节点完成后自动传播数据到下游
		_scheduler->spawnTask(_propagateFrom(nodeName, taskId, gate));
	}
}

Task<void> InferGraph::_propagateFrom(std::string nodeName, TaskId taskId,
									  std::shared_ptr<TaskGate> gate) {
	// [检查点 1] 入口：若 task 已终止，直接返回
	if (_isTerminated(taskId)) {
		co_return;
	}

	auto* src = _findNode(nodeName);
	if (!src)
		co_return;

	// co_await 挂起当前协程，等待该节点执行完成
	auto result = co_await src->whenComplete(taskId);
	if (!result.ok()) {
		_recordError(taskId, nodeName, "InferGraph::_propagateFrom",
					 "Node execution failed: " + result.message);
		co_return; // 失败则不传播
	}

	// [检查点 2] 节点完成 → 按输出端口累积计数（不依赖出边，终端节点也能触发）：
	// 若所有声明条件已满足，立即终止（不消费无用数据，避免传播到已清理的下游）
	for (const auto& outPort : src->schema().outputs) {
		if (!src->hasOutput(taskId, outPort.name))
			continue;
		if (_accumulateAndCheck(nodeName, outPort.name, taskId)) {
			gate->terminated.store(true, std::memory_order_release);
			_terminate(taskId);
			co_return;
		}
	}

	// 数据冒泡：消费输出 → 写入下游 → 提交到对应线程池
	for (const auto& edge : _edges) {
		if (edge.srcNode != nodeName)
			continue;

		if (!src->hasOutput(taskId, edge.srcPort))
			continue;

		Value data = src->getOutput(taskId, edge.srcPort);

		// [检查点 3] 写入下游前再确认一次未被终止
		if (_isTerminated(taskId))
			co_return;

		auto* dst = _findNode(edge.dstNode);
		if (!dst)
			continue;

		try {
			dst->setInput(taskId, edge.dstPort, std::move(data));
		} catch (const NodeException& e) {
			_recordError(taskId, edge.dstNode, "InferGraph::_propagateFrom",
						 "NodeException in setInput for port '" + edge.dstPort + "': " + std::string(e.what()));
			continue;
		}

		// 下游就绪 → 按 affinity 提交到相应线程池
		if (dst->isReady(taskId)) {
			switch (dst->affinity()) {
			case ThreadPoolAffinity::Compute:
				co_await _computePool.submitAsync(dst->tag(), [this, dst, taskId, dstNode = edge.dstNode] {
					try {
						dst->tryExecute(taskId);
					} catch (const NodeException& e) {
						_recordError(taskId, dstNode, "InferGraph::_propagateFrom<Compute>",
									 "NodeException in tryExecute: " + std::string(e.what()));
					}
				});
				break;
			case ThreadPoolAffinity::Operator:
				co_await _operatorPool.submitAsync(dst->tag(), [this, dst, taskId, dstNode = edge.dstNode] {
					try {
						dst->tryExecute(taskId);
					} catch (const NodeException& e) {
						_recordError(taskId, dstNode, "InferGraph::_propagateFrom<Operator>",
									 "NodeException in tryExecute: " + std::string(e.what()));
					}
				});
				break;
			case ThreadPoolAffinity::System:
				co_await _systemPool.submitAsync(dst->tag(), [this, dst, taskId, dstNode = edge.dstNode] {
					try {
						dst->tryExecute(taskId);
					} catch (const NodeException& e) {
						_recordError(taskId, dstNode, "InferGraph::_propagateFrom<System>",
									 "NodeException in tryExecute: " + std::string(e.what()));
					}
				});
				break;
			}

			// [检查点 1b] 提交完确认未被终止再 spawn 下游
			if (_isTerminated(taskId))
				co_return;

			// 创建下游传播协程（非阻塞：fire-and-forget spawn）
			_scheduler->spawnTask(_propagateFrom(edge.dstNode, taskId, gate));
		}
	}
}

// ════════════════════════════════════════════
// 结果获取
// ════════════════════════════════════════════

Value InferGraph::getOutput(const TaskId& taskId, const std::string& nodeName, const std::string& portName) {
	auto* n = _findNode(nodeName);
	if (!n) {
		throw GraphException(GraphException::ErrorType::NodeNotFound, "InferGraph::getOutput",
							 "node '" + nodeName + "' not found");
	}
	return n->getOutput(taskId, portName);
}

Tensor InferGraph::getOutputTensor(const TaskId& taskId, const std::string& nodeName, const std::string& portName) {
	auto* n = _findNode(nodeName);
	if (!n) {
		throw GraphException(GraphException::ErrorType::NodeNotFound, "InferGraph::getOutputTensor",
							 "node '" + nodeName + "' not found");
	}
	return n->getOutputTensor(taskId, portName);
}

bool InferGraph::hasOutput(const TaskId& taskId, const std::string& nodeName, const std::string& portName) const {
	auto* n = _findNode(nodeName);
	if (!n)
		return false;
	return n->hasOutput(taskId, portName);
}

// ════════════════════════════════════════════
// 查询
// ════════════════════════════════════════════

Node* InferGraph::node(const std::string& name) {
	return _findNode(name);
}

const Node* InferGraph::node(const std::string& name) const {
	return _findNode(name);
}

std::vector<std::string> InferGraph::nodeNames() const {
	std::vector<std::string> names;
	names.reserve(_nodes.size());
	for (const auto& [name, nodePtr] : _nodes) {
		names.push_back(name);
	}
	return names;
}

// ════════════════════════════════════════════
// 终止辅助
// ════════════════════════════════════════════

bool InferGraph::_accumulateAndCheck(const std::string& nodeName, const std::string& portName,
									 const TaskId& taskId) {
	std::lock_guard lk(_declarationMutex);
	auto declIt = _outputDeclarations.find(taskId);
	if (declIt == _outputDeclarations.end())
		return false; // 未声明（在 submit 中已被拦截，此处防御）

	// 递增累加器
	std::string key = nodeName + ":" + portName;
	++_accumulatedCounts[taskId][key];

	// 检查所有声明是否均已满足
	for (const auto& decl : declIt->second) {
		std::string dkey = decl.nodeName + ":" + decl.portName;
		size_t current = _accumulatedCounts[taskId][dkey];
		if (current < decl.count)
			return false;
	}
	return true;
}

bool InferGraph::_isTerminated(const TaskId& taskId) const {
	std::lock_guard lk(_terminationMutex);
	return _terminatedTasks.contains(taskId);
}

void InferGraph::_terminate(const TaskId& taskId) {
	{
		std::lock_guard lk(_terminationMutex);
		// 防止重复终止
		if (!_terminatedTasks.insert(taskId).second)
			return;
	}

	// 清理输出声明和累加器（同一 taskId 可安全重新提交）
	{
		std::lock_guard lk(_declarationMutex);
		_outputDeclarations.erase(taskId);
		_accumulatedCounts.erase(taskId);
	}

	// 通知 task 完成回调（在所有节点 cleanup 之前触发，保证调用方能安全读取输出）
	if (_taskCompleteCb) {
		_taskCompleteCb(taskId);
	}

	// 遍历所有节点，清理此 taskId 的 IO 缓冲区并通知等待者
	for (auto& [name, nodePtr] : _nodes) {
		if (nodePtr->hasTask(taskId))
			nodePtr->terminateTask(taskId);
	}
}

// ════════════════════════════════════════════
// 耗尽检测：TaskGate 析构或超时触发
// ════════════════════════════════════════════

void InferGraph::_onExhausted(const TaskId& taskId) {
	// 已终止则跳过
	if (_isTerminated(taskId)) {
		return;
	}

	// 检查声明是否已满足
	bool allMet = false;
	{
		std::lock_guard lk(_declarationMutex);
		auto declIt = _outputDeclarations.find(taskId);
		if (declIt != _outputDeclarations.end()) {
			allMet = true;
			for (const auto& decl : declIt->second) {
				std::string key = decl.nodeName + ":" + decl.portName;
				auto& counts = _accumulatedCounts[taskId];
				auto countIt = counts.find(key);
				size_t current = (countIt != counts.end()) ? countIt->second : 0;
				if (current < decl.count) {
					allMet = false;
					break;
				}
			}
		}
	}

	if (allMet) {
		// 声明已满足 → 正常终止（守护路径，主路径在 _accumulateAndCheck 中处理）
		_terminate(taskId);
		return;
	}

	// 声明未满足 → 诊断原因
	bool hasStuckData = false;
	for (const auto& [name, nodePtr] : _nodes) {
		if (nodePtr->hasTask(taskId)) {
			hasStuckData = true;
			break;
		}
	}

	if (hasStuckData) {
		_recordError(taskId, "<graph>", "InferGraph::_onExhausted",
					 "graph stalled: data remains in node buffers but no active propagation coroutines");
	} else {
		_recordError(taskId, "<graph>", "InferGraph::_onExhausted",
					 "graph exhausted without producing all declared outputs");
	}
	_terminate(taskId);
}

// ════════════════════════════════════════════
// 错误记录
// ════════════════════════════════════════════

void InferGraph::_recordError(const TaskId& taskId, std::string nodeName, std::string source,
							  std::string message) {
	std::lock_guard lk(_errorMutex);
	_taskErrors[taskId].push_back({std::move(nodeName), std::move(source), std::move(message)});
}

std::vector<TaskError> InferGraph::taskErrors(const TaskId& taskId) const {
	std::lock_guard lk(_errorMutex);
	auto it = _taskErrors.find(taskId);
	return it != _taskErrors.end() ? it->second : std::vector<TaskError>{};
}

void InferGraph::clearErrors() {
	std::lock_guard lk(_errorMutex);
	_taskErrors.clear();
}

bool InferGraph::hasErrors() const {
	std::lock_guard lk(_errorMutex);
	return !_taskErrors.empty();
}

} // namespace DC
