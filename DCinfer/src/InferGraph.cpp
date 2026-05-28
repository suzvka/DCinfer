#include "InferGraph.h"
#include "Connector.h"
#include "NodeException.h"

#include <algorithm>
#include <iostream>

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

	// DAG 环检测：若从 dstNode 出发已能到达 srcNode，则添加此边会形成环
	// 允许环路（由输出声明保证收敛），仅打印警告
	if (_wouldCreateCycle(srcNode, dstNode)) {
		std::cerr << "InferGraph::connect: cycle detected when connecting '" << srcNode << ":" << srcPort << "' -> '" << dstNode << ":" << dstPort << "'. Ensure output declarations exist before submit()." << std::endl;
	}

	_edges.push_back({srcNode, srcPort, dstNode, dstPort});
	return true;
}

size_t InferGraph::connectAll(const std::string& srcNode, const std::string& dstNode) {
	auto* src = _findNode(srcNode);
	auto* dst = _findNode(dstNode);
	if (!src || !dst)
		return 0;

	// DAG 环检测
	if (_wouldCreateCycle(srcNode, dstNode)) {
		std::cerr << "InferGraph::connectAll: cycle detected between '" << srcNode << "' and '" << dstNode << "'. Ensure output declarations exist before submit()." << std::endl;
	}

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
// wire：自动插入导线连接器
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

	// DAG 环检测：src→wire→dst 等效于 src→dst
	if (_wouldCreateCycle(srcNode, dstNode)) {
		std::cerr << "InferGraph::wire: cycle detected between '" << srcNode << "' and '" << dstNode << "'. Ensure output declarations exist before submit()." << std::endl;
	}

	// 自动创建导线连接器
	auto wireName = "__wire_" + std::to_string(_nextWireId.fetch_add(1));
	auto wireNode = std::make_unique<Node>("Connector.Wire", wireName, Connector::wireSchema(), Connector::wireRunFn(),
										   nullptr, ThreadPoolAffinity::System);
	wireNode->setConnector(true);
	auto* wirePtr = addNode(std::move(wireNode));
	if (!wirePtr)
		return nullptr;

	// 上游 → 导线
	_edges.push_back({srcNode, srcPort, wireName, "in"});
	// 导线 → 下游
	_edges.push_back({wireName, "out", dstNode, dstPort});

	return wirePtr;
}

// ════════════════════════════════════════════
// 数据注入
// ════════════════════════════════════════════

bool InferGraph::feedInput(const TaskId& taskId, const std::string& nodeName, const std::string& portName, Value data) {
	auto* n = _findNode(nodeName);
	if (!n)
		return false;
	_activeTasks.insert(taskId);
	try {
		n->setInput(taskId, portName, std::move(data));
		return true;
	} catch (const NodeException&) {
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
// 同步执行驱动
// ════════════════════════════════════════════

void InferGraph::run() {
	// 反复扫描全图：对每个活跃 task，检查每个节点是否就绪，
	// 就绪则执行并级联传播输出到下游，直到本轮无新执行发生
	bool anyExecuted = true;
	while (anyExecuted) {
		anyExecuted = false;

		for (const auto& tid : _activeTasks) {
			for (const auto& [nodeName, nodePtr] : _nodes) {
				if (!nodePtr->isReady(tid))
					continue;
				try {
					nodePtr->tryExecute(tid);
				} catch (const NodeException&) {
					continue;
				}

				anyExecuted = true;

				// 传播输出到所有下游
				for (const auto& edge : _edges) {
					if (edge.srcNode != nodeName)
						continue;
					if (!nodePtr->hasOutput(tid, edge.srcPort))
						continue;

					Value data = nodePtr->getOutput(tid, edge.srcPort);

					auto* dst = _findNode(edge.dstNode);
					if (!dst)
						continue;

					dst->setInput(tid, edge.dstPort, std::move(data));
				}
			}
		}
	}
}

// ════════════════════════════════════════════
// 异步提交：协程驱动的数据传播
// ════════════════════════════════════════════

void InferGraph::submit(const TaskId& taskId) {
	// 校验：若图中有环路但未声明输出，则拒绝
	{
		std::lock_guard lk(_declarationMutex);
		if (!_outputDeclarations.contains(taskId) && _hasCycle()) {
			std::cerr << "InferGraph::submit: cycle detected but no output declarations for task '" << taskId
					  << "'. Call declareOutput() before submit()." << std::endl;
			return;
		}
	}

	// 记录活跃任务，供同步 run() 扫描
	_activeTasks.insert(taskId);

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
				} catch (const NodeException&) {}
			});
			break;
		case ThreadPoolAffinity::Operator:
			_operatorPool.submit(nodePtr->tag(), [nodePtr = nodePtr.get(), taskId] {
				try {
					nodePtr->tryExecute(taskId);
				} catch (const NodeException&) {}
			});
			break;
		case ThreadPoolAffinity::System:
			_systemPool.submit(nodePtr->tag(), [nodePtr = nodePtr.get(), taskId] {
				try {
					nodePtr->tryExecute(taskId);
				} catch (const NodeException&) {}
			});
			break;
		}

		// 创建协程：等待节点完成后自动传播数据到下游
		_scheduler->spawn([this, nodeName, taskId]() -> Task<void> { co_await _propagateFrom(nodeName, taskId); });
	}
}

Task<void> InferGraph::_propagateFrom(const std::string& nodeName, const TaskId& taskId) {
	// [检查点 1] 入口：若 task 已终止，直接返回
	if (_isTerminated(taskId))
		co_return;

	auto* src = _findNode(nodeName);
	if (!src)
		co_return;

	// co_await 挂起当前协程，等待该节点执行完成
	auto result = co_await src->whenComplete(taskId);
	if (!result.ok()) {
		co_return; // 失败则不传播
	}

	// 节点完成 → 数据冒泡：消费输出 → 写入下游 → 提交到对应线程池
	for (const auto& edge : _edges) {
		if (edge.srcNode != nodeName)
			continue;

		if (!src->hasOutput(taskId, edge.srcPort))
			continue;

		// [检查点 2] 先累积计数，再消费数据：
		// 若所有声明条件已满足，立即终止（不消费无用数据，避免传播到已清理的下游）
		if (_accumulateAndCheck(edge.srcNode, edge.srcPort, taskId)) {
			_terminate(taskId);
			co_return;
		}

		Value data = src->getOutput(taskId, edge.srcPort);

		// [检查点 3] 写入下游前再确认一次未被终止
		if (_isTerminated(taskId))
			co_return;

		auto* dst = _findNode(edge.dstNode);
		if (!dst)
			continue;

		try {
			dst->setInput(taskId, edge.dstPort, std::move(data));
		} catch (const NodeException&) {
			continue;
		}

		// 下游就绪 → 按 affinity 提交到相应线程池
		if (dst->isReady(taskId)) {
			switch (dst->affinity()) {
			case ThreadPoolAffinity::Compute:
				co_await _computePool.submitAsync(dst->tag(), [dst, taskId] {
					try {
						dst->tryExecute(taskId);
					} catch (const NodeException&) {}
				});
				break;
			case ThreadPoolAffinity::Operator:
				co_await _operatorPool.submitAsync(dst->tag(), [dst, taskId] {
					try {
						dst->tryExecute(taskId);
					} catch (const NodeException&) {}
				});
				break;
			case ThreadPoolAffinity::System:
				co_await _systemPool.submitAsync(dst->tag(), [dst, taskId] {
					try {
						dst->tryExecute(taskId);
					} catch (const NodeException&) {}
				});
				break;
			}

			// [检查点 1b] 提交完确认未被终止再 spawn 下游
			if (_isTerminated(taskId))
				co_return;

			// 创建下游传播协程（非阻塞：fire-and-forget spawn）
			_scheduler->spawn(
				[this, dstNode = edge.dstNode, taskId]() -> Task<void> { co_await _propagateFrom(dstNode, taskId); });
		}
	}
}

// ════════════════════════════════════════════
// 结果获取
// ════════════════════════════════════════════

Value InferGraph::getOutput(const TaskId& taskId, const std::string& nodeName, const std::string& portName) {
	auto* n = _findNode(nodeName);
	if (!n) {
		throw std::out_of_range("InferGraph::getOutput: node '" + nodeName + "' not found");
	}
	return n->getOutput(taskId, portName);
}

Tensor InferGraph::getOutputTensor(const TaskId& taskId, const std::string& nodeName, const std::string& portName) {
	auto* n = _findNode(nodeName);
	if (!n) {
		throw std::out_of_range("InferGraph::getOutputTensor: node '" + nodeName + "' not found");
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

// ════════════════════════════════════════════
// _wouldCreateCycle：DAG 环检测
// ════════════════════════════════════════════

bool InferGraph::_wouldCreateCycle(const std::string& srcNode, const std::string& dstNode) const {
	// 自环
	if (srcNode == dstNode)
		return true;

	// 从 dstNode 出发 DFS，检查是否可达 srcNode
	std::unordered_set<std::string> visited;
	std::function<bool(const std::string&)> dfs = [&](const std::string& current) -> bool {
		if (current == srcNode)
			return true;
		if (!visited.insert(current).second)
			return false;
		for (const auto& edge : _edges) {
			if (edge.srcNode == current) {
				if (dfs(edge.dstNode))
					return true;
			}
		}
		return false;
	};
	return dfs(dstNode);
}

// ════════════════════════════════════════════
// _hasCycle：整图环检测
// ════════════════════════════════════════════

bool InferGraph::_hasCycle() const {
	// DFS 三色标记法：White=未访问, Gray=访问中(栈上), Black=已完成
	// 若从 Gray 节点出发可达另一个 Gray 节点，则存在环
	enum class Color : uint8_t { White, Gray, Black };
	std::unordered_map<std::string, Color> color;
	color.reserve(_nodes.size());
	for (const auto& [name, _] : _nodes) {
		color[name] = Color::White;
	}

	// 预构建邻接表
	std::unordered_map<std::string, std::vector<std::string>> adj;
	for (const auto& edge : _edges) {
		adj[edge.srcNode].push_back(edge.dstNode);
	}

	std::function<bool(const std::string&)> dfs = [&](const std::string& node) -> bool {
		color[node] = Color::Gray;
		auto it = adj.find(node);
		if (it != adj.end()) {
			for (const auto& next : it->second) {
				auto c = color[next];
				if (c == Color::Gray)
					return true; // 回边：发现环
				if (c == Color::White && dfs(next))
					return true;
			}
		}
		color[node] = Color::Black;
		return false;
	};

	for (const auto& [name, _] : _nodes) {
		if (color[name] == Color::White && dfs(name))
			return true;
	}
	return false;
}

// ════════════════════════════════════════════
// 终止辅助
// ════════════════════════════════════════════

bool InferGraph::_accumulateAndCheck(const std::string& nodeName, const std::string& portName,
									 const TaskId& taskId) {
	std::lock_guard lk(_declarationMutex);
	auto declIt = _outputDeclarations.find(taskId);
	if (declIt == _outputDeclarations.end())
		return false; // 未声明 → 永不终止

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

	// 遍历所有节点，清理此 taskId 的 IO 缓冲区并通知等待者
	for (auto& [name, nodePtr] : _nodes) {
		if (nodePtr->hasTask(taskId))
			nodePtr->terminateTask(taskId);
	}
}

} // namespace DC
