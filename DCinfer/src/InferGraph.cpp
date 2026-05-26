#include "InferGraph.h"
#include "Connector.h"

#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace DC {

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
	if (!node || node->name().empty()) return nullptr;

	const auto& name = node->name();
	if (_nodes.contains(name)) return nullptr;

	auto* raw = node.get();
	_nodes.emplace(name, std::move(node));
	return raw;
}

bool InferGraph::connect(
	const std::string& srcNode, const std::string& srcPort,
	const std::string& dstNode, const std::string& dstPort)
{
	auto* src = _findNode(srcNode);
	auto* dst = _findNode(dstNode);
	if (!src || !dst) return false;

	// 验证端口存在
	if (!src->schema().findOutput(srcPort)) return false;
	if (!dst->schema().findInput(dstPort))   return false;

	_edges.push_back({srcNode, srcPort, dstNode, dstPort});
	return true;
}

size_t InferGraph::connectAll(
	const std::string& srcNode,
	const std::string& dstNode)
{
	auto* src = _findNode(srcNode);
	auto* dst = _findNode(dstNode);
	if (!src || !dst) return 0;

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
// 数据注入
// ════════════════════════════════════════════

bool InferGraph::feedInput(
	const TaskId& taskId,
	const std::string& nodeName,
	const std::string& portName,
	Value data)
{
	auto* n = _findNode(nodeName);
	if (!n) return false;
	return n->setInput(taskId, portName, std::move(data));
}

bool InferGraph::feedInput(
	const TaskId& taskId,
	const std::string& nodeName,
	const std::string& portName,
	Tensor data)
{
	auto* p = new Tensor(std::move(data));
	return feedInput(taskId, nodeName, portName,
		Value(p, [](Tensor* ptr) { delete ptr; }));
}


// ════════════════════════════════════════════
// 执行驱动
// ════════════════════════════════════════════

void InferGraph::schedule(const std::string& nodeName, const TaskId& taskId) {
	_readyQueue.emplace(nodeName, taskId);
}

void InferGraph::run() {
	// 就绪队列为空时，扫描全图查找已就绪节点
	if (_readyQueue.empty()) {
		for (const auto& [nodeName, nodePtr] : _nodes) {
			if (nodePtr->taskCount() == 0) continue;
			// 对所有活跃 task 检查就绪
			// 注：Node 不暴露 taskId 列表，此扫描仅在初始入队时作为最佳努力
			// 正常流程中由 _dispatchOutputs 自动将下游加入队列
		}
	}

	while (!_readyQueue.empty()) {
		auto [nodeName, taskId] = _readyQueue.front();
		_readyQueue.pop();

		auto* n = _findNode(nodeName);
		if (!n) continue;

		// 尝试执行（获取重入锁 → 加载输入 → RunFn → 收集输出）
		if (!n->tryExecute(taskId)) {
			// 未能执行（未就绪或正忙），放回队列尾部重试
			_readyQueue.emplace(nodeName, taskId);
			// 避免死循环：如果队列只有一个元素且它反复失败，跳出
			// 注：单线程下此策略安全——外部需在 feedInput 后调用 run()
			if (_readyQueue.size() == 1) {
				break;  // 唯一的任务无法执行，等待外部注入更多数据
			}
			continue;
		}

		// 执行成功 → 分发输出到下游
		_dispatchOutputs(nodeName, taskId);
	}
}

void InferGraph::_dispatchOutputs(const std::string& nodeName, const TaskId& taskId) {
	auto* src = _findNode(nodeName);
	if (!src) return;

	for (const auto& edge : _edges) {
		if (edge.srcNode != nodeName) continue;

		// 检查上游是否还有该端口的输出
		if (!src->hasOutput(taskId, edge.srcPort)) continue;

		// 消费式取出
		Value data = src->getOutput(taskId, edge.srcPort);

		auto* dst = _findNode(edge.dstNode);
		if (!dst) continue;

		// 写入下游输入缓冲
		dst->setInput(taskId, edge.dstPort, std::move(data));

		// 若下游就绪，加入调度队列
		if (dst->isReady(taskId)) {
			_readyQueue.emplace(edge.dstNode, taskId);
		}
	}
}

void InferGraph::_scanReady(const TaskId& taskId) {
	// 扫描所有节点，将指定 taskId 就绪的节点入队
	for (const auto& [nodeName, nodePtr] : _nodes) {
		if (nodePtr->isReady(taskId)) {
			_readyQueue.emplace(nodeName, taskId);
		}
	}
}


// ════════════════════════════════════════════
// 结果获取
// ════════════════════════════════════════════

Value InferGraph::getOutput(
	const TaskId& taskId,
	const std::string& nodeName,
	const std::string& portName)
{
	auto* n = _findNode(nodeName);
	if (!n) {
		throw std::out_of_range("InferGraph::getOutput: node '" + nodeName + "' not found");
	}
	return n->getOutput(taskId, portName);
}

Tensor InferGraph::getOutputTensor(
	const TaskId& taskId,
	const std::string& nodeName,
	const std::string& portName)
{
	auto* n = _findNode(nodeName);
	if (!n) {
		throw std::out_of_range("InferGraph::getOutputTensor: node '" + nodeName + "' not found");
	}
	return n->getOutputTensor(taskId, portName);
}

bool InferGraph::hasOutput(
	const TaskId& taskId,
	const std::string& nodeName,
	const std::string& portName) const
{
	auto* n = _findNode(nodeName);
	if (!n) return false;
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

} // namespace DC
