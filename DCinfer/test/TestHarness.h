#pragma once

#include "InferGraph.h"
#include "GraphException.h"
#include "Tensor.hpp"
#include "SignalStore.h"

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DC {

/// @brief 多线程场景化测试夹具：封装 InferGraph + 同步等待机制
///
/// 使用方式：
///   1. 构建图：addNode / wire / connect / connectAll
///   2. 注入数据：feedInput
///   3. 声明输出：declareOutput（submit 前必须调用）
///   4. 异步提交：submit（内部设置 task 完成回调）
///   5. 等待完成：awaitCompletion（由 _terminate 中的回调触发）
///   6. 取结果：getOutputTensor（从回调捕获的缓存中读取）
///
/// 关键设计：task 完成回调在 _terminate 中、节点缓冲区 cleanup 前触发，
/// 确保回调中能安全读取输出。TestHarness 在回调中捕获输出数据到本地缓存，
/// 然后通知 awaitCompletion 返回。无需轮询。
class TestHarness {
public:
	using TaskId = InferGraph::TaskId;

	TestHarness() = default;

	// ── 图构建（透传）──

	Node* addNode(std::unique_ptr<Node> node) {
		return _graph.addNode(std::move(node));
	}

	bool connect(const std::string& srcNode, const std::string& srcPort, const std::string& dstNode,
				 const std::string& dstPort) {
		return _graph.connect(srcNode, srcPort, dstNode, dstPort);
	}

	size_t connectAll(const std::string& srcNode, const std::string& dstNode) {
		return _graph.connectAll(srcNode, dstNode);
	}

	Node* wire(const std::string& srcNode, const std::string& srcPort, const std::string& dstNode,
			   const std::string& dstPort) {
		return _graph.wire(srcNode, srcPort, dstNode, dstPort);
	}

	void bindOutput(const std::string& nodeName, const std::string& portName) {
		_graph.bindOutput(nodeName, portName);
	}

	// ── 数据注入 ──

	bool feedInput(const TaskId& taskId, const std::string& nodeName, const std::string& portName, Value data) {
		return _graph.feedInput(taskId, nodeName, portName, std::move(data));
	}

	bool feedInput(const TaskId& taskId, const std::string& nodeName, const std::string& portName, Tensor data) {
		return _graph.feedInput(taskId, nodeName, portName, std::move(data));
	}

	// ── 输出声明 ──

	void declareOutput(const TaskId& taskId, const std::string& nodeName,
					   const std::string& portName, size_t count = 1) {
		_graph.declareOutput(taskId, nodeName, portName, count);
		// 记录位置供回调捕获
		std::lock_guard lk(_declMutex);
		_declaredOutputs[taskId].emplace_back(nodeName, portName);
	}

	// ── 异步提交 ──

	/// @brief  提交 task。内部设置 task 完成回调：在 _terminate 触发时捕获输出并通知等待者。
	void submit(const TaskId& taskId, std::chrono::milliseconds timeout = std::chrono::milliseconds(0),
				uint32_t maxHops = InferGraph::kDefaultMaxHops) {
		// 注册 task 完成回调：在 _terminate（所有传播完成）时触发，捕获输出到本地缓存
		_graph.setTaskCompleteCallback([this](const TaskId& tid) {
			// 捕获所有声明输出到本地缓存
			{
				std::lock_guard lk(_declMutex);
				auto it = _declaredOutputs.find(tid);
				if (it != _declaredOutputs.end()) {
					auto& captured = _capturedOutputs[tid];
					for (auto& [nodeName, portName] : it->second) {
						std::string key = nodeName + ":" + portName;
						if (captured.contains(key))
							continue;

						// 优先从 OutputZone / Node 查找（_graph.hasOutput 会双端检查）
						if (!_graph.hasOutput(tid, nodeName, portName))
							continue;
						try {
							auto val = _graph.getOutput(tid, nodeName, portName);
							auto* t = val.as<Tensor>();
							if (t)
								captured[key] = std::move(*t);
						} catch (...) {
						}
					}
				}
			}
			// 注意：无需手动通知 CV，InferGraph::wait() 通过 _completionCv 同步
		});

		_graph.submit(taskId, timeout, maxHops);
	}

	// ── 同步等待 ──

	/// @brief  同步等待 task 完成（复用 InferGraph::wait）
	/// @return true 在超时前完成，false 超时
	bool awaitCompletion(const TaskId& taskId, std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
		return _graph.wait(taskId, timeout);
	}

	// ── 结果获取（从缓存读取）──

	Tensor getOutputTensor(const TaskId& taskId, const std::string& nodeName, const std::string& portName) {
		std::string key = nodeName + ":" + portName;
		auto taskIt = _capturedOutputs.find(taskId);
		if (taskIt == _capturedOutputs.end() || !taskIt->second.contains(key)) {
			throw GraphException(GraphException::ErrorType::Other, "TestHarness::getOutputTensor",
								"no captured output for task '" + taskId + "' at " + nodeName + "." + portName);
		}
		return std::move(taskIt->second.at(key));
	}

	bool hasOutput(const TaskId& taskId, const std::string& nodeName, const std::string& portName) const {
		std::string key = nodeName + ":" + portName;
		auto taskIt = _capturedOutputs.find(taskId);
		return taskIt != _capturedOutputs.end() && taskIt->second.contains(key);
	}

	// ── 查询 ──

	Node* node(const std::string& name) {
		return _graph.node(name);
	}

	const Node* node(const std::string& name) const {
		return _graph.node(name);
	}

	size_t nodeCount() const {
		return _graph.nodeCount();
	}

	size_t edgeCount() const {
		return _graph.edgeCount();
	}

	std::vector<TaskError> taskErrors(const TaskId& taskId) const {
		return _graph.taskErrors(taskId);
	}

	bool hasErrors() const {
		return _graph.hasErrors();
	}

	void clearErrors() {
		_graph.clearErrors();
	}

	/// @brief  获取底层 InferGraph 的只读引用（供序列化等场景遍历图结构）
	const InferGraph& graph() const {
		return _graph;
	}

	// ── 信号系统 ──

	void setSignal(const std::string& name, bool value) {
		_graph.setSignal(name, value);
	}

	void setSignal(const std::string& name, const TaskId& taskId, bool value) {
		_graph.setSignal(name, taskId, value);
	}

	bool getSignal(const std::string& name) const {
		return _graph.getSignal(name);
	}

	bool getSignal(const std::string& name, const TaskId& taskId) const {
		return _graph.getSignal(name, taskId);
	}

	std::shared_ptr<SignalStore> signalStore() {
		return _graph.signalStore();
	}

private:
	InferGraph _graph;

	// 声明输出位置记录（submit 时供回调使用）
	std::unordered_map<TaskId, std::vector<std::pair<std::string, std::string>>> _declaredOutputs;
	mutable std::mutex _declMutex;

	// 回调中捕获的输出缓存：taskId → {"nodeName:portName" → Tensor}
	std::unordered_map<TaskId, std::unordered_map<std::string, Tensor>> _capturedOutputs;


};

} // namespace DC
