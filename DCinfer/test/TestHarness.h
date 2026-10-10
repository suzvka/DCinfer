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
/// task 完成回调在 _terminate 中、节点缓冲区 cleanup 前触发并捕获输出，故无需轮询。
class TestHarness {
public:
	using TaskId = InferGraph::TaskId;

	TestHarness() = default;

	Node& addNode(std::unique_ptr<Node> node) {
		return _graph.addNode(std::move(node));
	}

	Node& connect(const std::string& srcNode, const std::string& srcPort, const std::string& dstNode,
				  const std::string& dstPort) {
		return _graph.connect(srcNode, srcPort, dstNode, dstPort);
	}

	void bindOutput(const std::string& alias, const std::string& nodeName, const std::string& portName) {
		_graph.bindOutput(alias, nodeName, portName);
	}

	void feedInput(const TaskId& taskId, const std::string& nodeName, const std::string& portName, Value data) {
		_graph.feedInput(taskId, nodeName, portName, std::move(data));
	}

	void feedInput(const TaskId& taskId, const std::string& nodeName, const std::string& portName, Tensor data) {
		_graph.feedInput(taskId, nodeName, portName, std::move(data));
	}

	void submit(const TaskId& taskId, const std::string& nodeName, const std::string& portName,
				size_t count = 1,
				uint32_t maxHops = InferGraph::kDefaultMaxHops) {
		{
			std::lock_guard lk(_declMutex);
			_declaredOutputs[taskId].emplace_back(nodeName, portName);
		}
		_setupCallback();
		_graph.submit(taskId, nodeName, portName, count, maxHops);
	}

	void submit(const TaskId& taskId, std::vector<OutputDeclaration> declarations,
				uint32_t maxHops = InferGraph::kDefaultMaxHops) {
		{
			std::lock_guard lk(_declMutex);
			for (auto& d : declarations) {
				_declaredOutputs[taskId].emplace_back(d.nodeName, d.portName);
			}
		}
		_setupCallback();
		_graph.submit(taskId, std::move(declarations), maxHops);
	}

	bool awaitCompletion(const TaskId& taskId, std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
		return _graph.waitForResult(taskId, timeout).status != TaskStatus::Running;
	}

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

	const InferGraph& graph() const {
		return _graph;
	}

	InferGraph& graph() {
		return _graph;
	}

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
	void _setupCallback() {
		_graph.setTaskCompleteCallback([this](const TaskId& tid) {
			std::lock_guard lk(_declMutex);
			auto it = _declaredOutputs.find(tid);
			if (it != _declaredOutputs.end()) {
				auto& captured = _capturedOutputs[tid];
				for (auto& [nodeName, portName] : it->second) {
					std::string key = nodeName + ":" + portName;
					if (captured.contains(key))
						continue;
					if (!_graph.hasOutput(tid, nodeName, portName))
						continue;
					try {
						auto val = _graph.takeOutput(tid, nodeName, portName);
						auto* t = val.as<Tensor>();
						if (t)
							captured[key] = std::move(*t);
					} catch (...) {
					}
				}
			}
		});
	}

	InferGraph _graph;

	std::unordered_map<TaskId, std::vector<std::pair<std::string, std::string>>> _declaredOutputs;
	mutable std::mutex _declMutex;

	std::unordered_map<TaskId, std::unordered_map<std::string, Tensor>> _capturedOutputs;


};

} // namespace DC
