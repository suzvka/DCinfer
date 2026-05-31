#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Value.h"

namespace DC {

// ── 审计信息：伴随每次 append 写入的元数据 ──
struct OutputAudit {
	std::string nodeName; // dc.node
	std::string portName; // dc.port
	std::string taskId;   // dc.task_id
	// 后续扩展: outputId, timestamp, iteration, lineage, dtype, shape, device
};

// ── 输出声明：某 task 期望哪个节点的哪个端口产出多少次 ──
struct OutputDeclaration {
	std::string nodeName;
	std::string portName;
	size_t count = 1;
};

// ── 输出绑定（序列化用）──
struct OutputBinding {
	std::string nodeName;
	std::string portName;
};

/// @brief OutputZone：append-only 输出区，聚合声明/累加/artifact 存储。
///
/// 语义：
/// - bind() 标记 node:port 的目的地是输出区（与边目的地互斥）
/// - declare() 声明 task 的期望产出
/// - append() 写入 artifact（数据 + 审计信息）
/// - accumulateAndCheck() 累加计数并检查所有声明是否满足
/// - take() 消费式读取 artifact
///
/// 所有公开方法线程安全（内部 mutex）。
class OutputZone {
public:
	using TaskId = std::string;

	// ── 绑定管理 ──

	void bind(const std::string& nodeName, const std::string& portName);
	bool isBound(const std::string& nodeName, const std::string& portName) const;
	const std::vector<OutputBinding>& bindings() const;

	// ── 声明管理 ──

	void declare(const TaskId& taskId, std::vector<OutputDeclaration> declarations);
	void declare(const TaskId& taskId, const std::string& nodeName,
				 const std::string& portName, size_t count = 1);
	bool hasDeclaration(const TaskId& taskId) const;

	// ── 累加与检查（RuleEngine — 当前仅做数量检查）──

	/// @brief  累加指定端口的产出计数，返回 true 表示所有声明均已满足
	bool accumulateAndCheck(const std::string& nodeName, const std::string& portName,
							const TaskId& taskId);

	/// @brief  静态检查所有声明是否满足（不累加，供 _onExhausted 使用）
	bool checkAllSatisfied(const TaskId& taskId) const;

	// ── Artifact 存储 ──

	/// @brief  append-only 写入 artifact（消费式，数据所有权转移）
	void append(const TaskId& taskId, const std::string& nodeName,
				const std::string& portName, Value data, OutputAudit audit);

	/// @brief  消费式读取 artifact（取出后内部清空）
	std::optional<Value> take(const TaskId& taskId, const std::string& nodeName,
							  const std::string& portName);

	/// @brief  检查 artifact 是否存在
	bool hasOutput(const TaskId& taskId, const std::string& nodeName,
				   const std::string& portName) const;

	// ── Task 清理 ──

	/// @brief  清理指定 task 的所有声明、累加器、artifact
	void clearTask(const TaskId& taskId);

private:
	static std::string _makeKey(const std::string& nodeName, const std::string& portName) {
		return nodeName + ":" + portName;
	}

	struct Artifact {
		Value data;
		OutputAudit audit;
	};

	mutable std::mutex _mutex;

	std::unordered_set<std::string> _bindings;   // "nodeName:portName"
	std::vector<OutputBinding> _bindingsList;    // 序列化用

	std::unordered_map<TaskId, std::vector<OutputDeclaration>> _declarations;
	std::unordered_map<TaskId, std::unordered_map<std::string, size_t>> _accumulated;
	std::unordered_map<TaskId, std::unordered_map<std::string, Artifact>> _artifacts;
};

// ════════════════════════════════════════════
// 内联实现
// ════════════════════════════════════════════

inline void OutputZone::bind(const std::string& nodeName, const std::string& portName) {
	std::lock_guard lk(_mutex);
	std::string key = _makeKey(nodeName, portName);
	_bindings.insert(key);
	_bindingsList.push_back({nodeName, portName});
}

inline bool OutputZone::isBound(const std::string& nodeName, const std::string& portName) const {
	std::lock_guard lk(_mutex);
	return _bindings.contains(_makeKey(nodeName, portName));
}

inline const std::vector<OutputBinding>& OutputZone::bindings() const {
	std::lock_guard lk(_mutex);
	return _bindingsList;
}

inline void OutputZone::declare(const TaskId& taskId, std::vector<OutputDeclaration> declarations) {
	std::lock_guard lk(_mutex);
	auto& existing = _declarations[taskId];
	existing.reserve(existing.size() + declarations.size());
	for (auto& decl : declarations) {
		existing.push_back(std::move(decl));
	}
}

inline void OutputZone::declare(const TaskId& taskId, const std::string& nodeName,
								const std::string& portName, size_t count) {
	std::lock_guard lk(_mutex);
	_declarations[taskId].push_back({nodeName, portName, count});
}

inline bool OutputZone::hasDeclaration(const TaskId& taskId) const {
	std::lock_guard lk(_mutex);
	return _declarations.contains(taskId);
}

inline bool OutputZone::accumulateAndCheck(const std::string& nodeName, const std::string& portName,
										   const TaskId& taskId) {
	std::lock_guard lk(_mutex);
	auto declIt = _declarations.find(taskId);
	if (declIt == _declarations.end())
		return false; // 防御：未声明（submit 中已被拦截）

	std::string key = _makeKey(nodeName, portName);
	++_accumulated[taskId][key];

	for (const auto& decl : declIt->second) {
		std::string dkey = _makeKey(decl.nodeName, decl.portName);
		size_t current = _accumulated[taskId][dkey];
		if (current < decl.count)
			return false;
	}
	return true;
}

inline bool OutputZone::checkAllSatisfied(const TaskId& taskId) const {
	std::lock_guard lk(_mutex);
	auto declIt = _declarations.find(taskId);
	if (declIt == _declarations.end())
		return false;

	auto accIt = _accumulated.find(taskId);
	for (const auto& decl : declIt->second) {
		std::string key = _makeKey(decl.nodeName, decl.portName);
		size_t current = 0;
		if (accIt != _accumulated.end()) {
			auto countIt = accIt->second.find(key);
			if (countIt != accIt->second.end())
				current = countIt->second;
		}
		if (current < decl.count)
			return false;
	}
	return true;
}

inline void OutputZone::append(const TaskId& taskId, const std::string& nodeName,
							   const std::string& portName, Value data, OutputAudit audit) {
	std::lock_guard lk(_mutex);
	std::string key = _makeKey(nodeName, portName);
	_artifacts[taskId][key] = {std::move(data), std::move(audit)};
}

inline std::optional<Value> OutputZone::take(const TaskId& taskId, const std::string& nodeName,
											 const std::string& portName) {
	std::lock_guard lk(_mutex);
	auto taskIt = _artifacts.find(taskId);
	if (taskIt == _artifacts.end())
		return std::nullopt;

	std::string key = _makeKey(nodeName, portName);
	auto artIt = taskIt->second.find(key);
	if (artIt == taskIt->second.end())
		return std::nullopt;

	Value val = std::move(artIt->second.data);
	taskIt->second.erase(artIt);
	return val;
}

inline bool OutputZone::hasOutput(const TaskId& taskId, const std::string& nodeName,
								  const std::string& portName) const {
	std::lock_guard lk(_mutex);
	auto taskIt = _artifacts.find(taskId);
	if (taskIt == _artifacts.end())
		return false;

	std::string key = _makeKey(nodeName, portName);
	auto artIt = taskIt->second.find(key);
	return artIt != taskIt->second.end() && artIt->second.data;
}

inline void OutputZone::clearTask(const TaskId& taskId) {
	std::lock_guard lk(_mutex);
	_declarations.erase(taskId);
	_accumulated.erase(taskId);
	_artifacts.erase(taskId);
}

} // namespace DC
