#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DC {

/// @brief 单条 task 级错误记录，包含节点名和异常信息
struct TaskError {
	std::string nodeName;  ///< 发生错误的节点名
	std::string source;    ///< 异常来源（如 "InferGraph::_propagateFrom"）
	std::string message;   ///< 错误详情
};

/// @brief 线程安全的 task 级错误收集器。
///
/// 从 InferGraph 提取的独立组件，负责记录和查询执行过程中的错误信息。
/// 所有公开方法线程安全（内部 mutex）。
class ErrorTracker {
public:
	using TaskId = std::string;

	/// @brief  记录一条 task 级错误（线程安全，可在协程/线程池中调用）
	void recordError(const TaskId& taskId, std::string nodeName, std::string source, std::string message);

	/// @brief  查询指定 task 的所有错误记录
	/// @return 按发生顺序排列的错误列表；若无错误则返回空向量
	std::vector<TaskError> taskErrors(const TaskId& taskId) const;

	/// @brief  清除所有 task 级错误记录（通常在重新 submit 前调用）
	void clearErrors();

	/// @brief  是否有任何 task 发生过错误
	bool hasErrors() const;

private:
	mutable std::mutex _mutex;
	std::unordered_map<TaskId, std::vector<TaskError>> _taskErrors;
};

// ════════════════════════════════════════════
// 内联实现
// ════════════════════════════════════════════

inline void ErrorTracker::recordError(const TaskId& taskId, std::string nodeName, std::string source,
									  std::string message) {
	std::lock_guard lk(_mutex);
	_taskErrors[taskId].push_back({std::move(nodeName), std::move(source), std::move(message)});
}

inline std::vector<TaskError> ErrorTracker::taskErrors(const TaskId& taskId) const {
	std::lock_guard lk(_mutex);
	auto it = _taskErrors.find(taskId);
	return it != _taskErrors.end() ? it->second : std::vector<TaskError>{};
}

inline void ErrorTracker::clearErrors() {
	std::lock_guard lk(_mutex);
	_taskErrors.clear();
}

inline bool ErrorTracker::hasErrors() const {
	std::lock_guard lk(_mutex);
	return !_taskErrors.empty();
}

} // namespace DC
