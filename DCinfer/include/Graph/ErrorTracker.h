#pragma once

#include "Diagnostic.h"

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DC {

/// @brief 诊断级别。
enum class DiagnosticLevel {
	Info,
	Warning,
	Error
};

/// @brief 单条 task 级诊断记录。
struct TaskError {
	DiagnosticLevel level = DiagnosticLevel::Error;
	std::string nodeName;
	std::string source;
	std::string message;
	std::optional<Diagnostic> diagnostic;
};

/// @brief 线程安全的 task 级错误收集器。
class ErrorTracker {
public:
	using TaskId = std::string;

	/// @brief 记录一条 task 级错误。
	void recordError(const TaskId& taskId, std::string nodeName, std::string source, std::string message);

	/// @brief 记录一条 task 级错误并附带领域诊断。
	void recordError(const TaskId& taskId, std::string nodeName, std::string source, std::string message,
					 std::optional<Diagnostic> diagnostic);

	/// @brief 记录一条 task 级警告。
	void recordWarning(const TaskId& taskId, std::string nodeName, std::string source, std::string message);

	/// @brief 记录一条 task 级信息。
	void recordInfo(const TaskId& taskId, std::string nodeName, std::string source, std::string message);

	/// @brief 查询指定 task 的所有诊断记录，按发生顺序。
	std::vector<TaskError> taskErrors(const TaskId& taskId) const;

	/// @brief 清除所有 task 级错误记录。
	void clearErrors();

	/// @brief 清除指定 task 的诊断记录。
	void clearTask(const TaskId& taskId) {
		std::lock_guard lk(_mutex);
		_taskErrors.erase(taskId);
	}

	/// @brief 是否有任何 task 发生过错误。
	bool hasErrors() const;

private:
	mutable std::mutex _mutex;
	std::unordered_map<TaskId, std::vector<TaskError>> _taskErrors;
};

inline void ErrorTracker::recordError(const TaskId& taskId, std::string nodeName, std::string source,
									  std::string message) {
	std::lock_guard lk(_mutex);
	_taskErrors[taskId].push_back({DiagnosticLevel::Error, std::move(nodeName), std::move(source), std::move(message)});
}

inline void ErrorTracker::recordError(const TaskId& taskId, std::string nodeName, std::string source,
									  std::string message, std::optional<Diagnostic> diagnostic) {
	std::lock_guard lk(_mutex);
	TaskError e;
	e.level = DiagnosticLevel::Error;
	e.nodeName = std::move(nodeName);
	e.source = std::move(source);
	e.message = std::move(message);
	e.diagnostic = std::move(diagnostic);
	_taskErrors[taskId].push_back(std::move(e));
}

inline void ErrorTracker::recordWarning(const TaskId& taskId, std::string nodeName, std::string source,
										std::string message) {
	std::lock_guard lk(_mutex);
	_taskErrors[taskId].push_back({DiagnosticLevel::Warning, std::move(nodeName), std::move(source), std::move(message)});
}

inline void ErrorTracker::recordInfo(const TaskId& taskId, std::string nodeName, std::string source,
									 std::string message) {
	std::lock_guard lk(_mutex);
	_taskErrors[taskId].push_back({DiagnosticLevel::Info, std::move(nodeName), std::move(source), std::move(message)});
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
