#pragma once

#include "ErrorTracker.h"

#include <string>
#include <vector>

namespace DC {

/// @brief task 生命周期状态；已终止的 taskId 允许复用，再次 submit 时重新开始。
enum class TaskStatus {
	Unknown,
	Running,
	Succeeded,
	Failed,
	Cancelled,
};

/// @brief task 终止后的结构化结果，由 InferGraph::waitForResult 返回。
///
/// 输出数据本体由 OutputZone 持有，经 takeOutput 消费式取出，存活至下一次同 ID submit 或 releaseTask。
struct TaskResult {
	TaskStatus status = TaskStatus::Unknown;
	std::vector<TaskError> errors;
};

} // namespace DC
