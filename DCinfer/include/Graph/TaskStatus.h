#pragma once

#include "ErrorTracker.h"

#include <string>
#include <vector>

namespace DC {

/// @brief task 生命周期状态（已终止的 taskId 允许复用：再次 submit 时重新开始）。
enum class TaskStatus {
	Unknown,    ///< 从未提交过该 taskId
	Running,    ///< 已提交、尚未终止
	Succeeded,  ///< 输出声明满足，正常终止
	Failed,     ///< 已终止且存在 Error 级诊断
	Cancelled,  ///< 被 cancel() 取消
};

/// @brief task 终止后的结构化结果（InferGraph::waitForResult 返回）。
///
/// 输出数据本体由 OutputZone 持有，经 takeOutput 消费式取出，存活至下一次同 ID submit 或 releaseTask。
struct TaskResult {
	TaskStatus status = TaskStatus::Unknown; ///< 终止状态；等待超时未终止为 Running
	std::vector<TaskError> errors;           ///< 诊断记录
};

} // namespace DC
