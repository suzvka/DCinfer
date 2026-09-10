#pragma once

#include "ErrorTracker.h"

#include <string>
#include <vector>

namespace DC {

/// @brief task 生命周期状态。
///
/// 状态迁移：
///   (未提交) --submit--> Running --声明满足--> Succeeded
///                                  --节点失败但已终止--> Failed
///                                  --cancel()--> Cancelled
/// 已终止的 taskId 允许复用：再次 submit 时清除上一轮状态重新开始。
enum class TaskStatus {
	Unknown,    ///< 从未提交过该 taskId
	Running,    ///< 已提交、尚未终止（仍在调度或传播中）
	Succeeded,  ///< 输出声明满足，正常终止
	Failed,     ///< 已终止且存在 Error 级诊断（部分节点执行失败）
	Cancelled,  ///< 被 cancel() 主动取消
};

/// @brief task 终止后的结构化结果（InferGraph::waitForResult 返回）。
///
/// 输出数据本体仍由 OutputZone 持有，终止后依然有效：
/// 经 takeOutput / takeOutputTensor 按 (nodeName, portName) 消费式取出（取出即消耗）；
/// 数据存活至下一次同 taskId 的 submit 或 releaseTask()。
struct TaskResult {
	TaskStatus status = TaskStatus::Unknown; ///< 终止状态（宿主等待超时未终止则为 Running）
	std::vector<TaskError> errors;           ///< 诊断记录（可能为空）
};

} // namespace DC
