#pragma once

#include <coroutine>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace DC {

class Node;          // 前向声明
class TaskBuffer;    // 前向声明
struct NodeSchema;   // 前向声明（定义见 Node.h）
struct NodeCompletion; // 前向声明（完整定义见 Node.h）

/// @brief 协程等待/通知桥：封装 Node 的 _waiters / _waitersMutex / _completedTasks。
///
/// 管理 co_await 的挂起/恢复语义：
/// - whenComplete() 返回可 co_await 的 NodeCompletion 对象
/// - notifyWaiters() 在任务完成后 resume 所有等待协程
/// - terminateTask() 在任务终止时清理并通知等待者
class CoroutineBridge {
public:
	using TaskId = std::string;

	CoroutineBridge() = default;

	/// @brief  创建可 co_await 的等待器，与指定任务的生命周期绑定。
	///         调用者需提供 TaskBuffer 和 Schema 用于就绪判断。
	NodeCompletion whenComplete(const TaskId& taskId, TaskBuffer& buffer,
							   const NodeSchema& schema);

	/// @brief  通知所有等待指定 task 完成的协程，resume 它们。
	void notifyWaiters(const TaskId& taskId);

	/// @brief  终止指定 task：从完成集合中移除，resume 所有等待协程（无结果）。
	void terminateTask(const TaskId& taskId);

	/// @brief  从已完成集合中移除指定 task（通常由 clearTask 调用）。
	void clearCompleted(const TaskId& taskId);

private:
	friend struct NodeCompletion;

	std::unordered_map<TaskId, std::vector<std::coroutine_handle<>>> _waiters;
	mutable std::mutex _mutex;
	std::unordered_set<TaskId> _completedTasks;
};

} // namespace DC
