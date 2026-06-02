#include "Node/internal/CoroutineBridge.h"
#include "Node/internal/TaskBuffer.h"
#include "Node.h"

namespace DC {

// ── CoroutineBridge ──

NodeCompletion CoroutineBridge::whenComplete(const TaskId& taskId, TaskBuffer& buffer,
															  const NodeSchema& schema) {
	return NodeCompletion(&buffer, this, taskId, &schema);
}

void CoroutineBridge::notifyWaiters(const TaskId& taskId) {
	std::vector<std::coroutine_handle<>> handles;
	{
		std::lock_guard lk(_mutex);
		_completedTasks.insert(taskId);
		auto it = _waiters.find(taskId);
		if (it == _waiters.end()) {
			return;
		}
		handles = std::move(it->second);
		_waiters.erase(it);
	}
	// 在锁外 resume，避免协程恢复后可能的死锁
	for (auto h : handles) {
		if (h) {
			h.resume();
			if (h.done()) {
				h.destroy();
			}
		}
	}
}

void CoroutineBridge::terminateTask(const TaskId& taskId) {
	std::vector<std::coroutine_handle<>> handles;
	{
		std::lock_guard lk(_mutex);
		_completedTasks.erase(taskId);
		auto it = _waiters.find(taskId);
		if (it != _waiters.end()) {
			handles = std::move(it->second);
			_waiters.erase(it);
		}
	}
	for (auto h : handles) {
		if (h) {
			h.resume();
			if (h.done()) {
				h.destroy();
			}
		}
	}
}

void CoroutineBridge::clearCompleted(const TaskId& taskId) {
	std::lock_guard lk(_mutex);
	_completedTasks.erase(taskId);
}

} // namespace DC
