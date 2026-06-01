#include "SignalStore.h"

#include <shared_mutex>

namespace DC {

// ── 内部辅助：复合键构造 ──
// 使用 '\0' 分隔信号名与 taskId，避免与合法信号名冲突
std::string SignalStore::_makeKey(const std::string& name, const std::string& taskId) {
	std::string key;
	key.reserve(name.size() + 1 + taskId.size());
	key.append(name);
	key.push_back('\0');
	key.append(taskId);
	return key;
}

// ── 全局信号 ──

void SignalStore::set(const std::string& name, bool value) {
	{
		std::shared_lock lock(_mutex);
		auto it = _signals.find(name);
		if (it != _signals.end()) {
			it->second.store(value, std::memory_order_relaxed);
			return;
		}
	}

	// 信号名不存在 → 升为写锁，插入新条目
	std::unique_lock lock(_mutex);
	auto it = _signals.find(name);
	if (it != _signals.end()) {
		it->second.store(value, std::memory_order_relaxed);
		return;
	}
	_signals[name].store(value, std::memory_order_relaxed);
}

bool SignalStore::get(const std::string& name, bool defaultVal) const {
	std::shared_lock lock(_mutex);
	auto it = _signals.find(name);
	if (it != _signals.end()) {
		return it->second.load(std::memory_order_relaxed);
	}
	return defaultVal;
}

// ── Task 级信号 ──

void SignalStore::set(const std::string& name, const std::string& taskId, bool value) {
	if (taskId.empty())
		return; // task 级信号必须有明确的 taskId

	const std::string key = _makeKey(name, taskId);

	{
		std::shared_lock lock(_mutex);
		auto it = _taskSignals.find(key);
		if (it != _taskSignals.end()) {
			it->second.store(value, std::memory_order_relaxed);
			return;
		}
	}

	// 不存在 → 升为写锁，插入新条目
	std::unique_lock lock(_mutex);
	auto it = _taskSignals.find(key);
	if (it != _taskSignals.end()) {
		it->second.store(value, std::memory_order_relaxed);
		return;
	}
	_taskSignals[key].store(value, std::memory_order_relaxed);
}

bool SignalStore::get(const std::string& name, const std::string& taskId, bool defaultVal) const {
	std::shared_lock lock(_mutex);

	// ① task 级信号优先
	if (!taskId.empty()) {
		const std::string key = _makeKey(name, taskId);
		auto it = _taskSignals.find(key);
		if (it != _taskSignals.end()) {
			return it->second.load(std::memory_order_relaxed);
		}
	}

	// ② 回退到全局信号
	auto it = _signals.find(name);
	if (it != _signals.end()) {
		return it->second.load(std::memory_order_relaxed);
	}

	// ③ 默认值
	return defaultVal;
}

void SignalStore::remove(const std::string& name, const std::string& taskId) {
	if (taskId.empty())
		return;

	std::unique_lock lock(_mutex);
	_taskSignals.erase(_makeKey(name, taskId));
}

void SignalStore::clearTask(const std::string& taskId) {
	if (taskId.empty())
		return;

	std::unique_lock lock(_mutex);

	// 遍历所有 task 级信号，移除属于该 taskId 的条目
	for (auto it = _taskSignals.begin(); it != _taskSignals.end();) {
		const auto& key = it->first;
		// 复合键格式：name\0taskId → 查找 \0 后的部分
		auto nullPos = key.find('\0');
		if (nullPos != std::string::npos && key.compare(nullPos + 1, std::string::npos, taskId) == 0) {
			it = _taskSignals.erase(it);
		} else {
			++it;
		}
	}
}

} // namespace DC
