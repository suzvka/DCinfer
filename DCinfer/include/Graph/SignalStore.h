#pragma once

#include <atomic>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace DC {

/// @brief 图级信号仓库：独立于数据流的键值对存储，atomic<bool> 无锁读写，任意线程可用。
///
/// 两级信号：全局广播信号与 task 级覆盖信号；查找顺序 task 级、全局、defaultVal。
class SignalStore {
public:
	SignalStore() = default;

	/// @brief 写入全局信号值，信号名不存在时自动创建。
	void set(const std::string& name, bool value);

	/// @brief 读取全局信号值，信号不存在时返回 defaultVal。
	bool get(const std::string& name, bool defaultVal = false) const;

	/// @brief 写入 task 级信号值，仅对指定 taskId 生效。
	void set(const std::string& name, const std::string& taskId, bool value);

	/// @brief 读取信号值；task 级优先，全局回退，最后 defaultVal。
	bool get(const std::string& name, const std::string& taskId, bool defaultVal = false) const;

	/// @brief 移除单个 task 级信号。
	void remove(const std::string& name, const std::string& taskId);

	/// @brief 清理指定 task 的所有 task 级信号。
	void clearTask(const std::string& taskId);

private:
	static std::string _makeKey(const std::string& name, const std::string& taskId);

	mutable std::shared_mutex _mutex;
	std::unordered_map<std::string, std::atomic<bool>> _signals;
	std::unordered_map<std::string, std::atomic<bool>> _taskSignals;
};

} // namespace DC
