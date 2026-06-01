#pragma once

#include <atomic>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace DC {

/// @brief 图级信号仓库：独立于数据流的键值对存储。
///
/// 信号与数据图正交解耦：不走数据边，不参与协程传播。
/// 任意时刻、任意线程可读写，通过 atomic<bool> 实现无锁读。
///
/// 典型用途：控制 Node 的 isBlocked() 判断，实现开关语义——
///   信号=false → 节点阻塞 → 搬运线程跳过 → 下游短路。
///
/// 两级信号模型：
///   - 全局信号（广播）：对所有 task 生效
///   - Task 级信号（覆盖）：仅对指定 taskId 生效，覆盖同名的全局信号
///   查找顺序：task 级 → 全局 → defaultVal
class SignalStore {
public:
	SignalStore() = default;

	// ── 全局信号（广播）──

	/// @brief 写入全局信号值，信号名不存在时自动创建。
	/// @param name  信号名。
	/// @param value 信号值。
	void set(const std::string& name, bool value);

	/// @brief 读取全局信号值。
	/// @param name       信号名。
	/// @param defaultVal 信号不存在时的默认值（默认 false = 导通）。
	/// @return 信号值，信号不存在时返回 defaultVal。
	bool get(const std::string& name, bool defaultVal = false) const;

	// ── Task 级信号（覆盖广播）──

	/// @brief 写入 task 级信号值（仅对指定 taskId 生效）。
	/// @param name   信号名。
	/// @param taskId 任务标识符（非空）。
	/// @param value  信号值。
	void set(const std::string& name, const std::string& taskId, bool value);

	/// @brief 读取信号值（task 级优先 → 全局回退 → defaultVal）。
	/// @param name       信号名。
	/// @param taskId     任务标识符（为空时退化为全局查询）。
	/// @param defaultVal 信号不存在时的默认值（默认 false = 导通）。
	/// @return 信号值。
	bool get(const std::string& name, const std::string& taskId, bool defaultVal = false) const;

	/// @brief 移除单个 task 级信号。
	/// @param name   信号名。
	/// @param taskId 任务标识符。
	void remove(const std::string& name, const std::string& taskId);

	/// @brief 清理指定 task 的所有 task 级信号（task 终止时调用）。
	/// @param taskId 任务标识符。
	void clearTask(const std::string& taskId);

private:
	// 内部辅助：按"signalName\0taskId"格式构造复合键
	static std::string _makeKey(const std::string& name, const std::string& taskId);

	mutable std::shared_mutex _mutex;
	std::unordered_map<std::string, std::atomic<bool>> _signals;           // 全局信号
	std::unordered_map<std::string, std::atomic<bool>> _taskSignals;       // task 级信号（复合键）
};

} // namespace DC
