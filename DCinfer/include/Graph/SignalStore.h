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
class SignalStore {
public:
	SignalStore() = default;

	/// @brief 写入信号值，信号名不存在时自动创建。
	/// @param name  信号名。
	/// @param value 信号值。
	void set(const std::string& name, bool value);

	/// @brief 读取信号值。
	/// @param name       信号名。
	/// @param defaultVal 信号不存在时的默认值（默认 false = 导通）。
	/// @return 信号值，信号不存在时返回 defaultVal。
	bool get(const std::string& name, bool defaultVal = false) const;

private:
	mutable std::shared_mutex _mutex;
	std::unordered_map<std::string, std::atomic<bool>> _signals;
};

} // namespace DC
