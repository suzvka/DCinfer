#include "SignalStore.h"

#include <shared_mutex>

namespace DC {

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

} // namespace DC
