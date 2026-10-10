#pragma once

#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace DC {

/// @brief 节点级执行闸：同一节点同一时刻只允许一个 task 进入执行流水线。
///        图路径由 TaskExecutionDomain 冻结期预建；单节点路径由 NodeExecutor 持有。
class NodeExecutionGate {
public:
	NodeExecutionGate() = default;

	NodeExecutionGate(const NodeExecutionGate&) = delete;
	NodeExecutionGate& operator=(const NodeExecutionGate&) = delete;

	/// @brief 尝试获取执行租约；false = 已有 task 在执行（调度侧应登记重试）。
	bool tryAcquire() noexcept {
		std::lock_guard lk(_m);
		if (_busy)
			return false;
		_busy = true;
		return true;
	}

	/// @brief 释放执行租约（全量投递已登记的重试）。
	void release() noexcept {
		std::vector<std::pair<const void*, std::function<void()>>> pending;
		{
			std::lock_guard lk(_m);
			_busy = false;
			pending.swap(_pending);
		}
		for (auto& [key, fn] : pending) {
			(void)key;
			try {
				fn();
			} catch (...) {
				// 重试投递失败不传播
			}
		}
	}

	/// @brief 登记一次重试（空闲时立即投递；忙时随某次 release 全量投递）。
	/// @param key 去重键；同一 key 重复登记后到覆盖
	void enqueueRetry(const void* key, std::function<void()> fn) {
		bool immediate = false;
		{
			std::lock_guard lk(_m);
			if (_busy) {
				for (auto& [k, pendingFn] : _pending) {
					if (k == key) {
						pendingFn = std::move(fn); // 后到覆盖
						return;
					}
				}
				_pending.emplace_back(key, std::move(fn));
				return;
			}
			immediate = true;
		}
		if (immediate) {
			try {
				fn();
			} catch (...) {
			}
		}
	}

private:
	std::mutex _m; ///< 保护 _busy 与 _pending
	bool _busy = false;
	std::vector<std::pair<const void*, std::function<void()>>> _pending;
};

} // namespace DC
