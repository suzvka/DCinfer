#pragma once

#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace DC {

/// @brief 节点级执行闸：同一节点同一时刻只允许一个 task 进入执行流水线。
///
/// 生命周期语义:
/// - 图路径:由 TaskExecutionDomain 在冻结期按节点集合预建(结构此后不变,
///   运行期只翻标志位),与 CompiledGraph 同生命周期;
/// - 单节点路径(NetServerAdapter / 测试):由 NodeExecutor 持有,随载体走。
class NodeExecutionGate {
public:
	NodeExecutionGate() = default;

	NodeExecutionGate(const NodeExecutionGate&) = delete;
	NodeExecutionGate& operator=(const NodeExecutionGate&) = delete;

	/// @brief  尝试获取执行租约
	/// @return false = 已有 task 在执行（重入拒绝；调度侧应登记重试而非判错）
	bool tryAcquire() noexcept {
		std::lock_guard lk(_m);
		if (_busy)
			return false;
		_busy = true;
		return true;
	}

	/// @brief  释放执行租约
	///         释放时全量投递已登记的重试
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

	/// @brief  登记一次重试
	/// @param  key 去重键
	/// @param  fn  重试动作
	/// @note   与 tryAcquire/release 同锁互斥：忙时登记必随某次 release 全量
	///         释放，空闲时立即投递——不存在登记后无人消费的窗口。
	///         同一 key 重复登记时以最新动作为准（后到覆盖，不静默丢弃——
	///         各次触发的 TTL/上下文以最后一次登记为最新）。
	void enqueueRetry(const void* key, std::function<void()> fn) {
		bool immediate = false;
		{
			std::lock_guard lk(_m);
			if (_busy) {
				for (auto& [k, pendingFn] : _pending) {
					if (k == key) {
						pendingFn = std::move(fn); // 后到覆盖：保留最新重试动作
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
	std::mutex _m; ///< 保护 _busy 与 _pending（互斥换 atomic_flag：重试登记需与闸状态同锁）
	bool _busy = false;
	/// 待重试登记：key→ 重试动作；release 时全量投递
	std::vector<std::pair<const void*, std::function<void()>>> _pending;
};

} // namespace DC
