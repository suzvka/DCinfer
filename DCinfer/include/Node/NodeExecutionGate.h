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
/// currentTask 的写入仅发生在持有租约期间。
///
/// 重试登记（H-1 修复）：闸忙时被拒的提交经 enqueueRetry 登记待重放，
/// 租约释放时按登记顺序投递重试——同一节点上的并发任务排队执行，
/// "节点正忙"不再等价于任务失败（Reentrant 判死路径移除）。
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

	/// @brief  释放执行租约（须与成功的 tryAcquire 配对）。
	///         释放时全量投递已登记的重试（锁外执行，防死锁）。
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
				// 重试投递失败不传播（任务的下个触发点/宿主护栏兜底）
			}
		}
	}

	/// @brief  登记一次重试（同一 key 去重；闸空闲时立即投递）。
	/// @param  key 去重键（引擎侧传轮次指针：同一轮次对同一节点至多一个待重试）
	/// @param  fn  重试动作（重新提交节点执行）
	/// @note   与 tryAcquire/release 同锁互斥：忙时登记必随某次 release 全量
	///         释放，空闲时立即投递——不存在登记后无人消费的窗口。
	void enqueueRetry(const void* key, std::function<void()> fn) {
		bool immediate = false;
		{
			std::lock_guard lk(_m);
			if (_busy) {
				for (const auto& [k, _] : _pending) {
					if (k == key)
						return; // 同一轮次已登记，合并
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

	/// @brief  记录当前执行中的 task ID(调用方须持有租约)
	void setCurrentTask(std::string taskId) {
		_currentTaskId = std::move(taskId);
	}

	/// @brief  清除当前 task ID(调用方须持有租约)
	void clearCurrentTask() noexcept {
		_currentTaskId.reset();
	}

private:
	std::mutex _m; ///< 保护 _busy 与 _pending（互斥换 atomic_flag：重试登记需与闸状态同锁）
	bool _busy = false;
	/// 待重试登记：key（轮次指针）→ 重试动作；release 时全量投递
	std::vector<std::pair<const void*, std::function<void()>>> _pending;
	std::optional<std::string> _currentTaskId;
};

} // namespace DC
