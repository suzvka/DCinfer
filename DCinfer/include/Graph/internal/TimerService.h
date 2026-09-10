#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <stop_token>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace DC {

/// @brief 引擎级共享超时定时器：单线程 + deadline 最小堆 + 条件变量。
///
/// 已 cancel 或已触发过的条目不会触发回调；回调在锁外调用
///（回调可能获取调用方其它锁，回调内禁止再调用本类）。
///
/// 析构：请求停止并 join 定时器线程，挂起条目的回调不再触发。
class TimerService {
public:
	using Handle = uint64_t;
	using Clock = std::chrono::steady_clock;
	using OnFire = std::function<void()>;

	TimerService() : _thread([this](std::stop_token stoken) { _run(stoken); }) {}

	TimerService(const TimerService&) = delete;
	TimerService& operator=(const TimerService&) = delete;

	/// @brief  登记 deadline 条目
	/// @return 条目句柄（cancel 用；单调递增，不复用）
	Handle schedule(Clock::time_point deadline, OnFire onFire) {
		std::lock_guard lk(_mutex);
		Handle handle = _nextHandle++;
		_live.insert(handle);
		_queue.push(Entry{deadline, handle, std::move(onFire)});
		_cv.notify_all(); // 新条目可能早于当前等待目标，唤醒重算
		return handle;
	}

	/// @brief  作废条目
	/// @return true  条目此前仍存活（本次调用将其作废）
	/// @return false 已触发/已作废/不存在
	bool cancel(Handle handle) {
		std::lock_guard lk(_mutex);
		return _live.erase(handle) > 0;
	}

private:
	struct Entry {
		Clock::time_point deadline;
		Handle handle;
		OnFire onFire;
	};

	// min-heap：deadline 早者优先
	struct LaterFirst {
		bool operator()(const Entry& a, const Entry& b) const noexcept {
			return a.deadline > b.deadline;
		}
	};

	void _run(std::stop_token stoken) {
		// stop 请求即时唤醒等待中的定时器线程（析构快速返回）
		std::stop_callback stopNotify(stoken, [this] {
			std::lock_guard lk(_mutex);
			_cv.notify_all();
		});

		std::unique_lock lk(_mutex);
		while (!stoken.stop_requested()) {
			if (_queue.empty()) {
				_cv.wait(lk, [&] { return stoken.stop_requested() || !_queue.empty(); });
				continue;
			}

			auto next = _queue.top().deadline;
			// 唤醒路径：更早条目入队 / stop / 到点；虚假唤醒由循环复查消化
			_cv.wait_until(lk, next, [&] {
				return stoken.stop_requested()
					   || (!_queue.empty() && _queue.top().deadline < next);
			});
			if (stoken.stop_requested())
				return;

			if (_queue.top().deadline > Clock::now())
				continue; // 尚未到点（被更早条目唤醒后重算目标）

			// 到点批处理：锁内认领（live → fired），锁外触发回调
			std::vector<OnFire> fired;
			while (!_queue.empty() && _queue.top().deadline <= Clock::now()) {
				// priority_queue::top() 返回 const 引用；队内对象实际非 const，
				// 弹出前移出回调是安全的惯用法
				auto& top = const_cast<Entry&>(_queue.top());
				Entry entry{top.deadline, top.handle, std::move(top.onFire)};
				_queue.pop();
				if (_live.erase(entry.handle) > 0) // 认领：仅存活条目触发
					fired.push_back(std::move(entry.onFire));
			}

			if (fired.empty())
				continue;

			lk.unlock();
			for (auto& fire : fired)
				fire();
			lk.lock();
		}
	}

	std::mutex _mutex;
	std::condition_variable _cv;
	std::priority_queue<Entry, std::vector<Entry>, LaterFirst> _queue;
	std::unordered_set<Handle> _live; ///< 存活条目句柄集（cancel 与 fire 认领的仲裁点）
	Handle _nextHandle = 0;
	/// 声明在最后：构造时所有状态先行就位；析构时最先 stop + join
	std::jthread _thread;
};

} // namespace DC
