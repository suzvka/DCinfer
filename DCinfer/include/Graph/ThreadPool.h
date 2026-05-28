#pragma once

#include <atomic>
#include <condition_variable>
#include <coroutine>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <semaphore>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace DC {

// ── 线程池配置 ──
struct PoolConfig {
	size_t totalThreads = 1;
	std::unordered_map<std::string, size_t> groupLimits; // 分组限流

	bool valid() const {
		return totalThreads > 0;
	}
};

// ── 前向声明 ──
class ThreadPool;

// ── PoolTicket：co_await-able 线程池提交句柄 ──
// co_await pool.submitAsync(tag, task) 挂起协程，
// 在线程池中执行 task 完成后 resume
struct PoolTicket {
	bool await_ready() const noexcept {
		return false;
	}
	void await_suspend(std::coroutine_handle<> h);
	/// @brief 恢复后检查是否因 shutdown 被取消
	void await_resume() const {
		if (_pool && _pool->_shuttingDown.load(std::memory_order_acquire)) {
			// 任务因 shutdown 被取消，调用方可按需检测
		}
	}

private:
	friend class ThreadPool;
	PoolTicket(ThreadPool& pool, std::string tag, std::function<void()> task);

	ThreadPool* _pool;
	std::string _tag;
	std::function<void()> _task;
	std::coroutine_handle<> _handle;
};

// ── 带分组信号量的线程池 ──
class ThreadPool {
public:
	explicit ThreadPool(const PoolConfig& config = {});
	~ThreadPool();

	ThreadPool(const ThreadPool&) = delete;
	ThreadPool& operator=(const ThreadPool&) = delete;

	/// @brief  传统 fire-and-forget 提交
	void submit(const std::string& nodeTag, std::function<void()> task);

	/// @brief  协程友好提交：co_await 等待任务在线程池中执行完成
	PoolTicket submitAsync(const std::string& nodeTag, std::function<void()> task);

	/// @brief  查询组当前活跃任务数
	size_t activeCount(const std::string& groupTag) const;

	/// @brief  优雅关闭（取消所有等待中的协程并 resume 句柄）
	void shutdown();

	size_t totalThreads() const {
		return _totalThreads;
	}

private:
	friend struct PoolTicket;

	struct PendingTask {
		std::function<void()> task;
		std::coroutine_handle<> handle; // 非空 = 需要 resume 的协程句柄
		std::string groupTag;
	};

	void _workerLoop();

	bool _tryAcquireGroup(const std::string& tag);
	void _releaseGroup(const std::string& tag);

	/// @brief  递增/递减组活跃任务计数（懒初始化原子计数器）
	void _incrementActive(const std::string& tag);
	void _decrementActive(const std::string& tag);

	PoolConfig _config;
	size_t _totalThreads;
	std::vector<std::thread> _workers;

	std::mutex _mutex;
	std::condition_variable _cv;
	std::queue<PendingTask> _taskQueue;

	std::atomic<bool> _running{true};

	// 分组信号量
	std::mutex _groupMutex;
	std::unordered_map<std::string, std::unique_ptr<std::counting_semaphore<>>> _groupSemaphores;
	std::unique_ptr<std::counting_semaphore<>> _globalSemaphore;

	// 分组活跃任务计数
	std::unordered_map<std::string, std::unique_ptr<std::atomic<size_t>>> _groupActiveCount;
	std::mutex _activeCountMutex;

	// 关闭标记：通知所有 awaiting 协程任务被取消
	std::atomic<bool> _shuttingDown{false};
};

} // namespace DC
