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
	/// @note 实现在 ThreadPool 完整定义之后，避免访问未完整类型
	void await_resume() const;

private:
	friend class ThreadPool;
	PoolTicket(ThreadPool& pool, std::string tag, std::function<void()> task);

	ThreadPool* _pool;
	std::string _tag;
	std::function<void()> _task;
	std::coroutine_handle<> _handle;
};

// ── 跨池共享的组信号量注册表 ──
// 由 ExecutionEngine 持有并注入所有线程池：同一 tag 的信号量被多个池共享，
// 从而实现跨池分组互斥（如混合 affinity 子图）。无该表或表中无 tag 时组不限流。
struct GroupSemaphoreRegistry {
	using Semaphore = std::counting_semaphore<>;

	/// @brief  查找分组信号量；不存在返回 nullptr（该组不限流）
	std::shared_ptr<Semaphore> find(const std::string& tag) {
		std::lock_guard lk(_mutex);
		auto it = _semaphores.find(tag);
		return it != _semaphores.end() ? it->second : nullptr;
	}

	/// @brief  创建或替换分组信号量（limit 为新初始计数）
	void setLimit(const std::string& tag, size_t limit) {
		std::lock_guard lk(_mutex);
		_semaphores[tag] = std::make_shared<Semaphore>(static_cast<std::ptrdiff_t>(limit));
	}

	mutable std::mutex _mutex;
	std::unordered_map<std::string, std::shared_ptr<Semaphore>> _semaphores;
};

// ── 带分组信号量的线程池 ──
class ThreadPool {
public:
	/// @brief  构造线程池
	/// @param  config        线程数 + 初始分组限流
	/// @param  sharedGroups  跨池共享的组信号量注册表；nullptr 时自建（独立使用）
	explicit ThreadPool(const PoolConfig& config = {},
						std::shared_ptr<GroupSemaphoreRegistry> sharedGroups = nullptr);
	~ThreadPool();

	ThreadPool(const ThreadPool&) = delete;
	ThreadPool& operator=(const ThreadPool&) = delete;

	/// @brief  传统 fire-and-forget 提交
	void submit(const std::string& nodeTag, std::function<void()> task);

	/// @brief  协程友好提交：co_await 等待任务在线程池中执行完成
	PoolTicket submitAsync(const std::string& nodeTag, std::function<void()> task);

	/// @brief  运行时注册分组限流（构造后追加，无需重建池）
	/// @param  tag    分组标识（与 Node::tag 对应）
	/// @param  limit  该分组最大并发执行数
	void registerGroupLimit(const std::string& tag, size_t limit);

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

	// 跨池共享的组信号量注册表（nullptr 时在构造内自建）
	std::shared_ptr<GroupSemaphoreRegistry> _sharedGroups;
	std::unique_ptr<std::counting_semaphore<>> _globalSemaphore;

	// 分组活跃任务计数
	std::unordered_map<std::string, std::unique_ptr<std::atomic<size_t>>> _groupActiveCount;
	std::mutex _activeCountMutex;

	// 关闭标记：通知所有 awaiting 协程任务被取消
	std::atomic<bool> _shuttingDown{false};
};

// ── PoolTicket::await_resume 需访问 ThreadPool 私有成员，
// 定义必须在 ThreadPool 完整定义之后 ──
inline void PoolTicket::await_resume() const {
	if (_pool && _pool->_shuttingDown.load(std::memory_order_acquire)) {
		// 任务因 shutdown 被取消，调用方可按需检测
	}
}

} // namespace DC
