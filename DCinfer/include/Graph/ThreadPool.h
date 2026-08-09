#pragma once

#include <atomic>
#include <condition_variable>
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

	/// @brief  fire-and-forget 提交
	void submit(const std::string& nodeTag, std::function<void()> task);

	/// @brief  运行时注册分组限流（构造后追加，无需重建池）
	/// @param  tag    分组标识（与 Node::tag 对应）
	/// @param  limit  该分组最大并发执行数
	void registerGroupLimit(const std::string& tag, size_t limit);

	/// @brief  查询组当前活跃任务数
	size_t activeCount(const std::string& groupTag) const;

	/// @brief  优雅关闭（丢弃队列中未执行的任务）
	void shutdown();

	size_t totalThreads() const {
		return _totalThreads;
	}

private:
	friend struct PoolTicket;

	struct PendingTask {
		std::function<void()> task;
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
};

} // namespace DC
