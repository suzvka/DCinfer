#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace DC {

// ── 线程池配置 ──
struct PoolConfig {
	size_t totalThreads = 1;

	bool valid() const {
		return totalThreads > 0;
	}
};

// ── 前向声明 ──
class ThreadPool;

// ── FIFO 工作线程池 ──
//
// 三层隔离（Compute / Operator / System）由 Engine 按节点 affinity 分发实现；
// 池本身只保证任务串行出队执行，并发上限即 worker 数量。
class ThreadPool {
public:
	/// @brief  构造线程池
	/// @param  config  线程数配置
	explicit ThreadPool(const PoolConfig& config = {});
	~ThreadPool();

	ThreadPool(const ThreadPool&) = delete;
	ThreadPool& operator=(const ThreadPool&) = delete;

	/// @brief  fire-and-forget 提交
	void submit(std::function<void()> task);

	/// @brief  优雅关闭（丢弃队列中未执行的任务）
	void shutdown();

	size_t totalThreads() const {
		return _totalThreads;
	}

private:
	void _workerLoop();

	size_t _totalThreads;
	std::vector<std::thread> _workers;

	std::mutex _mutex;
	std::condition_variable _cv;
	std::queue<std::function<void()>> _taskQueue;

	std::atomic<bool> _running{true};
};

} // namespace DC
