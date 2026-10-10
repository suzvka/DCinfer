#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace DC {

struct PoolConfig {
	size_t totalThreads = 1;

	bool valid() const {
		return totalThreads > 0;
	}
};

// FIFO 工作线程池：ResourceScheduler 的内部执行器。资源隔离由调度器按节点亲和
// 分发承载，池本身只保证任务串行出队执行，并发上限即 worker 数。
class ThreadPool {
public:
	explicit ThreadPool(const PoolConfig& config = {});
	/// 必须由外部非 worker 线程销毁，违反则 std::terminate。
	~ThreadPool();

	ThreadPool(const ThreadPool&) = delete;
	ThreadPool& operator=(const ThreadPool&) = delete;

	/// @brief 提交任务，fire-and-forget。
	/// @return true = 已入队；false = 池已关闭或入队失败，任务未被执行。
	/// @note   队列无界无背压，宿主须自行做有界准入控制内存规模；
	///         worker 不得阻塞等待由排队任务释放的准入票据。
	bool submit(std::function<void()> task);

	/// @brief 优雅关闭，丢弃队列中未执行的任务。
	/// @throws 在本池 worker 内调用抛 std::logic_error。
	void shutdown();

	/// @brief 当前线程是否为本池 worker。
	bool isWorkerThread() const noexcept;
	static const ThreadPool* currentWorkerPool() noexcept;

	size_t totalThreads() const {
		return _totalThreads;
	}

private:
	void _workerLoop();
	/// @brief 关停收尾，构造失败回收与 shutdown 共用；幂等。
	void _drainWorkers();

	size_t _totalThreads;
	std::vector<std::thread> _workers;

	std::mutex _mutex;
	std::condition_variable _cv;
	std::queue<std::function<void()>> _taskQueue;

	/// 串行化并发 shutdown：对同一 worker 并发 join 是 UB。
	std::mutex _drainMutex;

	std::atomic<bool> _running{true};

	// 测试注入点：返回 false 模拟 std::thread 构造失败；生产路径恒为空。
	static std::function<bool(size_t)> s_spawnFilter;
	friend struct ThreadPoolSpawnProbe;
};

} // namespace DC
