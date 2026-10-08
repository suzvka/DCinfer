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
// 作为 ResourceScheduler 的内部执行器：资源类隔离（Compute / Operator / System）
// 由调度器按节点亲和（Node::affinity）分发承载，池本身只保证任务串行出队执行，
// 并发上限即 worker 数量。
class ThreadPool {
public:
	/// @brief  构造线程池
	/// @param  config  线程数配置
	explicit ThreadPool(const PoolConfig& config = {});
	/// Destruction requires an external owner on a nonworker thread. Violations
	/// print a diagnostic and terminate (also in Release); workers are never detached.
	~ThreadPool();

	ThreadPool(const ThreadPool&) = delete;
	ThreadPool& operator=(const ThreadPool&) = delete;

	/// @brief  fire-and-forget 提交。
	/// @return true = 任务已入队；false = 池已关闭或入队失败（内存压力）——
	///         任务未被接受，调用方应按失败语义处理（不得假设任务会执行）
	/// @note   队列无界（无背压）：提交速率长期超过执行速率时内存占用随之
	///         增长；宿主须在分配载荷/提交前做有界准入，超载立即拒绝，
	///         RAII 准入票据持有至实际执行/载荷释放；增加线程数不等于内存界限。
	///         worker 不得阻塞等待由排队任务释放的准入票据。
	bool submit(std::function<void()> task);

	/// @brief  优雅关闭（丢弃队列中未执行的任务）
	/// @throws std::logic_error on this pool's worker, before any state change/lock.
	void shutdown();

	/// Identity includes task execution AND destruction of its captured payload.
	bool isWorkerThread() const noexcept;
	static const ThreadPool* currentWorkerPool() noexcept;

	size_t totalThreads() const {
		return _totalThreads;
	}

private:
	void _workerLoop();
	/// @brief  关停收尾（构造失败回收与 shutdown 共用）：置停 → 清队 →
	///         唤醒 → join 全部已启动 worker → 清空。幂等。
	void _drainWorkers();

	size_t _totalThreads;
	std::vector<std::thread> _workers;

	std::mutex _mutex;
	std::condition_variable _cv;
	std::queue<std::function<void()>> _taskQueue;

	/// 排水互斥：串行化并发 shutdown（P2-13）——两个线程同时对同一 worker
	/// joinable+join 是竞态（UB）；顺序重复调用本已幂等，本锁补齐并发面。
	/// 构造失败回收路径（单线程）无竞争，同锁复用。
	std::mutex _drainMutex;

	std::atomic<bool> _running{true};

	// 测试注入点：线程创建过滤器（返回 false 模拟 std::thread 构造失败）。
	// 仅测试 TU 经 friend 访问设置；生产路径恒为空，不参与公共契约。
	static std::function<bool(size_t)> s_spawnFilter;
	friend struct ThreadPoolSpawnProbe;
};

} // namespace DC
