#include "ThreadPool.h"

#include <cstdio>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <system_error>

namespace DC {

namespace {
	thread_local const ThreadPool* g_workerPool = nullptr;
}

const ThreadPool* ThreadPool::currentWorkerPool() noexcept { return g_workerPool; }
bool ThreadPool::isWorkerThread() const noexcept { return g_workerPool == this; }

// 测试注入钩子：生产路径恒为空
std::function<bool(size_t)> ThreadPool::s_spawnFilter = nullptr;

ThreadPool::ThreadPool(const PoolConfig& config)
	: _totalThreads(config.totalThreads) {
	if (!config.valid()) {
		throw std::invalid_argument("ThreadPool: config.totalThreads must be > 0");
	}

	// 任一线程构造失败时先回收已启动 worker 再传播异常，避免 joinable 触发 std::terminate
	_workers.reserve(_totalThreads);
	for (size_t i = 0; i < _totalThreads; ++i) {
		try {
			if (s_spawnFilter && !s_spawnFilter(i)) {
				throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again),
										"ThreadPool: thread creation failed");
			}
			_workers.emplace_back(&ThreadPool::_workerLoop, this);
		} catch (...) {
			_drainWorkers();
			throw;
		}
	}
}

ThreadPool::~ThreadPool() {
	if (isWorkerThread()) {
		// stderr 重定向后可能全缓冲，崩溃路径不经 flush 会丢诊断；显式 flush 兜底。
		std::fputs("ThreadPool: prohibited destruction on own worker; retain external ownership\n", stderr);
		std::fflush(stderr);
		std::terminate();
	}
	shutdown();
}

bool ThreadPool::submit(std::function<void()> task) {
	{
		std::lock_guard lk(_mutex);
		// 已关闭：任务入队后无消费者，返回失败让调用方收尾
		if (!_running.load(std::memory_order_acquire))
			return false;
		// 入队抛 bad_alloc 按拒绝处理：提交方据返回值回滚计数
		try {
			_taskQueue.push(std::move(task));
		} catch (...) {
			return false;
		}
	}
	_cv.notify_one();
	return true;
}

void ThreadPool::shutdown() {
	_drainWorkers();
}

void ThreadPool::_drainWorkers() {
	if (isWorkerThread())
		throw std::logic_error("ThreadPool::shutdown: prohibited on own worker");
	// 排水锁：并发 shutdown 同时 join 同一 worker 是 UB；串行化后幂等
	std::lock_guard drainLk(_drainMutex);
	_running.store(false, std::memory_order_release);

	{
		std::lock_guard lk(_mutex);
		// 丢弃所有等待中的任务
		std::queue<std::function<void()>> empty;
		_taskQueue.swap(empty);
	}
	_cv.notify_all();

	for (auto& t : _workers) {
		if (t.joinable())
			t.join();
	}
	_workers.clear();
}

void ThreadPool::_workerLoop() {
	struct WorkerScope {
		const ThreadPool* previous;
		explicit WorkerScope(const ThreadPool* pool) : previous(g_workerPool) { g_workerPool = pool; }
		~WorkerScope() { g_workerPool = previous; }
	} identity{this};
	while (true) {
		std::function<void()> task;
		{
			std::unique_lock lk(_mutex);
			_cv.wait(lk, [this] { return !_taskQueue.empty() || !_running.load(std::memory_order_acquire); });

			if (!_running.load(std::memory_order_acquire))
				break;

			task = std::move(_taskQueue.front());
			_taskQueue.pop();
		}

		try {
			task();
		} catch (const std::exception& e) {
			std::cerr << "ThreadPool: exception in task: " << e.what() << std::endl;
		} catch (...) {
			std::cerr << "ThreadPool: unknown exception in task" << std::endl;
		}
	}
}

} // namespace DC
