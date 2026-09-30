#include "ThreadPool.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <system_error>

namespace DC {

// ── ThreadPool ──

// 测试注入钩子（生产路径恒为空，见 ThreadPool.h 注释）
std::function<bool(size_t)> ThreadPool::s_spawnFilter = nullptr;

ThreadPool::ThreadPool(const PoolConfig& config)
	: _totalThreads(config.totalThreads) {
	if (!config.valid()) {
		throw std::invalid_argument("ThreadPool: config.totalThreads must be > 0");
	}

	// 启动工作线程：任一 std::thread 构造失败（含测试注入）时，先回收
	// 已启动 worker 再传播异常——否则构造未完成、成员 _workers 析构时
	// 对 joinable 线程触发 std::terminate（构造异常安全）
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
	shutdown();
}

bool ThreadPool::submit(std::function<void()> task) {
	{
		std::lock_guard lk(_mutex);
		// 已关闭（shutdown 后）拒绝：任务入队后无消费者，返回失败让调用方
		// 按失败语义收尾，避免任务静默滞留（#8-1）
		if (!_running.load(std::memory_order_acquire))
			return false;
		// 入队可能因内存压力抛 bad_alloc：捕获后按拒绝处理（#7）——
		// 提交方（_submitNodeRun）据返回值回滚在飞计数，不静默丢任务
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
	// 排水锁（P2-13）：并发 shutdown 时两个线程同时 joinable+join 同一
	// worker 是竞态（UB）——串行化后第二个调用者看到已清空的 _workers，
	// 顺序幂等性保持（joinable 检查 + clear）
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

		// 执行任务
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
