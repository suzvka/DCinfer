#include "ThreadPool.h"

#include <iostream>
#include <stdexcept>

namespace DC {

// ── ThreadPool ──

ThreadPool::ThreadPool(const PoolConfig& config)
	: _totalThreads(config.totalThreads) {
	if (!config.valid()) {
		throw std::invalid_argument("ThreadPool: config.totalThreads must be > 0");
	}

	// 启动工作线程
	_workers.reserve(_totalThreads);
	for (size_t i = 0; i < _totalThreads; ++i) {
		_workers.emplace_back(&ThreadPool::_workerLoop, this);
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
	_running = false;

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
