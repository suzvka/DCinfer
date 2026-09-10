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

void ThreadPool::submit(std::function<void()> task) {
	{
		std::lock_guard lk(_mutex);
		_taskQueue.push(std::move(task));
	}
	_cv.notify_one();
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
			if (_taskQueue.empty())
				continue;

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
