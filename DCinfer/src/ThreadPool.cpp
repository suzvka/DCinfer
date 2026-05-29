#include "ThreadPool.h"

#include <iostream>
#include <stdexcept>

namespace DC {

// ── PoolTicket ──

PoolTicket::PoolTicket(ThreadPool& pool, std::string tag, std::function<void()> task)
	: _pool(&pool), _tag(std::move(tag)), _task(std::move(task)) {}

void PoolTicket::await_suspend(std::coroutine_handle<> h) {
	_handle = h;

	std::lock_guard lk(_pool->_mutex);
	_pool->_taskQueue.push(ThreadPool::PendingTask{std::move(_task), h, _tag});
	_pool->_cv.notify_one();
}

// ── ThreadPool ──

ThreadPool::ThreadPool(const PoolConfig& config) : _config(config), _totalThreads(config.totalThreads) {
	if (!config.valid()) {
		throw std::invalid_argument("ThreadPool: config.totalThreads must be > 0");
	}

	// 初始化组信号量
	for (const auto& [tag, limit] : config.groupLimits) {
		_groupSemaphores[tag] = std::make_unique<std::counting_semaphore<>>(static_cast<std::ptrdiff_t>(limit));
	}

	// 全局信号量
	_globalSemaphore = std::make_unique<std::counting_semaphore<>>(static_cast<std::ptrdiff_t>(_totalThreads));

	// 启动工作线程
	_workers.reserve(_totalThreads);
	for (size_t i = 0; i < _totalThreads; ++i) {
		_workers.emplace_back(&ThreadPool::_workerLoop, this);
	}
}

ThreadPool::~ThreadPool() {
	shutdown();
}

void ThreadPool::submit(const std::string& nodeTag, std::function<void()> task) {
	std::lock_guard lk(_mutex);
	_taskQueue.push(ThreadPool::PendingTask{std::move(task), {}, nodeTag});
	_cv.notify_one();
}

PoolTicket ThreadPool::submitAsync(const std::string& nodeTag, std::function<void()> task) {
	return PoolTicket(*this, nodeTag, std::move(task));
}

size_t ThreadPool::activeCount(const std::string& groupTag) const {
	std::lock_guard lk(const_cast<ThreadPool*>(this)->_activeCountMutex);
	auto it = _groupActiveCount.find(groupTag);
	if (it == _groupActiveCount.end())
		return 0;
	return it->second->load(std::memory_order_acquire);
}

void ThreadPool::shutdown() {
	_running = false;
	_shuttingDown = true;

	std::vector<std::coroutine_handle<>> pendingHandles;
	{
		std::lock_guard lk(_mutex);
		// 收集所有等待中的协程句柄，以便在锁外 resume
		while (!_taskQueue.empty()) {
			auto& task = _taskQueue.front();
			if (task.handle) {
				pendingHandles.push_back(task.handle);
			}
			_taskQueue.pop();
		}
	}
	_cv.notify_all();

	// 在锁外批量 resume，避免协程恢复后可能的死锁
	for (auto h : pendingHandles) {
		if (h)
			h.resume();
	}

	for (auto& t : _workers) {
		if (t.joinable())
			t.join();
	}
	_workers.clear();
}

bool ThreadPool::_tryAcquireGroup(const std::string& tag) {
	if (tag.empty())
		return true; // 无分组，不限流

	std::lock_guard lk(_groupMutex);
	auto it = _groupSemaphores.find(tag);
	if (it == _groupSemaphores.end())
		return true; // 未配置的组不限流

	return it->second->try_acquire();
}

void ThreadPool::_releaseGroup(const std::string& tag) {
	if (tag.empty())
		return;

	std::lock_guard lk(_groupMutex);
	auto it = _groupSemaphores.find(tag);
	if (it != _groupSemaphores.end()) {
		it->second->release();
	}
}

void ThreadPool::_workerLoop() {
	while (_running.load(std::memory_order_acquire)) {
		ThreadPool::PendingTask pending;

		{
			std::unique_lock lk(_mutex);
			_cv.wait(lk, [this] { return !_taskQueue.empty() || !_running.load(std::memory_order_acquire); });

			if (!_running.load(std::memory_order_acquire))
				break;
			if (_taskQueue.empty())
				continue;

			// 遍历队列，找到可以获取到信号量的任务
			bool found = false;
			size_t queueSize = _taskQueue.size();
			for (size_t i = 0; i < queueSize; ++i) {
				auto& front = _taskQueue.front();
				if (_tryAcquireGroup(front.groupTag)) {
					// 尝试获取全局信号量
					if (_globalSemaphore->try_acquire()) {
						pending = std::move(front);
						_taskQueue.pop();
						found = true;
						break;
					} else {
						// 全局信号量不足，释放组信号量
						_releaseGroup(front.groupTag);
					}
				}
				// 移到队尾重试
				auto temp = std::move(front);
				_taskQueue.pop();
				_taskQueue.push(std::move(temp));
			}

			if (!found)
				continue;
		}

		// 递增活跃计数
		_incrementActive(pending.groupTag);

		// 执行任务
		try {
			pending.task();
		} catch (const std::exception& e) {
			std::cerr << "ThreadPool: exception in task: " << e.what() << std::endl;
		} catch (...) {
			std::cerr << "ThreadPool: unknown exception in task" << std::endl;
		}

		// 递减活跃计数
		_decrementActive(pending.groupTag);

		// 释放信号量
		_globalSemaphore->release();
		_releaseGroup(pending.groupTag);

		// 通知等待的协程
		{
			std::lock_guard lk(_mutex);
			_cv.notify_one(); // 通知其他工作线程可能有新槽位
		}

		// resume 协程句柄
		if (pending.handle) {
			pending.handle.resume();
			if (pending.handle.done()) {
				pending.handle.destroy();
			}
		}
	}
}

// ── 分组活跃计数 ──

void ThreadPool::_incrementActive(const std::string& tag) {
	if (tag.empty())
		return;

	std::lock_guard lk(_activeCountMutex);
	auto it = _groupActiveCount.find(tag);
	if (it == _groupActiveCount.end()) {
		it = _groupActiveCount.emplace(tag, std::make_unique<std::atomic<size_t>>(0)).first;
	}
	it->second->fetch_add(1, std::memory_order_release);
}

void ThreadPool::_decrementActive(const std::string& tag) {
	if (tag.empty())
		return;

	std::lock_guard lk(_activeCountMutex);
	auto it = _groupActiveCount.find(tag);
	if (it != _groupActiveCount.end()) {
		it->second->fetch_sub(1, std::memory_order_release);
	}
}

} // namespace DC
