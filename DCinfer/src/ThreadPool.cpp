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

ThreadPool::ThreadPool(const PoolConfig& config, std::shared_ptr<GroupSemaphoreRegistry> sharedGroups)
	: _config(config), _totalThreads(config.totalThreads), _sharedGroups(std::move(sharedGroups)) {
	if (!config.valid()) {
		throw std::invalid_argument("ThreadPool: config.totalThreads must be > 0");
	}

	// 独立使用时自建共享表（默认构造 / 未注入场景）
	if (!_sharedGroups) {
		_sharedGroups = std::make_shared<GroupSemaphoreRegistry>();
	}

	// 初始化组信号量（写入共享表）
	for (const auto& [tag, limit] : config.groupLimits) {
		_sharedGroups->setLimit(tag, limit);
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

void ThreadPool::registerGroupLimit(const std::string& tag, size_t limit) {
	// 语义升级：注册到跨池共享表，对共享该表的所有线程池同时生效
	_sharedGroups->setLimit(tag, limit);
}

void ThreadPool::shutdown() {
	_running = false;
	_shuttingDown = true;

	{
		std::lock_guard lk(_mutex);
		// 丢弃所有等待中的任务（含 submitAsync 挂起的协程句柄）。
		// 注意：不 resume 挂起协程——析构期间 ExecutionEngine 的状态成员
		// （_terminatedTasks/_watchdogs 等）可能已销毁，恢复协程会访问悬空对象。
		// 挂起的协程帧随之泄漏（进程退出时由 OS 回收），这是关闭期的安全取舍。
		_taskQueue = {};
	}
	_cv.notify_all();

	for (auto& t : _workers) {
		if (t.joinable())
			t.join();
	}
	_workers.clear();
}

bool ThreadPool::_tryAcquireGroup(const std::string& tag) {
	if (tag.empty())
		return true; // 无分组，不限流

	auto sem = _sharedGroups->find(tag);
	if (!sem)
		return true; // 未配置的组不限流

	return sem->try_acquire();
}

void ThreadPool::_releaseGroup(const std::string& tag) {
	if (tag.empty())
		return;

	auto sem = _sharedGroups->find(tag);
	if (sem) {
		sem->release();
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
