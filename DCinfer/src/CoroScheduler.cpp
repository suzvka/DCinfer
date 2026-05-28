#include "CoroScheduler.h"

#include <iostream>

namespace DC {

CoroScheduler::CoroScheduler(size_t numThreads) {
	_workers.reserve(numThreads);
	for (size_t i = 0; i < numThreads; ++i) {
		_workers.emplace_back(&CoroScheduler::_workerLoop, this);
	}
}

CoroScheduler::~CoroScheduler() {
	shutdown();
}

void CoroScheduler::enqueue(std::coroutine_handle<> h) {
	{
		std::lock_guard lk(_mutex);
		_readyQueue.push(h);
	}
	_cv.notify_one();
}

void CoroScheduler::run() {
	_workerLoop();
}

void CoroScheduler::shutdown() {
	_running = false;
	_cv.notify_all();

	for (auto& t : _workers) {
		if (t.joinable())
			t.join();
	}
	_workers.clear();
}

bool CoroScheduler::empty() const {
	return _activeCoroutines.load(std::memory_order_acquire) == 0;
}

void CoroScheduler::_workerLoop() {
	while (_running.load(std::memory_order_acquire)) {
		std::coroutine_handle<> h;

		{
			std::unique_lock lk(_mutex);
			_cv.wait(lk, [this] { return !_readyQueue.empty() || !_running.load(std::memory_order_acquire); });

			if (!_running.load(std::memory_order_acquire))
				break;

			if (_readyQueue.empty())
				continue;

			h = _readyQueue.front();
			_readyQueue.pop();
		}

		if (!h || h.done())
			continue;

		_activeCoroutines.fetch_add(1, std::memory_order_release);
		try {
			h.resume();
		} catch (const std::exception& e) {
			std::cerr << "CoroScheduler: exception in coroutine: " << e.what() << std::endl;
		} catch (...) {
			std::cerr << "CoroScheduler: unknown exception in coroutine" << std::endl;
		}
		_activeCoroutines.fetch_sub(1, std::memory_order_release);

		// 协程完成后销毁 handle
		if (h.done()) {
			h.destroy();
		}
	}
}

} // namespace DC
