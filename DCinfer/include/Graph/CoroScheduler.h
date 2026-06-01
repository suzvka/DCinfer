#pragma once

#include <atomic>
#include <condition_variable>
#include <coroutine>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <utility>

namespace DC {

// ── 协程 Task<T>：惰性启动、move-only 的协程类型 ──
template <typename T = void>
struct Task {
	struct promise_type {
		std::coroutine_handle<> _continuation;
		std::exception_ptr _error;
		T _value{};
		bool _ready = false;

		Task get_return_object() {
			return Task(std::coroutine_handle<promise_type>::from_promise(*this));
		}

		std::suspend_always initial_suspend() noexcept {
			return {};
		}
		std::suspend_always final_suspend() noexcept {
			_ready = true;
			if (_continuation)
				_continuation.resume();
			return {};
		}

		void unhandled_exception() {
			_error = std::current_exception();
			_ready = true;
		}

		void return_value(T v) {
			_value = std::move(v);
			_ready = true;
		}

		T result() {
			if (_error)
				std::rethrow_exception(_error);
			return std::move(_value);
		}
	};

	using handle_type = std::coroutine_handle<promise_type>;

	Task() : _handle(nullptr) {}
	explicit Task(handle_type h) : _handle(h) {}

	Task(const Task&) = delete;
	Task& operator=(const Task&) = delete;

	Task(Task&& other) noexcept : _handle(std::exchange(other._handle, nullptr)) {}
	Task& operator=(Task&& other) noexcept {
		if (this != &other) {
			if (_handle)
				_handle.destroy();
			_handle = std::exchange(other._handle, nullptr);
		}
		return *this;
	}

	~Task() {
		if (_handle && _handle.done())
			_handle.destroy();
	}

	bool await_ready() const noexcept {
		return false;
	}

	template <typename U>
	void await_suspend(std::coroutine_handle<U> continuation) noexcept {
		_handle.promise()._continuation = continuation;
	}

	auto await_resume() {
		return _handle.promise().result();
	}

	handle_type handle() const {
		return _handle;
	}

private:
	handle_type _handle;
};

// void 特化
template <>
struct Task<void> {
	struct promise_type {
		std::coroutine_handle<> _continuation;
		std::exception_ptr _error;
		bool _ready = false;

		Task get_return_object() {
			return Task(std::coroutine_handle<promise_type>::from_promise(*this));
		}

		std::suspend_always initial_suspend() noexcept {
			return {};
		}
		std::suspend_always final_suspend() noexcept {
			_ready = true;
			if (_continuation)
				_continuation.resume();
			return {};
		}

		void unhandled_exception() {
			_error = std::current_exception();
			_ready = true;
		}

		void return_void() {
			_ready = true;
		}

		void result() {
			if (_error)
				std::rethrow_exception(_error);
		}
	};

	using handle_type = std::coroutine_handle<promise_type>;

	Task() : _handle(nullptr) {}
	explicit Task(handle_type h) : _handle(h) {}

	Task(const Task&) = delete;
	Task& operator=(const Task&) = delete;

	Task(Task&& other) noexcept : _handle(std::exchange(other._handle, nullptr)) {}
	Task& operator=(Task&& other) noexcept {
		if (this != &other) {
			if (_handle)
				_handle.destroy();
			_handle = std::exchange(other._handle, nullptr);
		}
		return *this;
	}

	~Task() {
		if (_handle && _handle.done())
			_handle.destroy();
	}

	bool await_ready() const noexcept {
		return false;
	}

	template <typename U>
	void await_suspend(std::coroutine_handle<U> continuation) noexcept {
		_handle.promise()._continuation = continuation;
	}

	void await_resume() {
		_handle.promise().result();
	}

	handle_type handle() const {
		return _handle;
	}

private:
	handle_type _handle;
};

// ── 协程调度器：管理就绪协程队列，由系统线程驱动 ──
class CoroScheduler {
public:
	explicit CoroScheduler(size_t numThreads = 2);
	~CoroScheduler();

	CoroScheduler(const CoroScheduler&) = delete;
	CoroScheduler& operator=(const CoroScheduler&) = delete;

	/// @brief  生成一个协程并立即调度
	/// @param  返回 Task<void> 的工厂函数
	template <typename F>
	void spawn(F&& factory) {
		auto task = factory();
		auto h = task.handle();

		_activeCoroutines.fetch_add(1, std::memory_order_release);
		{
			std::lock_guard lk(_mutex);
			_readyQueue.push(h);
		}
		_cv.notify_one();
	}

	/// @brief  直接调度一个已创建的 Task<void>（移动语义）
	///         用于避免双重协程包装。Task 的协程帧直接入队调度。
	void spawnTask(Task<void> task) {
		auto h = task.handle();
		_activeCoroutines.fetch_add(1, std::memory_order_release);
		{
			std::lock_guard lk(_mutex);
			_readyQueue.push(h);
		}
		_cv.notify_one();
	}

	/// @brief  将协程 handle 重新入队（供 awaitable 内部使用）
	void enqueue(std::coroutine_handle<> h);

	/// @brief  系统线程主循环：阻塞当前线程，直到所有协程完成
	void run();

	/// @brief  优雅关闭：等待所有飞行中的协程完成
	void shutdown();

	/// @brief  是否所有协程已完成
	bool empty() const;

private:
	void _workerLoop();

	std::mutex _mutex;
	std::condition_variable _cv;
	std::queue<std::coroutine_handle<>> _readyQueue;

	std::atomic<bool> _running{true};
	std::atomic<size_t> _activeCoroutines{0};
	std::vector<std::thread> _workers;
};

} // namespace DC
