// TimerService 单元测试：登记触发顺序 / 作废仲裁 / 析构安全
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "Graph/internal/TimerService.h"

using namespace DC;
using Clock = std::chrono::steady_clock;

static void expect(bool cond, const std::string& msg) {
	if (!cond)
		throw std::runtime_error(msg);
}

static Clock::time_point deadlineIn(int ms) {
	return Clock::now() + std::chrono::milliseconds(ms);
}

static void runTests() {
	// ── Test 1: 乱序 deadline 依序触发，每条目恰好一次 ──
	// 回调全部在定时器线程上串行执行，触发顺序即 deadline 顺序
	{
		TimerService timer;
		std::mutex mtx;
		std::vector<std::string> order;

		timer.schedule(deadlineIn(150), [&] { std::lock_guard lk(mtx); order.push_back("A"); });
		timer.schedule(deadlineIn(50), [&] { std::lock_guard lk(mtx); order.push_back("B"); });
		timer.schedule(deadlineIn(100), [&] { std::lock_guard lk(mtx); order.push_back("C"); });

		std::this_thread::sleep_for(std::chrono::milliseconds(400));
		std::lock_guard lk(mtx);
		expect(order.size() == 3, "all three entries should fire exactly once");
		expect(order[0] == "B" && order[1] == "C" && order[2] == "A",
			   "entries should fire in deadline order (B, C, A)");
	}
	std::cout << "Test 1 passed: out-of-order deadlines fire in order" << std::endl;

	// ── Test 2: cancel 作废存活条目，不触发回调 ──
	{
		TimerService timer;
		std::atomic<int> fired{0};
		auto h = timer.schedule(deadlineIn(80), [&] { ++fired; });

		expect(timer.cancel(h), "cancel of a live entry should return true");
		expect(!timer.cancel(h), "second cancel of the same handle should return false");

		std::this_thread::sleep_for(std::chrono::milliseconds(250));
		expect(fired.load() == 0, "cancelled entry must not fire");
	}
	std::cout << "Test 2 passed: cancelled entry never fires" << std::endl;

	// ── Test 3: 触发后 cancel 返回 false（live → fired 认领不可逆）──
	{
		TimerService timer;
		std::atomic<int> fired{0};
		auto h = timer.schedule(deadlineIn(50), [&] { ++fired; });

		std::this_thread::sleep_for(std::chrono::milliseconds(250));
		expect(fired.load() == 1, "entry should fire exactly once");
		expect(!timer.cancel(h), "cancel after fire should return false");
	}
	std::cout << "Test 3 passed: fired handle can no longer be cancelled" << std::endl;

	// ── Test 4: 同刻多条目批处理，逐条触发且互不吞并 ──
	{
		TimerService timer;
		std::atomic<int> fired{0};
		auto t = deadlineIn(60);
		for (int i = 0; i < 8; ++i)
			timer.schedule(t, [&] { ++fired; });

		std::this_thread::sleep_for(std::chrono::milliseconds(300));
		expect(fired.load() == 8, "all same-deadline entries should fire exactly once each");
	}
	std::cout << "Test 4 passed: same-deadline batch fires every entry" << std::endl;

	// ── Test 5: 析构时挂起条目不触发，且 stop 即时返回 ──
	{
		std::atomic<int> fired{0};
		auto start = Clock::now();
		{
			TimerService timer;
			timer.schedule(deadlineIn(30'000), [&] { ++fired; });
		} // 析构：请求停止 + join，挂起条目的回调不再触发
		auto elapsed = Clock::now() - start;

		expect(fired.load() == 0, "pending entry must not fire after destruction");
		expect(elapsed < std::chrono::seconds(5),
			   "destructor should stop the timer thread promptly (long deadline pending)");
	}
	std::cout << "Test 5 passed: destruction stops timer promptly without firing" << std::endl;

	// ── Test 6: 未知句柄 cancel 安全返回 false ──
	{
		TimerService timer;
		expect(!timer.cancel(12345), "cancel of unknown handle should return false");
	}
	std::cout << "Test 6 passed: unknown handle cancel is safe" << std::endl;

	// ── Test 7: 空闲后重新登记仍可触发（等待/唤醒路径回归）──
	{
		TimerService timer;
		std::atomic<int> fired{0};

		std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 队列空转一段时间
		timer.schedule(deadlineIn(60), [&] { ++fired; });
		std::this_thread::sleep_for(std::chrono::milliseconds(250));
		expect(fired.load() == 1, "entry scheduled after idle period should fire");
	}
	std::cout << "Test 7 passed: scheduling after idle wakes the timer" << std::endl;

	std::cout << "\nAll TimerService tests passed!" << std::endl;
}

int main() {
	try {
		runTests();
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Test failure: " << e.what() << std::endl;
		return -1;
	}
}
