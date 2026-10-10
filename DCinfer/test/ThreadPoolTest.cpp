// ThreadPool 单元测试：submit 返回值语义与 shutdown 并发契约
#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <system_error>
#include <thread>

#include "ThreadPool.h"

using namespace DC;
using namespace std::chrono_literals;

static int failures = 0;

#define CHECK(cond, msg)                                                                                               \
	do {                                                                                                               \
		if (!(cond)) {                                                                                                 \
			std::cerr << "FAIL: " << msg << std::endl;                                                                 \
			++failures;                                                                                                \
			return;                                                                                                    \
		}                                                                                                              \
	} while (0)

#define TEST(name)                                                                                                     \
	std::cout << "Test: " << name << " ... " << std::flush;                                                            \
	[&]()
#define END_TEST()                                                                                                     \
	();                                                                                                                \
	std::cout << "PASSED" << std::endl

static void testSubmitExecutes() {
	TEST("submit during running -> true and task executes") {
		ThreadPool pool({2});
		std::atomic<int> ran{0};
		CHECK(pool.submit([&] { ++ran; }), "submit must return true while running");
		CHECK(pool.submit([&] { ++ran; }), "submit must return true while running");
		const auto deadline = std::chrono::steady_clock::now() + 2s;
		while (ran.load() < 2 && std::chrono::steady_clock::now() < deadline)
			std::this_thread::sleep_for(1ms);
		CHECK(ran.load() == 2, "both submitted tasks must execute");
	}
	END_TEST();
}

static void testSubmitAfterShutdownRejected() {
	TEST("submit after shutdown -> false, task never executes") {
		ThreadPool pool({1});
		pool.shutdown();
		std::atomic<bool> ran{false};
		CHECK(!pool.submit([&] { ran.store(true); }), "submit after shutdown must be rejected");
		std::this_thread::sleep_for(50ms);
		CHECK(!ran.load(), "rejected task must never execute");

		pool.shutdown();
	}
	END_TEST();
}

static void testConcurrentSubmitDuringShutdown() {
	TEST("concurrent submit x shutdown -> safe; post-shutdown submit rejected") {
		ThreadPool pool({2});
		std::atomic<bool> stop{false};
		std::atomic<int> accepted{0};
		std::thread submitter([&] {
			while (!stop.load()) {
				if (pool.submit([] {}))
					++accepted;
			}
		});
		std::this_thread::sleep_for(50ms);
		pool.shutdown();
		stop.store(true);
		submitter.join();

		CHECK(!pool.submit([] {}), "submit after shutdown returns must be rejected");
		CHECK(accepted.load() > 0, "some submits should be accepted before shutdown");
	}
	END_TEST();
}

// 测试注入入口：friend 声明于 DC::ThreadPool，故定义在 DC 命名空间内
namespace DC {
struct ThreadPoolSpawnProbe {
	static void set(std::function<bool(size_t)> f) { ThreadPool::s_spawnFilter = std::move(f); }
};
} // namespace DC

static void testConstructorPartialFailureRecovers() {
	TEST("thread creation failure mid-construction -> propagates exception, no terminate") {
		// 前 2 个成功，第 3 个注入失败
		ThreadPoolSpawnProbe::set([](size_t i) { return i < 2; });
		bool threw = false;
		try {
			ThreadPool pool({4});
			(void)pool;
		} catch (...) {
			threw = true;
		}
		ThreadPoolSpawnProbe::set(nullptr);
		CHECK(threw, "partial thread creation failure must propagate as exception");
		// 进程存活即主要断言：未 terminate、未挂死
	}
	END_TEST();
}

static void testConstructionAfterInjectionRecovers() {
	TEST("pool works normally after injection-cleared construction") {
		ThreadPoolSpawnProbe::set([](size_t) { return true; });
		ThreadPoolSpawnProbe::set(nullptr);
		ThreadPool pool({2});
		std::atomic<int> ran{0};
		CHECK(pool.submit([&] { ++ran; }), "submit must work after construction");
		const auto deadline = std::chrono::steady_clock::now() + 2s;
		while (ran.load() < 1 && std::chrono::steady_clock::now() < deadline)
			std::this_thread::sleep_for(1ms);
		CHECK(ran.load() == 1, "task must execute after recovered construction");
	}
	END_TEST();
}

static void testConcurrentShutdowns() {
TEST("concurrent shutdown x8 -> no crash, serial drain, post-shutdown submit rejected") {
		ThreadPool pool({4});
		std::atomic<int> ran{0};
		CHECK(pool.submit([&] { ++ran; }), "submit while running");
		std::vector<std::thread> shutdowners;
		for (int i = 0; i < 8; ++i)
			shutdowners.emplace_back([&pool] { pool.shutdown(); });
		for (auto& t : shutdowners)
			t.join();
		pool.shutdown(); // 收尾幂等
		CHECK(ran.load() <= 1, "at most the queued task may execute");
		CHECK(!pool.submit([] {}), "post-shutdown submit must be rejected");
	}
END_TEST();
}

int main() {
	try {
		testSubmitExecutes();
		testSubmitAfterShutdownRejected();
		testConcurrentSubmitDuringShutdown();
		testConstructorPartialFailureRecovers();
		testConstructionAfterInjectionRecovers();
		testConcurrentShutdowns();
	} catch (const std::exception& e) {
		std::cerr << "UNEXPECTED EXCEPTION: " << e.what() << std::endl;
		return 1;
	}

	if (failures != 0) {
		std::cerr << failures << " test(s) failed" << std::endl;
		return 1;
	}
	std::cout << "All thread pool tests passed" << std::endl;
	return 0;
}
