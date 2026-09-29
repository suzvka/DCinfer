// ResourceScheduler 单元测试（进程级共享调度器契约）
//
//   覆盖：
//   - 预算上限：1 worker 串行化互斥任务（峰值并发 == 1）；2 workers 峰值 == 2
//   - 资源类隔离：Compute 槽位被阻塞不影响 Operator 任务完成
//   - 配置校验：非法预算（0 worker）构造抛 std::invalid_argument
//   - 关闭拒绝：shutdown 后 submit 拒绝、isStopped 生效、幂等
//   - 全局实例：懒创建幂等 / configureInstance 仅首配生效 / resetInstance 恢复可配
//   - 引擎排水：慢节点在飞时析构图——引擎析构等待任务完成后返回
#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "InferGraph.h"
#include "ResourceScheduler.h"
#include "Tensor.hpp"

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

// ── 辅助 ──

static Tensor floatTensor(float value) {
	auto t = Tensor::Create<float>();
	t = value;
	return t;
}

static Node::Schema passSchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("x")};
	s.outputs = {Node::Port::out<float>("y")};
	return s;
}

/// @brief 轮询等待原子量到达目标值（超时返回 false）。
template <typename T>
static bool waitForValue(const std::atomic<T>& atom, T target, std::chrono::milliseconds timeout) {
	const auto deadline = std::chrono::steady_clock::now() + timeout;
	while (atom.load() != target && std::chrono::steady_clock::now() < deadline)
		std::this_thread::sleep_for(1ms);
	return atom.load() == target;
}

// ── 1. 预算上限：单 worker 串行 ──

static void testSerialLimit() {
	TEST("budget: single worker serializes tasks (peak concurrency == 1)") {
		ResourceScheduler sched(SchedulerConfig{1, 1, 1});

		std::atomic<int> running{0};
		std::atomic<int> peak{0};
		std::atomic<int> done{0};

		auto task = [&] {
			const int now = running.fetch_add(1, std::memory_order_acq_rel) + 1;
			int prev = peak.load(std::memory_order_relaxed);
			while (now > prev && !peak.compare_exchange_weak(prev, now, std::memory_order_acq_rel)) {
			}
			std::this_thread::sleep_for(30ms);
			running.fetch_sub(1, std::memory_order_acq_rel);
			done.fetch_add(1, std::memory_order_acq_rel);
		};

		bool accepted = true;
		for (int i = 0; i < 3; ++i)
			accepted = sched.submit(ResourceClass::Operator, task) && accepted;
		CHECK(accepted, "all three submits must be accepted");

		CHECK(waitForValue(done, 3, 5s), "all three tasks must complete");
		CHECK(peak.load() == 1, "peak concurrency must be 1 under a single-worker budget");
	}
	END_TEST();
}

// ── 2. 预算上限：双 worker 峰值饱和 ──

static void testConcurrentLimit() {
	TEST("budget: two workers saturate at peak concurrency == 2") {
		ResourceScheduler sched(SchedulerConfig{2, 2, 2});

		std::atomic<int> running{0};
		std::atomic<int> peak{0};
		std::atomic<int> done{0};

		auto task = [&] {
			const int now = running.fetch_add(1, std::memory_order_acq_rel) + 1;
			int prev = peak.load(std::memory_order_relaxed);
			while (now > prev && !peak.compare_exchange_weak(prev, now, std::memory_order_acq_rel)) {
			}
			std::this_thread::sleep_for(50ms); // 窗口足够长：两 worker 必然同拍
			running.fetch_sub(1, std::memory_order_acq_rel);
			done.fetch_add(1, std::memory_order_acq_rel);
		};

		bool accepted = true;
		for (int i = 0; i < 4; ++i)
			accepted = sched.submit(ResourceClass::Operator, task) && accepted;
		CHECK(accepted, "all four submits must be accepted");

		CHECK(waitForValue(done, 4, 5s), "all four tasks must complete");
		CHECK(peak.load() == 2, "peak concurrency must equal the configured budget (2)");
	}
	END_TEST();
}

// ── 3. 资源类隔离：Compute 阻塞不拖累 Operator ──

static void testClassIsolation() {
	TEST("isolation: blocked Compute slot must not stall Operator class") {
		ResourceScheduler sched(SchedulerConfig{1, 1, 1});

		std::atomic<bool> computeEntered{false};
		std::atomic<bool> releaseCompute{false};
		std::atomic<bool> operatorDone{false};

		// 同步用 atomic 轮询而非 promise/future：worker 等待有界（最长 ~5s 自愈退出），
		// 任何断言失败路径都不会令 sched 析构的 join 无限阻塞（挂起不再是可能结局）。
		const bool acceptedCompute = sched.submit(ResourceClass::Compute, [&] {
			computeEntered.store(true);
			const auto deadline = std::chrono::steady_clock::now() + 5s;
			while (!releaseCompute.load() && std::chrono::steady_clock::now() < deadline)
				std::this_thread::sleep_for(1ms);
		});
		const bool acceptedOperator = sched.submit(ResourceClass::Operator, [&] { operatorDone.store(true); });
		CHECK(acceptedCompute && acceptedOperator, "both submits must be accepted");

		// 确认 Compute 槽位已被占住（阻塞中）
		CHECK(waitForValue(computeEntered, true, 5s), "compute task must enter its blocking section");
		// Operator 类独立完成，不受 Compute 阻塞影响
		CHECK(waitForValue(operatorDone, true, 5s), "Operator task must complete while Compute slot is blocked");

		releaseCompute.store(true); // 解围（即使在此前断言失败，worker 也会在自身超时后退出）
	}
	END_TEST();
}

// ── 4. 配置校验：非法预算拒绝 ──

static void testConfigValidation() {
	TEST("config: zero-worker budget must be rejected with std::invalid_argument") {
		bool threw = false;
		try {
			ResourceScheduler bad(SchedulerConfig{0, 1, 1});
			static_cast<void>(bad);
		} catch (const std::invalid_argument&) {
			threw = true;
		}
		CHECK(threw, "constructing with computeWorkers == 0 must throw std::invalid_argument");
	}
	END_TEST();
}

// ── 5. 关闭拒绝：shutdown 后 submit 拒绝且幂等 ──

static void testShutdownRejection() {
	TEST("shutdown: submit rejected after shutdown, isStopped set, idempotent") {
		ResourceScheduler sched(SchedulerConfig{1, 1, 1});
		CHECK(!sched.isStopped(), "fresh scheduler must not be stopped");

		sched.shutdown();
		CHECK(sched.isStopped(), "isStopped must be true after shutdown");
		CHECK(!sched.submit(ResourceClass::System, [] {}), "submit after shutdown must be rejected");

		sched.shutdown(); // 幂等：重复关闭不崩溃
		CHECK(sched.isStopped(), "shutdown must be idempotent");
		CHECK(!sched.submit(ResourceClass::Operator, [] {}), "submit must stay rejected after repeated shutdown");
	}
	END_TEST();
}

// ── 6. 全局实例语义：懒创建 / 首配 / 重置 ──

static void testGlobalInstanceSemantics() {
	TEST("global instance: lazily created, pre-configurable once, resettable") {
		auto a = ResourceScheduler::instance();
		auto b = ResourceScheduler::instance();
		CHECK(a != nullptr && a == b, "instance() must be idempotent (same pointer)");
		CHECK(a->workersFor(ResourceClass::Compute) == 1 && a->workersFor(ResourceClass::Operator) == 1 &&
				  a->workersFor(ResourceClass::System) == 4,
			  "default instance budget must be 1/1/4 (System carries connectors + waiting nodes)");

		CHECK(!ResourceScheduler::configureInstance(SchedulerConfig{3, 2, 1}),
			  "configureInstance after lazy creation must be rejected");

		ResourceScheduler::resetInstance();
		CHECK(ResourceScheduler::configureInstance(SchedulerConfig{3, 2, 1}),
			  "configureInstance must be accepted after resetInstance");
		auto c = ResourceScheduler::instance();
		CHECK(c != a, "instance() after reset must create a fresh scheduler");
		CHECK(c->workersFor(ResourceClass::Compute) == 3 && c->workersFor(ResourceClass::Operator) == 2 &&
				  c->workersFor(ResourceClass::System) == 1,
			  "pre-configured budget must be honored");

		// 恢复进程干净状态：重置具备幂等性且恢复可配置
		ResourceScheduler::resetInstance();
		ResourceScheduler::resetInstance();
		CHECK(ResourceScheduler::configureInstance(SchedulerConfig{}), "reset must restore configurability");
		ResourceScheduler::resetInstance();
	}
	END_TEST();
}

// ── 7. 引擎排水：在飞慢任务完成前析构不得返回 ──

static void testEngineDestructorDrainsInflight() {
	TEST("drain: engine destructor waits for in-flight slow task before returning") {
		auto sched = std::make_shared<ResourceScheduler>(SchedulerConfig{2, 2, 2});
		std::atomic<bool> started{false};
		std::atomic<bool> finished{false};
		const auto begin = std::chrono::steady_clock::now();
		{
			// 慢节点任务在飞时析构图：引擎析构必须自排水（等待已派发 lambda 完成），
			// 否则共享调度器上的飞行任务会回访已销毁引擎
			InferGraph g(sched);
			g.addNode(std::make_unique<Node>("test", "slow", passSchema(),
				[&](Node::RunContext& ctx) -> Node::Result {
					started.store(true);
					std::this_thread::sleep_for(300ms);
					const auto* x = ctx.input<Tensor>("x");
					if (!x)
						return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
					ctx.output("y", Value(std::make_unique<Tensor>(*x)));
					finished.store(true);
					return ctx.success();
				}));
			g.bindOutput("y", "slow", "y");
			g.feedInput("t1", "slow", "x", Value(std::make_unique<Tensor>(floatTensor(1.0f))));
			g.submit("t1", "slow", "y");

			CHECK(waitForValue(started, true, 5s), "slow task must enter RunFn before graph teardown");
		}
		const auto elapsed = std::chrono::steady_clock::now() - begin;
		CHECK(finished.load(), "in-flight task must complete before engine destructor returns");
		CHECK(elapsed >= 300ms, "destructor must have waited for the slow task (elapsed >= sleep duration)");
	}
	END_TEST();
}

// ── 8. shutdown 死锁回归：worker 任务内递归提交 + 并发 shutdown ──
//
// 回归场景：worker 正在执行的任务 lambda 再次提交（经 _poolFor），若已穿过
// 第一道 _stopped 快检后遇 shutdown 持 _initMutex join——修复前为循环等待。
// 压力循环 + shared_future 看门狗：shutdown 未在时限内返回即判定失败。

static void testShutdownRacesRecursiveSubmit() {
	TEST("shutdown: concurrent shutdown x recursive submit must not deadlock") {
		for (int iter = 0; iter < 100; ++iter) {
			auto sched = std::make_unique<ResourceScheduler>(SchedulerConfig{2, 2, 2});
			// 放入一个持续经 _poolFor 递归提交的任务，然后立即并发 shutdown
			sched->submit(ResourceClass::Operator, [&] {
				for (int i = 0; i < 8; ++i) {
					sched->submit(ResourceClass::Operator, [] {});
					sched->submit(ResourceClass::Compute, [] {});
					std::this_thread::sleep_for(1ms);
				}
			});
			// shared_future 析构不阻塞：死锁分支下不拖死测试线程
			auto watchdog = std::async(std::launch::async, [&] { sched->shutdown(); }).share();
			if (watchdog.wait_for(30s) != std::future_status::ready) {
				std::cerr << "FAIL: shutdown deadlocked with recursive submit (iteration " << iter << ")"
						  << std::endl;
				++failures;
				sched.release(); // 故意泄漏：避免析构再次进入死锁的 shutdown
				return;
			}
		}
	}
	END_TEST();
}

// ── 9. 析构与调度器 shutdown 并发：排水与关停互相竞速仍必终止 ──
//
// 覆盖两条路径：
//   a) 引擎自排水（DrainTicket 递减）与在飞任务完成路径竞速；
//   b) 调度器 shutdown 弃置排队任务（票据随 function 析构回收）+ join 在飞。
// 任一侧挂起即失败（shared_future 看门狗 + 故意泄漏避免二次挂死）。

static void testTeardownRacingSchedulerShutdown() {
	TEST("teardown: engine drain racing scheduler shutdown must terminate") {
		for (int iter = 0; iter < 40; ++iter) {
			auto sched = std::make_unique<ResourceScheduler>(SchedulerConfig{2, 2, 2});
			std::atomic<bool> started{false};
			// 不拥有所有权的 shared_ptr 视图（InferGraph 接口要求 shared_ptr；
			// 生命期由 sched unique_ptr 独立管理，失败分支下二者同步泄漏）
			std::shared_ptr<ResourceScheduler> schedView(sched.get(), [](ResourceScheduler*) {});
			auto g = std::make_unique<InferGraph>(schedView);
			g->addNode(std::make_unique<Node>("test", "slow", passSchema(),
				[&](Node::RunContext& ctx) -> Node::Result {
					started.store(true);
					std::this_thread::sleep_for(2ms);
					const auto* x = ctx.input<Tensor>("x");
					if (!x)
						return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
					ctx.output("y", Value(std::make_unique<Tensor>(*x)));
					return ctx.success();
				}));
			g->bindOutput("y", "slow", "y");
			for (int t = 0; t < 4; ++t) {
				const std::string id = "t" + std::to_string(t);
				g->feedInput(id, "slow", "x", Value(std::make_unique<Tensor>(floatTensor(1.0f))));
				g->submit(id, "slow", "y");
			}
			CHECK(waitForValue(started, true, 5s), "slow task must enter RunFn before teardown");

			// 引擎析构（自排水）与调度器 shutdown（弃置排队票据 + join 在飞）并发
			auto drained = std::async(std::launch::async, [&] { g.reset(); }).share();
			auto stopped = std::async(std::launch::async, [&] { sched->shutdown(); }).share();
			if (drained.wait_for(30s) != std::future_status::ready
				|| stopped.wait_for(30s) != std::future_status::ready) {
				std::cerr << "FAIL: teardown/shutdown race hung (iteration " << iter << ")" << std::endl;
				++failures;
				g.release();   // 故意泄漏：避免析构再次进入挂起路径
				sched.release();
				return;
			}
		}
	}
	END_TEST();
}

int main() {
	try {
		testGlobalInstanceSemantics();
		testSerialLimit();
		testConcurrentLimit();
		testClassIsolation();
		testConfigValidation();
		testShutdownRejection();
		testEngineDestructorDrainsInflight();
		testShutdownRacesRecursiveSubmit();
		testTeardownRacingSchedulerShutdown();
	} catch (const std::exception& e) {
		std::cerr << "UNEXPECTED EXCEPTION: " << e.what() << std::endl;
		return 1;
	}

	if (failures != 0) {
		std::cerr << failures << " test(s) failed" << std::endl;
		return 1;
	}
	std::cout << "All ResourceScheduler tests passed" << std::endl;
	return 0;
}
