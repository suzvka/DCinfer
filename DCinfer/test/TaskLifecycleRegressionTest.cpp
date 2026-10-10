// 任务生命周期回归测试：轮次化任务状态下的复用/失败/收尾窗口/并发提交契约
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "InferGraph.h"
#include "ResourceScheduler.h"
#include "GraphException.h"
#include "Tensor.hpp"

using namespace DC;
using namespace std::chrono_literals;

using TensorType = DC::Tensor::TensorType;
using Tensor = DC::Tensor;

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

static Tensor floatTensor(float value) {
	auto t = Tensor::Create<float>();
	t = value;
	return t;
}

static Value floatValue(float value) {
	return Value(std::make_unique<Tensor>(floatTensor(value)));
}

static Node::Schema passSchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("x")};
	s.outputs = {Node::Port::out<float>("y")};
	return s;
}

static Node::RunFn passRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto* x = ctx.input<Tensor>("x");
		if (!x)
			return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
		ctx.output("y", Value(std::make_unique<Tensor>(*x)));
		return ctx.success();
	};
}

static bool hasMessage(const std::vector<TaskError>& errors, const std::string& fragment) {
	for (const auto& e : errors) {
		if (e.message.find(fragment) != std::string::npos)
			return true;
	}
	return false;
}

static void testReuseAfterCancelRace() {
	TEST("F04: reuse after cancel (queued stale lambda must not consume fresh input)") {
		// 默认 1/1/1 单槽位：阻塞节点占住 worker，旧轮次 lambda 滞留队列，确定性复刻
		std::promise<void> entered, go;
		auto ready = go.get_future().share();

		InferGraph g;
		g.addNode(std::make_unique<Node>("test", "block", passSchema(),
			[&](Node::RunContext& ctx) -> Node::Result {
				entered.set_value();
				ready.wait();
				const auto* x = ctx.input<Tensor>("x");
				if (!x)
					return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
				ctx.output("y", Value(std::make_unique<Tensor>(*x)));
				return ctx.success();
			}));
		g.addNode(std::make_unique<Node>("test", "n", passSchema(), passRunFn()));

		g.feedInput("a", "block", "x", floatValue(1.0f));
		g.submit("a", "block", "y");
		entered.get_future().wait();

		g.feedInput("b", "n", "x", floatValue(2.0f));
		g.submit("b", "n", "y");
		g.cancel("b");

		g.feedInput("b", "n", "x", floatValue(3.0f));
		g.submit("b", "n", "y");

		go.set_value();

		const auto r = g.waitForResult("b", 2s);
		CHECK(r.status == TaskStatus::Succeeded, "reused task should succeed with fresh input");
		CHECK(g.hasOutput("b", "n", "y"), "reused task should produce declared output");
		const float out = g.takeOutputTensor("b", "n", "y").item<float>();
		CHECK(std::abs(out - 3.0f) < 1e-6f, "reused task must reflect fresh input (3.0f), not the cancelled round");

		const auto ra = g.waitForResult("a", 2s);
		CHECK(ra.status == TaskStatus::Succeeded, "blocking task should complete normally");
	}
	END_TEST();
}

static void testFailureAfterOutput() {
	TEST("F05: node fails after emitting output -> task Failed, no success propagation") {
		InferGraph g;
		g.addNode(std::make_unique<Node>("test", "n", passSchema(),
			[](Node::RunContext& ctx) -> Node::Result {
				ctx.output("y", Value(std::make_unique<Tensor>(floatTensor(4.0f))));
				return ctx.failure(Node::Status::ExecutionFailed, "deliberate failure after output");
			}));
		g.bindInput("x", "n", "x");
		g.bindOutput("y", "n", "y");

		g.feedInput("t", "n", "x", floatValue(1.0f));
		g.submitBound("t");

		const auto r = g.waitForResult("t", 2s);
		CHECK(r.status == TaskStatus::Failed, "failure must not be reported as success just because output exists");
		CHECK(hasMessage(r.errors, "deliberate failure after output"),
			  "node failure diagnostic should be recorded");
	}
	END_TEST();
}

static void testMissingRequiredOutput() {
	TEST("F05: missing required output -> task Failed with completeness diagnostic") {
		InferGraph g;
		g.addNode(std::make_unique<Node>("test", "n", passSchema(),
			[](Node::RunContext& ctx) -> Node::Result {
				return ctx.success();
			}));
		g.bindInput("x", "n", "x");
		g.bindOutput("y", "n", "y");

		g.feedInput("t", "n", "x", floatValue(1.0f));
		g.submitBound("t");

		const auto r = g.waitForResult("t", 2s);
		CHECK(r.status == TaskStatus::Failed, "missing required output must fail the task");
		CHECK(hasMessage(r.errors, "Not all required outputs"),
			  "output completeness diagnostic should be recorded");
	}
	END_TEST();
}

static void testCallbackThrows() {
	TEST("F07: throwing completion callback must not break termination transaction") {
		InferGraph g;
		g.addNode(std::make_unique<Node>("test", "n", passSchema(), passRunFn()));
		g.bindInput("x", "n", "x");
		g.bindOutput("y", "n", "y");
		g.setTaskCompleteCallback([](const std::string&) { throw std::runtime_error("callback error"); });

		g.feedInput("t", "n", "x", floatValue(2.0f));
		g.submitBound("t");

		const auto start = std::chrono::steady_clock::now();
		const auto r = g.waitForResult("t", 2s);
		const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
								   std::chrono::steady_clock::now() - start)
								   .count();
		CHECK(r.status == TaskStatus::Succeeded, "callback exception must not affect task outcome");
		CHECK(elapsedMs < 1500, "wait must return promptly after termination completes (not ride the timeout)");
		CHECK(hasMessage(r.errors, "task complete callback threw"),
			  "isolated callback exception should be recorded as diagnostic");

		CHECK(g.hasOutput("t", "n", "y"), "declared output must be rescued despite callback throw");
		CHECK(std::abs(g.takeOutputTensor("t", "n", "y").item<float>() - 2.0f) < 1e-6f,
			  "result must be intact after callback exception");

		g.feedInput("t2", "n", "x", floatValue(5.0f));
		g.submitBound("t2");
		const auto r2 = g.waitForResult("t2"); // 无限等待
		CHECK(r2.status == TaskStatus::Succeeded, "infinite wait must not hang after callback exception");
	}
	END_TEST();
}

static void testReleaseActiveRejected() {
	TEST("F08: release on active task rejected; task still completes; terminal release clears state") {
		std::promise<void> entered, go;
		auto ready = go.get_future().share();

		InferGraph g;
		g.addNode(std::make_unique<Node>("test", "n", passSchema(),
			[&](Node::RunContext& ctx) -> Node::Result {
				entered.set_value();
				ready.wait();
				const auto* x = ctx.input<Tensor>("x");
				if (!x)
					return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
				ctx.output("y", Value(std::make_unique<Tensor>(*x)));
				return ctx.success();
			}));
		g.bindInput("x", "n", "x");
		g.bindOutput("y", "n", "y");

		g.feedInput("t", "n", "x", floatValue(1.0f));
		g.submitBound("t");
		entered.get_future().wait();

		g.releaseTask("t");
		CHECK(g.taskStatus("t") == TaskStatus::Running, "active task must not be released");

		go.set_value();
		const auto r = g.waitForResult("t", 2s);
		CHECK(r.status == TaskStatus::Succeeded, "task must complete normally after rejected release");
		CHECK(std::abs(g.takeOutputTensor("t", "n", "y").item<float>() - 1.0f) < 1e-6f,
			  "result must be intact after rejected release");

		g.releaseTask("t");
		CHECK(g.taskStatus("t") == TaskStatus::Unknown, "terminated task must be releasable");
		CHECK(!g.hasOutput("t", "n", "y"), "released result must be gone");
	}
	END_TEST();
}

static void testUnsubmittedDestructorFreesInput() {
	TEST("F09: unsubmitted handle destructor frees fed input immediately") {
		bool freed = false;

		InferGraph g;
		g.addNode(std::make_unique<Node>("test", "n", passSchema(), passRunFn()));
		g.bindInput("num", "n", "x");
		g.bindOutput("y", "n", "y");
		auto api = g.interface();

		{
			auto t = api.createTask();
			auto* raw = new Tensor(floatTensor(5.0f));
			t.feed("num", Value(raw, [&freed](Tensor* p) {
						freed = true;
						delete p;
					}));
		}
		CHECK(freed, "unsubmitted handle destructor must free fed input immediately");

		CHECK(g.taskStatus("_unused") == TaskStatus::Unknown, "sanity: unknown task ids stay unknown");
	}
	END_TEST();
}

static void testDiscardedCancelsAndReleases() {
	TEST("F09: discarded in-flight handle must cancel and release at destruction (cooperative)") {
		std::promise<void> entered, go, executed;
		auto ready = go.get_future().share();
		auto ran = executed.get_future();

		InferGraph g;
		g.addNode(std::make_unique<Node>("test", "n", passSchema(),
			[&](Node::RunContext& ctx) -> Node::Result {
				entered.set_value();
				ready.wait();
				executed.set_value();
				const auto* x = ctx.input<Tensor>("x");
				if (!x)
					return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
				ctx.output("y", Value(std::make_unique<Tensor>(*x)));
				return ctx.success();
			}));
		g.bindInput("num", "n", "x");
		g.bindOutput("y", "n", "y");
		auto api = g.interface();

		std::string tid;
		{
			auto t = api.createTask();
			tid = t.taskId();
			t.feed("num", floatTensor(7.0f));
			t.submit(); // 异步启动，_submitted = true
			entered.get_future().wait();
		}

		CHECK(g.taskStatus(tid) == TaskStatus::Unknown,
			  "discarded in-flight task must be cancelled and released at destruction");

		// 协作式取消不打断在飞节点：RunFn 照常返回，产出被丢弃
		go.set_value();
		CHECK(ran.wait_for(2s) == std::future_status::ready,
			  "in-flight node run must not be interrupted by discarding (cooperative)");
	}
	END_TEST();
}

static void testUnreachableSubmitRollback() {
	TEST("F10: unreachable submit leaves no Running residue and can be retried") {
		InferGraph g;
		g.addNode(std::make_unique<Node>("test", "a", passSchema(), passRunFn()));
		g.addNode(std::make_unique<Node>("test", "b", passSchema(), passRunFn()));

		g.feedInput("t", "a", "x", floatValue(1.0f));
		bool rejected = false;
		try {
			g.submit("t", "b", "y");
		} catch (const GraphException& e) {
			rejected = (e.getErrorType() == GraphException::ErrorType::UnreachableDeclaration);
		}
		CHECK(rejected, "unreachable declaration should throw UnreachableDeclaration");
		CHECK(g.taskStatus("t") == TaskStatus::Unknown, "failed submit must leave no Running residue");

		g.feedInput("t", "b", "x", floatValue(2.0f));
		g.submit("t", "b", "y");
		const auto r = g.waitForResult("t", 2s);
		CHECK(r.status == TaskStatus::Succeeded, "resubmit after rollback must succeed");
		CHECK(std::abs(g.takeOutputTensor("t", "b", "y").item<float>() - 2.0f) < 1e-6f,
			  "retried task result must match its input");
	}
	END_TEST();
}

static void testReuseDuringFinalizeRejected() {
	TEST("H-3: reuse during finalizing window rejected (blocked callback widens window)") {
		std::promise<void> cbEntered, cbGo;
		auto cbEnteredF = cbEntered.get_future();
		bool submitRejectedInCb = false, feedRejectedInCb = false, releaseKeptState = false;

		InferGraph g;
		g.addNode(std::make_unique<Node>("test", "n", passSchema(), passRunFn()));
		g.bindInput("x", "n", "x");
		g.bindOutput("y", "n", "y");

		// 回调阻塞拉长收尾窗口：终态已发布，清理未完成
		g.setTaskCompleteCallback([&](const std::string& tid) {
			cbEntered.set_value();
			cbGo.get_future().wait();

			try {
				g.submit(tid, "n", "y");
			} catch (const GraphException& e) {
				submitRejectedInCb = (e.getErrorType() == GraphException::ErrorType::DuplicateTask);
			}
			try {
				g.feedInput(tid, "n", "x", floatValue(9.0f));
			} catch (const GraphException& e) {
				feedRejectedInCb = (e.getErrorType() == GraphException::ErrorType::DuplicateTask);
			}
			g.releaseTask(tid);
			releaseKeptState = (g.taskStatus(tid) != TaskStatus::Unknown);
		});

		g.feedInput("t", "n", "x", floatValue(1.0f));
		g.submitBound("t");
		cbEnteredF.wait();

		CHECK(g.taskStatus("t") != TaskStatus::Running, "terminal status must be visible during finalize");

		bool feedRejected = false;
		try {
			g.feedInput("t", "n", "x", floatValue(5.0f));
		} catch (const GraphException& e) {
			feedRejected = (e.getErrorType() == GraphException::ErrorType::DuplicateTask);
		}
		CHECK(feedRejected, "feed during finalize must be rejected (input would be silently dropped)");

		bool submitRejected = false;
		try {
			g.submit("t", "n", "y");
		} catch (const GraphException& e) {
			submitRejected = (e.getErrorType() == GraphException::ErrorType::DuplicateTask);
		}
		CHECK(submitRejected, "submit during finalize must be rejected");

		g.releaseTask("t");
		CHECK(g.taskStatus("t") != TaskStatus::Unknown, "release during finalize must be rejected");

		cbGo.set_value();

		const auto r = g.waitForResult("t", 2s);
		CHECK(r.status == TaskStatus::Succeeded, "task must complete after finalize window closes");
		CHECK(submitRejectedInCb && feedRejectedInCb && releaseKeptState,
			  "in-callback reuse attempts must be rejected while finalizing");

		g.feedInput("t", "n", "x", floatValue(9.0f));
		g.submit("t", "n", "y");
		const auto r2 = g.waitForResult("t", 2s);
		CHECK(r2.status == TaskStatus::Succeeded, "reuse after finalize must succeed");
		CHECK(std::abs(g.takeOutputTensor("t", "n", "y").item<float>() - 9.0f) < 1e-6f,
			  "reused round must reflect fresh input (no stale artifact)");
	}
	END_TEST();
}

static void testDetachDuringFinalize() {
	TEST("H-3: detach during finalize auto-releases after cleanup completes") {
		std::promise<void> cbEntered, cbGo;
		auto cbEnteredF = cbEntered.get_future();

		InferGraph g;
		g.addNode(std::make_unique<Node>("test", "n", passSchema(), passRunFn()));
		g.bindInput("x", "n", "x");
		g.bindOutput("y", "n", "y");
		g.setTaskCompleteCallback([&](const std::string&) {
			cbEntered.set_value();
			cbGo.get_future().wait();
		});

		g.feedInput("t", "n", "x", floatValue(3.0f));
		g.submitBound("t");
		cbEnteredF.wait();

		g.detachTask("t");
		CHECK(g.taskStatus("t") != TaskStatus::Unknown, "detach during finalize must not release immediately");

		cbGo.set_value();

		const auto deadline = std::chrono::steady_clock::now() + 2s;
		while (g.taskStatus("t") != TaskStatus::Unknown && std::chrono::steady_clock::now() < deadline)
			std::this_thread::sleep_for(5ms);
		CHECK(g.taskStatus("t") == TaskStatus::Unknown, "detached finalizing task must be auto-released");
		CHECK(!g.hasOutput("t", "n", "y"), "auto-released task must clear artifacts");
	}
	END_TEST();
}

static void testConcurrentSubmitSameId() {
	TEST("H-5: concurrent submit on same taskId -> exactly one wins, winner intact") {
		std::promise<void> entered, go;
		auto enteredF = entered.get_future();
		auto goF = go.get_future().share();

		auto sched = std::make_shared<ResourceScheduler>(SchedulerConfig{1, 2, 1}); // 双 Operator 槽位：两线程真正并发进入 submit
		InferGraph g(sched);
		g.addNode(std::make_unique<Node>("test", "n", passSchema(),
			[&](Node::RunContext& ctx) -> Node::Result {
				entered.set_value();
				goF.wait();
				const auto* x = ctx.input<Tensor>("x");
				if (!x)
					return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
				ctx.output("y", Value(std::make_unique<Tensor>(*x)));
				return ctx.success();
			}));
		g.bindInput("x", "n", "x");
		g.bindOutput("y", "n", "y");

		g.feedInput("t", "n", "x", floatValue(7.0f));

		std::atomic<int> wins{0}, dups{0};
		std::atomic<bool> start{false};
		auto submitter = [&]() {
			while (!start.load())
				std::this_thread::yield();
			try {
				g.submit("t", "n", "y");
				++wins;
			} catch (const GraphException& e) {
				if (e.getErrorType() == GraphException::ErrorType::DuplicateTask)
					++dups;
			}
		};
		std::thread t1(submitter), t2(submitter);
		start.store(true);
		t1.join();
		t2.join();

		CHECK(wins.load() == 1, "exactly one concurrent submit must win");
		CHECK(dups.load() == 1, "the losing submit must fail with DuplicateTask");

		enteredF.wait();
		go.set_value();

		const auto r = g.waitForResult("t", 2s);
		CHECK(r.status == TaskStatus::Succeeded, "winner task must complete with intact declarations");
		CHECK(std::abs(g.takeOutputTensor("t", "n", "y").item<float>() - 7.0f) < 1e-6f,
			  "winner output must be intact (no cross-submit corruption)");
	}
	END_TEST();
}

static void testZombieRetryAfterFinalize() {
	TEST("#2: enqueued retry must not execute after its round finalized (zombie discarded)") {
		std::promise<void> entered, go;
		auto ready = go.get_future().share(); // shared_future：僵尸重试再次进入不挂起
		std::atomic<bool> firstEntry{true};

		auto sched = std::make_shared<ResourceScheduler>(SchedulerConfig{2, 2, 2}); // 多槽位：门被占期间 B 的提交可真实执行
		InferGraph g(sched);
		g.addNode(std::make_unique<Node>("test", "gate", passSchema(),
			[&](Node::RunContext& ctx) -> Node::Result {
				if (firstEntry.exchange(false))
					entered.set_value();
				ready.wait();
				const auto* x = ctx.input<Tensor>("x");
				if (!x)
					return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
				ctx.output("y", Value(std::make_unique<Tensor>(*x)));
				return ctx.success();
			}));
		g.addNode(std::make_unique<Node>("test", "err", passSchema(),
			[](Node::RunContext& ctx) -> Node::Result {
				return ctx.failure(Node::Status::ExecutionFailed, "deliberate error node");
			}));

		g.feedInput("A", "gate", "x", floatValue(1.0f));
		g.submit("A", "gate", "y");
		entered.get_future().wait();

		// B：gate 抛 Reentrant 重试排队，err 报错，B 收尾为 Failed，终态迁移必置 terminated
		g.feedInput("B", "gate", "x", floatValue(2.0f));
		g.feedInput("B", "err", "x", floatValue(2.0f));
		g.submit("B", {{"gate", "y"}, {"err", "y"}});

		const auto deadline = std::chrono::steady_clock::now() + 2s;
		while (g.taskStatus("B") == TaskStatus::Running && std::chrono::steady_clock::now() < deadline)
			std::this_thread::sleep_for(2ms);
		CHECK(g.taskStatus("B") == TaskStatus::Failed, "B must finalize as Failed (error node)");

		// 放行 A：B 的待重试派发被 terminated 拦截，僵尸不执行
		go.set_value();
		const auto ra = g.waitForResult("A", 2s);
		CHECK(ra.status == TaskStatus::Succeeded, "task A must complete normally");
		CHECK(std::abs(g.takeOutputTensor("A", "gate", "y").item<float>() - 1.0f) < 1e-6f,
			  "task A result must be intact");

		std::this_thread::sleep_for(150ms);
		CHECK(!g.hasOutput("B", "gate", "y"),
			  "zombie retry must not execute/accumulate into finalized round");
	}
	END_TEST();
}

static void testCancelVsCompletionRace() {
	TEST("#3: cancel x completion race loop with same-ID reuse (no state residue)") {
		auto sched = std::make_shared<ResourceScheduler>(SchedulerConfig{2, 2, 2});
		for (int round = 0; round < 30; ++round) {
			InferGraph g(sched);
			g.addNode(std::make_unique<Node>("test", "n", passSchema(),
				[](Node::RunContext& ctx) -> Node::Result {
					const auto* x = ctx.input<Tensor>("x");
					if (!x)
						return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
					ctx.output("y", Value(std::make_unique<Tensor>(*x)));
					// 产出后微睡：让 cancel 落在搬运/收尾未完成窗口
					std::this_thread::sleep_for(std::chrono::microseconds(200));
					return ctx.success();
				}));

			const std::string tid = "race";
			g.feedInput(tid, "n", "x", floatValue(1.0f));
			g.submit(tid, "n", "y");

			std::atomic<bool> go{false};
			std::thread canceller([&] {
				while (!go.load())
					std::this_thread::yield();
				g.cancel(tid);
			});
			go.store(true);

			const auto r = g.waitForResult(tid, 2s);
			canceller.join();
			CHECK(r.status == TaskStatus::Succeeded || r.status == TaskStatus::Cancelled,
				  "task must terminate as Succeeded or Cancelled under cancel race");

			g.feedInput(tid, "n", "x", floatValue(3.0f));
			g.submit(tid, "n", "y");
			const auto r2 = g.waitForResult(tid, 2s);
			CHECK(r2.status == TaskStatus::Succeeded, "reused round must succeed");
			CHECK(std::abs(g.takeOutputTensor(tid, "n", "y").item<float>() - 3.0f) < 1e-6f,
				  "reused round must reflect fresh input (no stale residue)");
		}
	}
	END_TEST();
}

static void testConcurrentFeedDuringFinalize() {
	TEST("#8-12: concurrent feed during active/finalizing task -> accepted or DuplicateTask only") {
		auto sched = std::make_shared<ResourceScheduler>(SchedulerConfig{2, 2, 2});
		for (int round = 0; round < 20; ++round) {
			InferGraph g(sched);
			g.addNode(std::make_unique<Node>("test", "n", passSchema(), passRunFn()));

			const std::string tid = "cf";
			g.feedInput(tid, "n", "x", floatValue(1.0f));
			g.submit(tid, "n", "y");

			std::atomic<bool> stop{false};
			std::atomic<int> protocolViolations{0};
			std::thread feeder([&] {
				while (!stop.load()) {
					try {
						g.feedInput(tid, "n", "x", floatValue(2.0f));
					} catch (const GraphException& e) {
						if (e.getErrorType() != GraphException::ErrorType::DuplicateTask)
							++protocolViolations;
					} catch (...) {
						++protocolViolations;
					}
				}
			});

			const auto r = g.waitForResult(tid, 2s);
			stop.store(true);
			feeder.join();
			CHECK(r.status == TaskStatus::Succeeded, "task must succeed despite concurrent feed");
			CHECK(protocolViolations.load() == 0,
				  "concurrent feed must only be accepted or rejected as DuplicateTask");

			g.feedInput(tid, "n", "x", floatValue(4.0f));
			g.submit(tid, "n", "y");
			const auto r2 = g.waitForResult(tid, 2s);
			CHECK(r2.status == TaskStatus::Succeeded, "reuse after concurrent feed must succeed");
			CHECK(std::abs(g.takeOutputTensor(tid, "n", "y").item<float>() - 4.0f) < 1e-6f,
				  "reused round must reflect fresh input");
		}
	}
	END_TEST();
}

static void testTypoDeclarationRejected() {
	TEST("#8-13: typo in declared node/port rejected at submit with clear error") {
		InferGraph g;
		g.addNode(std::make_unique<Node>("test", "n", passSchema(), passRunFn()));
		g.feedInput("t", "n", "x", floatValue(1.0f));

		bool portRejected = false;
		try {
			g.submit("t", "n", "y_typo");
		} catch (const GraphException& e) {
			portRejected = (e.getErrorType() == GraphException::ErrorType::PortNotFound);
		}
		CHECK(portRejected, "typo port must be rejected with PortNotFound at submit");
		CHECK(g.taskStatus("t") == TaskStatus::Unknown, "rejected submit must leave no Running residue");

		bool nodeRejected = false;
		try {
			g.submit("t", "n_typo", "y");
		} catch (const GraphException& e) {
			nodeRejected = (e.getErrorType() == GraphException::ErrorType::NodeNotFound);
		}
		CHECK(nodeRejected, "typo node must be rejected with NodeNotFound at submit");

		g.submit("t", "n", "y");
		const auto r = g.waitForResult("t", 2s);
		CHECK(r.status == TaskStatus::Succeeded, "corrected submit must succeed");
		CHECK(std::abs(g.takeOutputTensor("t", "n", "y").item<float>() - 1.0f) < 1e-6f,
			  "corrected submit result must be intact");
	}
	END_TEST();
}

int main() {
	try {
		testReuseAfterCancelRace();
		testFailureAfterOutput();
		testMissingRequiredOutput();
		testCallbackThrows();
		testReleaseActiveRejected();
		testUnsubmittedDestructorFreesInput();
		testDiscardedCancelsAndReleases();
		testUnreachableSubmitRollback();
		testReuseDuringFinalizeRejected();
		testDetachDuringFinalize();
		testConcurrentSubmitSameId();
		testZombieRetryAfterFinalize();
		testCancelVsCompletionRace();
		testConcurrentFeedDuringFinalize();
		testTypoDeclarationRejected();
	} catch (const std::exception& e) {
		std::cerr << "UNEXPECTED EXCEPTION: " << e.what() << std::endl;
		return 1;
	}

	if (failures != 0) {
		std::cerr << failures << " test(s) failed" << std::endl;
		return 1;
	}
	std::cout << "All lifecycle regression tests passed" << std::endl;
	return 0;
}
