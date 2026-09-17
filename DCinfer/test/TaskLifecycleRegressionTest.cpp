// 任务生命周期回归测试（发布前审查 F04-F10 修复 + H-3/H-5 修复的确定性用例）
//
// 锁定"轮次化任务状态"重构后的行为契约：
//   F04 同 ID 复用竞态：旧轮次排队 lambda 不得消费新一轮输入；取消后复用必须成功
//   F05 节点失败优先于输出存在性：先产出后失败 / 必需输出缺失必须 Failed 且不传播
//   F07 完成回调异常隔离：终止事务完整，wait 契约（返回即可读）不被破坏
//   F08 releaseTask 仅终态可释放：活动任务拒绝时声明/结果/诊断保持不动
//   F09 句柄析构：未提交立即释放输入；在飞弃置即请求取消并随即回收（协作式）
//   F10 提交失败（不可达声明）不留 Running 残留，可重试
//   H-3 收尾窗口：终态已发布、结果可读前的复用/注入/释放全部拒绝；
//       弃置（detach）登记自动回收，收尾完成后无残留
//   H-5 并发同 ID 提交：原子事务下恰一方成功，胜者任务状态/声明/输出完整
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

// ── 辅助 ──

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

/// 直通节点：输入 x 拷贝为输出 y
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

// ════════════════════════════════════════════
// F04：同 ID 复用竞态 —— 旧轮次排队 lambda 不得消费新一轮输入
// ════════════════════════════════════════════

static void testReuseAfterCancelRace() {
	TEST("F04: reuse after cancel (queued stale lambda must not consume fresh input)") {
		// 单线程池（默认）：阻塞节点占住唯一 worker，使 b 的旧轮次 lambda
		// 在取消与复用期间滞留于队列——报告复现场景的确定性复刻
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

		// a：阻塞任务占住 worker
		g.feedInput("a", "block", "x", floatValue(1.0f));
		g.submit("a", "block", "y");
		entered.get_future().wait();

		// b 第一轮：排队后取消
		g.feedInput("b", "n", "x", floatValue(2.0f));
		g.submit("b", "n", "y");
		g.cancel("b");

		// b 第二轮：复用同 ID，喂新输入后重新提交
		g.feedInput("b", "n", "x", floatValue(3.0f));
		g.submit("b", "n", "y");

		go.set_value(); // 解除 a 阻塞，队列开始消费

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

// ════════════════════════════════════════════
// F05：失败优先于输出存在性
// ════════════════════════════════════════════

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
				return ctx.success(); // 声明了输出 y 但从不产出
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

// ════════════════════════════════════════════
// F07：完成回调异常隔离 —— 终止事务完整
// ════════════════════════════════════════════

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

		// 结果可读契约：终止后声明输出必可取
		CHECK(g.hasOutput("t", "n", "y"), "declared output must be rescued despite callback throw");
		CHECK(std::abs(g.takeOutputTensor("t", "n", "y").item<float>() - 2.0f) < 1e-6f,
			  "result must be intact after callback exception");

		// 无限等待路径同样不得挂起（回调异常不阻断 resultsReady 发布与 notify）
		g.feedInput("t2", "n", "x", floatValue(5.0f));
		g.submitBound("t2");
		const auto r2 = g.waitForResult("t2"); // 无限等待
		CHECK(r2.status == TaskStatus::Succeeded, "infinite wait must not hang after callback exception");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// F08：releaseTask 仅终态可释放
// ════════════════════════════════════════════

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

		g.releaseTask("t"); // 活动任务：必须被拒绝（声明/结果/诊断保持不动）
		CHECK(g.taskStatus("t") == TaskStatus::Running, "active task must not be released");

		go.set_value();
		const auto r = g.waitForResult("t", 2s);
		CHECK(r.status == TaskStatus::Succeeded, "task must complete normally after rejected release");
		CHECK(std::abs(g.takeOutputTensor("t", "n", "y").item<float>() - 1.0f) < 1e-6f,
			  "result must be intact after rejected release");

		// 终态释放：状态与结果一并清空
		g.releaseTask("t");
		CHECK(g.taskStatus("t") == TaskStatus::Unknown, "terminated task must be releasable");
		CHECK(!g.hasOutput("t", "n", "y"), "released result must be gone");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// F09：句柄析构资源回收
// ════════════════════════════════════════════

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
			// 仅 feed，未 submit —— 句柄析构必须立即释放输入
		}
		CHECK(freed, "unsubmitted handle destructor must free fed input immediately");

		// 未提交任务不残留状态（Unknown，且无输出）
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
				executed.set_value(); // 协作式取消不打断在飞节点：RunFn 照常返回
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
			t.submit(); // 异步启动（_submitted = true）
			entered.get_future().wait();
		} // 句柄析构 → 在飞弃置：先请求取消（协作式），再回收兜底

		// 弃置即取消：无需等节点返回，任务在析构内同步终态并完成回收
		CHECK(g.taskStatus(tid) == TaskStatus::Unknown,
			  "discarded in-flight task must be cancelled and released at destruction");

		// 协作式：在飞节点调用不被中断，RunFn 照常返回；其产出被丢弃、不传播
		go.set_value();
		CHECK(ran.wait_for(2s) == std::future_status::ready,
			  "in-flight node run must not be interrupted by discarding (cooperative)");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// F10：提交失败无残留、可重试
// ════════════════════════════════════════════

static void testUnreachableSubmitRollback() {
	TEST("F10: unreachable submit leaves no Running residue and can be retried") {
		InferGraph g;
		g.addNode(std::make_unique<Node>("test", "a", passSchema(), passRunFn()));
		g.addNode(std::make_unique<Node>("test", "b", passSchema(), passRunFn()));

		// 只喂 a，却声明 b 的输出 —— 拓扑不可达，submit 必须抛且不留状态
		g.feedInput("t", "a", "x", floatValue(1.0f));
		bool rejected = false;
		try {
			g.submit("t", "b", "y");
		} catch (const GraphException& e) {
			rejected = (e.getErrorType() == GraphException::ErrorType::UnreachableDeclaration);
		}
		CHECK(rejected, "unreachable declaration should throw UnreachableDeclaration");
		CHECK(g.taskStatus("t") == TaskStatus::Unknown, "failed submit must leave no Running residue");

		// 修正输入后重试：同 ID 提交必须成功
		g.feedInput("t", "b", "x", floatValue(2.0f));
		g.submit("t", "b", "y");
		const auto r = g.waitForResult("t", 2s);
		CHECK(r.status == TaskStatus::Succeeded, "resubmit after rollback must succeed");
		CHECK(std::abs(g.takeOutputTensor("t", "b", "y").item<float>() - 2.0f) < 1e-6f,
			  "retried task result must match its input");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// H-3：收尾窗口（终态已发布、结果可读前）复用/注入/释放全部拒绝
// ════════════════════════════════════════════

static void testReuseDuringFinalizeRejected() {
	TEST("H-3: reuse during finalizing window rejected (blocked callback widens window)") {
		std::promise<void> cbEntered, cbGo;
		auto cbEnteredF = cbEntered.get_future();
		bool submitRejectedInCb = false, feedRejectedInCb = false, releaseKeptState = false;

		InferGraph g;
		g.addNode(std::make_unique<Node>("test", "n", passSchema(), passRunFn()));
		g.bindInput("x", "n", "x");
		g.bindOutput("y", "n", "y");

		// 回调阻塞：人为拉长收尾窗口（终态已发布，结果抢救/执行态清理未完成）
		g.setTaskCompleteCallback([&](const std::string& tid) {
			cbEntered.set_value();
			cbGo.get_future().wait();

			// 回调内复用尝试：提交/注入均被拒（DuplicateTask），释放被拒（状态保留）
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
			g.releaseTask(tid); // 收尾中：引擎拒绝，状态/结果保持不动
			releaseKeptState = (g.taskStatus(tid) != TaskStatus::Unknown);
		});

		g.feedInput("t", "n", "x", floatValue(1.0f));
		g.submitBound("t");
		cbEnteredF.wait();

		// 收尾窗口内（回调阻塞中）：终态可见，但复用/注入/释放全部拒绝
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

		g.releaseTask("t"); // 收尾中：拒绝
		CHECK(g.taskStatus("t") != TaskStatus::Unknown, "release during finalize must be rejected");

		cbGo.set_value(); // 放行回调 → 收尾完成（结果可读发布）

		const auto r = g.waitForResult("t", 2s);
		CHECK(r.status == TaskStatus::Succeeded, "task must complete after finalize window closes");
		CHECK(submitRejectedInCb && feedRejectedInCb && releaseKeptState,
			  "in-callback reuse attempts must be rejected while finalizing");

		// 收尾完成后：复用同 ID 合法，且新一轮结果无旧轮残留
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

		g.detachTask("t"); // 收尾中：登记自动回收（不立即释放、不破坏收尾事务）
		CHECK(g.taskStatus("t") != TaskStatus::Unknown, "detach during finalize must not release immediately");

		cbGo.set_value();

		// 收尾完成后自动回收：状态表条目/结果/诊断清空（轮询至回收完成）
		const auto deadline = std::chrono::steady_clock::now() + 2s;
		while (g.taskStatus("t") != TaskStatus::Unknown && std::chrono::steady_clock::now() < deadline)
			std::this_thread::sleep_for(5ms);
		CHECK(g.taskStatus("t") == TaskStatus::Unknown, "detached finalizing task must be auto-released");
		CHECK(!g.hasOutput("t", "n", "y"), "auto-released task must clear artifacts");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// H-5：并发同 ID 提交 —— 原子事务恰一方成功
// ════════════════════════════════════════════

static void testConcurrentSubmitSameId() {
	TEST("H-5: concurrent submit on same taskId -> exactly one wins, winner intact") {
		std::promise<void> entered, go;
		auto enteredF = entered.get_future();
		auto goF = go.get_future().share();

		InferGraph g({1}, {2}, {1}); // 双 Operator 线程：两线程真正并发进入 submit
		g.addNode(std::make_unique<Node>("test", "n", passSchema(),
			[&](Node::RunContext& ctx) -> Node::Result {
				entered.set_value();
				goF.wait(); // 首轮执行阻塞：胜者轮次保持 Running，败者必遭 DuplicateTask
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

		enteredF.wait(); // 胜者轮次执行中（阻塞）
		go.set_value();

		const auto r = g.waitForResult("t", 2s);
		CHECK(r.status == TaskStatus::Succeeded, "winner task must complete with intact declarations");
		CHECK(std::abs(g.takeOutputTensor("t", "n", "y").item<float>() - 7.0f) < 1e-6f,
			  "winner output must be intact (no cross-submit corruption)");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// #2 修复回归：僵尸重试 —— 收尾轮次的待重试经 terminated 拦截丢弃
// ════════════════════════════════════════════

static void testZombieRetryAfterFinalize() {
	TEST("#2: enqueued retry must not execute after its round finalized (zombie discarded)") {
		std::promise<void> entered, go;
		auto ready = go.get_future().share(); // shared_future：僵尸重试再次进入不挂起
		std::atomic<bool> firstEntry{true};

		InferGraph g({2}, {2}, {2}); // 多 worker：门被占期间 B 的提交可真实执行
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

		// A：占住 gate 节点执行门（长时间执行）
		g.feedInput("A", "gate", "x", floatValue(1.0f));
		g.submit("A", "gate", "y");
		entered.get_future().wait();

		// B：gate 提交抛 Reentrant → 重试排队（不计 inflight）；err 报错 →
		// B 经 _exhaustedCheck 收尾为 Failed（修复点：终态迁移必置 terminated）
		g.feedInput("B", "gate", "x", floatValue(2.0f));
		g.feedInput("B", "err", "x", floatValue(2.0f));
		g.submit("B", {{"gate", "y"}, {"err", "y"}});

		const auto deadline = std::chrono::steady_clock::now() + 2s;
		while (g.taskStatus("B") == TaskStatus::Running && std::chrono::steady_clock::now() < deadline)
			std::this_thread::sleep_for(2ms);
		CHECK(g.taskStatus("B") == TaskStatus::Failed, "B must finalize as Failed (error node)");

		// 放行 A：门释放 → 派发 B 的待重试 → terminated 拦截，僵尸不执行
		go.set_value();
		const auto ra = g.waitForResult("A", 2s);
		CHECK(ra.status == TaskStatus::Succeeded, "task A must complete normally");
		CHECK(std::abs(g.takeOutputTensor("A", "gate", "y").item<float>() - 1.0f) < 1e-6f,
			  "task A result must be intact");

		// 给异步路径留时间：修复后收尾轮次不得出现僵尸产出
		std::this_thread::sleep_for(150ms);
		CHECK(!g.hasOutput("B", "gate", "y"),
			  "zombie retry must not execute/accumulate into finalized round");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// #3 修复回归：cancel × 完成竞态 —— 收尾清理必然完成，复用无残留
// ════════════════════════════════════════════

static void testCancelVsCompletionRace() {
	TEST("#3: cancel x completion race loop with same-ID reuse (no state residue)") {
		for (int round = 0; round < 30; ++round) {
			InferGraph g({2}, {2}, {2});
			g.addNode(std::make_unique<Node>("test", "n", passSchema(),
				[](Node::RunContext& ctx) -> Node::Result {
					const auto* x = ctx.input<Tensor>("x");
					if (!x)
						return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
					ctx.output("y", Value(std::make_unique<Tensor>(*x)));
					// 产出后微睡：把 cancel 竞态落在“产出已写入、搬运/收尾未完成”窗口内
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

			// 复用同 ID：旧轮执行态清理必然完成，新一轮不得读入残留输入/输出
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

// ════════════════════════════════════════════
// #8-12 修复回归：finishing 窗口并发 feed —— 要么接受要么 DuplicateTask
// ════════════════════════════════════════════

static void testConcurrentFeedDuringFinalize() {
	TEST("#8-12: concurrent feed during active/finalizing task -> accepted or DuplicateTask only") {
		for (int round = 0; round < 20; ++round) {
			InferGraph g({2}, {2}, {2});
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
							++protocolViolations; // 只允许收尾窗口拒绝
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

			// 复用同 ID：新一轮结果干净
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

// ════════════════════════════════════════════
// #8-13 修复回归：声明坐标拼写错误在提交期即拒绝（不留 Running 残留）
// ════════════════════════════════════════════

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

		// 修正后提交成功（提交入口无状态污染）
		g.submit("t", "n", "y");
		const auto r = g.waitForResult("t", 2s);
		CHECK(r.status == TaskStatus::Succeeded, "corrected submit must succeed");
		CHECK(std::abs(g.takeOutputTensor("t", "n", "y").item<float>() - 1.0f) < 1e-6f,
			  "corrected submit result must be intact");
	}
	END_TEST();
}

// ════════════════════════════════════════════

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
