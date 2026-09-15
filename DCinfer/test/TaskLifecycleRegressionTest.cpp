// 任务生命周期回归测试（发布前审查 F04-F10 修复的确定性用例）
//
// 锁定"轮次化任务状态"重构后的行为契约：
//   F04 同 ID 复用竞态：旧轮次排队 lambda 不得消费新一轮输入；取消后复用必须成功
//   F05 节点失败优先于输出存在性：先产出后失败 / 必需输出缺失必须 Failed 且不传播
//   F07 完成回调异常隔离：终止事务完整，wait 契约（返回即可读）不被破坏
//   F08 releaseTask 仅终态可释放：活动任务拒绝时声明/结果/诊断保持不动
//   F09 句柄析构：未提交立即释放输入；在飞弃置即请求取消并随即回收（协作式）
//   F10 提交失败（不可达声明）不留 Running 残留，可重试
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
