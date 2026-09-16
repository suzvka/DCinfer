// 执行并发回归测试（发布前审查 H-1/H-2 修复的确定性用例）
//
//   H-1 多输入双触发：两上游并发传播到同一汇聚节点——写入与就绪判定原子化后，
//       节点恰执行一次、无 Error 级诊断、任务 Succeeded
//   H-1 节点闸竞争：两任务竞争同一节点执行租约——败者经重投排队执行，
//       不再被 Reentrant 误判为任务失败（伪失败回归）
//   H-2 失败闭环：完成回调连续抛出致非 NodeException 逃逸流水线——
//       统一记录 Error 诊断并收束 Failed，任务不再永久 Running
#include <atomic>
#include <chrono>
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

static bool hasErrorLevel(const std::vector<TaskError>& errors) {
	for (const auto& e : errors) {
		if (e.level == DiagnosticLevel::Error)
			return true;
	}
	return false;
}

// ════════════════════════════════════════════
// H-1：多输入节点并发双触发 —— 恰执行一次
// ════════════════════════════════════════════

static void testConcurrentFanInTriggersOnce() {
	TEST("H-1: concurrent fan-in triggers join node exactly once (atomic ready)") {
		// 双线程 Operator 池：两上游分支真正并发完成并同时向汇聚节点传播
		InferGraph g({1}, {2}, {1});

		std::atomic<int> joinRuns{0};
		std::promise<void> u1In, u2In;
		auto u1InF = u1In.get_future();
		auto u2InF = u2In.get_future();

		// 两上游在 RunFn 内互相等待对方进入——最大化"同拍传播"的概率
		g.addNode(std::make_unique<Node>("test", "u1", passSchema(),
			[&](Node::RunContext& ctx) -> Node::Result {
				u1In.set_value();
				u2InF.wait();
				const auto* x = ctx.input<Tensor>("x");
				if (!x)
					return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
				ctx.output("y", Value(std::make_unique<Tensor>(*x)));
				return ctx.success();
			}));
		g.addNode(std::make_unique<Node>("test", "u2", passSchema(),
			[&](Node::RunContext& ctx) -> Node::Result {
				u2In.set_value();
				u1InF.wait();
				const auto* x = ctx.input<Tensor>("x");
				if (!x)
					return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
				ctx.output("y", Value(std::make_unique<Tensor>(*x)));
				return ctx.success();
			}));

		// 汇聚节点：两输入齐 → 执行（计数执行次数，验证无重复触发）
		Node::Schema joinSchema;
		joinSchema.inputs = {Node::Port::in<float>("p1"), Node::Port::in<float>("p2")};
		joinSchema.outputs = {Node::Port::out<float>("z")};
		g.addNode(std::make_unique<Node>("test", "join", joinSchema,
			[&](Node::RunContext& ctx) -> Node::Result {
				++joinRuns;
				const auto* a = ctx.input<Tensor>("p1");
				const auto* b = ctx.input<Tensor>("p2");
				if (!a || !b)
					return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
				ctx.output("z", Value(std::make_unique<Tensor>(
					floatTensor(a->item<float>() + b->item<float>()))));
				return ctx.success();
			}));

		g.connect("u1", "y", "join", "p1");
		g.connect("u2", "y", "join", "p2");

		g.feedInput("t", "u1", "x", floatValue(1.0f));
		g.feedInput("t", "u2", "x", floatValue(2.0f));
		g.submit("t", "join", "z");

		const auto r = g.waitForResult("t", 3s);
		CHECK(r.status == TaskStatus::Succeeded, "fan-in task should succeed");
		CHECK(joinRuns.load() == 1, "join node must run exactly once (no duplicate trigger)");
		CHECK(!hasErrorLevel(r.errors), "no Error-level diagnostics expected on the happy path");
		CHECK(std::abs(g.takeOutputTensor("t", "join", "z").item<float>() - 3.0f) < 1e-6f,
			  "fan-in result must sum both inputs");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// H-1：共享图节点闸竞争 —— 败者重投不判死
// ════════════════════════════════════════════

static void testSharedGraphNodeContention() {
	TEST("H-1: two tasks contending on the same node both complete via retry") {
		InferGraph g({1}, {2}, {1});

		std::atomic<int> runs{0};
		std::promise<void> firstEntered, releaseFirst;
		auto firstEnteredF = firstEntered.get_future();
		auto releaseFirstF = releaseFirst.get_future().share();

		g.addNode(std::make_unique<Node>("test", "n", passSchema(),
			[&](Node::RunContext& ctx) -> Node::Result {
				// 首次执行阻塞：持闸期间另一任务对该节点的提交必然遭拒（Reentrant）
				if (runs.fetch_add(1) == 0) {
					firstEntered.set_value();
					releaseFirstF.wait();
				}
				const auto* x = ctx.input<Tensor>("x");
				if (!x)
					return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
				ctx.output("y", Value(std::make_unique<Tensor>(*x)));
				return ctx.success();
			}));

		// ta：占住节点执行租约（首执行阻塞中）
		g.feedInput("ta", "n", "x", floatValue(1.0f));
		g.submit("ta", "n", "y");
		firstEnteredF.wait();

		// tb：并发提交同一节点 → 闸忙 → 登记重投（不再记录 Error 判死）
		g.feedInput("tb", "n", "x", floatValue(2.0f));
		g.submit("tb", "n", "y");
		std::this_thread::sleep_for(100ms); // 让 tb 的提交经历"闸忙 → 登记"

		releaseFirst.set_value(); // 放行 ta → 闸释放 → tb 重投执行

		const auto ra = g.waitForResult("ta", 3s);
		CHECK(ra.status == TaskStatus::Succeeded, "first task should succeed");
		const auto rb = g.waitForResult("tb", 3s);
		CHECK(rb.status == TaskStatus::Succeeded,
			  "contending task must complete via retry (no pseudo-failure)");
		CHECK(std::abs(g.takeOutputTensor("ta", "n", "y").item<float>() - 1.0f) < 1e-6f,
			  "first task output must match");
		CHECK(std::abs(g.takeOutputTensor("tb", "n", "y").item<float>() - 2.0f) < 1e-6f,
			  "contending task output must match");
		CHECK(!hasErrorLevel(rb.errors), "contending task must carry no Error-level diagnostics");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// H-2：非 NodeException 逃逸 —— 失败闭环不中断
// ════════════════════════════════════════════

static void testNonNodeExceptionClosure() {
	TEST("H-2: escaping non-NodeException -> Failed with diagnostic (no permanent Running)") {
		InferGraph g;

		auto n = std::make_unique<Node>("test", "n", passSchema(), passRunFn());
		// 完成回调总是抛出：execute 异常分支的二次回调再次抛出 → 异常逃逸流水线。
		// 修复前穿过调度层直达池 worker（仅 stderr），任务无诊断永久挂起。
		n->setCompletionCallback([](const Node::TaskId&, const Node::Result&) {
			throw std::runtime_error("deliberate completion callback throw");
		});
		g.addNode(std::move(n));
		g.bindInput("x", "n", "x");
		g.bindOutput("y", "n", "y");

		g.feedInput("t", "n", "x", floatValue(1.0f));
		g.submitBound("t");

		const auto r = g.waitForResult("t", 2s);
		CHECK(r.status == TaskStatus::Failed,
			  "task must be finalized as Failed, not stuck Running");
		CHECK(hasMessage(r.errors, "non-NodeException escaped node execution"),
			  "escaping exception must be recorded as Error diagnostic");
	}
	END_TEST();
}

// ════════════════════════════════════════════

int main() {
	try {
		testConcurrentFanInTriggersOnce();
		testSharedGraphNodeContention();
		testNonNodeExceptionClosure();
	} catch (const std::exception& e) {
		std::cerr << "UNEXPECTED EXCEPTION: " << e.what() << std::endl;
		return 1;
	}

	if (failures != 0) {
		std::cerr << failures << " test(s) failed" << std::endl;
		return 1;
	}
	std::cout << "All execution concurrency tests passed" << std::endl;
	return 0;
}
