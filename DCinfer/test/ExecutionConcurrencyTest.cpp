// 执行并发回归测试：多输入双触发 / 节点闸竞争 / 非 NodeException 逃逸闭环
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

static void testConcurrentFanInTriggersOnce() {
	TEST("H-1: concurrent fan-in triggers join node exactly once (atomic ready)") {
		// 双 Operator 槽位：两上游真正并发传播到汇聚节点
		auto sched = std::make_shared<ResourceScheduler>(SchedulerConfig{1, 2, 1});
		InferGraph g(sched);

		std::atomic<int> joinRuns{0};
		std::promise<void> u1In, u2In;
		auto u1InF = u1In.get_future();
		auto u2InF = u2In.get_future();

		// 两上游互相等待对方进入，最大化同拍传播概率
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

static void testSharedGraphNodeContention() {
	TEST("H-1: two tasks contending on the same node both complete via retry") {
		auto sched = std::make_shared<ResourceScheduler>(SchedulerConfig{1, 2, 1});
		InferGraph g(sched);

		std::atomic<int> runs{0};
		std::promise<void> firstEntered, releaseFirst;
		auto firstEnteredF = firstEntered.get_future();
		auto releaseFirstF = releaseFirst.get_future().share();

		g.addNode(std::make_unique<Node>("test", "n", passSchema(),
			[&](Node::RunContext& ctx) -> Node::Result {
				// 首次执行阻塞：持闸期间另一任务提交遭 Reentrant 拒绝
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

		g.feedInput("ta", "n", "x", floatValue(1.0f));
		g.submit("ta", "n", "y");
		firstEnteredF.wait();

		g.feedInput("tb", "n", "x", floatValue(2.0f));
		g.submit("tb", "n", "y");
		std::this_thread::sleep_for(100ms);

		releaseFirst.set_value();

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

static void testNonNodeExceptionClosure() {
	TEST("H-2: escaping non-NodeException -> Failed with diagnostic (no permanent Running)") {
		InferGraph g;

		auto n = std::make_unique<Node>("test", "n", passSchema(), passRunFn());
		// 完成回调抛出后，catch 路径的重试通知必须为 no-op，回调至多一次
		std::atomic<int> callbackCalls{0};
		n->setCompletionCallback([&](const Node::TaskId&, const Node::Result&) {
			++callbackCalls;
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
		CHECK(callbackCalls.load() == 1,
			  "completion callback must be invoked exactly once even when it throws");
	}
	END_TEST();
}

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
