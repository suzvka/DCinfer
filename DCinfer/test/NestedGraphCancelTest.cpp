// 嵌套子图取消联动回归测试（发布前审查 H-4 修复的确定性用例）
//
//   1) 子图信号阻塞 + 父图 Compute 单线程：父任务挂起占住唯一 Compute 线程，
//      后继任务排队；宿主 cancel(父) 后 RunFn ≤100ms 感知、取消子图任务并
//      解围返回——线程释放、排队任务恢复执行（取消跨嵌套边界生效）
//   2) 子图正常路径：导出节点数据往返正确（子图任务由父任务 ID 驱动）
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

// ════════════════════════════════════════════
// H-4-1：父取消跨嵌套边界解围
// ════════════════════════════════════════════

static void testNestedCancelUnblocksParentThread() {
	TEST("H-4: cancel(parent) unwinds blocked subgraph and frees the compute thread") {
		// 子图：n → m（m 绑定信号，未置位 → 默认阻塞）→ 声明输出 m.y。
		// 信号阻塞使子图任务停滞挂起 → exportNode 的 RunFn 等待不再无限期占线程。
		InferGraph sub;
		sub.addNode(std::make_unique<Node>("test", "n", passSchema(), passRunFn()));
		auto m = std::make_unique<Node>("test", "m", passSchema(), passRunFn());
		m->bindSignal(sub.signalStore(), "gate"); // 绑定后未置位 → isBlocked(true)：默认阻断
		sub.addNode(std::move(m));
		sub.connect("n", "y", "m", "x");
		sub.bindInput("x", "n", "x");
		sub.bindOutput("y", "m", "y");

		auto subNode = sub.exportNode("sub");

		// 父图：Compute 单线程（默认）——挂起的父任务独占唯一 Compute 线程
		InferGraph g;
		g.addNode(std::move(subNode));
		// 独立直通节点 q（显式 Compute）：验证解围后线程恢复可用
		g.addNode(std::make_unique<Node>("test", "q", passSchema(), passRunFn(),
										 ThreadPoolAffinity::Compute));
		g.bindInput("x", "sub", "x");
		g.bindOutput("y", "sub", "y");

		// p1：经子图（内部信号阻塞 → RunFn 挂起，占住 Compute 线程）
		g.feedInput("p1", "sub", "x", floatValue(1.0f));
		g.submit("p1", "sub", "y");

		// p2：经独立节点 q（Compute）——排在 p1 之后等待线程释放
		g.feedInput("p2", "q", "x", floatValue(2.0f));
		g.submit("p2", "q", "y");

		// 挂起确认：p1 因内部阻塞无法完成；p2 因唯一 Compute 线程被占而排队
		CHECK(g.waitForResult("p1", 400ms).status == TaskStatus::Running,
			  "nested task blocked by subgraph signal must stay Running");
		CHECK(g.waitForResult("p2", 200ms).status == TaskStatus::Running,
			  "queued compute task must not run while the thread is held");

		// 宿主解围：cancel(p1) → RunFn 轮询感知 → 取消子图任务 → 返回 → 线程释放
		CHECK(g.cancel("p1"), "cancel on active task should be accepted");
		const auto r1 = g.waitForResult("p1", 2s);
		CHECK(r1.status == TaskStatus::Cancelled, "cancelled parent must reach Cancelled state");

		const auto r2 = g.waitForResult("p2", 3s);
		CHECK(r2.status == TaskStatus::Succeeded,
			  "queued task must complete after the compute thread is freed");
		CHECK(std::abs(g.takeOutputTensor("p2", "q", "y").item<float>() - 2.0f) < 1e-6f,
			  "freed-thread task output must match its input");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// H-4-2：子图正常路径（数据往返）
// ════════════════════════════════════════════

static void testNestedNormalRoundTrip() {
	TEST("H-4: nested export round-trip (subgraph task driven by parent task id)") {
		InferGraph sub;
		sub.addNode(std::make_unique<Node>("test", "n", passSchema(), passRunFn()));
		sub.bindInput("x", "n", "x");
		sub.bindOutput("y", "n", "y");

		auto subNode = sub.exportNode("sub");

		InferGraph g;
		g.addNode(std::move(subNode));
		g.bindInput("x", "sub", "x");
		g.bindOutput("y", "sub", "y");

		g.feedInput("p", "sub", "x", floatValue(5.0f));
		g.submit("p", "sub", "y");

		const auto r = g.waitForResult("p", 3s);
		CHECK(r.status == TaskStatus::Succeeded, "nested task should succeed");
		CHECK(g.hasOutput("p", "sub", "y"), "nested declared output must be retrievable");
		CHECK(std::abs(g.takeOutputTensor("p", "sub", "y").item<float>() - 5.0f) < 1e-6f,
			  "nested round-trip value must match");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// CORE-04：子图先析构 → 导出节点执行显式失败（非悬垂段错误）
// ════════════════════════════════════════════

static void testExportedNodeAfterSubgraphDestructionFailsExplicitly() {
	TEST("CORE-04: exported node executed after subgraph destruction fails explicitly") {
		std::unique_ptr<Node> subNode;
		{
			InferGraph sub;
			sub.addNode(std::make_unique<Node>("test", "n", passSchema(), passRunFn()));
			sub.bindInput("x", "n", "x");
			sub.bindOutput("y", "n", "y");
			subNode = sub.exportNode("sub");
		} // 子图析构：生命周期契约违规的实证场景（此前为悬垂 this 段错误）

		InferGraph g;
		g.addNode(std::move(subNode));
		g.bindInput("x", "sub", "x");
		g.bindOutput("y", "sub", "y");

		g.feedInput("p", "sub", "x", floatValue(1.0f));
		g.submit("p", "sub", "y");

		// 生命周期哨兵检测到子图已析构 → 节点失败 → 任务 Failed（而非崩溃）
		const auto r = g.waitForResult("p", 3s);
		CHECK(r.status == TaskStatus::Failed, "run against destroyed subgraph must fail explicitly");
	}
	END_TEST();
}

// ════════════════════════════════════════════

int main() {
	try {
		testNestedCancelUnblocksParentThread();
		testNestedNormalRoundTrip();
		testExportedNodeAfterSubgraphDestructionFailsExplicitly();
	} catch (const std::exception& e) {
		std::cerr << "UNEXPECTED EXCEPTION: " << e.what() << std::endl;
		return 1;
	}

	if (failures != 0) {
		std::cerr << failures << " test(s) failed" << std::endl;
		return 1;
	}
	std::cout << "All nested graph cancel tests passed" << std::endl;
	return 0;
}
