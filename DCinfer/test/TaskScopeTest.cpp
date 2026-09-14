// TaskScope 作用域句柄 单元测试
//
// 覆盖：同步 run 路径 / 多绑定输出收集 / 析构释放已完成任务 /
//       析构取消并释放在运行任务 / run 超时语义 / 移动语义 /
//       无绑定抛 NoDeclaration / 失败任务闭环
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <latch>
#include <memory>
#include <optional>
#include <thread>

#include "TaskScope.h"
#include "Tensor.hpp"

using namespace DC;

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

static Tensor makeFloatTensor(float value) {
	Tensor t(TensorType::Float, sizeof(float));
	t = value;
	return t;
}

static Value makeFloatValue(float value) {
	return Value(std::make_unique<Tensor>(makeFloatTensor(value)));
}

// ── 恒等算子 ──

static Node::Schema identitySchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("x")};
	s.outputs = {Node::Port::out<float>("y")};
	return s;
}

static Node::RunFn identityRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto& inVal = ctx.peek("x");
		const auto* t = inVal.as<Tensor>();
		if (!t)
			return ctx.failure(Node::Status::InvalidInput, "not a Tensor");

		ctx.output("y", Value(std::make_unique<Tensor>(*t)));
		return ctx.success();
	};
}

// ── 阻塞算子：进入在飞执行后等待测试放行（制造稳定 Running）──

static Node::RunFn blockingIdentityRunFn(std::atomic<bool>& started, std::latch& gate) {
	return [&started, &gate](Node::RunContext& ctx) -> Node::Result {
		started.store(true, std::memory_order_release);
		gate.wait();
		const auto& inVal = ctx.peek("x");
		const auto* t = inVal.as<Tensor>();
		if (!t)
			return ctx.failure(Node::Status::InvalidInput, "not a Tensor");

		ctx.output("y", Value(std::make_unique<Tensor>(*t)));
		return ctx.success();
	};
}

// ── 失败算子：自报执行失败 ──

static Node::RunFn failingRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		return ctx.failure(Node::Status::ExecutionFailed, "intentional failure");
	};
}

// ════════════════════════════════════════════
// 1. 同步 run：返回输出并释放任务
// ════════════════════════════════════════════

void testRunSync() {
	TEST("scope: run() returns outputs and releases task") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(), identityRunFn()));
		graph.bindOutput("y", "id1", "y");

		TaskScope task{graph, "t1"};
		task.feed("id1", "x", makeFloatValue(7.5f)); // Value 重载
		auto r = task.run();

		CHECK(r.status == TaskStatus::Succeeded, "run should succeed");
		CHECK(r.outputs.size() == 1, "exactly one bound output harvested");
		auto out = r.takeTensor("id1", "y");
		CHECK(std::abs(out.item<float>() - 7.5f) < 1e-6f, "output value should be 7.5");
		CHECK(graph.taskStatus("t1") == TaskStatus::Unknown, "run must release the task");
		CHECK(!graph.hasOutput("t1", "id1", "y"), "released task has no residual output");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 2. 多绑定输出：run 收集全部
// ════════════════════════════════════════════

void testRunHarvestsAllBoundOutputs() {
	TEST("scope: run() harvests all bound outputs") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		graph.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
		graph.bindOutput("yA", "id_a", "y");
		graph.bindOutput("yB", "id_b", "y");

		TaskScope task{graph, "t1"};
		task.feed("id_a", "x", makeFloatTensor(1.0f)) // Tensor 重载（链式）
			.feed("id_b", "x", makeFloatTensor(2.0f));
		auto r = task.run();

		CHECK(r.status == TaskStatus::Succeeded, "run should succeed");
		CHECK(r.outputs.size() == 2, "both bound outputs harvested");
		CHECK(std::abs(r.takeTensor("id_a", "y").item<float>() - 1.0f) < 1e-6f, "id_a output == 1.0");
		CHECK(std::abs(r.takeTensor("id_b", "y").item<float>() - 2.0f) < 1e-6f, "id_b output == 2.0");
		CHECK(graph.taskStatus("t1") == TaskStatus::Unknown, "task released after run");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 3. 析构释放已完成任务（wait 后不手动释放）
// ════════════════════════════════════════════

void testDestructorReleasesCompletedTask() {
	TEST("scope: destructor releases completed task") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(), identityRunFn()));
		graph.bindOutput("y", "id1", "y");

		TaskStatus waited = TaskStatus::Unknown;
		bool readableAfterWait = false;
		{
			TaskScope task{graph, "t1"};
			task.feed("id1", "x", makeFloatTensor(5.0f)).submit();
			waited = task.wait(std::chrono::milliseconds(5000)).status;
			readableAfterWait = graph.hasOutput("t1", "id1", "y");
		} // 不手动释放：析构负责清理

		CHECK(waited == TaskStatus::Succeeded, "task should succeed");
		CHECK(readableAfterWait, "output readable after wait");
		CHECK(graph.taskStatus("t1") == TaskStatus::Unknown, "dtor should release completed task");
		CHECK(!graph.hasOutput("t1", "id1", "y"), "released task has no residual output");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 4. 析构取消并释放在运行任务
// ════════════════════════════════════════════

void testDestructorCancelsRunningTask() {
	TEST("scope: destructor cancels and releases running task") {
		// 同步原语先于 graph 声明：确保其在图析构（线程池 join）之后才析构
		std::atomic<bool> started{false};
		std::latch gate{1};
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(),
											 blockingIdentityRunFn(started, gate)));
		graph.bindOutput("y", "id1", "y");

		{
			TaskScope task{graph, "t1"};
			task.feed("id1", "x", makeFloatTensor(1.0f)).submit();
			// 等待节点进入在飞执行（保证析构时任务处于稳定 Running）
			for (int i = 0; i < 5000 && !started.load(std::memory_order_acquire); ++i)
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			CHECK(started.load(std::memory_order_acquire), "blocking node should start");
		} // 析构：cancel（同步终止）→ releaseTask

		TaskStatus after = graph.taskStatus("t1");
		bool residual = graph.hasOutput("t1", "id1", "y");
		gate.count_down(); // 先放行在飞 lambda：任何断言失败都不遗留阻塞线程

		CHECK(after == TaskStatus::Unknown, "dtor should cancel+release running task");
		CHECK(!residual, "no residual output after release");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 5. run 超时：不取不释放，析构兜底
// ════════════════════════════════════════════

void testRunTimeoutLeavesTaskAlive() {
	TEST("scope: run(timeout) leaves running task; dtor cleans up") {
		std::atomic<bool> started{false};
		std::latch gate{1};
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(),
											 blockingIdentityRunFn(started, gate)));
		graph.bindOutput("y", "id1", "y");

		TaskStatus runStatus = TaskStatus::Unknown;
		size_t outputCount = 1;
		TaskStatus aliveStatus = TaskStatus::Unknown;
		bool residualBefore = true;
		{
			TaskScope task{graph, "t1"};
			task.feed("id1", "x", makeFloatTensor(9.0f));
			auto r = task.run(std::chrono::milliseconds(50));
			runStatus = r.status;
			outputCount = r.outputs.size();
			aliveStatus = task.status();
			residualBefore = graph.hasOutput("t1", "id1", "y");
			// 确保节点已进入在飞执行（析构走"取消在飞任务"路径）
			for (int i = 0; i < 5000 && !started.load(std::memory_order_acquire); ++i)
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
		} // 析构兜底：取消并释放

		TaskStatus after = graph.taskStatus("t1");
		gate.count_down(); // 先放行在飞 lambda

		CHECK(runStatus == TaskStatus::Running, "run timeout should report Running");
		CHECK(outputCount == 0, "no outputs harvested on timeout");
		CHECK(aliveStatus == TaskStatus::Running, "task still running after wait timeout");
		CHECK(!residualBefore, "result zone untouched on timeout");
		CHECK(after == TaskStatus::Unknown, "dtor should cancel+release after timeout");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 6. 移动语义：源析构空操作，目标析构完成清理
// ════════════════════════════════════════════

void testMoveSemantics() {
	TEST("scope: move semantics (ctor + assign)") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(), identityRunFn()));
		graph.bindOutput("y", "id1", "y");

		// move 构造
		{
			std::optional<TaskScope> keeper;
			{
				TaskScope source{graph, "t1"};
				source.feed("id1", "x", makeFloatTensor(3.0f)).submit();
				source.wait(std::chrono::milliseconds(5000));
				keeper.emplace(std::move(source));
			} // source（被移动）析构：空操作，不得释放任务
			CHECK(graph.taskStatus("t1") == TaskStatus::Succeeded,
				  "moved-from source dtor must not release the task");
			keeper.reset(); // 目标析构：释放
			CHECK(graph.taskStatus("t1") == TaskStatus::Unknown, "adopted scope releases on dtor");
		}

		// move 赋值：先清理自身持有，再接管对方
		{
			TaskScope first{graph, "tA"};
			first.feed("id1", "x", makeFloatTensor(1.0f)).submit();
			first.wait(std::chrono::milliseconds(5000));
			{
				TaskScope second{graph, "tB"};
				second.feed("id1", "x", makeFloatTensor(2.0f)).submit();
				second.wait(std::chrono::milliseconds(5000));

				second = std::move(first);
				CHECK(graph.taskStatus("tB") == TaskStatus::Unknown,
					  "move-assign should clean up the task it previously held");
				CHECK(graph.taskStatus("tA") == TaskStatus::Succeeded,
					  "adopted task must not be released by the move itself");
			} // second 析构：释放接管来的 tA
			CHECK(graph.taskStatus("tA") == TaskStatus::Unknown,
				  "adopted scope releases on dtor");
		} // first（被移动）析构：空操作
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 7. 无 bindOutput 时 run 抛 NoDeclaration
// ════════════════════════════════════════════

void testRunWithoutBindingsThrows() {
	TEST("scope: run() without bindOutput throws NoDeclaration") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(), identityRunFn()));

		TaskScope task{graph, "t1"};
		task.feed("id1", "x", makeFloatTensor(1.0f));

		bool threwNoDecl = false;
		try {
			task.run();
		} catch (const GraphException& e) {
			threwNoDecl = (e.getErrorType() == GraphException::ErrorType::NoDeclaration);
		}
		CHECK(threwNoDecl, "run() should throw GraphException(NoDeclaration)");
		CHECK(graph.taskStatus("t1") == TaskStatus::Unknown, "unsubmitted task stays Unknown");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 8. 失败任务 run：Failed 闭环并释放
// ════════════════════════════════════════════

void testRunFailedTask() {
	TEST("scope: run() on failing node returns Failed and releases") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(), failingRunFn()));
		graph.bindOutput("y", "id1", "y");

		TaskScope task{graph, "t1"};
		task.feed("id1", "x", makeFloatTensor(1.0f));
		auto r = task.run();

		CHECK(r.status == TaskStatus::Failed, "failing node should yield Failed");
		CHECK(!r.errors.empty(), "failure diagnostics should be reported");
		CHECK(r.outputs.empty(), "no outputs on failure");
		CHECK(graph.taskStatus("t1") == TaskStatus::Unknown, "failed task released by run");
	}
	END_TEST();
}

// ── 入口 ──

int main() {
	std::cout << "=== TaskScope tests ===" << std::endl;
	try {
		testRunSync();
		testRunHarvestsAllBoundOutputs();
		testDestructorReleasesCompletedTask();
		testDestructorCancelsRunningTask();
		testRunTimeoutLeavesTaskAlive();
		testMoveSemantics();
		testRunWithoutBindingsThrows();
		testRunFailedTask();

		if (failures == 0) {
			std::cout << "\nAll TaskScope tests passed!" << std::endl;
		} else {
			std::cout << "\n" << failures << " test(s) FAILED!" << std::endl;
		}
		return failures;
	} catch (const std::exception& e) {
		std::cerr << "Test failure: " << e.what() << std::endl;
		return -1;
	}
}
