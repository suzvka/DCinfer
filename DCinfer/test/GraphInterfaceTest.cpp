// GraphInterface 公开接口 单元测试
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "InferGraph.h"
#include "GraphException.h"
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

static Tensor floatTensor(float value) {
	auto t = Tensor::Create<float>();
	t = value;
	return t;
}

// ── 增 1 算子 ──

static Node::Schema incSchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("x")};
	s.outputs = {Node::Port::out<float>("y")};
	return s;
}

static Node::RunFn incRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto* x = ctx.input<Tensor>("x");
		if (!x)
			return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
		auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
		*t = x->item<float>() + 1.0f;
		ctx.output("y", Value(std::move(t)));
		return ctx.success();
	};
}

// ── 慢速增 1 算子（保证句柄析构发生在任务运行中）──

static Node::RunFn slowIncRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
		const auto* x = ctx.input<Tensor>("x");
		if (!x)
			return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
		auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
		*t = x->item<float>() + 1.0f;
		ctx.output("y", Value(std::move(t)));
		return ctx.success();
	};
}

// ── 加法算子 ──

static Node::Schema addSchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("a"), Node::Port::in<float>("b")};
	s.outputs = {Node::Port::out<float>("s")};
	return s;
}

static Node::RunFn addRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto* a = ctx.input<Tensor>("a");
		const auto* b = ctx.input<Tensor>("b");
		if (!a || !b)
			return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
		auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
		*t = a->item<float>() + b->item<float>();
		ctx.output("s", Value(std::move(t)));
		return ctx.success();
	};
}

// ════════════════════════════════════════════
// 别名路径 = 坐标路径（同一冻结图）
// ════════════════════════════════════════════

static void testAliasPathMatchesCoordinatePath() {
	InferGraph graph;
	graph.addNode(std::make_unique<Node>("Builtin", "inc", incSchema(), incRunFn()));
	graph.bindInput("num", "inc", "x");
	graph.bindOutput("result", "inc", "y");

	auto api = graph.interface();
	const auto inAliases = api.inputAliases();
	const auto outAliases = api.outputAliases();
	CHECK(inAliases.size() == 1 && inAliases[0] == "num", "input aliases should list 'num'");
	CHECK(outAliases.size() == 1 && outAliases[0] == "result", "output aliases should list 'result'");

	// 别名路径
	auto task = api.createTask();
	task.feed("num", floatTensor(41.0f));
	CHECK(task.run().status == TaskStatus::Succeeded, "alias path should complete");
	const auto out1 = task.take("result");
	CHECK(out1.as<Tensor>() != nullptr && std::abs(out1.as<Tensor>()->item<float>() - 42.0f) < 1e-6f,
		  "alias path should yield 42");

	auto task2 = api.createTask();
	task2.feed("num", floatTensor(7.0f));
	CHECK(task2.run().status == TaskStatus::Succeeded, "second task should complete");
	CHECK(std::abs(task2.takeTensor("result").item<float>() - 8.0f) < 1e-6f, "second task should yield 8");

	// 同一冻结图：坐标寻址照常（高级入口）
	graph.feedInput("t3", "inc", "x", floatTensor(1.0f));
	graph.submitBound("t3");
	CHECK(graph.waitForResult("t3").status == TaskStatus::Succeeded, "coordinate path should complete");
	CHECK(std::abs(graph.takeOutputTensor("t3", "inc", "y").item<float>() - 2.0f) < 1e-6f,
		  "coordinate path should yield 2");
}

// ════════════════════════════════════════════
// 未知别名：报错并列出全部可用别名
// ════════════════════════════════════════════

static void testUnknownAliasListsAvailable() {
	InferGraph graph;
	graph.addNode(std::make_unique<Node>("Builtin", "adder", addSchema(), addRunFn()));
	graph.bindInput("prompt", "adder", "a");
	graph.bindInput("context", "adder", "b");
	graph.bindOutput("answer", "adder", "s");

	auto api = graph.interface();
	auto task = api.createTask();

	bool badFeed = false;
	std::string feedMsg;
	try {
		task.feed("missing", floatTensor(1.0f));
	} catch (const GraphException& e) {
		badFeed = e.getErrorType() == GraphException::ErrorType::InvalidBinding;
		feedMsg = e.what();
	}
	CHECK(badFeed, "unknown input alias should throw InvalidBinding");
	CHECK(feedMsg.find("prompt") != std::string::npos && feedMsg.find("context") != std::string::npos,
		  "feed error should list available input aliases");

	bool badTake = false;
	std::string takeMsg;
	try {
		static_cast<void>(task.take("missing"));
	} catch (const GraphException& e) {
		badTake = e.getErrorType() == GraphException::ErrorType::InvalidBinding;
		takeMsg = e.what();
	}
	CHECK(badTake, "unknown output alias should throw InvalidBinding");
	CHECK(takeMsg.find("answer") != std::string::npos, "take error should list available output aliases");

	// 正确别名照常
	task.feed("prompt", floatTensor(2.0f));
	task.feed("context", floatTensor(3.0f));
	CHECK(task.run().status == TaskStatus::Succeeded, "valid aliases should run");
	CHECK(std::abs(task.takeTensor("answer").item<float>() - 5.0f) < 1e-6f, "result should be 5");
}

// ════════════════════════════════════════════
// 取接口即定型：interface() 冻结图
// ════════════════════════════════════════════

static void testInterfaceFreezesGraph() {
	InferGraph graph;
	graph.addNode(std::make_unique<Node>("Builtin", "inc", incSchema(), incRunFn()));
	graph.bindInput("num", "inc", "x");
	graph.bindOutput("result", "inc", "y");

	graph.interface(); // 取接口即定型

	bool bindRejected = false;
	try {
		graph.bindInput("extra", "inc", "x");
	} catch (const GraphException& e) {
		bindRejected = e.getErrorType() == GraphException::ErrorType::Frozen;
	}
	CHECK(bindRejected, "interface() should freeze the graph (bindInput rejected)");

	bool addRejected = false;
	try {
		graph.addNode(std::make_unique<Node>("Builtin", "inc2", incSchema(), incRunFn()));
	} catch (const GraphException& e) {
		addRejected = e.getErrorType() == GraphException::ErrorType::Frozen;
	}
	CHECK(addRejected, "interface() should freeze the graph (addNode rejected)");
}

// ════════════════════════════════════════════
// 绑定坐标在创建时校验（fail-fast）
// ════════════════════════════════════════════

static void testBindingValidationAtCreation() {
	// 输入绑定 → 不存在的节点
	{
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "inc", incSchema(), incRunFn()));
		graph.bindInput("num", "ghost", "x");
		bool threw = false;
		try {
			graph.interface();
		} catch (const GraphException& e) {
			threw = e.getErrorType() == GraphException::ErrorType::NodeNotFound;
		}
		CHECK(threw, "input binding on missing node should fail at interface()");
	}
	// 输出绑定 → 不存在的端口
	{
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "inc", incSchema(), incRunFn()));
		graph.bindOutput("result", "inc", "nope");
		bool threw = false;
		try {
			graph.interface();
		} catch (const GraphException& e) {
			threw = e.getErrorType() == GraphException::ErrorType::PortNotFound;
		}
		CHECK(threw, "output binding on missing port should fail at interface()");
	}
	// 输入绑定 → 方向错误的端口（输出端口）
	{
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "inc", incSchema(), incRunFn()));
		graph.bindInput("num", "inc", "y");
		bool threw = false;
		try {
			graph.interface();
		} catch (const GraphException& e) {
			threw = e.getErrorType() == GraphException::ErrorType::PortNotFound;
		}
		CHECK(threw, "input binding on output port should fail at interface()");
	}
}

// ════════════════════════════════════════════
// 任务句柄：移动交权 / 析构释放 / 不取消在飞任务
// ════════════════════════════════════════════

static void testTaskHandleLifecycle() {
	InferGraph graph;
	graph.addNode(std::make_unique<Node>("Builtin", "inc", incSchema(), incRunFn()));
	graph.bindInput("num", "inc", "x");
	graph.bindOutput("result", "inc", "y");
	auto api = graph.interface();

	// 移动语义：源句柄交权后不再持有
	std::string movedId;
	{
		auto moved = api.createTask();
		movedId = moved.taskId();
		auto owner = std::move(moved);
		owner.feed("num", floatTensor(3.0f));
		CHECK(owner.run().status == TaskStatus::Succeeded, "moved handle should run");
		CHECK(std::abs(owner.takeTensor("result").item<float>() - 4.0f) < 1e-6f,
			  "moved handle should yield 4");
	}
	CHECK(graph.taskStatus(movedId) == TaskStatus::Unknown, "moved handle should be released at scope end");

	// 终止后析构 → 资源释放（taskStatus 回到 Unknown）
	std::string doneId;
	{
		auto task = api.createTask();
		doneId = task.taskId();
		task.feed("num", floatTensor(9.0f));
		CHECK(task.run().status == TaskStatus::Succeeded, "scoped run should succeed");
		static_cast<void>(task.take("result"));
	}
	CHECK(graph.taskStatus(doneId) == TaskStatus::Unknown, "destructor should release terminated task");

	// 在飞任务的句柄析构不取消任务（慢算子保证析构时仍在运行）
	InferGraph slowGraph;
	slowGraph.addNode(std::make_unique<Node>("Builtin", "slow", incSchema(), slowIncRunFn()));
	slowGraph.bindInput("num", "slow", "x");
	slowGraph.bindOutput("result", "slow", "y");
	auto slowApi = slowGraph.interface();

	std::string runningId;
	{
		auto task = slowApi.createTask();
		runningId = task.taskId();
		task.feed("num", floatTensor(10.0f));
		slowGraph.submitBound(runningId); // 异步提交（句柄外），析构发生在运行中
	}
	const auto slowResult = slowGraph.waitForResult(runningId);
	CHECK(slowResult.status == TaskStatus::Succeeded, "destruction must not cancel the in-flight task");
	CHECK(std::abs(slowGraph.takeOutputTensor(runningId, "slow", "y").item<float>() - 11.0f) < 1e-6f,
		  "in-flight result should be intact");
}

// ════════════════════════════════════════════
// run() 无输出绑定：转发 NoDeclaration
// ════════════════════════════════════════════

static void testRunWithoutOutputBindings() {
	InferGraph graph;
	graph.addNode(std::make_unique<Node>("Builtin", "inc", incSchema(), incRunFn()));
	graph.bindInput("num", "inc", "x"); // 未声明输出绑定

	auto api = graph.interface();
	auto task = api.createTask();
	task.feed("num", floatTensor(1.0f));

	bool threw = false;
	try {
		task.run();
	} catch (const GraphException& e) {
		threw = e.getErrorType() == GraphException::ErrorType::NoDeclaration;
	}
	CHECK(threw, "run() without output bindings should throw NoDeclaration");
}

// ═══════════════════════════════════════════════════════════
// 统一任务句柄：同步与异步同级（submit / wait / status / cancel）
// ═══════════════════════════════════════════════════════════

// ── 必败算子（触发 Error 级诊断）──

static Node::RunFn failRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		return ctx.failure(Node::Status::InvalidInput, "boom");
	};
}

static void testAsyncSubmitWaitStatusCancel() {
	InferGraph graph;
	graph.addNode(std::make_unique<Node>("Builtin", "slow", incSchema(), slowIncRunFn()));
	graph.bindInput("num", "slow", "x");
	graph.bindOutput("result", "slow", "y");
	auto api = graph.interface();

	auto task = api.createTask();
	CHECK(task.status() == TaskStatus::Unknown, "fresh handle should be Unknown");

	task.feed("num", floatTensor(10.0f)).submit(); // 链式组装 + 异步启动
	CHECK(task.status() == TaskStatus::Running, "submitted task should be Running");

	auto done = task.wait(); // 异步配套：同步等待终止
	CHECK(done.status == TaskStatus::Succeeded, "async path should complete");
	CHECK(task.status() == TaskStatus::Succeeded, "status should be Succeeded after wait");
	CHECK(task.has("result"), "output should exist after completion");
	CHECK(std::abs(task.takeTensor("result").item<float>() - 11.0f) < 1e-6f,
		  "async result should be 11");
}

static void testRunWithTimeout() {
	InferGraph graph;
	graph.addNode(std::make_unique<Node>("Builtin", "slow", incSchema(), slowIncRunFn()));
	graph.bindInput("num", "slow", "x");
	graph.bindOutput("result", "slow", "y");
	auto api = graph.interface();

	auto task = api.createTask();
	task.feed("num", floatTensor(2.0f));
	// 慢算子 200ms > 超时 50ms：返回 Running 而不取消任务
	auto timedOut = task.run(std::chrono::milliseconds(50));
	CHECK(timedOut.status == TaskStatus::Running, "run(timeout) should return Running on timeout");

	auto done = task.wait(); // 任务未被取消：继续等待可完成
	CHECK(done.status == TaskStatus::Succeeded, "task should still complete after timeout");
	CHECK(std::abs(task.takeTensor("result").item<float>() - 3.0f) < 1e-6f,
		  "result should be 3 after timeout recovery");
}

static void testCancelAsyncTask() {
	InferGraph graph;
	graph.addNode(std::make_unique<Node>("Builtin", "slow", incSchema(), slowIncRunFn()));
	graph.bindInput("num", "slow", "x");
	graph.bindOutput("result", "slow", "y");
	auto api = graph.interface();

	auto task = api.createTask();
	task.feed("num", floatTensor(1.0f)).submit();
	CHECK(task.cancel(), "cancel on running task should return true");
	auto r = task.wait();
	CHECK(r.status == TaskStatus::Cancelled, "cancelled task should end Cancelled");
	CHECK(!task.cancel(), "second cancel on terminated task should return false");
}

static void testFailedTaskReportsErrors() {
	InferGraph graph;
	graph.addNode(std::make_unique<Node>("Builtin", "bad", incSchema(), failRunFn()));
	graph.bindInput("num", "bad", "x");
	graph.bindOutput("result", "bad", "y");
	auto api = graph.interface();

	auto task = api.createTask();
	task.feed("num", floatTensor(1.0f));
	auto r = task.run();
	CHECK(r.status == TaskStatus::Failed, "failed node should normalize to Failed");
	CHECK(!task.errors().empty(), "errors() should report node diagnostics");
}

// ════════════════════════════════════════════
// 内核 API 保持坐标唯一寻址（alias 不可用）
// ════════════════════════════════════════════

static void testCoreRemainsCoordinateOnly() {
	InferGraph graph;
	graph.addNode(std::make_unique<Node>("Builtin", "inc", incSchema(), incRunFn()));
	graph.bindInput("num", "inc", "x");
	graph.bindOutput("result", "inc", "y");

	// alias 不能冒充 nodeName 用于内核 API
	bool feedRejected = false;
	try {
		graph.feedInput("t1", "num", "x", floatTensor(1.0f));
	} catch (const GraphException& e) {
		feedRejected = e.getErrorType() == GraphException::ErrorType::NodeNotFound;
	}
	CHECK(feedRejected, "alias must not address core feedInput");

	bool takeRejected = false;
	try {
		static_cast<void>(graph.takeOutput("t2", "result", "y"));
	} catch (const GraphException& e) {
		takeRejected = e.getErrorType() == GraphException::ErrorType::NodeNotFound;
	}
	CHECK(takeRejected, "alias must not address core takeOutput");

	// 正路：坐标寻址
	graph.feedInput("t3", "inc", "x", floatTensor(5.0f));
	graph.submitBound("t3");
	CHECK(graph.waitForResult("t3").status == TaskStatus::Succeeded, "coordinate addressing should complete");
	CHECK(std::abs(graph.takeOutputTensor("t3", "inc", "y").item<float>() - 6.0f) < 1e-6f,
		  "coordinate addressing should yield 6");
}

int main() {
	try {
		testAliasPathMatchesCoordinatePath();
		testUnknownAliasListsAvailable();
		testInterfaceFreezesGraph();
		testBindingValidationAtCreation();
		testTaskHandleLifecycle();
		testRunWithoutOutputBindings();
		testAsyncSubmitWaitStatusCancel();
		testRunWithTimeout();
		testCancelAsyncTask();
		testFailedTaskReportsErrors();
		testCoreRemainsCoordinateOnly();

		if (failures == 0)
			std::cout << "\nAll GraphInterface tests passed!" << std::endl;
		else
			std::cout << "\n" << failures << " test(s) FAILED!" << std::endl;
		return failures;
	} catch (const std::exception& e) {
		std::cerr << "Test failure: " << e.what() << std::endl;
		return -1;
	}
}
