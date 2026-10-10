// 失败闭环 + 提交期拓扑守卫 单元测试
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>

#include "TestHarness.h"
#include "Connector.h"
#include "GraphException.h"

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

static Value makeFloatTensor(float value) {
	auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
	*t = value;
	return Value(std::move(t));
}

static Node::Schema identitySchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("x")};
	s.outputs = {Node::Port::out<float>("y")};
	return s;
}

static Node::RunFn identityRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto* t = ctx.peek("x").as<Tensor>();
		if (!t)
			return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
		ctx.output("y", Value(std::make_unique<Tensor>(*t)));
		return ctx.success();
	};
}

static Node::RunFn failRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		(void)ctx;
		return ctx.failure(Node::Status::ExecutionFailed, "intentional failure");
	};
}

static void testFailureClosesTaskAsFailed() {
	TEST("failure closure: sole declared source fails → task ends Failed") {
		TestHarness harness;
		harness.addNode(std::make_unique<Node>("Builtin", "fail", identitySchema(), failRunFn()));

		harness.feedInput("t1", "fail", "x", makeFloatTensor(1.0f));
		harness.submit("t1", "fail", "y");

		auto result = harness.graph().waitForResult("t1", std::chrono::milliseconds(3000));
		CHECK(result.status == TaskStatus::Failed, "declaration unmet + error diagnostic → Failed");
		CHECK(harness.graph().taskStatus("t1") == TaskStatus::Failed, "taskStatus should be Failed");

		bool failRecorded = false;
		for (const auto& e : harness.taskErrors("t1")) {
			if (e.nodeName == "fail" && e.level == DiagnosticLevel::Error)
				failRecorded = true;
		}
		CHECK(failRecorded, "failing node recorded in diagnostics");
	}
	END_TEST();
}

static void testParallelBranchPartialFailure() {
	TEST("parallel branch: one fails, other satisfies its declaration → Failed with partial outputs") {
		TestHarness harness;

		harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "id_c", identitySchema(), failRunFn()));

		auto bcSchema = Connector::broadcastSchema(2);
		auto bcNode = std::make_unique<Node>("Connector.Broadcast", "bc", bcSchema,
			Connector::broadcastRunFn(), ResourceClass::System);
		bcNode->setConnector(true);
		harness.addNode(std::move(bcNode));
		harness.connect("id_a", "y", "bc", "in");
		harness.connect("bc", "out_0", "id_b", "x");
		harness.connect("bc", "out_1", "id_c", "x");

		harness.feedInput("t1", "id_a", "x", makeFloatTensor(10.0f));
		harness.submit("t1", {{"id_b", "y", 1}, {"id_c", "y", 1}});

		auto result = harness.graph().waitForResult("t1", std::chrono::milliseconds(3000));
		CHECK(result.status == TaskStatus::Failed,
			  "id_c fails → overall declaration unmet → Failed (id_b alone insufficient)");

		CHECK(harness.hasOutput("t1", "id_b", "y"), "successful branch output should be readable");
		CHECK(!harness.hasOutput("t1", "id_c", "y"), "failed branch must not produce output");
	}
	END_TEST();
}

static void testTopologicallyUnreachableRejected() {
	TEST("submit-time guard: topologically unreachable declaration rejected") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "a", identitySchema(), identityRunFn()));
		graph.addNode(std::make_unique<Node>("Builtin", "orphan", identitySchema(), identityRunFn()));

		graph.feedInput("t1", "a", "x", makeFloatTensor(1.0f));

		bool rejected = false;
		try {
			graph.submit("t1", "orphan", "y");
		} catch (const GraphException& e) {
			rejected = (e.getErrorType() == GraphException::ErrorType::UnreachableDeclaration);
		}
		CHECK(rejected, "unreachable declaration should throw UnreachableDeclaration");
	}
	END_TEST();
}

static void testSignalBlockedButReachable() {
	TEST("submit-time guard: signal-blocked but topologically reachable → no throw, task stalls") {
		TestHarness harness;

		auto& a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto& b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
		harness.connect("id_a", "y", "id_b", "x");

		b.bindSignal(harness.signalStore(), "gate");
		harness.setSignal("gate", false);

		harness.feedInput("t1", "id_a", "x", makeFloatTensor(1.0f));

		harness.submit("t1", "id_b", "y");
		CHECK(harness.graph().taskStatus("t1") == TaskStatus::Running,
			  "signal-blocked task should be Running (stalled)");

		CHECK(harness.graph().cancel("t1"), "cancel should succeed on stalled task");
		CHECK(harness.graph().waitForResult("t1").status == TaskStatus::Cancelled,
			  "stalled task ends Cancelled via host cancel");
	}
	END_TEST();
}

int main() {
	try {
		testFailureClosesTaskAsFailed();
		testParallelBranchPartialFailure();
		testTopologicallyUnreachableRejected();
		testSignalBlockedButReachable();

		if (failures == 0) {
			std::cout << "\nAll FailureClosure tests passed!" << std::endl;
		} else {
			std::cout << "\n" << failures << " test(s) FAILED!" << std::endl;
		}
		return failures;
	} catch (const std::exception& e) {
		std::cerr << "Test failure: " << e.what() << std::endl;
		return -1;
	}
}