// 失败闭环 + 提交期拓扑守卫 单元测试
// 无执行超时语义下的行为契约：
//   ① 节点自报失败 → 传播耗尽 → 任务 Failed（宿主无需 wait+cancel 干预）
//   ② 声明目标拓扑不可达 → submit 立即抛 GraphException(UnreachableDeclaration)
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

// ── 辅助 ──

static Value makeFloatTensor(float value) {
	auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
	*t = value;
	return Value(std::move(t));
}

static Node::Schema identitySchema() {
	Node::Schema s;
	s.inputs = {{"x", TensorType::Float, sizeof(float), {}}};
	s.outputs = {{"y", TensorType::Float, sizeof(float), {}}};
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

// 失败节点：有输出 Schema 但执行永远失败、不产出（实现方自报失败）
static Node::RunFn failRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		(void)ctx;
		return ctx.failure(Node::Status::ExecutionFailed, "intentional failure");
	};
}

// ════════════════════════════════════════════
// ① 失败闭环
// ════════════════════════════════════════════

// 1. 唯一声明来源的节点失败 → 传播耗尽 → 任务 Failed
static void testFailureClosesTaskAsFailed() {
	TEST("failure closure: sole declared source fails → task ends Failed") {
		TestHarness harness;
		harness.addNode(std::make_unique<Node>("Builtin", "fail", identitySchema(), failRunFn()));

		harness.feedInput("t1", "fail", "x", makeFloatTensor(1.0f));
		harness.submit("t1", "fail", "y");

		// 等待护栏（宿主视角）：任务应在传播耗尽后自行终止为 Failed
		auto result = harness.graph().waitForResult("t1", std::chrono::milliseconds(3000));
		CHECK(result.status == TaskStatus::Failed, "declaration unmet + error diagnostic → Failed");
		CHECK(harness.graph().taskStatus("t1") == TaskStatus::Failed, "taskStatus should be Failed");

		// 失败原因归因到真实节点
		bool failRecorded = false;
		for (const auto& e : harness.taskErrors("t1")) {
			if (e.nodeName == "fail" && e.level == DiagnosticLevel::Error)
				failRecorded = true;
		}
		CHECK(failRecorded, "failing node recorded in diagnostics");
	}
	END_TEST();
}

// 2. 并行分支：一分支失败、另一分支正常满足声明 → Failed，部分输出仍可读
static void testParallelBranchPartialFailure() {
	TEST("parallel branch: one fails, other satisfies its declaration → Failed with partial outputs") {
		TestHarness harness;

		harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "id_c", identitySchema(), failRunFn()));

		auto bcSchema = Connector::broadcastSchema(2);
		auto bcNode = std::make_unique<Node>("Connector.Broadcast", "bc", bcSchema,
			Connector::broadcastRunFn(), ThreadPoolAffinity::System);
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

		// 正常分支的部分输出经终止清理抢救，仍可读
		CHECK(harness.hasOutput("t1", "id_b", "y"), "successful branch output should be readable");
		CHECK(!harness.hasOutput("t1", "id_c", "y"), "failed branch must not produce output");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// ② 提交期拓扑守卫
// ════════════════════════════════════════════

// 3. 声明目标拓扑不可达（孤立节点/断链）→ submit 立即抛错
static void testTopologicallyUnreachableRejected() {
	TEST("submit-time guard: topologically unreachable declaration rejected") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "a", identitySchema(), identityRunFn()));
		graph.addNode(std::make_unique<Node>("Builtin", "orphan", identitySchema(), identityRunFn()));

		graph.feedInput("t1", "a", "x", makeFloatTensor(1.0f));

		bool rejected = false;
		try {
			graph.submit("t1", "orphan", "y"); // orphan 无任何入边/连接
		} catch (const GraphException& e) {
			rejected = (e.getErrorType() == GraphException::ErrorType::UnreachableDeclaration);
		}
		CHECK(rejected, "unreachable declaration should throw UnreachableDeclaration");
	}
	END_TEST();
}

// 4. 信号阻断但拓扑可达 → submit 不抛、任务挂起由宿主解围（回归：勿误伤合法构图）
static void testSignalBlockedButReachable() {
	TEST("submit-time guard: signal-blocked but topologically reachable → no throw, task stalls") {
		TestHarness harness;

		auto& a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto& b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
		harness.connect("id_a", "y", "id_b", "x");

		b.bindSignal(harness.signalStore(), "gate");
		harness.setSignal("gate", false);

		harness.feedInput("t1", "id_a", "x", makeFloatTensor(1.0f));

		// 拓扑可达（id_a → id_b）→ 提交不抛；运行期信号阻断 → 挂起
		harness.submit("t1", "id_b", "y");
		CHECK(harness.graph().taskStatus("t1") == TaskStatus::Running,
			  "signal-blocked task should be Running (stalled)");

		// 宿主护栏：cancel 解围
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