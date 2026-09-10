// FreezeBoundaryTest：Build → Freeze → Execute 边界验收
// 验证：惰性冻结建立"执行期拓扑不可变"不变量；冻结后构建 API 抛 Frozen；
//       冻结前后内省一致；别名解析语义与冻结前完全一致。

#include "InferGraph.h"
#include "GraphBuilder.h"
#include "GraphException.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>

using namespace DC;

using TensorType = DC::Tensor::TensorType;
using Tensor = DC::Tensor;

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond, msg)                                                                                               \
	do {                                                                                                               \
		++g_checks;                                                                                                    \
		if (!(cond)) {                                                                                                 \
			++g_failures;                                                                                              \
			std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);                                                  \
		}                                                                                                              \
	} while (0)

// ── 测试用 identity 节点 ──

static Node::Schema idSchema() {
	Node::Schema s;
	s.inputs = {{"x", TensorType::Float, sizeof(float), {}}};
	s.outputs = {{"y", TensorType::Float, sizeof(float), {}}};
	return s;
}

static Node::RunFn idRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		ctx.output("y", ctx.pop("x"));
		return ctx.success();
	};
}

static std::unique_ptr<Node> makeId(const std::string& name) {
	return std::make_unique<Node>("Builtin", name, idSchema(), idRunFn());
}

static std::unique_ptr<Tensor> floatTensor(float v) {
	auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
	*t = v;
	return t;
}

static bool throwsFrozen(const std::function<void()>& fn) {
	try {
		fn();
	} catch (const GraphException& e) {
		return e.getErrorType() == GraphException::ErrorType::Frozen;
	} catch (...) {
		return false;
	}
	return false;
}

// ── 1. 冻结前后内省一致（源图视角） + freeze() 幂等 ──

static void test_introspectionConsistency() {
	InferGraph graph;
	graph.addNode(makeId("a"));
	graph.addNode(makeId("b"));
	graph.connect("a", "y", "b", "x");
	graph.bindInput("in", "a", "x");
	graph.bindOutput("out", "b", "y");

	// 冻结前内省
	const size_t nodesBefore = graph.nodeCount();
	const size_t edgesBefore = graph.edgeCount();
	const auto namesBefore = graph.nodeNames();
	const auto edgesBeforeVec = graph.edges();
	const auto inB4 = graph.inputBindings();
	const auto outB4 = graph.outputBindings();

	auto snapshot = graph.freeze();
	CHECK(snapshot != nullptr, "freeze() should return the compiled snapshot");

	// 冻结后内省：逐项一致（源图视角不变形）
	CHECK(graph.nodeCount() == nodesBefore, "nodeCount unchanged after freeze");
	CHECK(graph.edgeCount() == edgesBefore, "edgeCount unchanged after freeze");
	CHECK(graph.nodeNames() == namesBefore, "nodeNames unchanged after freeze");
	CHECK(graph.edges().size() == edgesBeforeVec.size(), "edges unchanged after freeze");
	CHECK(graph.inputBindings().size() == inB4.size(), "input bindings unchanged after freeze");
	CHECK(graph.outputBindings().size() == outB4.size(), "output bindings unchanged after freeze");
	CHECK(graph.inputBindings()[0].alias == "in", "input alias preserved in signature");
	CHECK(graph.outputBindings()[0].alias == "out", "output alias preserved in signature");

	// freeze() 幂等：重复调用返回同一快照
	CHECK(graph.freeze() == snapshot, "freeze() is idempotent (same snapshot)");

	// 快照视角一致：snapshot->store 与 graph 查询同源
	CHECK(snapshot->store().nodeCount() == nodesBefore, "snapshot store mirrors source graph");
	CHECK(snapshot->signature().inputs.size() == inB4.size(), "snapshot signature mirrors bindings");
}

// ── 2. 显式冻结后：全部构建 API 抛 Frozen ──

static void test_constructionRejectedAfterExplicitFreeze() {
	InferGraph graph;
	graph.addNode(makeId("a"));
	graph.freeze();

	CHECK(throwsFrozen([&] { graph.addNode(makeId("c")); }), "addNode after freeze throws Frozen");
	CHECK(throwsFrozen([&] { graph.connect("a", "y", "a", "x"); }), "connect after freeze throws Frozen");
	CHECK(throwsFrozen([&] { graph.bindInput("x", "a", "x"); }), "bindInput after freeze throws Frozen");
	CHECK(throwsFrozen([&] { graph.bindInput("alias", "a", "x"); }),
		  "bindInput(alias) after freeze throws Frozen");
	CHECK(throwsFrozen([&] { graph.bindOutput("y", "a", "y"); }), "bindOutput after freeze throws Frozen");
	CHECK(throwsFrozen([&] { graph.bindOutput("alias", "a", "y"); }),
		  "bindOutput(alias) after freeze throws Frozen");
}

// ── 3. 惰性冻结：submit 触发编译；运行期正常，构建面关闭 ──

static void test_lazyFreezeOnFirstSubmit() {
	InferGraph graph;
	auto& b = graph.addNode(makeId("b"));
	graph.addNode(makeId("a"));
	graph.connect("a", "y", "b", "x");
	b.bindSignal(graph.signalStore(), "gate");
	graph.setSignal("gate", true); // 运行期信号设置不触发冻结

	graph.feedInput("t1", "a", "x", floatTensor(41.0f)); // 惰性冻结在此触发
	graph.submit("t1", "b", "y");
	CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "task should complete after lazy freeze");

	// 冻结后：运行期 API 照常（信号、状态、结果读取）
	graph.setSignal("gate", false);
	CHECK(graph.taskStatus("t1") == TaskStatus::Succeeded, "task succeeded");
	CHECK(graph.hasOutput("t1", "b", "y"), "output should exist");
	auto r = graph.takeOutputTensor("t1", "b", "y");
	CHECK(std::abs(r.item<float>() - 41.0f) < 1e-6f, "value should propagate through frozen graph");

	// 冻结后：构建 API 一律拒绝（包括 submit 之后的新构建意图）
	CHECK(throwsFrozen([&] { graph.addNode(makeId("c")); }), "addNode after submit throws Frozen");
	CHECK(throwsFrozen([&] { graph.bindOutput("y", "b", "y"); }), "bindOutput after submit throws Frozen");
}

// ── 4. 冻结后别名/端口名解析语义与冻结前一致 ──

static void test_resolutionSemanticsPreserved() {
	InferGraph graph;
	graph.addNode(makeId("a"));
	graph.bindInput("num", "a", "x");   // 公共别名
	graph.bindOutput("res", "a", "y");  // 公共别名

	// 冻结前解析（构建期视图）
	bool preAliasUnknown = false;
	try {
		graph.feedBoundInput("t0", "no_such", floatTensor(0.0f));
	} catch (const GraphException& e) {
		preAliasUnknown = (e.getErrorType() == GraphException::ErrorType::NodeNotFound);
	}
	CHECK(preAliasUnknown, "unknown bound name throws NodeNotFound (pre-freeze)");

	graph.feedBoundInput("t1", "num", floatTensor(9.0f)); // 按别名注入
	graph.submitBound("t1");
	CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "bound flow should complete (lazy freeze)");

	// 冻结后解析（GraphSignature 视图，无锁）
	CHECK(graph.hasOutput("t1", "res"), "alias should resolve for hasOutput after freeze");
	bool portNameRejected = false;
	try {
		graph.hasOutput("t1", "y");
	} catch (const GraphException& e) {
		portNameRejected = (e.getErrorType() == GraphException::ErrorType::NodeNotFound);
	}
	CHECK(portNameRejected, "bare port name no longer resolves after freeze (alias-only addressing)");
	auto r = graph.takeOutputTensor("t1", "res");
	CHECK(std::abs(r.item<float>() - 9.0f) < 1e-6f, "alias retrieval after freeze should be 9.0");

	InferGraph g2;
	g2.addNode(makeId("a"));
	g2.addNode(makeId("b"));
	g2.bindInput("fa", "a", "x");
	g2.bindInput("fb", "b", "x");
	bool portNameUnresolved = false;
	try {
		g2.feedBoundInput("t1", "x", floatTensor(0.0f)); // 端口名 "x" 不再被解析
	} catch (const GraphException& e) {
		portNameUnresolved = (e.getErrorType() == GraphException::ErrorType::NodeNotFound);
	}
	CHECK(portNameUnresolved, "bare port name throws NodeNotFound (use alias)");
}

// ── 5. 取消/诊断等运行期 API 在冻结图上照常工作 ──

static void test_runtimeLifecycleOnFrozenGraph() {
	InferGraph graph;
	auto& b = graph.addNode(makeId("b"));
	graph.addNode(makeId("a"));
	graph.connect("a", "y", "b", "x");
	b.bindSignal(graph.signalStore(), "gate");

	graph.feedInput("t1", "a", "x", floatTensor(1.0f));
	graph.submit("t1", "b", "y"); // 无看门狗 + gate 阻塞 → 任务挂起
	graph.setSignal("gate", false);

	CHECK(graph.waitForResult("t1", std::chrono::milliseconds(80)).status == TaskStatus::Running,
		  "blocked task should not complete");
	CHECK(graph.taskStatus("t1") == TaskStatus::Running, "task running while blocked");
	CHECK(graph.cancel("t1"), "cancel on frozen graph should work");
	CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "wait should wake after cancel");
	CHECK(graph.taskStatus("t1") == TaskStatus::Cancelled, "status should be Cancelled");
	CHECK(graph.taskErrors("t1").empty(), "no errors expected on clean cancel");

	// 释放后状态归零（运行期 API 不受冻结影响）
	graph.releaseTask("t1");
	CHECK(graph.taskStatus("t1") == TaskStatus::Unknown, "released task should be Unknown");
}

int main() {
	test_introspectionConsistency();
	test_constructionRejectedAfterExplicitFreeze();
	test_lazyFreezeOnFirstSubmit();
	test_resolutionSemanticsPreserved();
	test_runtimeLifecycleOnFrozenGraph();

	if (g_failures == 0) {
		std::printf("All %d checks passed\n", g_checks);
		return 0;
	}
	std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
	return 1;
}
