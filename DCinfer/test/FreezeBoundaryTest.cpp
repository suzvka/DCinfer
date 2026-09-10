// FreezeBoundaryTest：Build → Freeze → Execute 边界验收
// 验证：惰性冻结建立"执行期拓扑不可变"不变量；冻结后构建 API 抛 Frozen；
//       冻结前后内省一致；数据 IO 语义与绑定签名在冻结前后完全一致。

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

// ── 4. 冻结前后数据 IO 语义一致（内部寻址）+ 绑定签名随快照固化 ──

static void test_ioSemanticsConsistentAcrossFreeze() {
	InferGraph graph;
	graph.addNode(makeId("a"));
	graph.bindInput("num", "a", "x");   // 图级签名（供 submitBound / 序列化）
	graph.bindOutput("res", "a", "y");

	// 冻结前：内部寻址注入 → submitBound（签名驱动声明）→ 内部寻址取用
	graph.feedInput("t0", "a", "x", floatTensor(9.0f));
	graph.submitBound("t0");
	CHECK(graph.waitForResult("t0").status != TaskStatus::Running,
		  "bound flow should complete (lazy freeze)");
	auto r0 = graph.takeOutputTensor("t0", "a", "y");
	CHECK(std::abs(r0.item<float>() - 9.0f) < 1e-6f, "result before freeze should be 9.0");

	// 冻结后（首次 submit 已惰性编译）：同一寻址方式照常工作
	graph.feedInput("t1", "a", "x", floatTensor(9.0f));
	graph.submitBound("t1");
	CHECK(graph.waitForResult("t1").status != TaskStatus::Running,
		  "frozen graph should complete the same way");
	CHECK(graph.hasOutput("t1", "a", "y"), "internal addressing resolves after freeze");
	auto r1 = graph.takeOutputTensor("t1", "a", "y");
	CHECK(std::abs(r1.item<float>() - 9.0f) < 1e-6f, "result after freeze should be 9.0");

	// 绑定签名随冻结快照固化（submitBound 声明来源 / 序列化契约保持有效）
	CHECK(graph.inputBindings().size() == 1 && graph.inputBindings()[0].alias == "num",
		  "input signature preserved across freeze");
	CHECK(graph.outputBindings().size() == 1 && graph.outputBindings()[0].alias == "res",
		  "output signature preserved across freeze");
}

// ── 5. 取消/诊断等运行期 API 在冻结图上照常工作 ──

static void test_runtimeLifecycleOnFrozenGraph() {
	InferGraph graph;
	auto& b = graph.addNode(makeId("b"));
	graph.addNode(makeId("a"));
	graph.connect("a", "y", "b", "x");
	b.bindSignal(graph.signalStore(), "gate");

	graph.feedInput("t1", "a", "x", floatTensor(1.0f));
	graph.submit("t1", "b", "y"); // 无执行超时 + gate 阻塞 → 挂起，宿主 wait+cancel 解围
	graph.setSignal("gate", false);

	CHECK(graph.waitForResult("t1", std::chrono::milliseconds(80)).status == TaskStatus::Running,
		  "blocked task should not complete");
	CHECK(graph.taskStatus("t1") == TaskStatus::Running, "task running while blocked");
	CHECK(graph.cancel("t1"), "cancel on frozen graph should work");
	CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "wait should wake after cancel");
	CHECK(graph.taskStatus("t1") == TaskStatus::Cancelled, "status should be Cancelled");
	// 新语义：传播耗尽时引擎写入 Warning 级停滞诊断（声明未满足/信号阻塞）；
	// clean cancel 允许诊断保留，但不得有 Error 级记录
	bool hasErrorLevel = false;
	for (const auto& e : graph.taskErrors("t1"))
		if (e.level == DiagnosticLevel::Error)
			hasErrorLevel = true;
	CHECK(!hasErrorLevel, "no error-level diagnostics expected on clean cancel");

	// 释放后状态归零（运行期 API 不受冻结影响）
	graph.releaseTask("t1");
	CHECK(graph.taskStatus("t1") == TaskStatus::Unknown, "released task should be Unknown");
}

int main() {
	test_introspectionConsistency();
	test_constructionRejectedAfterExplicitFreeze();
	test_lazyFreezeOnFirstSubmit();
	test_ioSemanticsConsistentAcrossFreeze();
	test_runtimeLifecycleOnFrozenGraph();

	if (g_failures == 0) {
		std::printf("All %d checks passed\n", g_checks);
		return 0;
	}
	std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
	return 1;
}
