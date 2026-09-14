// FreezeBoundaryTest：Build → Freeze → Execute 边界验收
// 验证：惰性冻结建立"执行期拓扑不可变"不变量；冻结后构建 API 抛 Frozen；
//       冻结前后内省一致；数据 IO 语义与绑定签名在冻结前后完全一致；
//       冻结前泄漏的引用（Node& / 构建面）在冻结后修改被拒绝；
//       并发首次冻结 / 并发首次运行 API / 冻结与内省并发均安全。

#include "InferGraph.h"
#include "GraphBuilder.h"
#include "GraphException.h"
#include "NodeException.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <latch>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

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
	s.inputs = {Node::Port::in<float>("x")};
	s.outputs = {Node::Port::out<float>("y")};
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

// 冻结后经泄漏 Node 引用修改配置：抛 NodeException(Frozen)
static bool throwsNodeFrozen(const std::function<void()>& fn) {
	try {
		fn();
	} catch (const NodeException& e) {
		return e.getErrorType() == NodeException::ErrorType::Frozen;
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

// ── 6. 冻结前泄漏的 Node&：冻结后修改配置被拒绝 ──

static void test_leakedNodeSettersRejectedAfterFreeze() {
	InferGraph graph;
	auto& leakedNode = graph.addNode(makeId("a"));
	Node* leakedPtr = graph.node("a"); // 构建期可写指针（冻结前获取）
	CHECK(leakedPtr == &leakedNode, "writable node() resolves before freeze");

	graph.freeze();

	// 全部公开可变入口：冻结后经泄漏引用调用一律抛 NodeException(Frozen)
	CHECK(throwsNodeFrozen([&] { leakedNode.setConnector(true); }),
		  "setConnector after freeze throws NodeException(Frozen)");
	CHECK(throwsNodeFrozen([&] { leakedNode.setTag("t"); }), "setTag after freeze throws NodeException(Frozen)");
	CHECK(throwsNodeFrozen([&] { leakedNode.setModelPath("m.onnx"); }),
		  "setModelPath after freeze throws NodeException(Frozen)");
	CHECK(throwsNodeFrozen([&] { leakedNode.setBlockedOverride([](const Node::TaskId&) { return false; }); }),
		  "setBlockedOverride after freeze throws NodeException(Frozen)");
	CHECK(throwsNodeFrozen([&] { leakedNode.setReadyOverride([](const Node::TaskId&) { return false; }); }),
		  "setReadyOverride after freeze throws NodeException(Frozen)");
	CHECK(throwsNodeFrozen([&] {
		leakedNode.setCompletionCallback([](const Node::TaskId&, const Node::Result&) {});
	}), "setCompletionCallback after freeze throws NodeException(Frozen)");
	CHECK(throwsNodeFrozen([&] { leakedNode.bindSignal(graph.signalStore(), "gate"); }),
		  "bindSignal after freeze throws NodeException(Frozen)");
	CHECK(throwsNodeFrozen([&] { leakedNode.bindEngine(nullptr, nullptr); }),
		  "bindEngine after freeze throws NodeException(Frozen)");

	// 同一节点经构建期可写指针修改：同样被拒绝（封印随节点对象走）
	CHECK(throwsNodeFrozen([&] { leakedPtr->setConnector(true); }),
		  "leaked writable pointer mutation rejected after freeze");

	// facade 的可写 node() 入口在冻结后抛 GraphException(Frozen)
	CHECK(throwsFrozen([&] { graph.node("a"); }), "writable node() after freeze throws GraphException(Frozen)");

	// 负样本：未入图的独立节点永不封印，配置语义不变
	auto standalone = makeId("solo");
	standalone->setConnector(true);
	standalone->setTag("solo-tag");
	standalone->setReadyOverride([](const Node::TaskId&) { return true; });
	CHECK(standalone->isConnector() && standalone->tag() == "solo-tag",
		  "standalone node setters stay mutable (never sealed)");
}

// ── 7. 构建面冻结后：类型层无可变入口；const 内省委托快照 ──

static void test_builderAccessorsAfterCompile() {
	// 类型层不变量：store() 仅存在 const 重载——冻结前泄漏可变 GraphStore&
	// 的路径在编译期不可达（若回归出非 const 重载，本断言失败）
	static_assert(std::is_same_v<decltype(std::declval<GraphBuilder&>().store()), const GraphStore&>,
				  "GraphBuilder::store() must be const-only (no mutable accessor)");

	GraphBuilder builder;
	builder.addNode(makeId("a"));
	builder.addNode(makeId("b"));
	builder.connect("a", "y", "b", "x"); // 自动插入 __wire（源图 3 节点）
	builder.bindOutput("b", "y", "out");
	auto snapshot = builder.compile();
	CHECK(snapshot != nullptr, "compile returns snapshot");

	// 冻结后 const 内省可用且与快照一致（不再空指针解引用）
	CHECK(builder.nodeCount() == 3, "builder nodeCount after compile reads snapshot view");
	CHECK(builder.nodeCount() == snapshot->store().nodeCount(), "builder introspection mirrors snapshot");
	CHECK(std::as_const(builder).node("a") != nullptr, "builder const node() works after compile");
	CHECK(builder.outputBindings().size() == 1 && builder.outputBindings()[0].alias == "out",
		  "builder output bindings readable after compile");
	CHECK(builder.edges().size() == 2, "builder edges readable after compile");

	// 冻结后全部可变入口抛 Frozen
	CHECK(throwsFrozen([&] { builder.addNode(makeId("c")); }), "builder addNode after compile throws Frozen");
	CHECK(throwsFrozen([&] { builder.connect("a", "y", "a", "x"); }),
		  "builder connect after compile throws Frozen");
	CHECK(throwsFrozen([&] { builder.bindInput("a", "x", "in"); }),
		  "builder bindInput after compile throws Frozen");
	CHECK(throwsFrozen([&] { builder.bindOutput("b", "y", "x"); }),
		  "builder bindOutput after compile throws Frozen");
	CHECK(throwsFrozen([&] { builder.node("a"); }), "builder writable node() after compile throws Frozen");
	CHECK(builder.compile() == snapshot, "compile idempotent under repeated calls");
}

// ── 8. 并发首次 freeze()：恰好一个快照，全部线程观察到同一实例 ──

static void test_concurrentFirstFreezeSingleSnapshot() {
	constexpr int kThreads = 8;
	constexpr int kRounds = 30;
	for (int round = 0; round < kRounds; ++round) {
		InferGraph graph;
		graph.addNode(makeId("a"));
		graph.addNode(makeId("b"));
		graph.connect("a", "y", "b", "x");

		std::latch startGate(kThreads);
		std::vector<std::shared_ptr<const CompiledGraph>> results(kThreads);
		std::vector<std::string> errors(kThreads);
		std::vector<std::thread> threads;
		for (int t = 0; t < kThreads; ++t) {
			threads.emplace_back([&, t] {
				startGate.arrive_and_wait();
				try {
					results[t] = graph.freeze();
				} catch (const std::exception& e) {
					errors[t] = e.what();
				}
			});
		}
		for (auto& th : threads)
			th.join();

		bool noExceptions = true, allNonEmpty = true, sameSnapshot = true;
		for (int t = 0; t < kThreads; ++t) {
			if (!errors[t].empty())
				noExceptions = false;
			if (!results[t])
				allNonEmpty = false;
			else if (results[t] != results[0])
				sameSnapshot = false;
		}
		CHECK(noExceptions, "concurrent freeze must not throw");
		CHECK(allNonEmpty, "every freeze() call returns a snapshot");
		CHECK(sameSnapshot, "concurrent first freeze yields exactly one snapshot instance");
		CHECK(graph.nodeCount() == 3, "introspection consistent after concurrent freeze");

		// 冻结后任务照常运行（快照 + 闸表完整就绪）
		graph.feedInput("t1", "a", "x", floatTensor(5.0f));
		graph.submit("t1", "b", "y");
		CHECK(graph.waitForResult("t1").status == TaskStatus::Succeeded, "task works after concurrent freeze");
	}
}

// ── 9. 并发首次运行 API：feedInput+submit 作为首操作竞争冻结 ──

static void test_concurrentFirstRunApis() {
	constexpr int kThreads = 8;
	InferGraph graph;
	for (int t = 0; t < kThreads; ++t)
		graph.addNode(makeId("n" + std::to_string(t))); // 每线程独立节点（节点级互斥是既有设计）

	std::latch startGate(kThreads);
	std::atomic<int> anomalies{0};
	std::vector<std::thread> threads;
	for (int t = 0; t < kThreads; ++t) {
		threads.emplace_back([&, t] {
			const std::string nodeName = "n" + std::to_string(t);
			const std::string tid = "t" + std::to_string(t);
			startGate.arrive_and_wait();
			try {
				graph.feedInput(tid, nodeName, "x", floatTensor(static_cast<float>(t)));
				graph.submit(tid, nodeName, "y");
				if (graph.waitForResult(tid, std::chrono::milliseconds(10000)).status != TaskStatus::Succeeded)
					++anomalies;
			} catch (...) {
				++anomalies;
			}
		});
	}
	for (auto& th : threads)
		th.join();
	CHECK(anomalies.load() == 0, "concurrent first run APIs complete on a single freeze");
	CHECK(graph.nodeCount() == kThreads, "source view intact after concurrent first run");
}

// ── 10. 冻结与内省并发：读者只观察到未冻结或完整冻结状态 ──

static void test_introspectionConcurrentWithFirstFreeze() {
	InferGraph graph;
	graph.addNode(makeId("a"));
	graph.bindOutput("out", "a", "y");

	std::atomic<bool> stop{false};
	std::atomic<int> anomalies{0};
	std::thread reader([&] {
		try {
			while (!stop.load(std::memory_order_relaxed)) {
				auto snap = graph.freeze(); // 幂等；与首次提交并发
				if (snap && snap->store().nodeCount() != 1)
					++anomalies;
				if (graph.nodeCount() != 1)
					++anomalies;
			}
		} catch (...) {
			++anomalies;
		}
	});

	graph.feedInput("t1", "a", "x", floatTensor(3.0f));
	graph.submitBound("t1");
	CHECK(graph.waitForResult("t1").status == TaskStatus::Succeeded,
		  "task completes while freeze is read concurrently");

	stop.store(true, std::memory_order_relaxed);
	reader.join();
	CHECK(anomalies.load() == 0, "concurrent introspection observes consistent state");
}

int main() {
	test_introspectionConsistency();
	test_constructionRejectedAfterExplicitFreeze();
	test_lazyFreezeOnFirstSubmit();
	test_ioSemanticsConsistentAcrossFreeze();
	test_runtimeLifecycleOnFrozenGraph();
	test_leakedNodeSettersRejectedAfterFreeze();
	test_builderAccessorsAfterCompile();
	test_concurrentFirstFreezeSingleSnapshot();
	test_concurrentFirstRunApis();
	test_introspectionConcurrentWithFirstFreeze();

	if (g_failures == 0) {
		std::printf("All %d checks passed\n", g_checks);
		return 0;
	}
	std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
	return 1;
}
