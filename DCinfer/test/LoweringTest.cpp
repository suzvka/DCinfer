// LoweringTest：Broadcast(1) lowering 验收
// 验证：1:1 wire 从运行时视图擦除（源图不变）；值/错误/取消传播语义不变；
//       TTL 只统计运行时顶点（wire 不再消耗 hop）；绑定防护与多出边防护。

#include "InferGraph.h"
#include "Connector.h"
#include "GraphException.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
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

// ── 测试节点 ──

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

static Node::RunFn incRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		auto in = ctx.pop("x");
		auto* t = in.as<Tensor>();
		float v = t->item<float>() + 1.0f;
		auto out = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
		*out = v;
		ctx.output("y", Value(std::move(out)));
		return ctx.success();
	};
}

static Node::RunFn failRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		(void)ctx;
		return ctx.failure(Node::Status::ExecutionFailed, "intentional failure");
	};
}

static std::unique_ptr<Node> makeId(const std::string& name) {
	return std::make_unique<Node>("Builtin", name, idSchema(), idRunFn());
}

static std::unique_ptr<Node> makeInc(const std::string& name) {
	return std::make_unique<Node>("Builtin", name, idSchema(), incRunFn());
}

static std::unique_ptr<Tensor> floatTensor(float v) {
	auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
	*t = v;
	return t;
}

// ── 1. 自动 wire（Broadcast(1)）被擦除：源图不变、运行时视图收缩、值传播不变 ──

static void test_autoWireErased() {
	InferGraph graph;
	graph.addNode(makeId("a"));
	graph.addNode(makeId("b"));
	graph.connect("a", "y", "b", "x"); // 自动插入 __wire_N（Broadcast(1)）
	graph.bindOutput("out", "b", "y");

	// 源图视角：3 节点（a、b、wire）2 边
	CHECK(graph.nodeCount() == 3, "source view: 3 nodes (incl. wire)");
	CHECK(graph.edgeCount() == 2, "source view: 2 edges");

	auto snap = graph.freeze();
	CHECK(snap->loweringStats().erasedConnectors == 1, "1 connector erased");
	CHECK(snap->runtimeNodeCount() == 2, "runtime view: wire erased (2 nodes)");
	CHECK(snap->runtimeEdgeCount() == 1, "runtime view: fused direct edge (1 edge)");
	// 源图视角冻结后不变形
	CHECK(graph.nodeCount() == 3, "source view unchanged after freeze");

	// 值传播：move 语义经直连边不变
	graph.feedInput("t1", "a", "x", floatTensor(7.0f));
	graph.submit("t1", "b", "y");
	CHECK(graph.wait("t1"), "task should complete through lowered edge");
	auto r = graph.takeOutputTensor("t1", "b", "y");
	CHECK(std::abs(r.item<float>() - 7.0f) < 1e-6f, "value should pass through unchanged");
}

// ── 2. Broadcast(N>1) 不擦除：1→2 广播保留连接器节点 ──

static void test_broadcastN2NotErased() {
	InferGraph graph;
	graph.addNode(makeId("a"));
	graph.addNode(makeId("b"));
	graph.addNode(makeId("c"));

	auto bcSchema = Connector::broadcastSchema(2);
	auto bcNode = std::make_unique<Node>("Connector.Broadcast", "bc", bcSchema,
										 Connector::broadcastRunFn(), ThreadPoolAffinity::System);
	bcNode->setConnector(true);
	graph.addNode(std::move(bcNode));

	graph.connectRaw("a", "y", "bc", "in");
	graph.connectRaw("bc", "out_0", "b", "x");
	graph.connectRaw("bc", "out_1", "c", "x");
	graph.bindOutput("ob", "b", "y");
	graph.bindOutput("oc", "c", "y");

	auto snap = graph.freeze();
	CHECK(snap->loweringStats().erasedConnectors == 0, "N>1 broadcast must not be erased");
	CHECK(snap->runtimeNodeCount() == 4, "runtime keeps the broadcast connector (4 nodes)");

	graph.feedInput("t1", "a", "x", floatTensor(5.0f));
	graph.submit("t1", {{"b", "y", 1}, {"c", "y", 1}});
	CHECK(graph.wait("t1"), "1→2 broadcast should complete");
	auto rb = graph.takeOutputTensor("t1", "b", "y");
	auto rc = graph.takeOutputTensor("t1", "c", "y");
	CHECK(std::abs(rb.item<float>() - 5.0f) < 1e-6f, "downstream 1 gets copy");
	CHECK(std::abs(rc.item<float>() - 5.0f) < 1e-6f, "downstream 2 gets copy");
}

// ── 3. TTL 语义：wire 不再消耗 hop（maxHops 只统计运行时顶点）──

static void test_ttlCountsRuntimeVertices() {
	// 3 源节点链（a → wire → c）：旧语义需要 3 hops（c 完成后 pf 检查需要 >0），
	// 新语义只需 2 hops（每业务节点完成后 1 次）。maxHops=2 成功即固化新语义。
	InferGraph graph;
	graph.addNode(makeId("a"));
	graph.addNode(makeId("c"));
	graph.connect("a", "y", "c", "x");
	graph.bindOutput("out", "c", "y");

	graph.feedInput("t1", "a", "x", floatTensor(3.0f));
	graph.submit("t1", "c", "y", 1, std::chrono::milliseconds(2000), /*maxHops=*/2);
	CHECK(graph.wait("t1"), "maxHops=2 suffices for 2-runtime-vertex chain (wire consumes no TTL)");
	CHECK(graph.taskStatus("t1") == TaskStatus::Succeeded, "status Succeeded");

	// 反向固化：maxHops=1 时 2 顶点链 TTL 不足 → Failed（hops exhausted）
	InferGraph g2;
	g2.addNode(makeId("a"));
	g2.addNode(makeId("c"));
	g2.connect("a", "y", "c", "x");
	g2.bindOutput("out", "c", "y");
	g2.feedInput("t1", "a", "x", floatTensor(3.0f));
	g2.submit("t1", "c", "y", 1, std::chrono::milliseconds(2000), /*maxHops=*/1);
	g2.wait("t1");
	CHECK(g2.taskStatus("t1") == TaskStatus::Failed, "maxHops=1 exhausts TTL on 2-vertex chain");
	bool ttlMsg = false;
	for (const auto& e : g2.taskErrors("t1"))
		if (e.message.find("hops exhausted") != std::string::npos)
			ttlMsg = true;
	CHECK(ttlMsg, "diagnostic reports hops exhausted");
}

// ── 4. 成环图：擦除后 hop 预算延长（方向安全），TTL 仍兜底 ──

static void test_cycleTtlStillBounded() {
	InferGraph graph;
	std::atomic<int> aRuns{0};
	std::atomic<int> bRuns{0};

	{
		Node::RunFn incA = [&aRuns](Node::RunContext& ctx) -> Node::Result {
			++aRuns;
			return incRunFn()(ctx);
		};
		Node::RunFn incB = [&bRuns](Node::RunContext& ctx) -> Node::Result {
			++bRuns;
			return incRunFn()(ctx);
		};
		graph.addNode(std::make_unique<Node>("Builtin", "a", idSchema(), incA));
		graph.addNode(std::make_unique<Node>("Builtin", "b", idSchema(), incB));
	}
	graph.connect("a", "y", "b", "x");
	graph.connect("b", "y", "a", "x");
	graph.bindInput("seed", "a", "x");

	auto snap = graph.freeze();
	CHECK(snap->loweringStats().erasedConnectors == 2, "both cycle wires erased");
	CHECK(snap->runtimeNodeCount() == 2, "cycle runtime: 2 business nodes");
	CHECK(graph.nodeCount() == 4, "source view keeps 4 nodes");

	// maxHops=6：新语义下预算 6 → a 执行 ≥3 次（旧语义每业务 hop 耗 2，只能 2 次）。
	// 声明一个永不产出的端口，避免首次环回即满足声明提前 Succeeded——
	// 由 TTL 兑底终止（本用例验证的就是 TTL 行为）。
	graph.feedInput("t1", "a", "x", floatTensor(0.0f));
	graph.submit("t1", "never", "y", 1, std::chrono::milliseconds(2000), /*maxHops=*/6);
	CHECK(graph.wait("t1"), "cycle should terminate by TTL");
	CHECK(graph.taskStatus("t1") == TaskStatus::Failed, "cycle ends Failed (TTL exhausted)");
	CHECK(aRuns.load() >= 3, "TTL budget stretches after lowering (a runs >= 3 times)");
}

// ── 5. 绑定防护：wire 承担图级输入/输出契约时不擦除 ──

static void test_bindingProtectionKeepsWire() {
	{
		// wire 的 out_0 被绑定为图级输出 → 保留
		InferGraph graph;
		graph.addNode(makeId("a"));
		graph.addNode(makeId("c"));
		auto& w = graph.connect("a", "y", "c", "x");
		graph.bindOutput(w.name(), "out_0");

		auto snap = graph.freeze();
		CHECK(snap->loweringStats().erasedConnectors == 0, "wire bound as output is kept");
		CHECK(snap->runtimeNodeCount() == 3, "runtime keeps the contract-bearing wire");

		graph.feedInput("t1", "a", "x", floatTensor(2.0f));
		graph.submit("t1", w.name(), "out_0");
		CHECK(graph.wait("t1"), "declaration on wire should be satisfied");
		auto r = graph.takeOutputTensor("t1", w.name(), "out_0");
		CHECK(std::abs(r.item<float>() - 2.0f) < 1e-6f, "wire artifact value correct");
	}
	{
		// wire 的 in 被绑定为图级输入 → 保留（feedInput 直喂 wire）
		InferGraph graph;
		graph.addNode(makeId("a"));
		graph.addNode(makeId("c"));
		auto& w = graph.connect("a", "y", "c", "x");
		graph.bindInput(w.name(), "in");
		graph.bindOutput("out", "c", "y");

		auto snap = graph.freeze();
		CHECK(snap->loweringStats().erasedConnectors == 0, "wire bound as input is kept");
		CHECK(snap->runtimeNodeCount() == 3, "runtime keeps the input-bearing wire");

		graph.feedInput("t1", w.name(), "in", floatTensor(4.0f));
		graph.submit("t1", "c", "y");
		CHECK(graph.wait("t1"), "wire-fed flow should complete");
		auto r = graph.takeOutputTensor("t1", "c", "y");
		CHECK(std::abs(r.item<float>() - 4.0f) < 1e-6f, "value should reach downstream via wire");
	}
}

// ── 6. 错误传播：上游失败 → 下游不执行（直连边路径语义不变）──

static void test_errorPropagationThroughLoweredEdge() {
	InferGraph graph;
	graph.addNode(std::make_unique<Node>("Builtin", "fail", idSchema(), failRunFn()));
	graph.addNode(makeId("down"));
	graph.connect("fail", "y", "down", "x");
	graph.bindOutput("out", "down", "y");

	graph.feedInput("t1", "fail", "x", floatTensor(1.0f));
	graph.submit("t1", "down", "y", 1, std::chrono::milliseconds(500));
	CHECK(graph.wait("t1"), "watchdog should terminate the stuck task");
	CHECK(graph.taskStatus("t1") == TaskStatus::TimedOut, "upstream failure → declaration unmet → TimedOut");
	bool failRecorded = false;
	for (const auto& e : graph.taskErrors("t1"))
		if (e.nodeName == "fail" && e.level == DiagnosticLevel::Error)
			failRecorded = true;
	CHECK(failRecorded, "failing node recorded in diagnostics");
	CHECK(!graph.hasOutput("t1", "down", "y"), "downstream must not produce output");
}

int main() {
	test_autoWireErased();
	test_broadcastN2NotErased();
	test_ttlCountsRuntimeVertices();
	test_cycleTtlStillBounded();
	test_bindingProtectionKeepsWire();
	test_errorPropagationThroughLoweredEdge();

	if (g_failures == 0) {
		std::printf("All %d checks passed\n", g_checks);
		return 0;
	}
	std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
	return 1;
}
