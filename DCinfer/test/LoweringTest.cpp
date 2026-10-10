// LoweringTest：Broadcast(1) lowering 验收

#include "InferGraph.h"
#include "Connector.h"
#include "GraphException.h"
#include "GraphStore.h"
#include "GraphLowering.h"
#include "GraphSignature.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
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

// fanOut=1 为等效导线，可被 lowering 擦除
static std::unique_ptr<Node> makeWire(const std::string& name, size_t fanOut = 1) {
	auto w = std::make_unique<Node>("Connector.Broadcast", name, Connector::broadcastSchema(fanOut),
								 Connector::broadcastRunFn(), ResourceClass::System);
	w->setConnector(true);
	return w;
}

static void test_autoWireErased() {
	InferGraph graph;
	graph.addNode(makeId("a"));
	graph.addNode(makeId("b"));
	graph.connect("a", "y", "b", "x");
	graph.bindOutput("out", "b", "y");

	CHECK(graph.nodeCount() == 3, "source view: 3 nodes (incl. wire)");
	CHECK(graph.edgeCount() == 2, "source view: 2 edges");

	auto snap = graph.freeze();
	CHECK(snap->loweringStats().erasedConnectors == 1, "1 connector erased");
	CHECK(snap->runtimeNodeCount() == 2, "runtime view: wire erased (2 nodes)");
	CHECK(snap->runtimeEdgeCount() == 1, "runtime view: fused direct edge (1 edge)");
	CHECK(graph.nodeCount() == 3, "source view unchanged after freeze");

	graph.feedInput("t1", "a", "x", floatTensor(7.0f));
	graph.submit("t1", "b", "y");
	CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "task should complete through lowered edge");
	auto r = graph.takeOutputTensor("t1", "b", "y");
	CHECK(std::abs(r.item<float>() - 7.0f) < 1e-6f, "value should pass through unchanged");
}

static void test_broadcastN2NotErased() {
	InferGraph graph;
	graph.addNode(makeId("a"));
	graph.addNode(makeId("b"));
	graph.addNode(makeId("c"));

	auto bcSchema = Connector::broadcastSchema(2);
	auto bcNode = std::make_unique<Node>("Connector.Broadcast", "bc", bcSchema,
										 Connector::broadcastRunFn(), ResourceClass::System);
	bcNode->setConnector(true);
	graph.addNode(std::move(bcNode));

	graph.connect("a", "y", "bc", "in");
	graph.connect("bc", "out_0", "b", "x");
	graph.connect("bc", "out_1", "c", "x");
	graph.bindOutput("ob", "b", "y");
	graph.bindOutput("oc", "c", "y");

	auto snap = graph.freeze();
	CHECK(snap->loweringStats().erasedConnectors == 3, "wrapping wires erased, N>1 broadcast kept");
	CHECK(snap->runtimeNodeCount() == 4, "runtime keeps the broadcast connector (4 nodes)");
	CHECK(snap->runtimeEdgeCount() == 3, "fused a→bc + bc's two out-edges (3 edges)");

	graph.feedInput("t1", "a", "x", floatTensor(5.0f));
	graph.submit("t1", {{"b", "y", 1}, {"c", "y", 1}});
	CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "1→2 broadcast should complete");
	auto rb = graph.takeOutputTensor("t1", "b", "y");
	auto rc = graph.takeOutputTensor("t1", "c", "y");
	CHECK(std::abs(rb.item<float>() - 5.0f) < 1e-6f, "downstream 1 gets copy");
	CHECK(std::abs(rc.item<float>() - 5.0f) < 1e-6f, "downstream 2 gets copy");
}

static void test_ttlCountsRuntimeVertices() {
	// 新语义 TTL 只计运行时顶点：2 跳链 maxHops=2 成功（旧语义需 3）
	InferGraph graph;
	graph.addNode(makeId("a"));
	graph.addNode(makeId("c"));
	graph.connect("a", "y", "c", "x");
	graph.bindOutput("out", "c", "y");

	graph.feedInput("t1", "a", "x", floatTensor(3.0f));
	graph.submit("t1", "c", "y", 1, /*maxHops=*/2);
	CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "maxHops=2 suffices for 2-runtime-vertex chain (wire consumes no TTL)");
	CHECK(graph.taskStatus("t1") == TaskStatus::Succeeded, "status Succeeded");

	InferGraph g2;
	g2.addNode(makeId("a"));
	g2.addNode(makeId("c"));
	g2.connect("a", "y", "c", "x");
	g2.bindOutput("out", "c", "y");
	g2.feedInput("t1", "a", "x", floatTensor(3.0f));
	g2.submit("t1", "c", "y", 1, /*maxHops=*/1);
	g2.waitForResult("t1");
	CHECK(g2.taskStatus("t1") == TaskStatus::Failed, "maxHops=1 exhausts TTL on 2-vertex chain");
	bool ttlMsg = false;
	for (const auto& e : g2.taskErrors("t1"))
		if (e.message.find("hops exhausted") != std::string::npos)
			ttlMsg = true;
	CHECK(ttlMsg, "diagnostic reports hops exhausted");
}

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

	// 声明 count=1000 避免首次环回即满足：6 跳内不可能达成，由 TTL 兜底终止
	graph.feedInput("t1", "a", "x", floatTensor(0.0f));
	graph.submit("t1", "a", "y", 1000, /*maxHops=*/6);
	CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "cycle should terminate by TTL");
	CHECK(graph.taskStatus("t1") == TaskStatus::Failed, "cycle ends Failed (TTL exhausted)");
	CHECK(aRuns.load() >= 3, "TTL budget stretches after lowering (a runs >= 3 times)");
}

static void test_bindingProtectionKeepsWire() {
	{
		InferGraph graph;
		graph.addNode(makeId("a"));
		graph.addNode(makeId("c"));
		auto& w = graph.connect("a", "y", "c", "x");
		graph.bindOutput("out_0", w.name(), "out_0");

		bool rejected = false;
		try {
			graph.freeze();
		} catch (const GraphException& e) {
			rejected = (e.getErrorType() == GraphException::ErrorType::NonTerminalPort);
		}
		CHECK(rejected, "binding a wire output port (with out-edges) must be rejected at freeze");
	}
	{
		InferGraph graph;
		graph.addNode(makeId("a"));
		graph.addNode(makeId("c"));
		auto& w = graph.connect("a", "y", "c", "x");
		graph.bindInput("in", w.name(), "in");
		graph.bindOutput("out", "c", "y");

		auto snap = graph.freeze();
		CHECK(snap->loweringStats().erasedConnectors == 0, "wire bound as input is kept");
		CHECK(snap->runtimeNodeCount() == 3, "runtime keeps the input-bearing wire");

		graph.feedInput("t1", w.name(), "in", floatTensor(4.0f));
		graph.submit("t1", "c", "y");
		CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "wire-fed flow should complete");
		auto r = graph.takeOutputTensor("t1", "c", "y");
		CHECK(std::abs(r.item<float>() - 4.0f) < 1e-6f, "value should reach downstream via wire");
	}
}

static void test_errorPropagationThroughLoweredEdge() {
	InferGraph graph;
	graph.addNode(std::make_unique<Node>("Builtin", "fail", idSchema(), failRunFn()));
	graph.addNode(makeId("down"));
	graph.connect("fail", "y", "down", "x");
	graph.bindOutput("out", "down", "y");

	graph.feedInput("t1", "fail", "x", floatTensor(1.0f));
	graph.submit("t1", "down", "y", 1);
	CHECK(graph.waitForResult("t1").status != TaskStatus::Running,
		  "failed upstream → propagation exhausted → task terminates");
	CHECK(graph.taskStatus("t1") == TaskStatus::Failed, "upstream failure → declaration unmet → Failed");
	bool failRecorded = false;
	for (const auto& e : graph.taskErrors("t1"))
		if (e.nodeName == "fail" && e.level == DiagnosticLevel::Error)
			failRecorded = true;
	CHECK(failRecorded, "failing node recorded in diagnostics");
	CHECK(!graph.hasOutput("t1", "down", "y"), "downstream must not produce output");
}

static void test_chainedWiresErased() {
	InferGraph graph;
	graph.addNode(makeId("a"));
	graph.addNode(makeId("b"));
	graph.addNode(makeWire("w1"));
	graph.addNode(makeWire("w2"));

	// 逐链融合至首个保留节点：不产生指向已擦除节点的悬空边
	graph.connect("a", "y", "w1", "in");
	graph.connect("w1", "out_0", "w2", "in");
	graph.connect("w2", "out_0", "b", "x");

	auto snap = graph.freeze();
	CHECK(snap->loweringStats().erasedConnectors == 5, "all chained wires erased (w1,w2 + 3 wrapping)");
	CHECK(snap->runtimeNodeCount() == 2, "runtime view: 2 business nodes");
	CHECK(snap->runtimeEdgeCount() == 1, "runtime view: single fused edge a→b");

	graph.feedInput("t1", "a", "x", floatTensor(9.0f));
	graph.submit("t1", "b", "y");
	CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "chained wires should complete (no dangling edge)");
	CHECK(graph.taskStatus("t1") == TaskStatus::Succeeded, "status Succeeded");
	auto r = graph.takeOutputTensor("t1", "b", "y");
	CHECK(std::abs(r.item<float>() - 9.0f) < 1e-6f, "value should traverse the wire chain");
}

static void test_chainThroughKeptConnector() {
	InferGraph graph;
	graph.addNode(makeId("a"));
	graph.addNode(makeId("b"));
	graph.addNode(makeId("c"));
	graph.addNode(makeWire("w1"));
	graph.addNode(makeWire("bc", 2)); // Broadcast(2)：1→2 分发，不擦除

	graph.connect("a", "y", "w1", "in");
	graph.connect("w1", "out_0", "bc", "in");
	graph.connect("bc", "out_0", "b", "x");
	graph.connect("bc", "out_1", "c", "x");

	auto snap = graph.freeze();
	CHECK(snap->loweringStats().erasedConnectors == 5, "w1 + wrapping wires erased (bc kept)");
	CHECK(snap->runtimeNodeCount() == 4, "runtime keeps bc (4 nodes)");
	CHECK(snap->runtimeEdgeCount() == 3, "fused a→bc + bc's two out-edges");

	graph.feedInput("t1", "a", "x", floatTensor(6.0f));
	graph.submit("t1", {{"b", "y", 1}, {"c", "y", 1}});
	CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "chain through kept broadcast should complete");
	auto rb = graph.takeOutputTensor("t1", "b", "y");
	auto rc = graph.takeOutputTensor("t1", "c", "y");
	CHECK(std::abs(rb.item<float>() - 6.0f) < 1e-6f, "downstream 1 gets value through fused chain");
	CHECK(std::abs(rc.item<float>() - 6.0f) < 1e-6f, "downstream 2 gets value through kept broadcast");
}

// connect() 无法构建裸环（二次 connect 引入第二条入边），故经 connectRaw 构建裸拓扑
static void test_wireOnlyCycleDropsFusedEdge() {
	GraphStore store;
	store.addNode(makeId("x"));
	store.addNode(makeWire("w1"));
	store.addNode(makeWire("w2"));

	store.connectRaw("x", "y", "w1", "in");
	store.connectRaw("w1", "out_0", "w2", "in");
	store.connectRaw("w2", "out_0", "w1", "in");

	GraphSignature signature;
	std::unordered_map<std::string, const Node*> runtimeNodes;
	std::vector<GraphStore::Edge> runtimeEdges;
	GraphLoweringStats stats;
	buildRuntimeView(store, signature, runtimeNodes, runtimeEdges, stats);

	CHECK(stats.erasedConnectors == 2, "both cycle wires erased");
	CHECK(runtimeNodes.size() == 1, "runtime view: only x");
	CHECK(runtimeEdges.empty(), "fused edge into wire-only cycle is dropped");

	bool directRejected = false;
	try {
		GraphStore gs;
		gs.addNode(makeId("p"));
		gs.addNode(makeId("q"));
		gs.connectRaw("p", "y", "q", "x");
	} catch (const GraphException&) {
		directRejected = true;
	}
	CHECK(directRejected, "direct connect between non-connectors should be rejected");
}

static void test_expandedFanOutKept() {
	InferGraph graph;
	graph.addNode(makeId("a"));
	graph.addNode(makeInc("b"));
	graph.addNode(makeInc("c"));
	graph.connect("a", "y", "b", "x");
	graph.connect("a", "y", "c", "x");
	graph.bindOutput("ob", "b", "y");
	graph.bindOutput("oc", "c", "y");

	CHECK(graph.nodeCount() == 4, "source view: 3 biz + 1 expanded wire (4 nodes)");
	CHECK(graph.edgeCount() == 3, "source view: 3 edges");

	auto snap = graph.freeze();
	CHECK(snap->loweringStats().erasedConnectors == 0, "expanded wire is kept (not erased)");
	CHECK(snap->runtimeNodeCount() == 4, "runtime keeps the expanded wire (4 nodes)");
	CHECK(snap->runtimeEdgeCount() == 3, "runtime: a→wire + 2 out-edges (3 edges)");

	graph.feedInput("t1", "a", "x", floatTensor(10.0f));
	graph.submit("t1", {{"b", "y"}, {"c", "y"}});
	CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "task should complete");
	auto rb = graph.takeOutputTensor("t1", "b", "y");
	auto rc = graph.takeOutputTensor("t1", "c", "y");
	CHECK(std::abs(rb.item<float>() - 11.0f) < 1e-6f, "branch b receives a+1");
	CHECK(std::abs(rc.item<float>() - 11.0f) < 1e-6f, "branch c receives a+1");
}

int main() {
	test_autoWireErased();
	test_broadcastN2NotErased();
	test_ttlCountsRuntimeVertices();
	test_cycleTtlStillBounded();
	test_bindingProtectionKeepsWire();
	test_errorPropagationThroughLoweredEdge();

	test_chainedWiresErased();
	test_chainThroughKeptConnector();
	test_wireOnlyCycleDropsFusedEdge();
	test_expandedFanOutKept();

	if (g_failures == 0) {
		std::printf("All %d checks passed\n", g_checks);
		return 0;
	}
	std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
	return 1;
}
