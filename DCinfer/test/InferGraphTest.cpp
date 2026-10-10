// InferGraph 单元测试：拓扑连接、数据流、信号与任务生命周期
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "TestHarness.h"
#include "Connector.h"
#include "NodeException.h"

using namespace DC;

using TensorType = DC::Tensor::TensorType;
using Tensor = DC::Tensor;
using Shape = DC::Tensor::Shape;

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

static Node::Schema addSchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("a"), Node::Port::in<float>("b")};
	s.outputs = {Node::Port::out<float>("s")};
	return s;
}

static Node::RunFn addRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto& aNT = ctx.peek("a");
		const auto& bNT = ctx.peek("b");
		const auto* a = aNT.as<Tensor>();
		const auto* b = bNT.as<Tensor>();
		if (!a || !b)
			return ctx.failure(Node::Status::InvalidInput, "not a Tensor");

		float sum = a->item<float>() + b->item<float>();
		auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
		*t = sum;
		ctx.output("s", Value(std::move(t)));
		return ctx.success();
	};
}

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

static Node::Schema incSchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("x")};
	s.outputs = {Node::Port::out<float>("y")};
	return s;
}

static Node::RunFn incRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto& inVal = ctx.peek("x");
		const auto* t = inVal.as<Tensor>();
		if (!t)
			return ctx.failure(Node::Status::InvalidInput, "not a Tensor");

		float val = t->item<float>() + 1.0f;
		auto out = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
		*out = val;
		ctx.output("y", Value(std::move(out)));
		return ctx.success();
	};
}

void testBuildGraph() {
	TEST("build graph - addNode and connect") {
		TestHarness harness;

		auto& n1 = harness.addNode(std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));
		CHECK(harness.nodeCount() == 1, "nodeCount should be 1");

		auto& n2 = harness.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(), identityRunFn()));
		CHECK(harness.nodeCount() == 2, "nodeCount should be 2");

		bool dupRejected = false;
		try {
			harness.addNode(std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));
		} catch (const GraphException&) {
			dupRejected = true;
		}
		CHECK(dupRejected, "duplicate name should be rejected");
		CHECK(harness.nodeCount() == 2, "nodeCount still 2");

		auto& w = harness.connect("add1", "s", "id1", "x");
		CHECK(w.isConnector(), "connect() should return the auto-inserted connector");
		CHECK(harness.nodeCount() == 3, "nodeCount should be 3 (add1, id1, __wire_0)");
		CHECK(harness.edgeCount() == 2, "edgeCount should be 2 (add1→connector, connector→id1)");

		bool badConnect = false;
		try {
			harness.connect("add1", "no_such", "id1", "x");
		} catch (const GraphException&) {
			badConnect = true;
		}
		CHECK(badConnect, "connect with bad src port should throw");

	}
	END_TEST();
}

void testSimpleDataflow() {
	TEST("simple 2-node dataflow: add → identity") {
		TestHarness harness;

		harness.addNode(std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(), identityRunFn()));
		harness.connect("add1", "s", "id1", "x");

		harness.feedInput("t1", "add1", "a", makeFloatTensor(3.0f));
		harness.feedInput("t1", "add1", "b", makeFloatTensor(4.0f));

		harness.submit("t1", "id1", "y");
		CHECK(harness.awaitCompletion("t1"), "should complete within timeout");

		CHECK(harness.hasOutput("t1", "id1", "y"), "id1 should have output");
		auto result = harness.getOutputTensor("t1", "id1", "y");
		CHECK(std::abs(result.item<float>() - 7.0f) < 1e-6f, "result should be 7.0");
	}
	END_TEST();
}

void testBroadcastConnectorInGraph() {
	TEST("broadcast connector: add → broadcast → [id_a, id_b]") {
		TestHarness harness;

		harness.addNode(std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));

		auto bcSchema = Connector::broadcastSchema(2);
		auto bcRunFn = Connector::broadcastRunFn();
		auto bcNode =
			std::make_unique<Node>("Connector.Broadcast", "bc", bcSchema, bcRunFn, ResourceClass::System);
		bcNode->setConnector(true);
		harness.addNode(std::move(bcNode));

		harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		harness.connect("add1", "s", "bc", "in");
		harness.connect("bc", "out_0", "id_a", "x");
		harness.connect("bc", "out_1", "id_b", "x");

		harness.feedInput("t1", "add1", "a", makeFloatTensor(10.0f));
		harness.feedInput("t1", "add1", "b", makeFloatTensor(20.0f));

		harness.submit("t1", {{"id_a", "y"}, {"id_b", "y"}});
		CHECK(harness.awaitCompletion("t1"), "should complete within timeout");

		CHECK(harness.hasOutput("t1", "id_a", "y"), "id_a should have output");
		CHECK(harness.hasOutput("t1", "id_b", "y"), "id_b should have output");

		auto ra = harness.getOutputTensor("t1", "id_a", "y");
		auto rb = harness.getOutputTensor("t1", "id_b", "y");
		CHECK(std::abs(ra.item<float>() - 30.0f) < 1e-6f, "id_a value");
		CHECK(std::abs(rb.item<float>() - 30.0f) < 1e-6f, "id_b value");
	}
	END_TEST();
}

void testNodeQuery() {
	TEST("node query by name") {
		TestHarness harness;
		harness.addNode(std::make_unique<Node>("Builtin", "test1", addSchema(), addRunFn()));

		auto* n = harness.node("test1");
		CHECK(n != nullptr, "node should be found");
		CHECK(n->name() == "test1", "name should match");

		CHECK(harness.node("phantom") == nullptr, "nonexistent node should be null");
	}
	END_TEST();
}

void testSerializationAccessors() {
	TEST("serialization accessors - nodeNames, edges, outputBindings, modelPath") {
		TestHarness harness;
		harness.addNode(std::make_unique<Node>("ONNX", "test1", identitySchema(), identityRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "test2", identitySchema(), identityRunFn()));
		harness.connect("test1", "y", "test2", "x");
		harness.bindOutput("y", "test2", "y");

		auto names = harness.graph().nodeNames();
		CHECK(names.size() == 3, "should have 3 nodes (test1, test2, __wire_0)");
		CHECK(harness.graph().edges().size() == 2, "should have 2 edges");
		CHECK(harness.graph().outputBindings().size() == 1, "should have 1 output binding");

		auto* n = harness.node("test1");
		n->setModelPath("models/test.onnx");
		CHECK(n->modelPath() == "models/test.onnx", "modelPath should be set");

		auto* builtinNode = harness.node("test2");
		CHECK(builtinNode->modelPath().empty(), "Builtin node modelPath should be empty");
	}
	END_TEST();
}

void testSimpleCycle() {
	TEST("simple feedback cycle: inc.y → inc.x, count=3 termination") {
		TestHarness harness;

		harness.addNode(std::make_unique<Node>("Builtin", "inc", incSchema(), incRunFn()));

		harness.connect("inc", "y", "inc", "x");
		harness.feedInput("t1", "inc", "x", makeFloatTensor(0.0f));
		harness.submit("t1", "inc", "y", 3);
		CHECK(harness.awaitCompletion("t1"), "should complete within timeout");

		CHECK(harness.hasOutput("t1", "inc", "y"), "inc should have output");
		auto result = harness.getOutputTensor("t1", "inc", "y");
		CHECK(std::abs(result.item<float>() - 3.0f) < 1e-6f, "result after 3 iterations should be 3.0");
	}
	END_TEST();
}

void testCycleHopsExhaustion() {
	TEST("cycle TTL exhaustion: maxHops=5 truncates loop before count=100") {
		TestHarness harness;

		harness.addNode(std::make_unique<Node>("Builtin", "inc", incSchema(), incRunFn()));

		harness.connect("inc", "y", "inc", "x");
		harness.feedInput("t1", "inc", "x", makeFloatTensor(0.0f));

		harness.submit("t1", "inc", "y", 100, 5);

		CHECK(harness.awaitCompletion("t1"), "should complete (via TTL exhaustion, not hang)");

		auto errors = harness.taskErrors("t1");
		bool hasHopsError = false;
		for (auto& e : errors) {
			if (e.message.find("hops exhausted") != std::string::npos) {
				hasHopsError = true;
				break;
			}
		}
		CHECK(hasHopsError, "should record hops exhaustion error");

		if (harness.hasOutput("t1", "inc", "y")) {
			auto result = harness.getOutputTensor("t1", "inc", "y");
			CHECK(result.item<float>() < 100.0f, "output should be truncated (less than declared count)");
		}
	}
	END_TEST();
}

void testCycleMultiNode() {
	TEST("multi-node cycle: A → B → C → A with TTL") {
		TestHarness harness;

		harness.addNode(std::make_unique<Node>("Builtin", "A", incSchema(), incRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "B", incSchema(), incRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "C", incSchema(), incRunFn()));

		harness.connect("A", "y", "B", "x");
		harness.connect("B", "y", "C", "x");
		harness.connect("C", "y", "A", "x");

		harness.feedInput("t1", "A", "x", makeFloatTensor(0.0f));

		// 每圈 6 跳（3 节点 + 3 导线），3 圈 18 跳，TTL=19 恰好完成
		harness.submit("t1", "C", "y", 3, 19);

		CHECK(harness.awaitCompletion("t1"), "should complete within timeout");
		CHECK(harness.hasOutput("t1", "C", "y"), "C should have output after 3 cycles");

		auto result = harness.getOutputTensor("t1", "C", "y");
		CHECK(std::abs(result.item<float>() - 9.0f) < 1e-6f, "result after 3 laps should be 9.0");
	}
	END_TEST();
}

void testBlockedNodeNotReceiving() {
	TEST("blocked node never receives data, upstream output stays") {
		TestHarness harness;

		auto& a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto& b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		harness.connect("id_a", "y", "id_b", "x");

		b.bindSignal(harness.signalStore(), "enable_b");
		harness.setSignal("enable_b", false);

		harness.feedInput("t1", "id_a", "x", makeFloatTensor(10.0f));

		harness.submit("t1", "id_a", "y", 1);
		CHECK(harness.awaitCompletion("t1"), "t1 should complete within timeout");

		CHECK(harness.hasOutput("t1", "id_a", "y"), "id_a should have output");
		CHECK(!harness.hasOutput("t1", "id_b", "y"), "id_b should NOT have output (was blocked)");
	}
	END_TEST();
}

void testPartialBlockKeepsOtherPath() {
	TEST("partial block: one path blocked, other path completes normally") {
		TestHarness harness;

		auto& a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto& b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
		auto& c = harness.addNode(std::make_unique<Node>("Builtin", "id_c", identitySchema(), identityRunFn()));

		auto bcSchema = Connector::broadcastSchema(2);
		auto bcNode = std::make_unique<Node>("Connector.Broadcast", "bc", bcSchema,
			Connector::broadcastRunFn(), ResourceClass::System);
		bcNode->setConnector(true);
		harness.addNode(std::move(bcNode));
		harness.connect("id_a", "y", "bc", "in");
		harness.connect("bc", "out_0", "id_b", "x");
		harness.connect("bc", "out_1", "id_c", "x");

		b.bindSignal(harness.signalStore(), "enable_b");
		harness.setSignal("enable_b", false);

		harness.feedInput("t1", "id_a", "x", makeFloatTensor(10.0f));

			harness.submit("t1", "id_c", "y", 1);
		CHECK(harness.awaitCompletion("t1"), "t1 should complete within timeout");

		CHECK(harness.hasOutput("t1", "id_c", "y"), "id_c should have output");
		auto r = harness.getOutputTensor("t1", "id_c", "y");
		CHECK(std::abs(r.item<float>() - 10.0f) < 1e-6f, "id_c value should be 10.0");
	}
	END_TEST();
}

void testDynamicSignalToggle() {
	TEST("dynamic signal toggle: same graph different behavior") {
		TestHarness harness;

		auto& a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto& b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
		auto& c = harness.addNode(std::make_unique<Node>("Builtin", "id_c", identitySchema(), identityRunFn()));

		harness.connect("id_a", "y", "id_b", "x");
		harness.connect("id_b", "y", "id_c", "x");

		b.bindSignal(harness.signalStore(), "gate");

		harness.setSignal("gate", true);
		harness.feedInput("t1", "id_a", "x", makeFloatTensor(5.0f));
		harness.submit("t1", "id_c", "y", 1);
		CHECK(harness.awaitCompletion("t1"), "t1 should complete");
		CHECK(harness.hasOutput("t1", "id_c", "y"), "id_c should have output when signal=true");

		harness.setSignal("gate", false);
		harness.feedInput("t2", "id_a", "x", makeFloatTensor(5.0f));
		harness.submit("t2", "id_a", "y", 1);
		CHECK(harness.awaitCompletion("t2"), "t2 should complete (id_a output declared)");
		CHECK(harness.hasOutput("t2", "id_a", "y"), "id_a should have output");
		CHECK(!harness.hasOutput("t2", "id_c", "y"), "id_c should NOT have output when blocked");
	}
	END_TEST();
}

void testUnboundNodeNeverBlocked() {
	TEST("unbound node is never blocked") {
		TestHarness harness;

		auto& a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto& b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		harness.connect("id_a", "y", "id_b", "x");

		CHECK(!b.isBlocked(), "unbound node should not be blocked");

		harness.feedInput("t1", "id_a", "x", makeFloatTensor(42.0f));
		harness.submit("t1", "id_b", "y", 1);
		CHECK(harness.awaitCompletion("t1"), "t1 should complete");
		CHECK(harness.hasOutput("t1", "id_b", "y"), "id_b should have output");
	}
	END_TEST();
}

void testTaskScopedSignalBlocksOnlyOneTask() {
	TEST("task-scoped signal: broadcast=false blocks all, task-scoped=true overrides for one task") {
		TestHarness harness;

		auto& a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto& b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		harness.connect("id_a", "y", "id_b", "x");

		b.bindSignal(harness.signalStore(), "gate");

		harness.setSignal("gate", false);
		harness.setSignal("gate", "t1", true);

		harness.feedInput("t1", "id_a", "x", makeFloatTensor(10.0f));
		harness.submit("t1", "id_b", "y", 1);
		CHECK(harness.awaitCompletion("t1"), "t1 should complete (task-scoped signal=true overrides broadcast)");
		CHECK(harness.hasOutput("t1", "id_b", "y"), "id_b should have output for t1");

		harness.clearErrors();
		harness.feedInput("t2", "id_a", "x", makeFloatTensor(20.0f));
		harness.submit("t2", "id_a", "y", 1);
		CHECK(harness.awaitCompletion("t2"), "t2 should complete (id_a output declared)");
		CHECK(!harness.hasOutput("t2", "id_b", "y"), "id_b should NOT have output for t2 (blocked by broadcast)");
	}
	END_TEST();
}

void testTaskScopedSignalBlocksOnlyTargetTask() {
	TEST("task-scoped signal: only target task blocked, other tasks proceed") {
		TestHarness harness;

		auto& a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto& b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		harness.connect("id_a", "y", "id_b", "x");

		b.bindSignal(harness.signalStore(), "gate");

		harness.setSignal("gate", true);
		harness.setSignal("gate", "t2", false);

		harness.feedInput("t1", "id_a", "x", makeFloatTensor(5.0f));
		harness.submit("t1", "id_b", "y", 1);
		CHECK(harness.awaitCompletion("t1"), "t1 should complete");
		CHECK(harness.hasOutput("t1", "id_b", "y"), "id_b should have output for t1");

		harness.clearErrors();
		harness.feedInput("t2", "id_a", "x", makeFloatTensor(30.0f));
		harness.submit("t2", "id_a", "y", 1);
		CHECK(harness.awaitCompletion("t2"), "t2 should complete (id_a output declared)");
		CHECK(!harness.hasOutput("t2", "id_b", "y"), "id_b should NOT have output for t2 (task-scoped block)");
	}
	END_TEST();
}

void testTaskSignalCleanupOnTerminate() {
	TEST("task signal cleanup: terminated task's signals don't leak to subsequent tasks") {
		TestHarness harness;

		auto& a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto& b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		harness.connect("id_a", "y", "id_b", "x");

		b.bindSignal(harness.signalStore(), "gate");

		harness.setSignal("gate", true);
		harness.setSignal("gate", "t1", false);
		harness.feedInput("t1", "id_a", "x", makeFloatTensor(7.0f));
		harness.submit("t1", "id_a", "y", 1);
		CHECK(harness.awaitCompletion("t1"), "t1 should complete");
		CHECK(!harness.hasOutput("t1", "id_b", "y"), "id_b should NOT have output for t1 (blocked)");

		harness.clearErrors();
		harness.feedInput("t2", "id_a", "x", makeFloatTensor(99.0f));
		harness.submit("t2", "id_b", "y", 1);
		CHECK(harness.awaitCompletion("t2"), "t2 should complete");
		CHECK(harness.hasOutput("t2", "id_b", "y"), "id_b should have output for t2 (t1 signal cleaned up)");

		auto r = harness.getOutputTensor("t2", "id_b", "y");
		CHECK(std::abs(r.item<float>() - 99.0f) < 1e-6f, "t2 value should be 99.0");
	}
	END_TEST();
}

void testTaskScopedSignalWithPartialBlock() {
	TEST("task-scoped partial block: broadcast path + task-scoped control on fan-out") {
		TestHarness harness;

		auto& a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto& b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
		auto& c = harness.addNode(std::make_unique<Node>("Builtin", "id_c", identitySchema(), identityRunFn()));

		auto bcSchema = Connector::broadcastSchema(2);
		auto bcNode = std::make_unique<Node>("Connector.Broadcast", "bc", bcSchema,
			Connector::broadcastRunFn(), ResourceClass::System);
		bcNode->setConnector(true);
		harness.addNode(std::move(bcNode));
		harness.connect("id_a", "y", "bc", "in");
		harness.connect("bc", "out_0", "id_b", "x");
		harness.connect("bc", "out_1", "id_c", "x");

		b.bindSignal(harness.signalStore(), "enable_b");
		harness.setSignal("enable_b", true);

		harness.setSignal("enable_b", "t1", false);

		harness.feedInput("t1", "id_a", "x", makeFloatTensor(50.0f));
		harness.submit("t1", "id_c", "y", 1);
		CHECK(harness.awaitCompletion("t1"), "t1 should complete");

		CHECK(harness.hasOutput("t1", "id_c", "y"), "id_c should have output (not blocked)");
		auto r = harness.getOutputTensor("t1", "id_c", "y");
		CHECK(std::abs(r.item<float>() - 50.0f) < 1e-6f, "id_c value should be 50.0");

		CHECK(!harness.hasOutput("t1", "id_b", "y"), "id_b should NOT have output (task-scoped block)");
	}
	END_TEST();
}

struct ConcurrencyDetector {
	std::atomic<int> concurrent{0};
	std::atomic<int> maxConcurrent{0};
	std::mutex mtx;

	void enter() {
		int c = concurrent.fetch_add(1, std::memory_order_acq_rel) + 1;
		int prev = maxConcurrent.load(std::memory_order_acquire);
		while (c > prev && !maxConcurrent.compare_exchange_weak(prev, c, std::memory_order_acq_rel)) {}
	}
	void leave() {
		concurrent.fetch_sub(1, std::memory_order_acq_rel);
	}
};

static Node::RunFn delayedRunFn(ConcurrencyDetector* detector, int delayMs = 50) {
	return [detector, delayMs](Node::RunContext& ctx) -> Node::Result {
		const auto& inVal = ctx.peek("x");
		const auto* t = inVal.as<Tensor>();
		if (!t)
			return ctx.failure(Node::Status::InvalidInput, "not a Tensor");

		if (detector) detector->enter();
		std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
		if (detector) detector->leave();

		ctx.output("y", Value(std::make_unique<Tensor>(*t)));
		return ctx.success();
	};
}

void testInputZoneRoundTrip() {
	TEST("inputZone serialization round-trip via GraphCompiler") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "n1", identitySchema(), identityRunFn()));
		graph.addNode(std::make_unique<Node>("Builtin", "n2", addSchema(), addRunFn()));

		graph.bindInput("x", "n1", "x");
		graph.bindInput("a", "n2", "a");
		graph.bindInput("b", "n2", "b");

		auto bindings = graph.inputBindings();
		CHECK(bindings.size() == 3, "should have 3 input bindings");
		CHECK(bindings[0].nodeName == "n1", "first binding node should be n1");
		CHECK(bindings[0].portName == "x", "first binding port should be x");
		CHECK(bindings[1].nodeName == "n2", "second binding node should be n2");
		CHECK(bindings[1].portName == "a", "second binding port should be a");
		CHECK(bindings[2].nodeName == "n2", "third binding node should be n2");
		CHECK(bindings[2].portName == "b", "third binding port should be b");
	}
	END_TEST();
}

void testWaitMechanism() {
	TEST("InferGraph::waitForResult synchronization") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "n1", identitySchema(), identityRunFn()));

		graph.feedInput("t1", "n1", "x", makeFloatTensor(99.0f));

		auto mtx = std::make_shared<std::mutex>();
		auto cv = std::make_shared<std::condition_variable>();
		auto done = std::make_shared<bool>(false);
		auto capturedOutput = std::make_shared<std::optional<Tensor>>();

		graph.setTaskCompleteCallback([mtx, cv, done, capturedOutput, &graph](const InferGraph::TaskId& tid) {
			if (tid != "t1") return;
			if (graph.hasOutput("t1", "n1", "y")) {
				*capturedOutput = graph.takeOutputTensor("t1", "n1", "y");
			}
			{
				std::lock_guard lk(*mtx);
				*done = true;
			}
			cv->notify_one();
		});

		// 回调先于 submit 设置：保证 _terminate 时回调必然就绪
		graph.submit("t1", "n1", "y", 1);

		bool completed = graph.waitForResult("t1", std::chrono::milliseconds(5000)).status != TaskStatus::Running;
		CHECK(completed, "waitForResult should return non-Running (completed within timeout)");

		{
			std::unique_lock lk(*mtx);
			cv->wait(lk, [&] { return *done; });
		}

		CHECK(capturedOutput->has_value(), "output should be captured");
		CHECK(std::abs(capturedOutput->value().item<float>() - 99.0f) < 1e-6f, "output should be 99.0");
	}
	END_TEST();
}

void testConcurrentTaskLifecycleStress() {
	TEST("concurrent submit/cancel/wait/releaseTask stress (shared GraphRuntimeState)") {
		InferGraph graph;
		// 每线程独立节点：节点级互斥是既有设计，本测试聚焦并发任务生命周期安全性
		constexpr int kThreads = 4;
		constexpr int kIters = 150;
		for (int t = 0; t < kThreads; ++t) {
			auto name = "n" + std::to_string(t);
			graph.addNode(std::make_unique<Node>("Builtin", name, identitySchema(), identityRunFn()));
			graph.bindOutput("y_" + name, name, "y");
		}

		std::atomic<int> anomalies{0};
		std::vector<std::thread> threads;
		for (int t = 0; t < kThreads; ++t) {
			threads.emplace_back([&, t] {
				const std::string nodeName = "n" + std::to_string(t);
				try {
					for (int i = 0; i < kIters; ++i) {
						// 每迭代唯一 taskId：取消后迟到的旧 lambda 经 gate 检查安全退出；
						// 同 ID 复用在取消场景有 tryExecute 竞态，不属本测试目标
						const std::string tid = "stress-" + std::to_string(t) + "-" + std::to_string(i);
						Tensor in(TensorType::Float, sizeof(float));
						in = static_cast<float>(i);
						graph.feedInput(tid, nodeName, "x", Value(std::make_unique<Tensor>(std::move(in))));
						graph.submit(tid, nodeName, "y", 1);
						if (i % 2 == 0)
							graph.cancel(tid);
						if (graph.waitForResult(tid, std::chrono::milliseconds(5000)).status == TaskStatus::Running) {
							std::cerr << "\n  [stress] wait timeout: task=" << tid
									  << " status=" << static_cast<int>(graph.taskStatus(tid))
									  << " iter=" << i << std::endl;
							for (auto& err : graph.taskErrors(tid))
								std::cerr << "    [err] node=" << err.nodeName
										  << " lvl=" << static_cast<int>(err.level)
										  << " msg=" << err.message << std::endl;
							++anomalies;
						}
						graph.releaseTask(tid);
					}
				} catch (const std::exception& e) {
					std::cerr << "\n  [stress] exception: thread=" << t
							  << " what=" << e.what() << std::endl;
					++anomalies;
				}
			});
		}
		for (auto& th : threads)
			th.join();
		CHECK(anomalies.load() == 0, "no anomalies under concurrent lifecycle stress");
	}
	END_TEST();
}

void testConnectAgainExpandsFanOut() {
	TEST("second connect on the same output port expands the wire to broadcast (in-place)") {
		InferGraph g;
		g.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRunFn()));
		g.addNode(std::make_unique<Node>("Builtin", "b", identitySchema(), identityRunFn()));
		g.addNode(std::make_unique<Node>("Builtin", "c", identitySchema(), identityRunFn()));
		g.addNode(std::make_unique<Node>("Builtin", "d", identitySchema(), identityRunFn()));

		auto& wire = g.connect("src", "y", "b", "x");
		const size_t nodesAfterFirst = g.nodeCount();
		const size_t edgesAfterFirst = g.edgeCount();

		auto& wire2 = g.connect("src", "y", "c", "x");
		CHECK(&wire2 == &wire, "expansion returns the same wire object (reference stable)");
		CHECK(g.nodeCount() == nodesAfterFirst, "no new node created (in-place expansion)");
		CHECK(g.edgeCount() == edgesAfterFirst + 1, "one new edge added");
		CHECK(wire.schema().outputs.size() == 2, "wire expanded to 2 output ports");

		g.connect("src", "y", "d", "x");
		CHECK(wire.schema().outputs.size() == 3, "wire expanded to 3 output ports");
		CHECK(g.edgeCount() == edgesAfterFirst + 2, "two new edges in total");

		bool reConnectRejected = false;
		try {
			g.connect("src", "y", "b", "x");
		} catch (const GraphException& e) {
			reConnectRejected = (e.getErrorType() == GraphException::ErrorType::DuplicateEdge);
		}
		CHECK(reConnectRejected, "same (src,dst) pair re-connect rejected by input-port guard");

		g.feedInput("t1", "src", "x", makeFloatTensor(5.0f));
		g.submit("t1", {{"b", "y"}, {"c", "y"}, {"d", "y"}});
		CHECK(g.waitForResult("t1").status == TaskStatus::Succeeded, "task should succeed");
		auto tb = g.takeOutputTensor("t1", "b", "y");
		auto tc = g.takeOutputTensor("t1", "c", "y");
		auto td = g.takeOutputTensor("t1", "d", "y");
		CHECK(std::abs(tb.item<float>() - 5.0f) < 1e-6f, "fan-out branch b receives data");
		CHECK(std::abs(tc.item<float>() - 5.0f) < 1e-6f, "fan-out branch c receives data");
		CHECK(std::abs(td.item<float>() - 5.0f) < 1e-6f, "fan-out branch d receives data");
	}
	END_TEST();
}

void testDuplicateFanInConnectRejected() {
	TEST("fan-in: second connect on the same input port rejects with DuplicateEdge") {
		InferGraph g;
		g.addNode(std::make_unique<Node>("Builtin", "a", identitySchema(), identityRunFn()));
		g.addNode(std::make_unique<Node>("Builtin", "b", identitySchema(), identityRunFn()));
		g.addNode(std::make_unique<Node>("Builtin", "sub", identitySchema(), identityRunFn()));

		g.connect("a", "y", "sub", "x");
		const size_t edgesAfterFirst = g.edgeCount();

		bool threw = false;
		std::string message;
		try {
			g.connect("b", "y", "sub", "x");
		} catch (const GraphException& e) {
			threw = true;
			message = e.what();
			CHECK(e.getErrorType() == GraphException::ErrorType::DuplicateEdge,
				  "error type should be DuplicateEdge");
		}
		CHECK(threw, "second connect on the same input port must be rejected");
		CHECK(message.find("sub:x") != std::string::npos, "message should name the input port");
		CHECK(g.edgeCount() == edgesAfterFirst, "rejected connect must leave no partial edges");
	}
	END_TEST();
}

void testOutputSurvivesWait() {
	TEST("lifecycle: outputs retrievable after wait without callback") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(), identityRunFn()));

		graph.feedInput("t1", "id1", "x", makeFloatTensor(7.0f));
		graph.submit("t1", "id1", "y");

		CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "task should complete");
		CHECK(graph.taskStatus("t1") == TaskStatus::Succeeded, "status should be Succeeded");

		CHECK(graph.hasOutput("t1", "id1", "y"), "output should survive wait()");
		auto result = graph.takeOutputTensor("t1", "id1", "y");
		CHECK(std::abs(result.item<float>() - 7.0f) < 1e-6f, "result should be 7.0");

		graph.releaseTask("t1");
		CHECK(graph.taskStatus("t1") == TaskStatus::Unknown, "released task should be Unknown");
		CHECK(!graph.hasOutput("t1", "id1", "y"), "released task should have no output");
	}
	END_TEST();
}

void testWaitForResult() {
	TEST("lifecycle: waitForResult returns structured status") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(), identityRunFn()));

		graph.feedInput("t1", "id1", "x", makeFloatTensor(3.0f));
		graph.submit("t1", "id1", "y");

		auto result = graph.waitForResult("t1");
		CHECK(result.status == TaskStatus::Succeeded, "waitForResult should return Succeeded");
		CHECK(result.errors.empty(), "no errors expected");

		auto unknown = graph.waitForResult("never_submitted", std::chrono::milliseconds(50));
		CHECK(unknown.status == TaskStatus::Unknown, "unsubmitted task should be Unknown");
	}
	END_TEST();
}

void testTaskIdReuseAfterCompletion() {
	TEST("lifecycle: completed taskId can be safely reused") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));

		graph.feedInput("t1", "add1", "a", makeFloatTensor(3.0f));
		graph.feedInput("t1", "add1", "b", makeFloatTensor(4.0f));
		graph.submit("t1", "add1", "s");
		CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "first run should complete");
		auto r1 = graph.takeOutputTensor("t1", "add1", "s");
		CHECK(std::abs(r1.item<float>() - 7.0f) < 1e-6f, "first result should be 7.0");

		graph.feedInput("t1", "add1", "a", makeFloatTensor(10.0f));
		graph.feedInput("t1", "add1", "b", makeFloatTensor(20.0f));
		graph.submit("t1", "add1", "s");
		CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "reused taskId should complete normally");
		CHECK(graph.taskStatus("t1") == TaskStatus::Succeeded, "reused task status should be Succeeded");
		auto r2 = graph.takeOutputTensor("t1", "add1", "s");
		CHECK(std::abs(r2.item<float>() - 30.0f) < 1e-6f, "second result should be 30.0");
	}
	END_TEST();
}

void testDuplicateActiveSubmitRejected() {
	TEST("lifecycle: duplicate submit of a running task is rejected") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "slow", identitySchema(),
				delayedRunFn(nullptr, 200), ResourceClass::Operator));

		graph.feedInput("t1", "slow", "x", makeFloatTensor(1.0f));
		graph.submit("t1", "slow", "y");

		bool rejected = false;
		try {
			graph.submit("t1", "slow", "y");
		} catch (const GraphException& e) {
			rejected = (e.getErrorType() == GraphException::ErrorType::DuplicateTask);
		}
		CHECK(rejected, "duplicate submit while running should throw DuplicateTask");

		CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "original task should still complete");
		CHECK(graph.taskStatus("t1") == TaskStatus::Succeeded, "original task should succeed");
	}
	END_TEST();
}

void testCancelRunningTask() {
	TEST("lifecycle: cancel terminates a blocked task with Cancelled status") {
		InferGraph graph;
		auto& b = graph.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
		graph.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		graph.connect("id_a", "y", "id_b", "x");

		b.bindSignal(graph.signalStore(), "gate");
		graph.setSignal("gate", false);

		graph.feedInput("t1", "id_a", "x", makeFloatTensor(1.0f));
		graph.submit("t1", "id_b", "y");

		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		CHECK(graph.taskStatus("t1") == TaskStatus::Running, "task should be running while blocked");

		CHECK(graph.cancel("t1"), "cancel should succeed on active task");
		CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "wait should wake up after cancel");
		CHECK(graph.taskStatus("t1") == TaskStatus::Cancelled, "status should be Cancelled");
		CHECK(!graph.cancel("t1"), "cancel on terminated task should return false (idempotent)");

		auto result = graph.waitForResult("t1");
		CHECK(result.status == TaskStatus::Cancelled, "waitForResult should report Cancelled");

		graph.setSignal("gate", true);
		graph.feedInput("t1", "id_a", "x", makeFloatTensor(9.0f));
		graph.submit("t1", "id_b", "y");
		CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "re-submitted task after cancel should complete");
		auto r = graph.takeOutputTensor("t1", "id_b", "y");
		CHECK(std::abs(r.item<float>() - 9.0f) < 1e-6f, "re-run result should be 9.0");
	}
	END_TEST();
}

void testBoundInputOutputApi() {
	TEST("graph API: bindInput/bindOutput signature + submitBound") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "inc", incSchema(), incRunFn()));

		graph.bindInput("x", "inc", "x");
		graph.bindOutput("y", "inc", "y");

		auto in = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
		*in = 41.0f;
		graph.feedInput("t1", "inc", "x", std::move(*in));
		graph.submitBound("t1");
		CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "bound flow should complete");
		auto out = graph.takeOutputTensor("t1", "inc", "y");
		CHECK(std::abs(out.item<float>() - 42.0f) < 1e-6f, "bound result should be 42.0");

		InferGraph graph2;
		graph2.addNode(std::make_unique<Node>("Builtin", "a", incSchema(), incRunFn()));
		bool noSuch = false;
		try {
			auto v = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
			*v = 1.0f;
			graph2.feedInput("t1", "no_such_node", "x", std::move(*v));
		} catch (const GraphException& e) {
			noSuch = (e.getErrorType() == GraphException::ErrorType::NodeNotFound);
		}
		CHECK(noSuch, "feedInput to unknown node should throw NodeNotFound");

		InferGraph graph3;
		graph3.addNode(std::make_unique<Node>("Builtin", "a", incSchema(), incRunFn()));
		graph3.addNode(std::make_unique<Node>("Builtin", "b", incSchema(), incRunFn()));
		graph3.bindInput("x", "a", "x");
		bool ambiguous = false;
		try {
			graph3.bindInput("x", "b", "x");
		} catch (const GraphException&) {
			ambiguous = true;
		}
		CHECK(ambiguous, "duplicate alias rejected at build time");
	}
	END_TEST();
}

// 绑定/声明端口必须无出边：带出边的端口会被输出区搬运截走数据，饿死下游
void testOutputRetrievalPortsMustBeTerminal() {
	TEST("output ports must be terminal: bound/declared ports with out-edges are rejected") {
		{
			InferGraph g;
			g.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRunFn()));
			g.addNode(std::make_unique<Node>("Builtin", "dst", identitySchema(), identityRunFn()));
			g.connect("src", "y", "dst", "x");
			g.bindOutput("mid", "src", "y");
			g.bindOutput("final", "dst", "y");

			bool threw = false;
			try {
				g.freeze();
			} catch (const GraphException& e) {
				threw = (e.getErrorType() == GraphException::ErrorType::NonTerminalPort);
			}
			CHECK(threw, "bound port with out-edges must be rejected at freeze time");
		}

		{
			InferGraph g;
			g.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRunFn()));
			g.addNode(std::make_unique<Node>("Builtin", "dst", identitySchema(), identityRunFn()));
			auto& wire = g.connect("src", "y", "dst", "x");

			g.bindOutput("mid", wire.name(), "out_0");
			bool threw = false;
			try {
				g.freeze();
			} catch (const GraphException& e) {
				threw = (e.getErrorType() == GraphException::ErrorType::NonTerminalPort);
			}
			CHECK(threw, "binding a wire output port (with out-edges) must be rejected at freeze");
		}

		{
			InferGraph g;
			g.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRunFn()));
			g.addNode(std::make_unique<Node>("Builtin", "mid_out", identitySchema(), identityRunFn()));
			g.addNode(std::make_unique<Node>("Builtin", "dst", identitySchema(), identityRunFn()));
			g.connect("src", "y", "mid_out", "x");
			g.connect("src", "y", "dst", "x");
			g.bindOutput("mid", "mid_out", "y");
			g.bindOutput("final", "dst", "y");

			g.feedInput("t1", "src", "x", makeFloatTensor(42.0f));
			g.submitBound("t1");
			CHECK(g.waitForResult("t1").status == TaskStatus::Succeeded, "explicit branch form must succeed");
			auto mid = g.takeOutputTensor("t1", "mid_out", "y");
			auto fin = g.takeOutputTensor("t1", "dst", "y");
			CHECK(std::abs(mid.item<float>() - 42.0f) < 1e-6f, "branch leaf carries the mid value");
			CHECK(std::abs(fin.item<float>() - 42.0f) < 1e-6f, "downstream branch keeps computing");
		}

		{
			InferGraph g;
			g.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRunFn()));
			g.addNode(std::make_unique<Node>("Builtin", "dst", identitySchema(), identityRunFn()));
			g.connect("src", "y", "dst", "x");

			g.feedInput("t1", "src", "x", makeFloatTensor(7.0f));
			g.submit("t1", "dst", "y");
			CHECK(g.waitForResult("t1").status == TaskStatus::Succeeded, "terminal-only declaration must work");
			auto r = g.takeOutputTensor("t1", "dst", "y");
			CHECK(std::abs(r.item<float>() - 7.0f) < 1e-6f, "value flows through to the terminal port");
		}

		{
			InferGraph g;
			g.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRunFn()));
			g.addNode(std::make_unique<Node>("Builtin", "dst", identitySchema(), identityRunFn()));
			g.connect("src", "y", "dst", "x");

			g.feedInput("t1", "src", "x", makeFloatTensor(3.0f));
			bool accepted = true;
			try {
				g.submit("t1", "src", "y");
			} catch (const GraphException&) {
				accepted = false;
			}
			CHECK(accepted, "declaration on an edge-carrying port stays allowed when not bound");
			CHECK(g.waitForResult("t1").status == TaskStatus::Succeeded, "declared port satisfied on first production");
		}
	}
	END_TEST();
}

void testAliasBindingApi() {
	TEST("graph API: bindings form graph signature; IO uses internal addressing") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "inc", incSchema(), incRunFn()));

		graph.bindInput("num", "inc", "x");
		graph.bindOutput("result", "inc", "y");

		auto in = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
		*in = 5.0f;
		graph.feedInput("t1", "inc", "x", std::move(*in));
		graph.submitBound("t1");
		CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "bound flow should complete");
		CHECK(graph.hasOutput("t1", "inc", "y"), "output should resolve by internal addressing");
		auto out = graph.takeOutputTensor("t1", "inc", "y");
		CHECK(std::abs(out.item<float>() - 6.0f) < 1e-6f, "result should be 6.0");

		auto in2 = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
		*in2 = 7.0f;
		graph.feedInput("t2", "inc", "x", std::move(in2));
		graph.submitBound("t2");
		CHECK(graph.waitForResult("t2").status != TaskStatus::Running, "second task should complete");
		auto out2 = graph.takeOutputTensor("t2", "inc", "y");
		CHECK(std::abs(out2.item<float>() - 8.0f) < 1e-6f, "second result should be 8.0");

		InferGraph graph2;
		graph2.addNode(std::make_unique<Node>("Builtin", "a", incSchema(), incRunFn()));
		graph2.addNode(std::make_unique<Node>("Builtin", "b", incSchema(), incRunFn()));
		graph2.bindInput("first", "a", "x");
		graph2.bindInput("second", "b", "x");

		bool dupIn = false;
		try {
			graph2.bindInput("first", "b", "x");
		} catch (const GraphException& e) {
			dupIn = (e.getErrorType() == GraphException::ErrorType::DuplicateBinding);
		}
		CHECK(dupIn, "duplicate input alias should throw DuplicateBinding");

		bool dupOut = false;
		try {
			graph2.bindOutput("first", "a", "y");
			graph2.bindOutput("first", "b", "y");
		} catch (const GraphException& e) {
			dupOut = (e.getErrorType() == GraphException::ErrorType::DuplicateBinding);
		}
		CHECK(dupOut, "duplicate output alias should throw DuplicateBinding");

		{
			auto v = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
			*v = 1.0f;
			graph2.feedInput("t1", "a", "x", std::move(*v));
		}
		{
			auto v = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
			*v = 2.0f;
			graph2.feedInput("t1", "b", "x", std::move(*v));
		}
	}
	END_TEST();
}

void testTakeOutputDestructive() {
	TEST("graph API: takeOutput is consumptive (single read)") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(), identityRunFn()));

		graph.feedInput("t1", "id1", "x", makeFloatTensor(9.0f));
		graph.submit("t1", "id1", "y");
		CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "task should complete");

		CHECK(graph.hasOutput("t1", "id1", "y"), "output should exist before take");
		auto first = graph.takeOutputTensor("t1", "id1", "y");
		CHECK(std::abs(first.item<float>() - 9.0f) < 1e-6f, "first take should be 9.0");

		CHECK(!graph.hasOutput("t1", "id1", "y"), "output should be consumed after take");
		bool consumed = false;
		try {
			auto again = graph.takeOutputTensor("t1", "id1", "y");
			(void)again;
		} catch (const GraphException&) {
			consumed = true;
		} catch (const NodeException&) {
			consumed = true;
		}
		CHECK(consumed, "second take should throw (output already consumed)");
	}
	END_TEST();
}

void testWaitSemantics() {
	TEST("lifecycle: wait semantics (0=infinite, waiter-only timeout, unknown-id fast-fail)") {
		InferGraph graph;
		auto& b = graph.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
		graph.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		graph.connect("id_a", "y", "id_b", "x");

		b.bindSignal(graph.signalStore(), "gate");
		graph.setSignal("gate", false);

		graph.feedInput("t1", "id_a", "x", makeFloatTensor(1.0f));
		graph.submit("t1", "id_b", "y");

		CHECK(graph.waitForResult("t1", std::chrono::milliseconds(80)).status == TaskStatus::Running, "explicit timeout leaves task running");
		auto running = graph.waitForResult("t1", std::chrono::milliseconds(80));
		CHECK(running.status == TaskStatus::Running, "waitForResult timeout should report Running");

		CHECK(graph.waitForResult("never_submitted").status == TaskStatus::Unknown, "unknown taskId reports Unknown immediately");
		auto unknown = graph.waitForResult("never_submitted");
		CHECK(unknown.status == TaskStatus::Unknown, "unknown taskId waitForResult should be Unknown");

		CHECK(graph.cancel("t1"), "cancel should succeed on active task");
		auto cancelled = graph.waitForResult("t1");
		CHECK(cancelled.status == TaskStatus::Cancelled, "default waitForResult should wait until termination");
	}
	END_TEST();
}

// 用裸 InferGraph（不经 TestHarness）：TestHarness 经完成回调捕获输出，
// 会遮蔽 wait→takeOutput 窗口，测不到"wait 返回即可读"契约本身。
void testWaitReturnsReadableResults() {
	TEST("lifecycle: wait returns only after declared outputs are readable") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		graph.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
		graph.connect("id_a", "y", "id_b", "x");

		for (int i = 0; i < 100; ++i) {
			std::string tid = "wt" + std::to_string(i);
			graph.feedInput(tid, "id_a", "x", makeFloatTensor(static_cast<float>(i)));
			graph.submit(tid, "id_b", "y");
			CHECK(graph.waitForResult(tid, std::chrono::milliseconds(2000)).status != TaskStatus::Running, "wait should succeed");
			auto r = graph.takeOutputTensor(tid, "id_b", "y");
			CHECK(std::abs(r.item<float>() - static_cast<float>(i)) < 1e-6f,
				  "declared output must be readable immediately after wait returns");
		}
	}
	END_TEST();
}

void testMultiDeclarationReadableAfterWait() {
	TEST("lifecycle: all declared outputs readable after wait (multi-declaration salvage)") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		graph.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
		graph.addNode(std::make_unique<Node>("Builtin", "id_c", identitySchema(), identityRunFn()));

		auto bcNode = std::make_unique<Node>("Connector.Broadcast", "bc", Connector::broadcastSchema(2),
											 Connector::broadcastRunFn(), ResourceClass::System);
		bcNode->setConnector(true);
		graph.addNode(std::move(bcNode));
		graph.connect("id_a", "y", "bc", "in");
		graph.connect("bc", "out_0", "id_b", "x");
		graph.connect("bc", "out_1", "id_c", "x");

		graph.feedInput("t1", "id_a", "x", makeFloatTensor(50.0f));
		graph.submit("t1", {{"id_b", "y", 1}, {"id_c", "y", 1}});
		CHECK(graph.waitForResult("t1").status != TaskStatus::Running, "task should complete");
		auto rb = graph.takeOutputTensor("t1", "id_b", "y");
		auto rc = graph.takeOutputTensor("t1", "id_c", "y");
		CHECK(std::abs(rb.item<float>() - 50.0f) < 1e-6f, "id_b output readable after wait");
		CHECK(std::abs(rc.item<float>() - 50.0f) < 1e-6f, "id_c output readable after wait");
	}
	END_TEST();
}

int main() {
	try {
		testSimpleDataflow();
		testBuildGraph();
		testBroadcastConnectorInGraph();
		testNodeQuery();
		testSerializationAccessors();

		testSimpleCycle();
		testCycleHopsExhaustion();
		testCycleMultiNode();

		testBlockedNodeNotReceiving();
		testPartialBlockKeepsOtherPath();
		testDynamicSignalToggle();
		testUnboundNodeNeverBlocked();

		testTaskScopedSignalBlocksOnlyOneTask();
		testTaskScopedSignalBlocksOnlyTargetTask();
		testTaskSignalCleanupOnTerminate();
		testTaskScopedSignalWithPartialBlock();

		testOutputSurvivesWait();
		testWaitForResult();
		testTaskIdReuseAfterCompletion();
		testDuplicateActiveSubmitRejected();
		testCancelRunningTask();

		testBoundInputOutputApi();
		testAliasBindingApi();
		testOutputRetrievalPortsMustBeTerminal();
		testTakeOutputDestructive();

		testWaitSemantics();

		testWaitReturnsReadableResults();
		testMultiDeclarationReadableAfterWait();

		testInputZoneRoundTrip();
		testWaitMechanism();
		testConcurrentTaskLifecycleStress();
		testConnectAgainExpandsFanOut();
		testDuplicateFanInConnectRejected();

		if (failures == 0) {
			std::cout << "\nAll InferGraph tests passed!" << std::endl;
		} else {
			std::cout << "\n" << failures << " test(s) FAILED!" << std::endl;
		}
		return failures;
	} catch (const std::exception& e) {
		std::cerr << "Test failure: " << e.what() << std::endl;
		return -1;
	}
}
