// InferGraph 拓扑连接与数据流 单元测试（异步场景化版本）
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "TestHarness.h"
#include "Connector.h"

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

// ── 辅助 ──

static Value makeFloatTensor(float value) {
	auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
	*t = value;
	return Value(std::move(t));
}

// ── 加法算子 Schema + RunFn ──
static Node::Schema addSchema() {
	Node::Schema s;
	s.inputs = {{"a", TensorType::Float, sizeof(float), {}}, {"b", TensorType::Float, sizeof(float), {}}};
	s.outputs = {{"s", TensorType::Float, sizeof(float), {}}};
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

// ── 恒等算子 ──
static Node::Schema identitySchema() {
	Node::Schema s;
	s.inputs = {{"x", TensorType::Float, sizeof(float), {}}};
	s.outputs = {{"y", TensorType::Float, sizeof(float), {}}};
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

// ── 增 1 算子（用于反馈环测试）──
static Node::Schema incSchema() {
	Node::Schema s;
	s.inputs = {{"x", TensorType::Float, sizeof(float), {}}};
	s.outputs = {{"y", TensorType::Float, sizeof(float), {}}};
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

// ════════════════════════════════════════════
// 测试用例
// ════════════════════════════════════════════

void testBuildGraph() {
	TEST("build graph - addNode and wire") {
		TestHarness harness;

		auto* n1 = harness.addNode(std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));
		CHECK(n1 != nullptr, "addNode should succeed");
		CHECK(harness.nodeCount() == 1, "nodeCount should be 1");

		auto* n2 = harness.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(), identityRunFn()));
		CHECK(n2 != nullptr, "second addNode");
		CHECK(harness.nodeCount() == 2, "nodeCount should be 2");

		// 重名应拒绝
		auto* dup = harness.addNode(std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));
		CHECK(dup == nullptr, "duplicate name should be rejected");
		CHECK(harness.nodeCount() == 2, "nodeCount still 2");

		// 接线：两个业务节点之间 → wire() 自动插入导线连接器
		auto* w = harness.wire("add1", "s", "id1", "x");
		CHECK(w != nullptr, "wire should succeed");
		CHECK(w->isConnector(), "wire node should be a connector");
		CHECK(harness.nodeCount() == 3, "nodeCount should be 3 (add1, id1, __wire_0)");
		CHECK(harness.edgeCount() == 2, "edgeCount should be 2 (add1→wire, wire→id1)");

		// 无效接线：端口不存在
		auto* bad = harness.wire("add1", "no_such", "id1", "x");
		CHECK(bad == nullptr, "wire with bad src port should fail");

		// 业务节点直连应被 connect() 拒绝
		bool direct = harness.connect("add1", "s", "id1", "x");
		CHECK(!direct, "direct connect between non-connectors should be rejected");
	}
	END_TEST();
}

void testSimpleDataflow() {
	TEST("simple 2-node dataflow: add → identity") {
		TestHarness harness;

		harness.addNode(std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(), identityRunFn()));
		harness.wire("add1", "s", "id1", "x");

		// 注入输入
		CHECK(harness.feedInput("t1", "add1", "a", makeFloatTensor(3.0f)), "feed a");
		CHECK(harness.feedInput("t1", "add1", "b", makeFloatTensor(4.0f)), "feed b");

		// 检查就绪
		CHECK(harness.node("add1")->isReady("t1"), "add1 should be ready");

		// 异步驱动执行
		harness.declareOutput("t1", "id1", "y");
		harness.submit("t1");
		CHECK(harness.awaitCompletion("t1"), "should complete within timeout");

		// 验证最终结果
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

		// 广播连接器：1 输入 → 2 输出
		auto bcSchema = Connector::broadcastSchema(2);
		auto bcRunFn = Connector::broadcastRunFn();
		auto bcNode =
			std::make_unique<Node>("Connector.Broadcast", "bc", bcSchema, bcRunFn, nullptr, ThreadPoolAffinity::System);
		bcNode->setConnector(true);
		harness.addNode(std::move(bcNode));

		harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		// 接线：add1 → bc → [id_a, id_b]
		harness.connect("add1", "s", "bc", "in");
		harness.connect("bc", "out_0", "id_a", "x");
		harness.connect("bc", "out_1", "id_b", "x");

		// 注入
		harness.feedInput("t1", "add1", "a", makeFloatTensor(10.0f));
		harness.feedInput("t1", "add1", "b", makeFloatTensor(20.0f));

		// 声明两个下游输出
		harness.declareOutput("t1", "id_a", "y");
		harness.declareOutput("t1", "id_b", "y");
		harness.submit("t1");
		CHECK(harness.awaitCompletion("t1"), "should complete within timeout");

		// 两个下游都应该有结果
		CHECK(harness.hasOutput("t1", "id_a", "y"), "id_a should have output");
		CHECK(harness.hasOutput("t1", "id_b", "y"), "id_b should have output");

		auto ra = harness.getOutputTensor("t1", "id_a", "y");
		auto rb = harness.getOutputTensor("t1", "id_b", "y");
		CHECK(std::abs(ra.item<float>() - 30.0f) < 1e-6f, "id_a value");
		CHECK(std::abs(rb.item<float>() - 30.0f) < 1e-6f, "id_b value");
	}
	END_TEST();
}

void testRoutingConnectorInGraph() {
	TEST("routing connector: add → routing → [id_a, id_b]") {
		TestHarness harness;

		harness.addNode(std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));

		auto rtSchema = Connector::routingSchema(2);
		auto rtRunFn = Connector::routingRunFn();
		auto rtNode =
			std::make_unique<Node>("Connector.Routing", "rt", rtSchema, rtRunFn, nullptr, ThreadPoolAffinity::System);
		rtNode->setConnector(true);
		harness.addNode(std::move(rtNode));

		harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		harness.connect("add1", "s", "rt", "in");
		harness.connect("rt", "out_0", "id_a", "x");
		harness.connect("rt", "out_1", "id_b", "x");

		// 第一轮：t1 → out_0 → id_a
		harness.feedInput("t1", "add1", "a", makeFloatTensor(1.0f));
		harness.feedInput("t1", "add1", "b", makeFloatTensor(2.0f));
		harness.declareOutput("t1", "id_a", "y");
		harness.submit("t1");
		CHECK(harness.awaitCompletion("t1"), "t1 should complete within timeout");

		CHECK(harness.hasOutput("t1", "id_a", "y"), "t1 should route to id_a (out_0)");
		CHECK(!harness.hasOutput("t1", "id_b", "y"), "t1 should NOT route to id_b");

		auto r1 = harness.getOutputTensor("t1", "id_a", "y");
		CHECK(std::abs(r1.item<float>() - 3.0f) < 1e-6f, "t1 value");

		// 第二轮：t2 → out_1 → id_b
		harness.feedInput("t2", "add1", "a", makeFloatTensor(5.0f));
		harness.feedInput("t2", "add1", "b", makeFloatTensor(6.0f));
		harness.declareOutput("t2", "id_b", "y");
		harness.submit("t2");
		CHECK(harness.awaitCompletion("t2"), "t2 should complete within timeout");

		CHECK(!harness.hasOutput("t2", "id_a", "y"), "t2 should NOT route to id_a");
		CHECK(harness.hasOutput("t2", "id_b", "y"), "t2 should route to id_b (out_1)");

		auto r2 = harness.getOutputTensor("t2", "id_b", "y");
		CHECK(std::abs(r2.item<float>() - 11.0f) < 1e-6f, "t2 value");
	}
	END_TEST();
}

void testConnectAll() {
	TEST("connectAll auto-matches output ports to input ports") {
		TestHarness harness;

		auto bcSchema = Connector::broadcastSchema(2);
		auto bcRunFn = Connector::broadcastRunFn();
		auto bcNode =
			std::make_unique<Node>("Connector.Broadcast", "bc", bcSchema, bcRunFn, nullptr, ThreadPoolAffinity::System);
		bcNode->setConnector(true);
		harness.addNode(std::move(bcNode));

		// 创建一个有两个输入端口的节点，命名为与 Connector 输出同名的 Schema
		Node::Schema dualInSchema;
		dualInSchema.inputs = {{"out_0", TensorType::Float, sizeof(float), {}},
							   {"out_1", TensorType::Float, sizeof(float), {}}};
		dualInSchema.outputs = {{"sum", TensorType::Float, sizeof(float), {}}};

		auto dualRunFn = [](Node::RunContext& ctx) -> Node::Result {
			const auto& aNT = ctx.peek("out_0");
			const auto& bNT = ctx.peek("out_1");
			const auto* a = aNT.as<Tensor>();
			const auto* b = bNT.as<Tensor>();
			if (!a || !b)
				return ctx.failure(Node::Status::InvalidInput, "not Tensor");
			float sum = a->item<float>() + b->item<float>();
			auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
			*t = sum;
			ctx.output("sum", Value(std::move(t)));
			return ctx.success();
		};

		harness.addNode(std::make_unique<Node>("Builtin", "adder", dualInSchema, dualRunFn));

		size_t matched = harness.connectAll("bc", "adder");
		CHECK(matched == 2, "connectAll should match 2 ports");
		CHECK(harness.edgeCount() == 2, "edgeCount should be 2");

		// 验证数据流
		harness.feedInput("t1", "bc", "in", makeFloatTensor(5.0f));
		harness.declareOutput("t1", "adder", "sum");
		harness.submit("t1");
		CHECK(harness.awaitCompletion("t1"), "should complete within timeout");

		CHECK(harness.hasOutput("t1", "adder", "sum"), "adder should have output");
		auto r = harness.getOutputTensor("t1", "adder", "sum");
		CHECK(std::abs(r.item<float>() - 10.0f) < 1e-6f, "sum should be 5+5=10");
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
		harness.wire("test1", "y", "test2", "x");
		harness.bindOutput("test2", "y");

		// 遍历
		auto names = harness.graph().nodeNames();
		CHECK(names.size() == 3, "should have 3 nodes (test1, test2, __wire_0)");
		CHECK(harness.graph().edges().size() == 2, "should have 2 edges");
		CHECK(harness.graph().outputBindings().size() == 1, "should have 1 output binding");

		// modelPath
		auto* n = harness.node("test1");
		n->setModelPath("models/test.onnx");
		CHECK(n->modelPath() == "models/test.onnx", "modelPath should be set");

		// Builtin 节点 modelPath 默认为空
		auto* builtinNode = harness.node("test2");
		CHECK(builtinNode->modelPath().empty(), "Builtin node modelPath should be empty");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 循环测试
// ════════════════════════════════════════════

void testSimpleCycle() {
	TEST("simple feedback cycle: inc.y → inc.x, count=3 termination") {
		TestHarness harness;

		harness.addNode(std::make_unique<Node>("Builtin", "inc", incSchema(), incRunFn()));

		// 反馈环：inc.y → inc.x
		harness.wire("inc", "y", "inc", "x");

		// 注入初始值
		harness.feedInput("t1", "inc", "x", makeFloatTensor(0.0f));

		// 期望产出 3 次
		harness.declareOutput("t1", "inc", "y", 3);
		harness.submit("t1");
		CHECK(harness.awaitCompletion("t1"), "should complete within timeout");

		CHECK(harness.hasOutput("t1", "inc", "y"), "inc should have output");
		auto result = harness.getOutputTensor("t1", "inc", "y");
		// 3 iterations: 0→1→2→3
		CHECK(std::abs(result.item<float>() - 3.0f) < 1e-6f, "result after 3 iterations should be 3.0");
	}
	END_TEST();
}

void testCycleHopsExhaustion() {
	TEST("cycle TTL exhaustion: maxHops=5 truncates loop before count=100") {
		TestHarness harness;

		harness.addNode(std::make_unique<Node>("Builtin", "inc", incSchema(), incRunFn()));

		// 反馈环
		harness.wire("inc", "y", "inc", "x");
		harness.feedInput("t1", "inc", "x", makeFloatTensor(0.0f));

		// 声明一个极大的 count，不可能在 5 跳内完成
		harness.declareOutput("t1", "inc", "y", 100);
		harness.submit("t1", std::chrono::milliseconds(0), 5);

		CHECK(harness.awaitCompletion("t1"), "should complete (via TTL exhaustion, not hang)");

		// 应该有 TTL 耗尽错误记录
		auto errors = harness.taskErrors("t1");
		bool hasHopsError = false;
		for (auto& e : errors) {
			if (e.message.find("hops exhausted") != std::string::npos) {
				hasHopsError = true;
				break;
			}
		}
		CHECK(hasHopsError, "should record hops exhaustion error");

		// 输出应少于声明（5 跳意味着最多 5 次迭代，实际约 ≤5）
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

		// 三个恒等节点成环: A → B → C → A
		harness.addNode(std::make_unique<Node>("Builtin", "A", incSchema(), incRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "B", incSchema(), incRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "C", incSchema(), incRunFn()));

		harness.wire("A", "y", "B", "x");
		harness.wire("B", "y", "C", "x");
		harness.wire("C", "y", "A", "x");

		harness.feedInput("t1", "A", "x", makeFloatTensor(0.0f));

		// 每圈 3 节点 + 3 导线 = 6 跳，3 圈 = 18 跳 → TTL=19 刚好完成
		harness.declareOutput("t1", "C", "y", 3);
		harness.submit("t1", std::chrono::milliseconds(0), 19);

		CHECK(harness.awaitCompletion("t1"), "should complete within timeout");
		CHECK(harness.hasOutput("t1", "C", "y"), "C should have output after 3 cycles");

		auto result = harness.getOutputTensor("t1", "C", "y");
		// 3 nodes + 3 wires per lap = 6 hops/lap，3 laps = +9, starting from 0
		CHECK(std::abs(result.item<float>() - 9.0f) < 1e-6f, "result after 3 laps should be 9.0");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 信号 + 阻塞标志测试
// ════════════════════════════════════════════

void testBlockedNodeNotReceiving() {
	TEST("blocked node never receives data, upstream output stays") {
		TestHarness harness;

		auto* a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto* b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		harness.wire("id_a", "y", "id_b", "x");

		b->bindSignal(harness.signalStore(), "enable_b");
		harness.setSignal("enable_b", false);

		harness.declareOutput("t1", "id_a", "y");
		harness.feedInput("t1", "id_a", "x", makeFloatTensor(10.0f));

		harness.submit("t1", std::chrono::milliseconds(2000));
		CHECK(harness.awaitCompletion("t1"), "t1 should complete within timeout");

		CHECK(harness.hasOutput("t1", "id_a", "y"), "id_a should have output");
		CHECK(!harness.hasOutput("t1", "id_b", "y"), "id_b should NOT have output (was blocked)");
	}
	END_TEST();
}

void testPartialBlockKeepsOtherPath() {
	TEST("partial block: one path blocked, other path completes normally") {
		TestHarness harness;

		auto* a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto* b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
		auto* c = harness.addNode(std::make_unique<Node>("Builtin", "id_c", identitySchema(), identityRunFn()));

		// 使用广播连接器扇出（避免 wire 同端口 getOutput 抢消费）
		auto bcSchema = Connector::broadcastSchema(2);
		auto bcNode = std::make_unique<Node>("Connector.Broadcast", "bc", bcSchema,
			Connector::broadcastRunFn(), nullptr, ThreadPoolAffinity::System);
		bcNode->setConnector(true);
		harness.addNode(std::move(bcNode));
		harness.connect("id_a", "y", "bc", "in");
		harness.connect("bc", "out_0", "id_b", "x");
		harness.connect("bc", "out_1", "id_c", "x");

		b->bindSignal(harness.signalStore(), "enable_b");
		harness.setSignal("enable_b", false);

		harness.declareOutput("t1", "id_c", "y");
		harness.feedInput("t1", "id_a", "x", makeFloatTensor(10.0f));

		harness.submit("t1", std::chrono::milliseconds(2000));
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

		auto* a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto* b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
		auto* c = harness.addNode(std::make_unique<Node>("Builtin", "id_c", identitySchema(), identityRunFn()));

		harness.wire("id_a", "y", "id_b", "x");
		harness.wire("id_b", "y", "id_c", "x");

		b->bindSignal(harness.signalStore(), "gate");

		// run 1: signal=true (conducting), full chain
		harness.setSignal("gate", true);
		harness.declareOutput("t1", "id_c", "y");
		harness.feedInput("t1", "id_a", "x", makeFloatTensor(5.0f));
		harness.submit("t1", std::chrono::milliseconds(2000));
		CHECK(harness.awaitCompletion("t1"), "t1 should complete");
		CHECK(harness.hasOutput("t1", "id_c", "y"), "id_c should have output when signal=true");

		// run 2: signal=false (blocked), id_b and downstream don't run
		harness.setSignal("gate", false);
		harness.declareOutput("t2", "id_a", "y");
		harness.feedInput("t2", "id_a", "x", makeFloatTensor(5.0f));
		harness.submit("t2", std::chrono::milliseconds(2000));
		CHECK(harness.awaitCompletion("t2"), "t2 should complete (id_a output declared)");
		CHECK(harness.hasOutput("t2", "id_a", "y"), "id_a should have output");
		CHECK(!harness.hasOutput("t2", "id_c", "y"), "id_c should NOT have output when blocked");
	}
	END_TEST();
}

void testUnboundNodeNeverBlocked() {
	TEST("unbound node is never blocked") {
		TestHarness harness;

		auto* a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto* b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		harness.wire("id_a", "y", "id_b", "x");

		CHECK(!b->isBlocked(), "unbound node should not be blocked");

		harness.declareOutput("t1", "id_b", "y");
		harness.feedInput("t1", "id_a", "x", makeFloatTensor(42.0f));
		harness.submit("t1", std::chrono::milliseconds(2000));
		CHECK(harness.awaitCompletion("t1"), "t1 should complete");
		CHECK(harness.hasOutput("t1", "id_b", "y"), "id_b should have output");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// Task 级信号测试
// ════════════════════════════════════════════

void testTaskScopedSignalBlocksOnlyOneTask() {
	TEST("task-scoped signal: broadcast=false blocks all, task-scoped=true overrides for one task") {
		TestHarness harness;

		auto* a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto* b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		harness.wire("id_a", "y", "id_b", "x");

		b->bindSignal(harness.signalStore(), "gate");

		// 广播阻塞所有 task
		harness.setSignal("gate", false);

		// task1 单独覆盖：导通
		harness.setSignal("gate", "t1", true);

		// task1: 应该能跑通（task 级覆盖=true）
		harness.declareOutput("t1", "id_b", "y");
		harness.feedInput("t1", "id_a", "x", makeFloatTensor(10.0f));
		harness.submit("t1", std::chrono::milliseconds(2000));
		CHECK(harness.awaitCompletion("t1"), "t1 should complete (task-scoped signal=true overrides broadcast)");
		CHECK(harness.hasOutput("t1", "id_b", "y"), "id_b should have output for t1");

		// task2: 被广播阻塞（无 task 级覆盖）
		harness.clearErrors();
		harness.declareOutput("t2", "id_a", "y");
		harness.feedInput("t2", "id_a", "x", makeFloatTensor(20.0f));
		harness.submit("t2", std::chrono::milliseconds(2000));
		CHECK(harness.awaitCompletion("t2"), "t2 should complete (id_a output declared)");
		CHECK(!harness.hasOutput("t2", "id_b", "y"), "id_b should NOT have output for t2 (blocked by broadcast)");
	}
	END_TEST();
}

void testTaskScopedSignalBlocksOnlyTargetTask() {
	TEST("task-scoped signal: only target task blocked, other tasks proceed") {
		TestHarness harness;

		auto* a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto* b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		harness.wire("id_a", "y", "id_b", "x");

		b->bindSignal(harness.signalStore(), "gate");

		// 全局导通（无广播阻塞）
		harness.setSignal("gate", true);

		// 只阻塞 task2
		harness.setSignal("gate", "t2", false);

		// task1: 不受影响
		harness.declareOutput("t1", "id_b", "y");
		harness.feedInput("t1", "id_a", "x", makeFloatTensor(5.0f));
		harness.submit("t1", std::chrono::milliseconds(2000));
		CHECK(harness.awaitCompletion("t1"), "t1 should complete");
		CHECK(harness.hasOutput("t1", "id_b", "y"), "id_b should have output for t1");

		// task2: 被 task 级信号阻塞
		harness.clearErrors();
		harness.declareOutput("t2", "id_a", "y");
		harness.feedInput("t2", "id_a", "x", makeFloatTensor(30.0f));
		harness.submit("t2", std::chrono::milliseconds(2000));
		CHECK(harness.awaitCompletion("t2"), "t2 should complete (id_a output declared)");
		CHECK(!harness.hasOutput("t2", "id_b", "y"), "id_b should NOT have output for t2 (task-scoped block)");
	}
	END_TEST();
}

void testTaskSignalCleanupOnTerminate() {
	TEST("task signal cleanup: terminated task's signals don't leak to subsequent tasks") {
		TestHarness harness;

		auto* a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto* b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		harness.wire("id_a", "y", "id_b", "x");

		b->bindSignal(harness.signalStore(), "gate");

		// 全局导通
		harness.setSignal("gate", true);

		// task1: 设置 task 级阻塞
		harness.setSignal("gate", "t1", false);
		harness.declareOutput("t1", "id_a", "y");
		harness.feedInput("t1", "id_a", "x", makeFloatTensor(7.0f));
		harness.submit("t1", std::chrono::milliseconds(2000));
		CHECK(harness.awaitCompletion("t1"), "t1 should complete");
		// t1 被阻塞，id_b 无输出
		CHECK(!harness.hasOutput("t1", "id_b", "y"), "id_b should NOT have output for t1 (blocked)");

		// task2: 使用相同的 signal name "gate"，但不应受 t1 的 task 级信号影响
		harness.clearErrors();
		harness.declareOutput("t2", "id_b", "y");
		harness.feedInput("t2", "id_a", "x", makeFloatTensor(99.0f));
		harness.submit("t2", std::chrono::milliseconds(2000));
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

		auto* a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto* b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
		auto* c = harness.addNode(std::make_unique<Node>("Builtin", "id_c", identitySchema(), identityRunFn()));

		// 扇出：id_a → bc → [id_b, id_c]
		auto bcSchema = Connector::broadcastSchema(2);
		auto bcNode = std::make_unique<Node>("Connector.Broadcast", "bc", bcSchema,
			Connector::broadcastRunFn(), nullptr, ThreadPoolAffinity::System);
		bcNode->setConnector(true);
		harness.addNode(std::move(bcNode));
		harness.connect("id_a", "y", "bc", "in");
		harness.connect("bc", "out_0", "id_b", "x");
		harness.connect("bc", "out_1", "id_c", "x");

		// id_b 绑定信号
		b->bindSignal(harness.signalStore(), "enable_b");
		// 全局允许
		harness.setSignal("enable_b", true);

		// task1: task 级阻塞 id_b
		harness.setSignal("enable_b", "t1", false);

		harness.declareOutput("t1", "id_c", "y");
		harness.feedInput("t1", "id_a", "x", makeFloatTensor(50.0f));
		harness.submit("t1", std::chrono::milliseconds(2000));
		CHECK(harness.awaitCompletion("t1"), "t1 should complete");

		// id_c 应该有输出（未阻塞）
		CHECK(harness.hasOutput("t1", "id_c", "y"), "id_c should have output (not blocked)");
		auto r = harness.getOutputTensor("t1", "id_c", "y");
		CHECK(std::abs(r.item<float>() - 50.0f) < 1e-6f, "id_c value should be 50.0");

		// id_b 被 task 级信号阻塞
		CHECK(!harness.hasOutput("t1", "id_b", "y"), "id_b should NOT have output (task-scoped block)");
	}
	END_TEST();
}

int main() {
	try {
		testSimpleDataflow();
		testBuildGraph();
		testBroadcastConnectorInGraph();
		testRoutingConnectorInGraph();
		testConnectAll();
		testNodeQuery();
		testSerializationAccessors();

		// 循环测试
		testSimpleCycle();
		testCycleHopsExhaustion();
		testCycleMultiNode();

		// 信号 + 阻塞标志测试
		testBlockedNodeNotReceiving();
		testPartialBlockKeepsOtherPath();
		testDynamicSignalToggle();
		testUnboundNodeNeverBlocked();

		// Task 级信号测试
		testTaskScopedSignalBlocksOnlyOneTask();
		testTaskScopedSignalBlocksOnlyTargetTask();
		testTaskSignalCleanupOnTerminate();
		testTaskScopedSignalWithPartialBlock();

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
