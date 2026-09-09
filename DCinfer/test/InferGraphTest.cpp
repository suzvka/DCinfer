// InferGraph 拓扑连接与数据流 单元测试（异步场景化版本）
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
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
	TEST("build graph - addNode and connect") {
		TestHarness harness;

		auto& n1 = harness.addNode(std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));
		CHECK(harness.nodeCount() == 1, "nodeCount should be 1");

		auto& n2 = harness.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(), identityRunFn()));
		CHECK(harness.nodeCount() == 2, "nodeCount should be 2");

		// 重名应拒绝
		bool dupRejected = false;
		try {
			harness.addNode(std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));
		} catch (const GraphException&) {
			dupRejected = true;
		}
		CHECK(dupRejected, "duplicate name should be rejected");
		CHECK(harness.nodeCount() == 2, "nodeCount still 2");

		// 接线：两个业务节点之间 → connect() 自动插入导线连接器
		auto& w = harness.connect("add1", "s", "id1", "x");
		CHECK(w.isConnector(), "connect() should return the auto-inserted connector");
		CHECK(harness.nodeCount() == 3, "nodeCount should be 3 (add1, id1, __wire_0)");
		CHECK(harness.edgeCount() == 2, "edgeCount should be 2 (add1→connector, connector→id1)");

		// 无效接线：端口不存在
		bool badConnect = false;
		try {
			harness.connect("add1", "no_such", "id1", "x");
		} catch (const GraphException&) {
			badConnect = true;
		}
		CHECK(badConnect, "connect with bad src port should throw");

		// 业务节点直连应被 connectRaw() 拒绝
		bool directRejected = false;
		try {
			harness.connectRaw("add1", "s", "id1", "x");
		} catch (const GraphException&) {
			directRejected = true;
		}
		CHECK(directRejected, "direct connect between non-connectors should be rejected");
	}
	END_TEST();
}

void testSimpleDataflow() {
	TEST("simple 2-node dataflow: add → identity") {
		TestHarness harness;

		harness.addNode(std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(), identityRunFn()));
		harness.connect("add1", "s", "id1", "x");

		// 注入输入
		harness.feedInput("t1", "add1", "a", makeFloatTensor(3.0f));
		harness.feedInput("t1", "add1", "b", makeFloatTensor(4.0f));

		// 检查就绪：task 态已归 task 执行域，就绪性由 submit 后传播链验证
		// （未就绪则入口节点不会被调度，awaitCompletion 将失败）

		// 异步驱动执行
		harness.submit("t1", "id1", "y");
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
			std::make_unique<Node>("Connector.Broadcast", "bc", bcSchema, bcRunFn, ThreadPoolAffinity::System);
		bcNode->setConnector(true);
		harness.addNode(std::move(bcNode));

		harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		// 接线：add1 → bc → [id_a, id_b]
		harness.connectRaw("add1", "s", "bc", "in");
		harness.connectRaw("bc", "out_0", "id_a", "x");
		harness.connectRaw("bc", "out_1", "id_b", "x");

		// 注入
		harness.feedInput("t1", "add1", "a", makeFloatTensor(10.0f));
		harness.feedInput("t1", "add1", "b", makeFloatTensor(20.0f));

		// 声明两个下游输出
		harness.submit("t1", {{"id_a", "y"}, {"id_b", "y"}});
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

void testConnectAll() {
	TEST("connectAll auto-matches output ports to input ports") {
		TestHarness harness;

		auto bcSchema = Connector::broadcastSchema(2);
		auto bcRunFn = Connector::broadcastRunFn();
		auto bcNode =
			std::make_unique<Node>("Connector.Broadcast", "bc", bcSchema, bcRunFn, ThreadPoolAffinity::System);
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
		harness.submit("t1", "adder", "sum");
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
		harness.connect("test1", "y", "test2", "x");
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
		harness.connect("inc", "y", "inc", "x");

		// 注入初始值
		harness.feedInput("t1", "inc", "x", makeFloatTensor(0.0f));

		// 期望产出 3 次
		harness.submit("t1", "inc", "y", 3);
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
		harness.connect("inc", "y", "inc", "x");
		harness.feedInput("t1", "inc", "x", makeFloatTensor(0.0f));

		// 声明一个极大的 count，不可能在 5 跳内完成
		harness.submit("t1", "inc", "y", 100, std::chrono::milliseconds(0), 5);

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

		harness.connect("A", "y", "B", "x");
		harness.connect("B", "y", "C", "x");
		harness.connect("C", "y", "A", "x");

		harness.feedInput("t1", "A", "x", makeFloatTensor(0.0f));

		// 每圈 3 节点 + 3 导线 = 6 跳，3 圈 = 18 跳 → TTL=19 刚好完成
		harness.submit("t1", "C", "y", 3, std::chrono::milliseconds(0), 19);

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

		auto& a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto& b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		harness.connect("id_a", "y", "id_b", "x");

		b.bindSignal(harness.signalStore(), "enable_b");
		harness.setSignal("enable_b", false);

		harness.feedInput("t1", "id_a", "x", makeFloatTensor(10.0f));

		harness.submit("t1", "id_a", "y", 1, std::chrono::milliseconds(2000));
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

		// 使用广播连接器扇出（避免 connect 同端口 takeOutput 抢消费）
		auto bcSchema = Connector::broadcastSchema(2);
		auto bcNode = std::make_unique<Node>("Connector.Broadcast", "bc", bcSchema,
			Connector::broadcastRunFn(), ThreadPoolAffinity::System);
		bcNode->setConnector(true);
		harness.addNode(std::move(bcNode));
		harness.connectRaw("id_a", "y", "bc", "in");
		harness.connectRaw("bc", "out_0", "id_b", "x");
		harness.connectRaw("bc", "out_1", "id_c", "x");

		b.bindSignal(harness.signalStore(), "enable_b");
		harness.setSignal("enable_b", false);

		harness.feedInput("t1", "id_a", "x", makeFloatTensor(10.0f));

		harness.submit("t1", "id_c", "y", 1, std::chrono::milliseconds(2000));
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

		// run 1: signal=true (conducting), full chain
		harness.setSignal("gate", true);
		harness.feedInput("t1", "id_a", "x", makeFloatTensor(5.0f));
		harness.submit("t1", "id_c", "y", 1, std::chrono::milliseconds(2000));
		CHECK(harness.awaitCompletion("t1"), "t1 should complete");
		CHECK(harness.hasOutput("t1", "id_c", "y"), "id_c should have output when signal=true");

		// run 2: signal=false (blocked), id_b and downstream don't run
		harness.setSignal("gate", false);
		harness.feedInput("t2", "id_a", "x", makeFloatTensor(5.0f));
		harness.submit("t2", "id_a", "y", 1, std::chrono::milliseconds(2000));
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
		harness.submit("t1", "id_b", "y", 1, std::chrono::milliseconds(2000));
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

		auto& a = harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		auto& b = harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		harness.connect("id_a", "y", "id_b", "x");

		b.bindSignal(harness.signalStore(), "gate");

		// 广播阻塞所有 task
		harness.setSignal("gate", false);

		// task1 单独覆盖：导通
		harness.setSignal("gate", "t1", true);

		// task1: 应该能跑通（task 级覆盖=true）
		harness.feedInput("t1", "id_a", "x", makeFloatTensor(10.0f));
		harness.submit("t1", "id_b", "y", 1, std::chrono::milliseconds(2000));
		CHECK(harness.awaitCompletion("t1"), "t1 should complete (task-scoped signal=true overrides broadcast)");
		CHECK(harness.hasOutput("t1", "id_b", "y"), "id_b should have output for t1");

		// task2: 被广播阻塞（无 task 级覆盖）
		harness.clearErrors();
		harness.feedInput("t2", "id_a", "x", makeFloatTensor(20.0f));
		harness.submit("t2", "id_a", "y", 1, std::chrono::milliseconds(2000));
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

		// 全局导通（无广播阻塞）
		harness.setSignal("gate", true);

		// 只阻塞 task2
		harness.setSignal("gate", "t2", false);

		// task1: 不受影响
		harness.feedInput("t1", "id_a", "x", makeFloatTensor(5.0f));
		harness.submit("t1", "id_b", "y", 1, std::chrono::milliseconds(2000));
		CHECK(harness.awaitCompletion("t1"), "t1 should complete");
		CHECK(harness.hasOutput("t1", "id_b", "y"), "id_b should have output for t1");

		// task2: 被 task 级信号阻塞
		harness.clearErrors();
		harness.feedInput("t2", "id_a", "x", makeFloatTensor(30.0f));
		harness.submit("t2", "id_a", "y", 1, std::chrono::milliseconds(2000));
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

		// 全局导通
		harness.setSignal("gate", true);

		// task1: 设置 task 级阻塞
		harness.setSignal("gate", "t1", false);
		harness.feedInput("t1", "id_a", "x", makeFloatTensor(7.0f));
		harness.submit("t1", "id_a", "y", 1, std::chrono::milliseconds(2000));
		CHECK(harness.awaitCompletion("t1"), "t1 should complete");
		// t1 被阻塞，id_b 无输出
		CHECK(!harness.hasOutput("t1", "id_b", "y"), "id_b should NOT have output for t1 (blocked)");

		// task2: 使用相同的 signal name "gate"，但不应受 t1 的 task 级信号影响
		harness.clearErrors();
		harness.feedInput("t2", "id_a", "x", makeFloatTensor(99.0f));
		harness.submit("t2", "id_b", "y", 1, std::chrono::milliseconds(2000));
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

		// 扇出：id_a → bc → [id_b, id_c]
		auto bcSchema = Connector::broadcastSchema(2);
		auto bcNode = std::make_unique<Node>("Connector.Broadcast", "bc", bcSchema,
			Connector::broadcastRunFn(), ThreadPoolAffinity::System);
		bcNode->setConnector(true);
		harness.addNode(std::move(bcNode));
		harness.connectRaw("id_a", "y", "bc", "in");
		harness.connectRaw("bc", "out_0", "id_b", "x");
		harness.connectRaw("bc", "out_1", "id_c", "x");

		// id_b 绑定信号
		b.bindSignal(harness.signalStore(), "enable_b");
		// 全局允许
		harness.setSignal("enable_b", true);

		// task1: task 级阻塞 id_b
		harness.setSignal("enable_b", "t1", false);

		harness.feedInput("t1", "id_a", "x", makeFloatTensor(50.0f));
		harness.submit("t1", "id_c", "y", 1, std::chrono::milliseconds(2000));
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

// ── 看门狗超时（回归）──
// 曾因看门狗线程在 _terminate 中 erase 自身 jthread（自 join → noexcept
// 析构内抛 resource_deadlock_would_occur）触发 std::terminate 使进程崩溃。
// 信号阻塞使输出声明永远无法满足，强制走看门狗超时路径。
void testWatchdogTimeoutTerminates() {
	TEST("watchdog timeout: blocked task terminated, no process crash") {
		TestHarness harness;

		harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		harness.connect("id_a", "y", "id_b", "x");

		// id_b 被信号阻塞 → 声明永远无法满足 → 看门狗超时必然触发
		harness.node("id_b")->bindSignal(harness.signalStore(), "enable_b");
		harness.setSignal("enable_b", false);

		harness.feedInput("t1", "id_a", "x", makeFloatTensor(10.0f));

		harness.submit("t1", "id_b", "y", 1, std::chrono::milliseconds(200));
		CHECK(harness.awaitCompletion("t1", std::chrono::milliseconds(3000)),
			  "watchdog should terminate the task and notify waiters");

		// 看门狗应记录超时错误（recordError 的第二个参数是 nodeName）
		auto errors = harness.taskErrors("t1");
		bool hasWatchdogError = false;
		for (auto& e : errors) {
			if (e.nodeName == "<watchdog>" || e.message.find("task timed out") != std::string::npos) {
				hasWatchdogError = true;
				break;
			}
		}
		CHECK(hasWatchdogError, "watchdog timeout error should be recorded");

		// 被阻塞节点不应产出
		CHECK(!harness.hasOutput("t1", "id_b", "y"), "blocked node should not produce output");

		// 看门狗终止的任务应处于 TimedOut 状态
		CHECK(harness.graph().taskStatus("t1") == TaskStatus::TimedOut,
			  "watchdog-terminated task should be TimedOut");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 子图（分组互斥）测试
// ════════════════════════════════════════════

// 用于检测并发执行的共享状态
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

// 带延迟的算子（模拟耗时推理），执行时记录并发度
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

void testSubgraphSerializesExecution() {
	TEST("subgraph serializes execution: two independent nodes never run concurrently") {
		TestHarness harness;

		ConcurrencyDetector detector;

		// 两个独立节点（无数据依赖），相同 affinity
		harness.addNode(std::make_unique<Node>("Builtin", "sg_a", identitySchema(),
			delayedRunFn(&detector, 30), ThreadPoolAffinity::Operator));
		harness.addNode(std::make_unique<Node>("Builtin", "sg_b", identitySchema(),
			delayedRunFn(&detector, 30), ThreadPoolAffinity::Operator));

		// 声明子图：两个节点互斥
		harness.graph().declareSubgraph("sg", {"sg_a", "sg_b"});

		// 同时注入两个 task
		harness.feedInput("t1", "sg_a", "x", makeFloatTensor(1.0f));
		harness.feedInput("t2", "sg_b", "x", makeFloatTensor(2.0f));

		// 同时提交
		harness.submit("t1", "sg_a", "y", 1, std::chrono::milliseconds(3000));
		harness.submit("t2", "sg_b", "y", 1, std::chrono::milliseconds(3000));

		CHECK(harness.awaitCompletion("t1"), "t1 should complete");
		CHECK(harness.awaitCompletion("t2"), "t2 should complete");

		// 核心断言：最大并发度不超过 1
		CHECK(detector.maxConcurrent.load() <= 1,
			  "subgraph nodes must not run concurrently (maxConcurrent="
				  + std::to_string(detector.maxConcurrent.load()) + ")");
	}
	END_TEST();
}

void testSubgraphDoesNotAffectOthers() {
	TEST("subgraph does not affect nodes outside the group") {
		TestHarness harness;

		ConcurrencyDetector detector;

		// 子图内节点
		harness.addNode(std::make_unique<Node>("Builtin", "sg_a", identitySchema(),
			delayedRunFn(&detector, 30), ThreadPoolAffinity::Operator));
		// 子图外节点（同 affinity，但不在子图中）
		harness.addNode(std::make_unique<Node>("Builtin", "free", identitySchema(),
			delayedRunFn(&detector, 30), ThreadPoolAffinity::Operator));

		// 只把 sg_a 放入子图（单节点子图，不影响 free）
		harness.graph().declareSubgraph("sg", {"sg_a"});

		harness.feedInput("t1", "sg_a", "x", makeFloatTensor(1.0f));
		harness.feedInput("t2", "free", "x", makeFloatTensor(2.0f));

		harness.submit("t1", "sg_a", "y", 1, std::chrono::milliseconds(3000));
		harness.submit("t2", "free", "y", 1, std::chrono::milliseconds(3000));

		CHECK(harness.awaitCompletion("t1"), "t1 should complete");
		CHECK(harness.awaitCompletion("t2"), "t2 should complete");

		// 两个节点应该都有输出（自由节点不受子图约束）
		CHECK(harness.hasOutput("t1", "sg_a", "y"), "sg_a should have output");
		CHECK(harness.hasOutput("t2", "free", "y"), "free should have output");
	}
	END_TEST();
}

void testSubgraphAllowsMixedAffinity() {
	TEST("subgraph allows mixed affinity: cross-pool nodes serialize via shared group semaphore") {
		TestHarness harness;

		ConcurrencyDetector detector;

		// 两个独立节点（无数据依赖），分属不同线程池（Compute / Operator）
		harness.addNode(std::make_unique<Node>("Builtin", "sg_a", identitySchema(),
			delayedRunFn(&detector, 30), ThreadPoolAffinity::Compute));
		harness.addNode(std::make_unique<Node>("Builtin", "sg_b", identitySchema(),
			delayedRunFn(&detector, 30), ThreadPoolAffinity::Operator));

		// 混合 affinity 子图：不再抛异常，跨池共享信号量实现全局互斥
		harness.graph().declareSubgraph("sg", {"sg_a", "sg_b"});

		// 同时注入两个 task（不同池本可并行；互斥后必须串行）
		harness.feedInput("t1", "sg_a", "x", makeFloatTensor(1.0f));
		harness.feedInput("t2", "sg_b", "x", makeFloatTensor(2.0f));

		harness.submit("t1", "sg_a", "y", 1, std::chrono::milliseconds(3000));
		harness.submit("t2", "sg_b", "y", 1, std::chrono::milliseconds(3000));

		CHECK(harness.awaitCompletion("t1"), "t1 should complete");
		CHECK(harness.awaitCompletion("t2"), "t2 should complete");

		// 核心断言：跨池互斥生效，最大并发度不超过 1
		CHECK(detector.maxConcurrent.load() <= 1,
			  "cross-pool subgraph nodes must not run concurrently (maxConcurrent="
				  + std::to_string(detector.maxConcurrent.load()) + ")");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// GraphNode 状态代理：声明通路检测
// ════════════════════════════════════════════

void testGraphNodeBranchBlocking() {
	TEST("GraphNode: branch blocked, alternate path satisfies declaration -> not blocked") {
		// 内部图：入口 id_in → Broadcast(2) → {idA(信号阻塞), idB}，声明 idB.y
		InferGraph sub;
		sub.addNode(std::make_unique<Node>("Builtin", "id_in", identitySchema(), identityRunFn(),
			ThreadPoolAffinity::Operator));
		auto bcNode = std::make_unique<Node>("Connector.Broadcast", "bc",
			Connector::broadcastSchema(2), Connector::broadcastRunFn(), ThreadPoolAffinity::System);
		bcNode->setConnector(true);
		sub.addNode(std::move(bcNode));
		sub.addNode(std::make_unique<Node>("Builtin", "idA", identitySchema(), identityRunFn(),
			ThreadPoolAffinity::Operator));
		sub.addNode(std::make_unique<Node>("Builtin", "idB", identitySchema(), identityRunFn(),
			ThreadPoolAffinity::Operator));

		sub.connectRaw("id_in", "y", "bc", "in");
		sub.connectRaw("bc", "out_0", "idA", "x");
		sub.connectRaw("bc", "out_1", "idB", "x");

		sub.bindInput("id_in", "x");
		sub.bindOutput("idB", "y");

		// idA 绑定信号并阻塞；idB 正常（旁路存在 → 子图不阻塞）
		sub.node("idA")->bindSignal(sub.signalStore(), "enableA");
		sub.setSignal("enableA", false);

		auto gn = sub.exportNode("gn");
		CHECK(!gn->isBlocked("t1"), "GraphNode should NOT be blocked when alternate path exists");

		// 集成：父图 src → GraphNode，task 应经旁路正常完成
		TestHarness parent;
		parent.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRunFn()));
		parent.addNode(std::move(gn));
		parent.connect("src", "y", "gn", "x");

		parent.feedInput("t1", "src", "x", makeFloatTensor(42.0f));
		parent.submit("t1", "gn", "y", 1, std::chrono::milliseconds(3000));
		CHECK(parent.awaitCompletion("t1"), "t1 should complete via alternate path");
		CHECK(parent.hasOutput("t1", "gn", "y"), "gn should have output");
		auto r = parent.getOutputTensor("t1", "gn", "y");
		CHECK(std::abs(r.item<float>() - 42.0f) < 1e-6f, "value should be 42.0");
	}
	END_TEST();

	TEST("GraphNode: sole path blocked -> blocked; recovery after signal restore") {
		InferGraph sub;
		sub.addNode(std::make_unique<Node>("Builtin", "id_in", identitySchema(), identityRunFn(),
			ThreadPoolAffinity::Operator));
		auto bcNode = std::make_unique<Node>("Connector.Broadcast", "bc",
			Connector::broadcastSchema(2), Connector::broadcastRunFn(), ThreadPoolAffinity::System);
		bcNode->setConnector(true);
		sub.addNode(std::move(bcNode));
		sub.addNode(std::make_unique<Node>("Builtin", "idA", identitySchema(), identityRunFn(),
			ThreadPoolAffinity::Operator));
		sub.addNode(std::make_unique<Node>("Builtin", "idB", identitySchema(), identityRunFn(),
			ThreadPoolAffinity::Operator));

		sub.connectRaw("id_in", "y", "bc", "in");
		sub.connectRaw("bc", "out_0", "idA", "x");
		sub.connectRaw("bc", "out_1", "idB", "x");

		sub.bindInput("id_in", "x");
		sub.bindOutput("idB", "y");

		// 两条分支全部阻塞（唯一通路切断）→ 子图边界应答阻塞
		sub.node("idA")->bindSignal(sub.signalStore(), "enableA");
		sub.node("idB")->bindSignal(sub.signalStore(), "enableB");
		sub.setSignal("enableA", false);
		sub.setSignal("enableB", false);

		auto gn = sub.exportNode("gn");
		CHECK(gn->isBlocked("t1"), "GraphNode should be blocked when no path to declaration exists");

		// 集成：父级传播跳过边，task 不完成
		TestHarness parent;
		parent.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRunFn()));
		parent.addNode(std::move(gn));
		parent.connect("src", "y", "gn", "x");

		parent.feedInput("t1", "src", "x", makeFloatTensor(7.0f));
		parent.submit("t1", "gn", "y", 1); // 无看门狗：阻塞期间 task 保持挂起
		CHECK(!parent.awaitCompletion("t1", std::chrono::milliseconds(600)),
			  "should NOT complete while path is blocked");
		CHECK(!parent.hasOutput("t1", "gn", "y"), "blocked task should not produce output");

		// 恢复信号：通路口径重新导通
		sub.setSignal("enableB", true);
		CHECK(!std::as_const(parent).node("gn")->isBlocked("t1"), "GraphNode should unblock after signal restore");

		// 新 task：恢复后正常完成
		parent.feedInput("t2", "src", "x", makeFloatTensor(7.0f));
		parent.submit("t2", "gn", "y", 1, std::chrono::milliseconds(3000));
		CHECK(parent.awaitCompletion("t2"), "t2 should complete after signal restore");
		CHECK(parent.hasOutput("t2", "gn", "y"), "gn should have output for t2");
		auto r = parent.getOutputTensor("t2", "gn", "y");
		CHECK(std::abs(r.item<float>() - 7.0f) < 1e-6f, "value should be 7.0");
	}
	END_TEST();

	TEST("GraphNode: declared node itself blocked -> blocked") {
		InferGraph sub;
		sub.addNode(std::make_unique<Node>("Builtin", "idC", identitySchema(), identityRunFn(),
			ThreadPoolAffinity::Operator));
		sub.bindInput("idC", "x");
		sub.bindOutput("idC", "y");
		sub.node("idC")->bindSignal(sub.signalStore(), "enableC");
		sub.setSignal("enableC", false);

		auto gn = sub.exportNode("gn");
		CHECK(gn->isBlocked("t1"), "declared node blocked -> GraphNode blocked");

		sub.setSignal("enableC", true);
		CHECK(!gn->isBlocked("t1"), "signal restore -> GraphNode not blocked");
	}
	END_TEST();
}

void testSubgraphDataflowCorrect() {
	TEST("subgraph serial chain: data propagates correctly through A→B→C") {
		TestHarness harness;

		// 串行链：A → B → C（全部 Operator affinity）
		harness.addNode(std::make_unique<Node>("Builtin", "A", incSchema(), incRunFn(),
			ThreadPoolAffinity::Operator));
		harness.addNode(std::make_unique<Node>("Builtin", "B", incSchema(), incRunFn(),
			ThreadPoolAffinity::Operator));
		harness.addNode(std::make_unique<Node>("Builtin", "C", incSchema(), incRunFn(),
			ThreadPoolAffinity::Operator));

		harness.connect("A", "y", "B", "x");
		harness.connect("B", "y", "C", "x");

		// 声明子图
		harness.graph().declareSubgraph("chain", {"A", "B", "C"});

		// 注入初始值 0 → +1 → +1 → +1 = 3
		harness.feedInput("t1", "A", "x", makeFloatTensor(0.0f));
		harness.submit("t1", "C", "y", 1, std::chrono::milliseconds(3000));
		CHECK(harness.awaitCompletion("t1"), "t1 should complete");

		CHECK(harness.hasOutput("t1", "C", "y"), "C should have output");
		auto result = harness.getOutputTensor("t1", "C", "y");
		CHECK(std::abs(result.item<float>() - 3.0f) < 1e-6f, "result should be 3.0 (0+1+1+1)");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 任务生命周期：结果保留 / taskId 复用 / 状态机 / 取消
// ════════════════════════════════════════════

// 回归：终止流程曾先清理 OutputZone 与节点缓冲、最后才 notify wait()，
// 导致 submit → wait → takeOutput 取不到结果（只能在回调内读）。
void testOutputSurvivesWait() {
	TEST("lifecycle: outputs retrievable after wait without callback") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(), identityRunFn()));

		graph.feedInput("t1", "id1", "x", makeFloatTensor(7.0f));
		graph.submit("t1", "id1", "y");

		CHECK(graph.wait("t1"), "task should complete");
		CHECK(graph.taskStatus("t1") == TaskStatus::Succeeded, "status should be Succeeded");

		// 核心断言：wait 返回后无需回调即可取结果
		CHECK(graph.hasOutput("t1", "id1", "y"), "output should survive wait()");
		auto result = graph.takeOutputTensor("t1", "id1", "y");
		CHECK(std::abs(result.item<float>() - 7.0f) < 1e-6f, "result should be 7.0");

		// releaseTask 回收：状态与结果一并释放
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

		// 从未提交的 task：超时后返回 Unknown（区别于 Running）
		auto unknown = graph.waitForResult("never_submitted", std::chrono::milliseconds(50));
		CHECK(unknown.status == TaskStatus::Unknown, "unsubmitted task should be Unknown");
	}
	END_TEST();
}

// 回归：_terminatedTasks 只增不减，复用 ID 时 wait 立即返回、
// 传播被拦截、回调不触发、集合无限增长。
void testTaskIdReuseAfterCompletion() {
	TEST("lifecycle: completed taskId can be safely reused") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));

		// 第一轮
		graph.feedInput("t1", "add1", "a", makeFloatTensor(3.0f));
		graph.feedInput("t1", "add1", "b", makeFloatTensor(4.0f));
		graph.submit("t1", "add1", "s");
		CHECK(graph.wait("t1"), "first run should complete");
		auto r1 = graph.takeOutputTensor("t1", "add1", "s");
		CHECK(std::abs(r1.item<float>() - 7.0f) < 1e-6f, "first result should be 7.0");

		// 复用同一 taskId：重新注入并提交
		graph.feedInput("t1", "add1", "a", makeFloatTensor(10.0f));
		graph.feedInput("t1", "add1", "b", makeFloatTensor(20.0f));
		graph.submit("t1", "add1", "s");
		CHECK(graph.wait("t1"), "reused taskId should complete normally");
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
				delayedRunFn(nullptr, 200), ThreadPoolAffinity::Operator));

		graph.feedInput("t1", "slow", "x", makeFloatTensor(1.0f));
		graph.submit("t1", "slow", "y");

		bool rejected = false;
		try {
			graph.submit("t1", "slow", "y");
		} catch (const GraphException& e) {
			rejected = (e.getErrorType() == GraphException::ErrorType::DuplicateTask);
		}
		CHECK(rejected, "duplicate submit while running should throw DuplicateTask");

		CHECK(graph.wait("t1"), "original task should still complete");
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
		graph.setSignal("gate", false); // id_b 永久阻塞，声明无法满足

		graph.feedInput("t1", "id_a", "x", makeFloatTensor(1.0f));
		graph.submit("t1", "id_b", "y"); // 无看门狗：任务将一直挂起

		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		CHECK(graph.taskStatus("t1") == TaskStatus::Running, "task should be running while blocked");

		CHECK(graph.cancel("t1"), "cancel should succeed on active task");
		CHECK(graph.wait("t1"), "wait should wake up after cancel");
		CHECK(graph.taskStatus("t1") == TaskStatus::Cancelled, "status should be Cancelled");
		CHECK(!graph.cancel("t1"), "cancel on terminated task should return false (idempotent)");

		auto result = graph.waitForResult("t1");
		CHECK(result.status == TaskStatus::Cancelled, "waitForResult should report Cancelled");

		// 复用已取消的 taskId：解除信号后可重新执行
		graph.setSignal("gate", true);
		graph.feedInput("t1", "id_a", "x", makeFloatTensor(9.0f));
		graph.submit("t1", "id_b", "y");
		CHECK(graph.wait("t1"), "re-submitted task after cancel should complete");
		auto r = graph.takeOutputTensor("t1", "id_b", "y");
		CHECK(std::abs(r.item<float>() - 9.0f) < 1e-6f, "re-run result should be 9.0");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 图 API 便捷绑定：feedBoundInput / submitBound
// ════════════════════════════════════════════

void testBoundInputOutputApi() {
	TEST("graph API: feedBoundInput / submitBound follow bindInput/bindOutput") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "inc", incSchema(), incRunFn()));

		graph.bindInput("inc", "x");
		graph.bindOutput("inc", "y");

		auto in = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
		*in = 41.0f;
		graph.feedBoundInput("t1", "x", std::move(*in));
		graph.submitBound("t1");
		CHECK(graph.wait("t1"), "bound flow should complete");
		auto out = graph.takeOutputTensor("t1", "inc", "y");
		CHECK(std::abs(out.item<float>() - 42.0f) < 1e-6f, "bound result should be 42.0");

		// 错误路径：未绑定的端口名
		bool noSuch = false;
		try {
			auto v = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
			*v = 1.0f;
			graph.feedBoundInput("t1", "no_such_port", std::move(*v));
		} catch (const GraphException&) {
			noSuch = true;
		}
		CHECK(noSuch, "unbound port name should throw");

		// 错误路径：绑定名跨节点歧义
		InferGraph graph2;
		graph2.addNode(std::make_unique<Node>("Builtin", "a", incSchema(), incRunFn()));
		graph2.addNode(std::make_unique<Node>("Builtin", "b", incSchema(), incRunFn()));
		graph2.bindInput("a", "x");
		graph2.bindInput("b", "x");
		bool ambiguous = false;
		try {
			auto v = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
			*v = 1.0f;
			graph2.feedBoundInput("t1", "x", std::move(*v));
		} catch (const GraphException&) {
			ambiguous = true;
		}
		CHECK(ambiguous, "ambiguous bound name should throw");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 图级公共别名：bindInput/bindOutput 三参重载 + 按别名注入/取用
// ════════════════════════════════════════════

void testAliasBindingApi() {
	TEST("graph API: public aliases decouple callers from internal topology") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "inc", incSchema(), incRunFn()));

		// 输入/输出均带公共别名
		graph.bindInput("num", "inc", "x");
		graph.bindOutput("result", "inc", "y");

		auto in = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
		*in = 5.0f;
		graph.feedBoundInput("t1", "num", std::move(*in)); // 按别名注入
		graph.submitBound("t1");
		CHECK(graph.wait("t1"), "alias-bound flow should complete");
		CHECK(graph.hasOutput("t1", "result"), "alias should resolve for hasOutput");
		auto out = graph.takeOutputTensor("t1", "result"); // 按别名取，无需内部节点名
		CHECK(std::abs(out.item<float>() - 6.0f) < 1e-6f, "alias result should be 6.0");

		// 2 参重载也接受唯一绑定端口名（向后兼容）
		auto in2 = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
		*in2 = 7.0f;
		graph.feedBoundInput("t2", "num", std::move(*in2));
		graph.submitBound("t2");
		CHECK(graph.wait("t2"), "second task should complete");
		auto out2 = graph.takeOutputTensor("t2", "y"); // 按唯一绑定端口名取
		CHECK(std::abs(out2.item<float>() - 8.0f) < 1e-6f, "port-name retrieval should be 8.0");

		// 跨节点同名端口：唯一别名消除注入歧义（同名端口仍歧义，但别名不歧义）
		InferGraph graph2;
		graph2.addNode(std::make_unique<Node>("Builtin", "a", incSchema(), incRunFn()));
		graph2.addNode(std::make_unique<Node>("Builtin", "b", incSchema(), incRunFn()));
		graph2.bindInput("first", "a", "x");
		graph2.bindInput("second", "b", "x");

		// 别名唯一性校验：输入别名重复（构建期 API：需在首次运行期调用前完成，
		// feedBoundInput 触发惰性冻结后构建面关闭，另见 FreezeBoundaryTest）
		bool dupIn = false;
		try {
			graph2.bindInput("first", "b", "x");
		} catch (const GraphException& e) {
			dupIn = (e.getErrorType() == GraphException::ErrorType::DuplicateBinding);
		}
		CHECK(dupIn, "duplicate input alias should throw DuplicateBinding");

		// 别名唯一性校验：输出别名重复（输出别名与输入别名是独立命名空间）
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
			graph2.feedBoundInput("t1", "first", std::move(*v));
		}
		{
			auto v = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
			*v = 2.0f;
			graph2.feedBoundInput("t1", "second", std::move(*v));
		}
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 消费式取用语义：takeOutput 取出即消耗，不可重复读取
// ════════════════════════════════════════════

void testTakeOutputDestructive() {
	TEST("graph API: takeOutput is consumptive (single read)") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(), identityRunFn()));

		graph.feedInput("t1", "id1", "x", makeFloatTensor(9.0f));
		graph.submit("t1", "id1", "y");
		CHECK(graph.wait("t1"), "task should complete");

		CHECK(graph.hasOutput("t1", "id1", "y"), "output should exist before take");
		auto first = graph.takeOutputTensor("t1", "id1", "y");
		CHECK(std::abs(first.item<float>() - 9.0f) < 1e-6f, "first take should be 9.0");

		// 取出即消耗：OutputZone 已清空且节点缓冲已随终止清理 → 再次取出抛异常
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

// ════════════════════════════════════════════
// wait/waitForResult 超时语义：默认无限等待；显式超时只放弃等待不取消
// ════════════════════════════════════════════

void testWaitSemantics() {
	TEST("lifecycle: wait semantics (0=infinite, waiter-only timeout, unknown-id fast-fail)") {
		InferGraph graph;
		auto& b = graph.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
		graph.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		graph.connect("id_a", "y", "id_b", "x");

		b.bindSignal(graph.signalStore(), "gate");
		graph.setSignal("gate", false); // id_b 阻塞，任务保持 Running

		graph.feedInput("t1", "id_a", "x", makeFloatTensor(1.0f));
		graph.submit("t1", "id_b", "y");

		// 显式超时：只放弃等待，不取消任务（任务仍 Running）
		CHECK(!graph.wait("t1", std::chrono::milliseconds(80)), "explicit timeout should return false while blocked");
		auto running = graph.waitForResult("t1", std::chrono::milliseconds(80));
		CHECK(running.status == TaskStatus::Running, "waitForResult timeout should report Running");

		// 未知 taskId：立即返回（无限等待模式下防误拼写挂死）
		CHECK(!graph.wait("never_submitted"), "unknown taskId should return false immediately");
		auto unknown = graph.waitForResult("never_submitted");
		CHECK(unknown.status == TaskStatus::Unknown, "unknown taskId waitForResult should be Unknown");

		// 默认无限等待：cancel 唤醒后返回（配合 testCancelRunningTask 的 wait 覆盖）
		CHECK(graph.cancel("t1"), "cancel should succeed on active task");
		auto cancelled = graph.waitForResult("t1");
		CHECK(cancelled.status == TaskStatus::Cancelled, "default waitForResult should wait until termination");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 共享 Timer 超时（回归）：deadline 精确触发 / 旧条目不误杀复用任务 / cancel 竞态
// ════════════════════════════════════════════

// 超时触发不得早于请求的 deadline（共享 Timer 精确唤醒语义）
void testTimeoutLowerBound() {
	TEST("shared timer: timeout fires no earlier than the requested deadline") {
		TestHarness harness;

		harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), delayedRunFn(nullptr, 500)));
		harness.connect("id_a", "y", "id_b", "x");

		harness.feedInput("t1", "id_a", "x", makeFloatTensor(10.0f));

		auto start = std::chrono::steady_clock::now();
		harness.submit("t1", "id_b", "y", 1, std::chrono::milliseconds(150));
		CHECK(harness.awaitCompletion("t1", std::chrono::milliseconds(3000)),
			  "watchdog should terminate the task");
		auto elapsed = std::chrono::steady_clock::now() - start;

		CHECK(harness.graph().taskStatus("t1") == TaskStatus::TimedOut, "task should be TimedOut");
		CHECK(elapsed >= std::chrono::milliseconds(150),
			  "timeout must not fire before the requested deadline");
	}
	END_TEST();
}

// 同 ID 复用：上一轮超时条目到点后必须失配退出，不得终止新一轮任务
void testTimeoutThenTaskIdReuse() {
	TEST("shared timer: stale timeout entry must not kill a resubmitted task") {
		TestHarness harness;

		harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
		harness.connect("id_a", "y", "id_b", "x");

		// 第 1 轮：id_b 信号阻塞 → 声明无法满足 → 必然超时
		harness.node("id_b")->bindSignal(harness.signalStore(), "enable_b");
		harness.setSignal("enable_b", false);
		harness.feedInput("t1", "id_a", "x", makeFloatTensor(1.0f));
		harness.submit("t1", "id_b", "y", 1, std::chrono::milliseconds(120));
		CHECK(harness.awaitCompletion("t1", std::chrono::milliseconds(3000)),
			  "round 1 should be terminated by timeout");
		CHECK(harness.graph().taskStatus("t1") == TaskStatus::TimedOut,
			  "round 1 should end as TimedOut");

		// 第 2~5 轮：同 ID 复用、解除阻塞、不限时提交——
		// 旧超时条目到点时经活动门控身份校验失配退出，不得误杀新任务
		for (int i = 2; i <= 5; ++i) {
			harness.setSignal("enable_b", true);
			harness.feedInput("t1", "id_a", "x", makeFloatTensor(static_cast<float>(i)));
			harness.submit("t1", "id_b", "y");
			CHECK(harness.awaitCompletion("t1", std::chrono::milliseconds(3000)),
				  "resubmitted task should complete normally");
			CHECK(harness.graph().taskStatus("t1") == TaskStatus::Succeeded,
				  "resubmitted task must not be killed by the stale timeout entry");
		}

		// 产出值校验：TestHarness 输出缓存为一次性消费式读取（取出后不重捕），
		// 缓存条目在首轮复用（第 2 轮）就位后保持不变，故在此统一校验数值
		auto r = harness.getOutputTensor("t1", "id_b", "y");
		CHECK(std::abs(r.item<float>() - 2.0f) < 1e-6f,
			  "first reused task result should be correct");
	}
	END_TEST();
}

// cancel 与超时竞争终止权：终态二选一，无崩溃、无双重终止
void testCancelVsTimeoutRace() {
	TEST("shared timer: cancel racing timeout yields exactly one terminal state") {
		for (int round = 0; round < 10; ++round) {
			TestHarness harness;

			harness.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
			harness.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));
			harness.connect("id_a", "y", "id_b", "x");

			harness.node("id_b")->bindSignal(harness.signalStore(), "enable_b");
			harness.setSignal("enable_b", false);

			// 30ms 处请求取消，与 60ms 超时竞争；cancel 对未提交/已终止任务幂等
			std::thread canceller([&harness] {
				std::this_thread::sleep_for(std::chrono::milliseconds(30));
				harness.graph().cancel("t1");
			});

			harness.feedInput("t1", "id_a", "x", makeFloatTensor(1.0f));
			harness.submit("t1", "id_b", "y", 1, std::chrono::milliseconds(60));
			canceller.join();

			auto st = harness.graph().taskStatus("t1");
			CHECK(st == TaskStatus::Cancelled || st == TaskStatus::TimedOut,
				  "final status must be exactly one of Cancelled/TimedOut");
		}
	}
	END_TEST();
}

int main() {
	try {
		testSimpleDataflow();
		testBuildGraph();
		testBroadcastConnectorInGraph();
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

		// 看门狗超时（回归：曾因看门狗线程自 join 触发 std::terminate）
		testWatchdogTimeoutTerminates();

		// 任务生命周期（结果保留 / 复用 / 状态机 / 取消）
		testOutputSurvivesWait();
		testWaitForResult();
		testTaskIdReuseAfterCompletion();
		testDuplicateActiveSubmitRejected();
		testCancelRunningTask();

		// 图 API 便捷绑定
		testBoundInputOutputApi();
		testAliasBindingApi();
		testTakeOutputDestructive();

		// wait/waitForResult 超时语义
		testWaitSemantics();

		// 共享 Timer 超时（deadline 精确触发 / 复用隔离 / cancel 竞态）
		testTimeoutLowerBound();
		testTimeoutThenTaskIdReuse();
		testCancelVsTimeoutRace();

		// 子图（分组互斥）测试
		testSubgraphSerializesExecution();
		testSubgraphDoesNotAffectOthers();
		testSubgraphAllowsMixedAffinity();
		testSubgraphDataflowCorrect();
		testGraphNodeBranchBlocking();

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
