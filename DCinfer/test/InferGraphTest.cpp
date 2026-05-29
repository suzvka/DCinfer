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
		const auto& aNT = ctx.input("a");
		const auto& bNT = ctx.input("b");
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
		const auto& inVal = ctx.input("x");
		const auto* t = inVal.as<Tensor>();
		if (!t)
			return ctx.failure(Node::Status::InvalidInput, "not a Tensor");

		ctx.output("y", Value(std::make_unique<Tensor>(*t)));
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
			const auto& aNT = ctx.input("out_0");
			const auto& bNT = ctx.input("out_1");
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

int main() {
	try {
		testSimpleDataflow();
		testBuildGraph();
		testBroadcastConnectorInGraph();
		testRoutingConnectorInGraph();
		testConnectAll();
		testNodeQuery();

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
