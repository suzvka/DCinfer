// InferGraph 拓扑连接与数据流 单元测试
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "InferGraph.h"
#include "Connector.h"

using namespace DC;

using TensorType = DC::Tensor::TensorType;
using Tensor     = DC::Tensor;
using Shape      = DC::Tensor::Shape;

static int failures = 0;

#define CHECK(cond, msg) do { \
	if (!(cond)) { \
		std::cerr << "FAIL: " << msg << std::endl; \
		++failures; \
		return; \
	} \
} while(0)

#define TEST(name) std::cout << "Test: " << name << " ... " << std::flush; \
	[&]()
#define END_TEST() (); \
	std::cout << "PASSED" << std::endl

// ── 辅助 ──

static Value makeFloatTensor(float value) {
	auto* p = new Tensor(TensorType::Float, sizeof(float));
	*p = value;
	return Value(p, [](Tensor* ptr) { delete ptr; });
}

// ── 加法算子 Schema + RunFn ──
static Node::Schema addSchema() {
	Node::Schema s;
	s.inputs  = {{"a", TensorType::Float, sizeof(float), {}},
	              {"b", TensorType::Float, sizeof(float), {}}};
	s.outputs = {{"s", TensorType::Float, sizeof(float), {}}};
	return s;
}

static Node::RunFn addRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto& aNT = ctx.input("a");
		const auto& bNT = ctx.input("b");
		const auto* a = aNT.as<Tensor>();
		const auto* b = bNT.as<Tensor>();
		if (!a || !b) return ctx.failure(Node::Status::InvalidInput, "not a Tensor");

		float sum = a->item<float>() + b->item<float>();
		auto* p = new Tensor(TensorType::Float, sizeof(float));
		*p = sum;
		ctx.output("s", Value(p, [](Tensor* ptr) { delete ptr; }));
		return ctx.success();
	};
}

// ── 恒等算子 ──
static Node::Schema identitySchema() {
	Node::Schema s;
	s.inputs  = {{"x", TensorType::Float, sizeof(float), {}}};
	s.outputs = {{"y", TensorType::Float, sizeof(float), {}}};
	return s;
}

static Node::RunFn identityRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto& inVal = ctx.input("x");
		const auto* t = inVal.as<Tensor>();
		if (!t) return ctx.failure(Node::Status::InvalidInput, "not a Tensor");

		auto* copy = new Tensor(*t);
		ctx.output("y", Value(copy, [](Tensor* ptr) { delete ptr; }));
		return ctx.success();
	};
}


// ════════════════════════════════════════════
// 测试用例
// ════════════════════════════════════════════

void testBuildGraph() {
	TEST("build graph - addNode and connect") {
		InferGraph graph;

		auto* n1 = graph.addNode(
			std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));
		CHECK(n1 != nullptr, "addNode should succeed");
		CHECK(graph.nodeCount() == 1, "nodeCount should be 1");

		auto* n2 = graph.addNode(
			std::make_unique<Node>("Builtin", "id1", identitySchema(), identityRunFn()));
		CHECK(n2 != nullptr, "second addNode");
		CHECK(graph.nodeCount() == 2, "nodeCount should be 2");

		// 重名应拒绝
		auto* dup = graph.addNode(
			std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));
		CHECK(dup == nullptr, "duplicate name should be rejected");
		CHECK(graph.nodeCount() == 2, "nodeCount still 2");

		// 接线
		bool ok = graph.connect("add1", "s", "id1", "x");
		CHECK(ok, "connect should succeed");
		CHECK(graph.edgeCount() == 1, "edgeCount should be 1");

		// 无效接线
		bool bad = graph.connect("add1", "no_such", "id1", "x");
		CHECK(!bad, "connect with bad src port should fail");
	}
	END_TEST();
}

void testSimpleDataflow() {
	TEST("simple 2-node dataflow: add → identity") {
		InferGraph graph;

		graph.addNode(std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));
		graph.addNode(std::make_unique<Node>("Builtin", "id1", identitySchema(), identityRunFn()));
		graph.connect("add1", "s", "id1", "x");

		// 注入输入
		CHECK(graph.feedInput("t1", "add1", "a", makeFloatTensor(3.0f)), "feed a");
		CHECK(graph.feedInput("t1", "add1", "b", makeFloatTensor(4.0f)), "feed b");

		// 检查就绪
		CHECK(graph.node("add1")->isReady("t1"), "add1 should be ready");

		// 调度入口节点
		graph.schedule("add1", "t1");

		// 驱动执行
		graph.run();

		// 验证最终结果
		CHECK(graph.hasOutput("t1", "id1", "y"), "id1 should have output");
		auto result = graph.getOutputTensor("t1", "id1", "y");
		CHECK(std::abs(result.item<float>() - 7.0f) < 1e-6f, "result should be 7.0");
	}
	END_TEST();
}

void testBroadcastConnectorInGraph() {
	TEST("broadcast connector: add → broadcast → [id1, id2]") {
		InferGraph graph;

		graph.addNode(std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));

		// 广播连接器：1 输入 → 2 输出
		auto bcSchema = Connector::broadcastSchema(2);
		auto bcRunFn  = Connector::broadcastRunFn();
		graph.addNode(std::make_unique<Node>("Builtin", "bc", bcSchema, bcRunFn));

		graph.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		graph.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		// 接线：add1 → bc → [id_a, id_b]
		graph.connect("add1", "s", "bc", "in");
		graph.connect("bc", "out_0", "id_a", "x");
		graph.connect("bc", "out_1", "id_b", "x");

		// 注入
		graph.feedInput("t1", "add1", "a", makeFloatTensor(10.0f));
		graph.feedInput("t1", "add1", "b", makeFloatTensor(20.0f));

		graph.schedule("add1", "t1");
		graph.run();

		// 两个下游都应该有结果
		CHECK(graph.hasOutput("t1", "id_a", "y"), "id_a should have output");
		CHECK(graph.hasOutput("t1", "id_b", "y"), "id_b should have output");

		auto ra = graph.getOutputTensor("t1", "id_a", "y");
		auto rb = graph.getOutputTensor("t1", "id_b", "y");
		CHECK(std::abs(ra.item<float>() - 30.0f) < 1e-6f, "id_a value");
		CHECK(std::abs(rb.item<float>() - 30.0f) < 1e-6f, "id_b value");
	}
	END_TEST();
}

void testRoutingConnectorInGraph() {
	TEST("routing connector: add → routing → [id_a, id_b]") {
		InferGraph graph;

		graph.addNode(std::make_unique<Node>("Builtin", "add1", addSchema(), addRunFn()));

		auto rtSchema = Connector::routingSchema(2);
		auto rtRunFn  = Connector::routingRunFn();
		graph.addNode(std::make_unique<Node>("Builtin", "rt", rtSchema, rtRunFn));

		graph.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
		graph.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

		graph.connect("add1", "s", "rt", "in");
		graph.connect("rt", "out_0", "id_a", "x");
		graph.connect("rt", "out_1", "id_b", "x");

		// 第一轮：t1 → out_0 → id_a
		graph.feedInput("t1", "add1", "a", makeFloatTensor(1.0f));
		graph.feedInput("t1", "add1", "b", makeFloatTensor(2.0f));
		graph.schedule("add1", "t1");
		graph.run();

		CHECK(graph.hasOutput("t1", "id_a", "y"), "t1 should route to id_a (out_0)");
		CHECK(!graph.hasOutput("t1", "id_b", "y"), "t1 should NOT route to id_b");

		auto r1 = graph.getOutputTensor("t1", "id_a", "y");
		CHECK(std::abs(r1.item<float>() - 3.0f) < 1e-6f, "t1 value");

		// 第二轮：t2 → out_1 → id_b
		graph.feedInput("t2", "add1", "a", makeFloatTensor(5.0f));
		graph.feedInput("t2", "add1", "b", makeFloatTensor(6.0f));
		graph.schedule("add1", "t2");
		graph.run();

		CHECK(!graph.hasOutput("t2", "id_a", "y"), "t2 should NOT route to id_a");
		CHECK(graph.hasOutput("t2", "id_b", "y"), "t2 should route to id_b (out_1)");

		auto r2 = graph.getOutputTensor("t2", "id_b", "y");
		CHECK(std::abs(r2.item<float>() - 11.0f) < 1e-6f, "t2 value");
	}
	END_TEST();
}

void testConnectAll() {
	TEST("connectAll auto-matches output ports to input ports") {
		InferGraph graph;

		auto bcSchema = Connector::broadcastSchema(2);
		auto bcRunFn  = Connector::broadcastRunFn();
		graph.addNode(std::make_unique<Node>("Builtin", "bc", bcSchema, bcRunFn));

		// 创建一个有两个输入端口的节点：in_0, in_1（注意与 Connector 的 out_0, out_1 命名不同则不会匹配）
		// 改为与 Connector 输出同名的 Schema
		Node::Schema dualInSchema;
		dualInSchema.inputs  = {{"out_0", TensorType::Float, sizeof(float), {}},
		                         {"out_1", TensorType::Float, sizeof(float), {}}};
		dualInSchema.outputs = {{"sum", TensorType::Float, sizeof(float), {}}};

		auto dualRunFn = [](Node::RunContext& ctx) -> Node::Result {
			const auto& aNT = ctx.input("out_0");
			const auto& bNT = ctx.input("out_1");
			const auto* a = aNT.as<Tensor>();
			const auto* b = bNT.as<Tensor>();
			if (!a || !b) return ctx.failure(Node::Status::InvalidInput, "not Tensor");
			float sum = a->item<float>() + b->item<float>();
			auto* p = new Tensor(TensorType::Float, sizeof(float));
			*p = sum;
			ctx.output("sum", Value(p, [](Tensor* ptr) { delete ptr; }));
			return ctx.success();
		};

		graph.addNode(std::make_unique<Node>("Builtin", "adder", dualInSchema, dualRunFn));

		size_t matched = graph.connectAll("bc", "adder");
		CHECK(matched == 2, "connectAll should match 2 ports");
		CHECK(graph.edgeCount() == 2, "edgeCount should be 2");

		// 验证数据流
		graph.feedInput("t1", "bc", "in", makeFloatTensor(5.0f));
		graph.schedule("bc", "t1");
		graph.run();

		CHECK(graph.hasOutput("t1", "adder", "sum"), "adder should have output");
		auto r = graph.getOutputTensor("t1", "adder", "sum");
		CHECK(std::abs(r.item<float>() - 10.0f) < 1e-6f, "sum should be 5+5=10");
	}
	END_TEST();
}

void testNodeQuery() {
	TEST("node query by name") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "test1", addSchema(), addRunFn()));

		auto* n = graph.node("test1");
		CHECK(n != nullptr, "node should be found");
		CHECK(n->name() == "test1", "name should match");

		CHECK(graph.node("phantom") == nullptr, "nonexistent node should be null");
	}
	END_TEST();
}

int main() {
	try {
		testBuildGraph();
		testSimpleDataflow();
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
