// GraphOperator 组合算子集成测试
//
//   组合语义：把一整张 InferGraph 包装成普通 Node（构造 → makeNode → addNode），
//   覆盖：基本往返 / 分支子图 / 三层嵌套 / 环 + TTL 截断 / 链式复用 /
//   Schema 推导（端口名 = 绑定 alias）/ 构造校验（fail-fast）/ 构造即冻结 /
//   父取消解围（协作式取消跨组合边界保留）/ 同一子图多节点并发（旧
//   DuplicateTask 限制解除）/ 共享所有权生命周期 / 内层诊断转发。
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "Connector.h"
#include "GraphException.h"
#include "GraphOperator.h"
#include "Tensor.hpp"

using namespace DC;
using namespace std::chrono_literals;

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

static Tensor floatTensor(float value) {
	auto t = Tensor::Create<float>();
	t = value;
	return t;
}

static Value floatValue(float value) {
	return Value(std::make_unique<Tensor>(floatTensor(value)));
}

static Node::Schema identitySchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("x")};
	s.outputs = {Node::Port::out<float>("y")};
	return s;
}

static Node::RunFn identityRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto* x = ctx.peek("x").as<Tensor>();
		if (!x)
			return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
		ctx.output("y", Value(std::make_unique<Tensor>(*x)));
		return ctx.success();
	};
}

static Node::Schema addSchema() {
	Node::Schema s;
	s.inputs = {Node::Port::in<float>("a"), Node::Port::in<float>("b")};
	s.outputs = {Node::Port::out<float>("s")};
	return s;
}

static Node::RunFn addRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto* a = ctx.peek("a").as<Tensor>();
		const auto* b = ctx.peek("b").as<Tensor>();
		if (!a || !b)
			return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
		ctx.output("s", Value(std::make_unique<Tensor>(floatTensor(a->item<float>() + b->item<float>()))));
		return ctx.success();
	};
}

// ════════════════════════════════════════════
// 1. 基本往返：add → identity 子图嵌入父图
// ════════════════════════════════════════════

static void testBasicEmbedding() {
	TEST("basic embedding: subgraph(add→identity) composed as a node in parent") {
		auto sub = std::make_shared<InferGraph>();
		sub->addNode(std::make_unique<Node>("Builtin", "sub_add", addSchema(), addRunFn()));
		sub->addNode(std::make_unique<Node>("Builtin", "sub_id", identitySchema(), identityRunFn()));
		sub->connect("sub_add", "s", "sub_id", "x");
		sub->bindInput("a", "sub_add", "a");
		sub->bindInput("b", "sub_add", "b");
		sub->bindOutput("y", "sub_id", "y");

		GraphOperator op(sub); // 构造即冻结子图

		InferGraph parent;
		parent.addNode(std::make_unique<Node>("Builtin", "source", identitySchema(), identityRunFn()));
		parent.addNode(op.makeNode("SubAdder"));
		parent.addNode(std::make_unique<Node>("Builtin", "sink", identitySchema(), identityRunFn()));

		auto bcNode = std::make_unique<Node>("Connector.Broadcast", "source_bc", Connector::broadcastSchema(2),
											 Connector::broadcastRunFn(), ThreadPoolAffinity::System);
		bcNode->setConnector(true);
		parent.addNode(std::move(bcNode));
		parent.connect("source", "y", "source_bc", "in");
		parent.connect("source_bc", "out_0", "SubAdder", "a");
		parent.connect("source_bc", "out_1", "SubAdder", "b");
		parent.connect("SubAdder", "y", "sink", "x");

		parent.feedInput("t1", "source", "x", floatValue(3.0f));
		parent.submit("t1", "sink", "y", 1);

		const auto res = parent.waitForResult("t1", 5s);
		CHECK(res.status == TaskStatus::Succeeded, "composed graph should complete");
		CHECK(parent.hasOutput("t1", "sink", "y"), "sink should have output");
		CHECK(std::abs(parent.takeOutputTensor("t1", "sink", "y").item<float>() - 6.0f) < 1e-6f,
			  "result should be 3+3=6 via composed node");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 2. 分支子图：identity → Broadcast(2)，仅绑定一个输出
// ════════════════════════════════════════════

static void testBranchSubgraph() {
	TEST("branch subgraph: identity → broadcast → single bound output") {
		auto sub = std::make_shared<InferGraph>();
		sub->addNode(std::make_unique<Node>("Builtin", "sub_src", identitySchema(), identityRunFn()));

		auto bcNode = std::make_unique<Node>("Connector.Broadcast", "sub_bc", Connector::broadcastSchema(2),
											 Connector::broadcastRunFn(), ThreadPoolAffinity::System);
		bcNode->setConnector(true);
		sub->addNode(std::move(bcNode));
		sub->addNode(std::make_unique<Node>("Builtin", "sub_a", identitySchema(), identityRunFn()));
		sub->addNode(std::make_unique<Node>("Builtin", "sub_b", identitySchema(), identityRunFn()));

		sub->connect("sub_src", "y", "sub_bc", "in");
		sub->connect("sub_bc", "out_0", "sub_a", "x");
		sub->connect("sub_bc", "out_1", "sub_b", "x");

		sub->bindInput("x", "sub_src", "x");
		sub->bindOutput("y", "sub_a", "y"); // 只收集一个输出验证广播数据流

		GraphOperator op(sub);

		InferGraph parent;
		parent.addNode(std::make_unique<Node>("Builtin", "source", identitySchema(), identityRunFn()));
		parent.addNode(op.makeNode("FanOutGraph"));
		parent.addNode(std::make_unique<Node>("Builtin", "sink", identitySchema(), identityRunFn()));
		parent.connect("source", "y", "FanOutGraph", "x");
		parent.connect("FanOutGraph", "y", "sink", "x");

		parent.feedInput("t1", "source", "x", floatValue(7.0f));
		parent.submit("t1", "sink", "y", 1);

		const auto res = parent.waitForResult("t1", 5s);
		CHECK(res.status == TaskStatus::Succeeded, "branch subgraph should complete");
		CHECK(std::abs(parent.takeOutputTensor("t1", "sink", "y").item<float>() - 7.0f) < 1e-6f,
			  "value should pass through broadcast subgraph");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 3. 三层嵌套：LevelC ⊂ LevelB ⊂ LevelA
// ════════════════════════════════════════════

static void testThreeLevelNesting() {
	TEST("three-level nesting: composed node inside composed node") {
		// 最内层 C：identity
		auto graphC = std::make_shared<InferGraph>();
		graphC->addNode(std::make_unique<Node>("Builtin", "c_id", identitySchema(), identityRunFn()));
		graphC->bindInput("x", "c_id", "x");
		graphC->bindOutput("y", "c_id", "y");
		GraphOperator opC(graphC);

		// 中间层 B：LevelC → identity
		auto graphB = std::make_shared<InferGraph>();
		graphB->addNode(opC.makeNode("LevelC"));
		graphB->addNode(std::make_unique<Node>("Builtin", "b_id", identitySchema(), identityRunFn()));
		graphB->connect("LevelC", "y", "b_id", "x");
		graphB->bindInput("x", "LevelC", "x");
		graphB->bindOutput("y", "b_id", "y");
		GraphOperator opB(graphB);

		// 最外层 A：LevelB → identity
		auto graphA = std::make_shared<InferGraph>();
		graphA->addNode(opB.makeNode("LevelB"));
		graphA->addNode(std::make_unique<Node>("Builtin", "a_id", identitySchema(), identityRunFn()));
		graphA->connect("LevelB", "y", "a_id", "x");
		graphA->bindInput("x", "LevelB", "x");
		graphA->bindOutput("y", "a_id", "y");
		GraphOperator opA(graphA);

		InferGraph parent;
		parent.addNode(std::make_unique<Node>("Builtin", "source", identitySchema(), identityRunFn()));
		parent.addNode(opA.makeNode("LevelA"));
		parent.addNode(std::make_unique<Node>("Builtin", "sink", identitySchema(), identityRunFn()));
		parent.connect("source", "y", "LevelA", "x");
		parent.connect("LevelA", "y", "sink", "x");

		parent.feedInput("t1", "source", "x", floatValue(42.0f));
		parent.submit("t1", "sink", "y", 1);

		const auto res = parent.waitForResult("t1", 5s);
		CHECK(res.status == TaskStatus::Succeeded, "three-level nesting should complete");
		CHECK(std::abs(parent.takeOutputTensor("t1", "sink", "y").item<float>() - 42.0f) < 1e-6f,
			  "result should pass through 3 levels unchanged");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 4. 子图内环 + TTL 截断（Options.maxHops 传递到子图提交）
// ════════════════════════════════════════════

static void testLoopTTLBounded() {
	TEST("subgraph loop: TTL (Options.maxHops) bounds iterations, no hang") {
		auto sub = std::make_shared<InferGraph>();

		// 自增节点（反馈环用）
		Node::Schema incSchema;
		incSchema.inputs = {Node::Port::in<float>("x")};
		incSchema.outputs = {Node::Port::out<float>("y")};
		auto incRunFn = [](Node::RunContext& ctx) -> Node::Result {
			const auto* t = ctx.peek("x").as<Tensor>();
			if (!t)
				return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
			ctx.output("y", Value(std::make_unique<Tensor>(floatTensor(t->item<float>() + 1.0f))));
			return ctx.success();
		};

		sub->addNode(std::make_unique<Node>("Builtin", "loop", incSchema, incRunFn));
		sub->connect("loop", "y", "loop", "x"); // 自反馈环
		sub->bindInput("x", "loop", "x");
		sub->bindOutput("y", "loop", "y");

		GraphOperator::Options opts;
		opts.maxHops = 3; // 很小的跳数限制：子图内很快终止
		GraphOperator op(sub, opts);

		InferGraph parent;
		parent.addNode(std::make_unique<Node>("Builtin", "source", identitySchema(), identityRunFn()));
		parent.addNode(op.makeNode("LoopGraph"));
		parent.connect("source", "y", "LoopGraph", "x");

		parent.feedInput("t1", "source", "x", floatValue(0.0f));
		parent.submit("t1", "LoopGraph", "y", 1);

		const auto res = parent.waitForResult("t1", 5s);
		CHECK(res.status != TaskStatus::Running, "loop must terminate via TTL, not hang");
		if (parent.hasOutput("t1", "LoopGraph", "y")) {
			CHECK(parent.takeOutputTensor("t1", "LoopGraph", "y").item<float>() < 10.0f,
				  "iteration count should be bounded by TTL=3");
		}
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 5. 链式复用：两个独立组合节点在同一父任务并行
// ════════════════════════════════════════════

static void testChainedSubgraphs() {
	TEST("chained subgraphs: two independent composed nodes in one parent") {
		auto sub1 = std::make_shared<InferGraph>();
		sub1->addNode(std::make_unique<Node>("Builtin", "add_a", addSchema(), addRunFn()));
		sub1->bindInput("a", "add_a", "a");
		sub1->bindInput("b", "add_a", "b");
		sub1->bindOutput("s", "add_a", "s");
		GraphOperator op1(sub1);

		auto sub2 = std::make_shared<InferGraph>();
		sub2->addNode(std::make_unique<Node>("Builtin", "add_b", addSchema(), addRunFn()));
		sub2->bindInput("a", "add_b", "a");
		sub2->bindInput("b", "add_b", "b");
		sub2->bindOutput("s", "add_b", "s");
		GraphOperator op2(sub2);

		InferGraph parent;
		parent.addNode(std::make_unique<Node>("Builtin", "src1", identitySchema(), identityRunFn()));
		parent.addNode(std::make_unique<Node>("Builtin", "src2", identitySchema(), identityRunFn()));
		parent.addNode(op1.makeNode("Adder1"));
		parent.addNode(op2.makeNode("Adder2"));

		// src1 扇出到 Adder1.a 与 Adder2.a
		auto bc1Node = std::make_unique<Node>("Connector.Broadcast", "bc1", Connector::broadcastSchema(2),
											  Connector::broadcastRunFn(), ThreadPoolAffinity::System);
		bc1Node->setConnector(true);
		parent.addNode(std::move(bc1Node));
		parent.connect("src1", "y", "bc1", "in");
		parent.connect("bc1", "out_0", "Adder1", "a");
		parent.connect("bc1", "out_1", "Adder2", "a");

		// src2 扇出到 Adder1.b 与 Adder2.b
		auto bc2Node = std::make_unique<Node>("Connector.Broadcast", "bc2", Connector::broadcastSchema(2),
											  Connector::broadcastRunFn(), ThreadPoolAffinity::System);
		bc2Node->setConnector(true);
		parent.addNode(std::move(bc2Node));
		parent.connect("src2", "y", "bc2", "in");
		parent.connect("bc2", "out_0", "Adder1", "b");
		parent.connect("bc2", "out_1", "Adder2", "b");

		parent.feedInput("t1", "src1", "x", floatValue(10.0f));
		parent.feedInput("t1", "src2", "x", floatValue(20.0f));
		parent.submit("t1", {{"Adder1", "s"}, {"Adder2", "s"}});

		const auto res = parent.waitForResult("t1", 5s);
		CHECK(res.status == TaskStatus::Succeeded, "both composed nodes should complete");
		CHECK(parent.hasOutput("t1", "Adder1", "s") && parent.hasOutput("t1", "Adder2", "s"),
			  "both adders should have output");
		CHECK(std::abs(parent.takeOutputTensor("t1", "Adder1", "s").item<float>() - 30.0f) < 1e-6f, "Adder1: 10+20=30");
		CHECK(std::abs(parent.takeOutputTensor("t1", "Adder2", "s").item<float>() - 30.0f) < 1e-6f, "Adder2: 10+20=30");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 6. Schema 推导：端口名 = 绑定 alias（alias ≠ 目标端口名）
// ════════════════════════════════════════════

static void testSchemaDerivation() {
	TEST("schema derivation: port name = binding alias; type/size/required copied") {
		auto sub = std::make_shared<InferGraph>();
		sub->addNode(std::make_unique<Node>("Builtin", "sub_add", addSchema(), addRunFn()));
		sub->addNode(std::make_unique<Node>("Builtin", "sub_id", identitySchema(), identityRunFn()));
		sub->connect("sub_add", "s", "sub_id", "x");

		sub->bindInput("left", "sub_add", "a");   // alias ≠ 目标端口名 "a"
		sub->bindInput("right", "sub_add", "b");  // alias ≠ 目标端口名 "b"
		sub->bindOutput("sum", "sub_id", "y");    // alias ≠ 目标端口名 "y"

		GraphOperator op(sub);
		auto node = op.makeNode("Calc");

		CHECK(node->name() == "Calc", "node name should match");
		CHECK(node->type() == "Builtin", "composed node type should be Builtin (operator parity)");

		const auto& schema = node->schema();
		CHECK(schema.inputs.size() == 2, "should have 2 input ports");
		CHECK(schema.inputs[0].name == "left" && schema.inputs[1].name == "right",
			  "input port names must be binding aliases");
		CHECK(schema.inputs[0].type == Tensor::TensorType::Float && schema.inputs[0].typeSize == sizeof(float),
			  "input type/size must be copied from target port");
		CHECK(schema.inputs[0].required, "required flag must be copied from target port");
		CHECK(schema.outputs.size() == 1 && schema.outputs[0].name == "sum",
			  "output port name must be the binding alias");

		// 端到端：按 alias 寻址喂入/取出
		InferGraph parent;
		parent.addNode(std::move(node));
		parent.feedInput("t1", "Calc", "left", floatValue(2.0f));
		parent.feedInput("t1", "Calc", "right", floatValue(5.0f));
		parent.submit("t1", "Calc", "sum", 1);

		const auto res = parent.waitForResult("t1", 5s);
		CHECK(res.status == TaskStatus::Succeeded, "alias-addressed task should complete");
		CHECK(std::abs(parent.takeOutputTensor("t1", "Calc", "sum").item<float>() - 7.0f) < 1e-6f, "2+5=7");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 7. 空接口拒绝（fail-fast）
// ════════════════════════════════════════════

static void testEmptyInterfaceRejected() {
	TEST("no-interface graph is rejected at construction") {
		auto sub = std::make_shared<InferGraph>();
		sub->addNode(std::make_unique<Node>("Builtin", "n1", identitySchema(), identityRunFn()));

		bool threw = false;
		try {
			GraphOperator op(sub);
			static_cast<void>(op);
		} catch (const GraphException& e) {
			threw = e.getErrorType() == GraphException::ErrorType::Other;
		}
		CHECK(threw, "graph without bindInput/bindOutput must be rejected");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 8. 构造即冻结：Schema 定型后子图不可再改
// ════════════════════════════════════════════

static void testEagerFreezeOnConstruction() {
	TEST("subgraph is frozen at construction; composed node still runs") {
		auto sub = std::make_shared<InferGraph>();
		sub->addNode(std::make_unique<Node>("Builtin", "n", identitySchema(), identityRunFn()));
		sub->bindInput("x", "n", "x");
		sub->bindOutput("y", "n", "y");

		GraphOperator op(sub);

		bool frozen = false;
		try {
			sub->addNode(std::make_unique<Node>("Builtin", "late", identitySchema(), identityRunFn()));
		} catch (const GraphException& e) {
			frozen = e.getErrorType() == GraphException::ErrorType::Frozen;
		}
		CHECK(frozen, "subgraph build API must be rejected after composition");

		InferGraph parent;
		parent.addNode(op.makeNode("B"));
		parent.feedInput("t1", "B", "x", floatValue(4.0f));
		parent.submit("t1", "B", "y", 1);
		const auto res = parent.waitForResult("t1", 5s);
		CHECK(res.status == TaskStatus::Succeeded, "frozen subgraph should still execute");
		CHECK(std::abs(parent.takeOutputTensor("t1", "B", "y").item<float>() - 4.0f) < 1e-6f, "value round-trip");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 9. 父取消解围：内层信号阻塞不再永久占住池线程（协作式取消跨组合边界）
// ════════════════════════════════════════════

static void testParentCancelUnwindsBlockedSubgraph() {
	TEST("cancel(parent) unwinds blocked composed node and frees the pool thread") {
		// 子图：n → m（m 绑定信号，未置位 → 默认阻塞）→ 声明输出 m.y。
		auto sub = std::make_shared<InferGraph>();
		sub->addNode(std::make_unique<Node>("test", "n", identitySchema(), identityRunFn()));
		auto m = std::make_unique<Node>("test", "m", identitySchema(), identityRunFn());
		m->bindSignal(sub->signalStore(), "gate"); // 绑定后未置位 → 默认阻断
		sub->addNode(std::move(m));
		sub->connect("n", "y", "m", "x");
		sub->bindInput("x", "n", "x");
		sub->bindOutput("y", "m", "y");

		GraphOperator op(sub);
		auto blockNode = op.makeNode("sub", ThreadPoolAffinity::Compute); // 复刻旧导出节点的池占位

		// 父图：默认 Compute 单线程——挂起的父任务独占唯一 Compute 线程
		InferGraph parent;
		parent.addNode(std::move(blockNode));
		// 独立直通节点 q（显式 Compute）：验证解围后线程恢复可用
		parent.addNode(std::make_unique<Node>("test", "q", identitySchema(), identityRunFn(),
											  ThreadPoolAffinity::Compute));
		parent.bindInput("x", "sub", "x");
		parent.bindOutput("y", "sub", "y");

		// p1：经组合节点（内部信号阻塞 → RunFn 挂起，占住 Compute 线程）
		parent.feedInput("p1", "sub", "x", floatValue(1.0f));
		parent.submit("p1", "sub", "y", 1);

		// p2：经独立节点 q（Compute）——排在 p1 之后等待线程释放
		parent.feedInput("p2", "q", "x", floatValue(2.0f));
		parent.submit("p2", "q", "y", 1);

		// 挂起确认：p1 因内部阻塞无法完成；p2 因唯一 Compute 线程被占而排队
		CHECK(parent.waitForResult("p1", 400ms).status == TaskStatus::Running,
			  "composed task blocked by inner signal must stay Running");
		CHECK(parent.waitForResult("p2", 200ms).status == TaskStatus::Running,
			  "queued compute task must not run while the thread is held");

		// 宿主解围：cancel(p1) → RunFn 轮询感知 → 取消子图任务 → 返回 → 线程释放
		CHECK(parent.cancel("p1"), "cancel on active task should be accepted");
		CHECK(parent.waitForResult("p1", 2s).status == TaskStatus::Cancelled,
			  "cancelled parent must reach Cancelled state");

		const auto r2 = parent.waitForResult("p2", 3s);
		CHECK(r2.status == TaskStatus::Succeeded, "queued task must complete after the thread is freed");
		CHECK(std::abs(parent.takeOutputTensor("p2", "q", "y").item<float>() - 2.0f) < 1e-6f,
			  "freed-thread task output must match its input");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 10. 同一子图多节点并发（旧 DuplicateTask 限制解除）
// ════════════════════════════════════════════

static void testSharedSubgraphConcurrentNodes() {
	TEST("one subgraph shared by two composed nodes in the same parent task") {
		auto sub = std::make_shared<InferGraph>();
		sub->addNode(std::make_unique<Node>("test", "id", identitySchema(), identityRunFn()));
		sub->bindInput("x", "id", "x");
		sub->bindOutput("y", "id", "y");

		GraphOperator op(sub);

		InferGraph parent;
		parent.addNode(op.makeNode("BlockA"));
		parent.addNode(op.makeNode("BlockB"));

		// 同一父任务内两个组合节点并发驱动同一子图（子任务 ID 命名空间独立）
		parent.feedInput("t1", "BlockA", "x", floatValue(1.5f));
		parent.feedInput("t1", "BlockB", "x", floatValue(2.5f));
		parent.submit("t1", {{"BlockA", "y"}, {"BlockB", "y"}});

		const auto res = parent.waitForResult("t1", 5s);
		CHECK(res.status == TaskStatus::Succeeded, "shared subgraph must serve concurrent nodes");
		CHECK(std::abs(parent.takeOutputTensor("t1", "BlockA", "y").item<float>() - 1.5f) < 1e-6f,
			  "BlockA value must not be cross-contaminated");
		CHECK(std::abs(parent.takeOutputTensor("t1", "BlockB", "y").item<float>() - 2.5f) < 1e-6f,
			  "BlockB value must not be cross-contaminated");

		// 跨任务复用：新父任务再次驱动同一组合节点
		parent.feedInput("t2", "BlockA", "x", floatValue(9.0f));
		parent.submit("t2", "BlockA", "y", 1);
		const auto res2 = parent.waitForResult("t2", 5s);
		CHECK(res2.status == TaskStatus::Succeeded, "same node must be reusable across tasks");
		CHECK(std::abs(parent.takeOutputTensor("t2", "BlockA", "y").item<float>() - 9.0f) < 1e-6f,
			  "second task round-trip");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 11. 共享所有权：GraphOperator 先析构，节点仍可执行
// ════════════════════════════════════════════

static void testOperatorDestroyedNodeStillRuns() {
	TEST("node keeps working after GraphOperator is destroyed (shared ownership)") {
		std::unique_ptr<Node> keep;
		{
			auto sub = std::make_shared<InferGraph>();
			sub->addNode(std::make_unique<Node>("test", "n", identitySchema(), identityRunFn()));
			sub->bindInput("x", "n", "x");
			sub->bindOutput("y", "n", "y");
			GraphOperator op(sub);
			keep = op.makeNode("Persist");
		} // GraphOperator 与局部 shared_ptr 均析构；子图由节点捕获的共享句柄维持存活

		InferGraph parent;
		parent.addNode(std::move(keep));
		parent.feedInput("t1", "Persist", "x", floatValue(9.0f));
		parent.submit("t1", "Persist", "y", 1);
		const auto res = parent.waitForResult("t1", 5s);
		CHECK(res.status == TaskStatus::Succeeded, "node must outlive its operator");
		CHECK(std::abs(parent.takeOutputTensor("t1", "Persist", "y").item<float>() - 9.0f) < 1e-6f,
			  "value round-trip after operator destruction");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 12. 内层失败诊断转发
// ════════════════════════════════════════════

static void testInnerFailureDiagnosticsForwarded() {
	TEST("inner node failure is forwarded with context to the parent task") {
		auto sub = std::make_shared<InferGraph>();
		sub->addNode(std::make_unique<Node>("test", "boom", identitySchema(),
											[](Node::RunContext& ctx) -> Node::Result {
												return ctx.failure(Node::Status::ExecutionFailed, "inner boom");
											}));
		sub->bindInput("x", "boom", "x");
		sub->bindOutput("y", "boom", "y");

		GraphOperator op(sub);

		InferGraph parent;
		parent.addNode(op.makeNode("Bad"));
		parent.feedInput("t1", "Bad", "x", floatValue(1.0f));
		parent.submit("t1", "Bad", "y", 1);

		const auto res = parent.waitForResult("t1", 5s);
		CHECK(res.status == TaskStatus::Failed, "inner failure must fail the parent task");

		bool forwarded = false;
		for (const auto& e : parent.taskErrors("t1")) {
			if (e.message.find("inner boom") != std::string::npos) {
				forwarded = true;
				break;
			}
		}
		CHECK(forwarded, "parent diagnostics must carry the inner failure message");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 13. 构造校验：非法输入 fail-fast（NodeNotFound / PortNotFound / Other）
// ════════════════════════════════════════════

static void testConstructorValidation() {
	TEST("constructor rejects null graph / missing node / missing port / connector / bad poll") {
		// ① null graph
		{
			bool ok = false;
			try {
				GraphOperator op(nullptr);
				static_cast<void>(op);
			} catch (const GraphException& e) {
				ok = e.getErrorType() == GraphException::ErrorType::Other;
			}
			CHECK(ok, "null graph must be rejected");
		}

		// ② 绑定引用的节点不存在
		{
			auto sub = std::make_shared<InferGraph>();
			sub->addNode(std::make_unique<Node>("test", "n", identitySchema(), identityRunFn()));
			sub->bindInput("x", "ghost", "x");
			sub->bindOutput("y", "n", "y");
			bool ok = false;
			try {
				GraphOperator op(sub);
				static_cast<void>(op);
			} catch (const GraphException& e) {
				ok = e.getErrorType() == GraphException::ErrorType::NodeNotFound;
			}
			CHECK(ok, "binding to missing node must be rejected with NodeNotFound");
		}

		// ③ 绑定引用的端口不存在
		{
			auto sub = std::make_shared<InferGraph>();
			sub->addNode(std::make_unique<Node>("test", "n", identitySchema(), identityRunFn()));
			sub->bindInput("x", "n", "nope");
			sub->bindOutput("y", "n", "y");
			bool ok = false;
			try {
				GraphOperator op(sub);
				static_cast<void>(op);
			} catch (const GraphException& e) {
				ok = e.getErrorType() == GraphException::ErrorType::PortNotFound;
			}
			CHECK(ok, "binding to missing port must be rejected with PortNotFound");
		}

		// ④ 绑定目标为连接器
		{
			auto sub = std::make_shared<InferGraph>();
			auto bcNode = std::make_unique<Node>("Connector.Broadcast", "wire", Connector::broadcastSchema(2),
												 Connector::broadcastRunFn(), ThreadPoolAffinity::System);
			bcNode->setConnector(true);
			sub->addNode(std::move(bcNode));
			sub->bindInput("x", "wire", "in");
			bool ok = false;
			try {
				GraphOperator op(sub);
				static_cast<void>(op);
			} catch (const GraphException& e) {
				ok = e.getErrorType() == GraphException::ErrorType::Other;
			}
			CHECK(ok, "connector binding target must be rejected");
		}

		// ⑤ pollInterval <= 0
		{
			auto sub = std::make_shared<InferGraph>();
			sub->addNode(std::make_unique<Node>("test", "n", identitySchema(), identityRunFn()));
			sub->bindInput("x", "n", "x");
			sub->bindOutput("y", "n", "y");
			GraphOperator::Options opts;
			opts.pollInterval = std::chrono::milliseconds(0);
			bool ok = false;
			try {
				GraphOperator op(sub, opts);
				static_cast<void>(op);
			} catch (const GraphException& e) {
				ok = e.getErrorType() == GraphException::ErrorType::Other;
			}
			CHECK(ok, "non-positive pollInterval must be rejected");
		}
	}
	END_TEST();
}

// ════════════════════════════════════════════

int main() {
	try {
		testBasicEmbedding();
		testBranchSubgraph();
		testThreeLevelNesting();
		testLoopTTLBounded();
		testChainedSubgraphs();
		testSchemaDerivation();
		testEmptyInterfaceRejected();
		testEagerFreezeOnConstruction();
		testParentCancelUnwindsBlockedSubgraph();
		testSharedSubgraphConcurrentNodes();
		testOperatorDestroyedNodeStillRuns();
		testInnerFailureDiagnosticsForwarded();
		testConstructorValidation();
	} catch (const std::exception& e) {
		std::cerr << "UNEXPECTED EXCEPTION: " << e.what() << std::endl;
		return 1;
	}

	if (failures != 0) {
		std::cerr << failures << " test(s) failed" << std::endl;
		return 1;
	}
	std::cout << "All GraphOperator tests passed" << std::endl;
	return 0;
}
