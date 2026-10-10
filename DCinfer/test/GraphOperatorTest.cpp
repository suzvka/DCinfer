// GraphOperator 组合算子集成测试：子图包装为 Node 的完整组合语义
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "Connector.h"
#include "GraphException.h"
#include "GraphOperator.h"
#include "ResourceScheduler.h"
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

static void testBasicEmbedding() {
	TEST("basic embedding: subgraph(add→identity) composed as a node in parent") {
		auto sub = std::make_shared<InferGraph>();
		sub->addNode(std::make_unique<Node>("Builtin", "sub_add", addSchema(), addRunFn()));
		sub->addNode(std::make_unique<Node>("Builtin", "sub_id", identitySchema(), identityRunFn()));
		sub->connect("sub_add", "s", "sub_id", "x");
		sub->bindInput("a", "sub_add", "a");
		sub->bindInput("b", "sub_add", "b");
		sub->bindOutput("y", "sub_id", "y");

		GraphOperator op(sub);

		InferGraph parent;
		parent.addNode(std::make_unique<Node>("Builtin", "source", identitySchema(), identityRunFn()));
		parent.addNode(op.makeNode("SubAdder"));
		parent.addNode(std::make_unique<Node>("Builtin", "sink", identitySchema(), identityRunFn()));

		auto bcNode = std::make_unique<Node>("Connector.Broadcast", "source_bc", Connector::broadcastSchema(2),
											 Connector::broadcastRunFn(), ResourceClass::System);
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

static void testBranchSubgraph() {
	TEST("branch subgraph: identity → broadcast → single bound output") {
		auto sub = std::make_shared<InferGraph>();
		sub->addNode(std::make_unique<Node>("Builtin", "sub_src", identitySchema(), identityRunFn()));

		auto bcNode = std::make_unique<Node>("Connector.Broadcast", "sub_bc", Connector::broadcastSchema(2),
											 Connector::broadcastRunFn(), ResourceClass::System);
		bcNode->setConnector(true);
		sub->addNode(std::move(bcNode));
		sub->addNode(std::make_unique<Node>("Builtin", "sub_a", identitySchema(), identityRunFn()));
		sub->addNode(std::make_unique<Node>("Builtin", "sub_b", identitySchema(), identityRunFn()));

		sub->connect("sub_src", "y", "sub_bc", "in");
		sub->connect("sub_bc", "out_0", "sub_a", "x");
		sub->connect("sub_bc", "out_1", "sub_b", "x");

		sub->bindInput("x", "sub_src", "x");
		sub->bindOutput("y", "sub_a", "y");

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

static void testThreeLevelNesting() {
	TEST("three-level nesting: composed node inside composed node") {
		auto graphC = std::make_shared<InferGraph>();
		graphC->addNode(std::make_unique<Node>("Builtin", "c_id", identitySchema(), identityRunFn()));
		graphC->bindInput("x", "c_id", "x");
		graphC->bindOutput("y", "c_id", "y");
		GraphOperator opC(graphC);

		auto graphB = std::make_shared<InferGraph>();
		graphB->addNode(opC.makeNode("LevelC"));
		graphB->addNode(std::make_unique<Node>("Builtin", "b_id", identitySchema(), identityRunFn()));
		graphB->connect("LevelC", "y", "b_id", "x");
		graphB->bindInput("x", "LevelC", "x");
		graphB->bindOutput("y", "b_id", "y");
		GraphOperator opB(graphB);

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

static void testLoopTTLBounded() {
	TEST("subgraph loop: TTL (Options.maxHops) bounds iterations, no hang") {
		auto sub = std::make_shared<InferGraph>();

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
		sub->addNode(std::make_unique<Node>("Builtin", "out", identitySchema(), identityRunFn()));
		// 绑定不得挂在环边上：分支叶承接绑定，环边只作反馈
		sub->connect("loop", "y", "out", "x");
		sub->connect("loop", "y", "loop", "x");
		sub->bindInput("x", "loop", "x");
		sub->bindOutput("y", "out", "y");

		GraphOperator::Options opts;
		opts.maxHops = 3;
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

		auto bc1Node = std::make_unique<Node>("Connector.Broadcast", "bc1", Connector::broadcastSchema(2),
											  Connector::broadcastRunFn(), ResourceClass::System);
		bc1Node->setConnector(true);
		parent.addNode(std::move(bc1Node));
		parent.connect("src1", "y", "bc1", "in");
		parent.connect("bc1", "out_0", "Adder1", "a");
		parent.connect("bc1", "out_1", "Adder2", "a");

		auto bc2Node = std::make_unique<Node>("Connector.Broadcast", "bc2", Connector::broadcastSchema(2),
											  Connector::broadcastRunFn(), ResourceClass::System);
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

static void testSchemaDerivation() {
	TEST("schema derivation: port name = binding alias; type/size/required copied") {
		auto sub = std::make_shared<InferGraph>();
		sub->addNode(std::make_unique<Node>("Builtin", "sub_add", addSchema(), addRunFn()));
		sub->addNode(std::make_unique<Node>("Builtin", "sub_id", identitySchema(), identityRunFn()));
		sub->connect("sub_add", "s", "sub_id", "x");

		sub->bindInput("left", "sub_add", "a");
		sub->bindInput("right", "sub_add", "b");
		sub->bindOutput("sum", "sub_id", "y");

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

static void testParentCancelUnwindsBlockedSubgraph() {
	TEST("cancel(parent) unwinds blocked composed node and frees the compute slot") {
		// 专属 1/1/1 调度器：父 block 占唯一 Compute 槽位，q 排队；子图节点属 Operator 类，类间隔离
		auto sched = std::make_shared<ResourceScheduler>(SchedulerConfig{1, 1, 1});
		auto sub = std::make_shared<InferGraph>(sched);
		sub->addNode(std::make_unique<Node>("test", "n", identitySchema(), identityRunFn()));
		auto m = std::make_unique<Node>("test", "m", identitySchema(), identityRunFn());
		m->bindSignal(sub->signalStore(), "gate");
		sub->addNode(std::move(m));
		sub->connect("n", "y", "m", "x");
		sub->bindInput("x", "n", "x");
		sub->bindOutput("y", "m", "y");

		GraphOperator op(sub);
		auto blockNode = op.makeNode("sub", ResourceClass::Compute);

		InferGraph parent(sched);
		parent.addNode(std::move(blockNode));
		parent.addNode(std::make_unique<Node>("test", "q", identitySchema(), identityRunFn(),
											  ResourceClass::Compute));
		parent.bindInput("x", "sub", "x");
		parent.bindOutput("y", "sub", "y");

		parent.feedInput("p1", "sub", "x", floatValue(1.0f));
		parent.submit("p1", "sub", "y", 1);

		parent.feedInput("p2", "q", "x", floatValue(2.0f));
		parent.submit("p2", "q", "y", 1);

		CHECK(parent.waitForResult("p1", 400ms).status == TaskStatus::Running,
			  "composed task blocked by inner signal must stay Running");
		CHECK(parent.waitForResult("p2", 200ms).status == TaskStatus::Running,
			  "queued compute task must not run while the slot is held");

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

		parent.feedInput("t1", "BlockA", "x", floatValue(1.5f));
		parent.feedInput("t1", "BlockB", "x", floatValue(2.5f));
		parent.submit("t1", {{"BlockA", "y"}, {"BlockB", "y"}});

		const auto res = parent.waitForResult("t1", 5s);
		CHECK(res.status == TaskStatus::Succeeded, "shared subgraph must serve concurrent nodes");
		CHECK(std::abs(parent.takeOutputTensor("t1", "BlockA", "y").item<float>() - 1.5f) < 1e-6f,
			  "BlockA value must not be cross-contaminated");
		CHECK(std::abs(parent.takeOutputTensor("t1", "BlockB", "y").item<float>() - 2.5f) < 1e-6f,
			  "BlockB value must not be cross-contaminated");

		parent.feedInput("t2", "BlockA", "x", floatValue(9.0f));
		parent.submit("t2", "BlockA", "y", 1);
		const auto res2 = parent.waitForResult("t2", 5s);
		CHECK(res2.status == TaskStatus::Succeeded, "same node must be reusable across tasks");
		CHECK(std::abs(parent.takeOutputTensor("t2", "BlockA", "y").item<float>() - 9.0f) < 1e-6f,
			  "second task round-trip");
	}
	END_TEST();
}

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
		}

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

static void testConstructorValidation() {
	TEST("constructor rejects null graph / missing node / missing port / connector / bad poll") {
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

		{
			auto sub = std::make_shared<InferGraph>();
			auto bcNode = std::make_unique<Node>("Connector.Broadcast", "wire", Connector::broadcastSchema(2),
												 Connector::broadcastRunFn(), ResourceClass::System);
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

static void testMissingBoundOutputFailFast() {
	TEST("missing bound output after subgraph success -> explicit failure with port info (#6)") {
		auto sub = std::make_shared<InferGraph>();
		sub->addNode(std::make_unique<Node>("test", "n", identitySchema(), identityRunFn()));
		sub->bindInput("x", "n", "x");
		sub->bindOutput("y", "n", "y");

		// 完成回调消费掉绑定输出：声明满足但产物不可取时必须显式失败
		std::atomic<bool> consumed{false};
		sub->setTaskCompleteCallback([&](const std::string& tid) {
			try {
				sub->takeOutput(tid, "n", "y");
				consumed.store(true);
			} catch (...) {
			}
		});

		GraphOperator op(sub);
		InferGraph parent;
		parent.addNode(op.makeNode("Blk"));
		parent.feedInput("t1", "Blk", "x", floatValue(1.0f));
		parent.submit("t1", "Blk", "y", 1);

		const auto res = parent.waitForResult("t1", 5s);
		CHECK(consumed.load(), "sanity: callback must consume the bound output");
		CHECK(res.status == TaskStatus::Failed, "missing bound output must fail explicitly");

		bool carriesPort = false;
		for (const auto& e : parent.taskErrors("t1")) {
			if (e.message.find("bound outputs were not produced") != std::string::npos
				&& e.message.find("y") != std::string::npos)
				carriesPort = true;
		}
		CHECK(carriesPort, "failure diagnostic must carry the missing port info");
	}
	END_TEST();
}

// 等待型组合节点默认归 System 类，默认预算 4：与子图 Operator 类隔离，
// 避免父节点占满 Operator 槽位后子图同池排队自锁
static void testDefaultBudgetNesting() {
	TEST("default budget: parent-subgraph nesting runs out of the box") {
		auto sub = std::make_shared<InferGraph>();
		sub->addNode(std::make_unique<Node>("Builtin", "sub_id", identitySchema(), identityRunFn()));
		sub->bindInput("x", "sub_id", "x");
		sub->bindOutput("y", "sub_id", "y");

		GraphOperator op(sub);

		InferGraph parent;
		parent.addNode(std::make_unique<Node>("Builtin", "source", identitySchema(), identityRunFn()));
		parent.addNode(op.makeNode("Block"));
		parent.connect("source", "y", "Block", "x");
		parent.feedInput("t1", "source", "x", floatValue(3.0f));
		parent.submit("t1", "Block", "y", 1);

		const auto res = parent.waitForResult("t1", 5s);
		CHECK(res.status == TaskStatus::Succeeded,
			  "nested graph must complete under default budget (no operator-slot self-starvation)");
		const auto out = parent.takeOutput("t1", "Block", "y");
		const auto* t = out.as<Tensor>();
		CHECK(t && std::fabs(t->item<float>() - 3.0f) < 1e-5f, "subgraph passthrough result must be 3.0");
	}
	END_TEST();
}

int main() {
	try {
		// 必须最先运行：使用进程默认预算 1/1/1，早于下方放宽配置
		testDefaultBudgetNesting();

		// 等待型节点在等待期间占住资源类槽位：嵌套等待链需同类预算覆盖全部
		// 并发等待节点，本测试放宽为 1/8/8；configureInstance 仅在实例首次
		// 创建前有效，检查返回值。
		ResourceScheduler::resetInstance(); // 回归用例已创建默认实例，重置后才可重新预配置
		if (!ResourceScheduler::configureInstance(SchedulerConfig{1, 8, 8})) {
			std::cerr << "FAIL: ResourceScheduler::configureInstance rejected (instance pre-created)" << std::endl;
			return 1;
		}
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
		testMissingBoundOutputFailFast();
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
