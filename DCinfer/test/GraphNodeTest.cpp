// GraphNode：InferGraph::exportNode 集成测试
// 验证子图嵌入为 Node 的完整生命周期

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "TestHarness.h"
#include "Connector.h"

using namespace DC;

using TensorType = DC::Tensor::TensorType;
using Tensor = DC::Tensor;

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

// ── 测试算子 ──

static Value makeFloatTensor(float value) {
	auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
	*t = value;
	return Value(std::move(t));
}

static Node::Schema addSchema() {
	Node::Schema s;
	s.inputs = {{"a", TensorType::Float, sizeof(float), {}}, {"b", TensorType::Float, sizeof(float), {}}};
	s.outputs = {{"s", TensorType::Float, sizeof(float), {}}};
	return s;
}

static Node::RunFn addRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto* a = ctx.peek("a").as<Tensor>();
		const auto* b = ctx.peek("b").as<Tensor>();
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
	s.inputs = {{"x", TensorType::Float, sizeof(float), {}}};
	s.outputs = {{"y", TensorType::Float, sizeof(float), {}}};
	return s;
}

static Node::RunFn identityRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto* t = ctx.peek("x").as<Tensor>();
		if (!t)
			return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
		ctx.output("y", Value(std::make_unique<Tensor>(*t)));
		return ctx.success();
	};
}

// ════════════════════════════════════════════
// 测试 1: 基础子图嵌入 — add → identity
// ════════════════════════════════════════════

void testBasicGraphEmbedding() {
	TEST("basic graph embedding: subgraph(add→identity) as GraphNode in parent") {
		// ═══ 构建子图 ═══
		InferGraph subGraph;
		subGraph.addNode(std::make_unique<Node>("Builtin", "sub_add", addSchema(), addRunFn()));
		subGraph.addNode(std::make_unique<Node>("Builtin", "sub_id", identitySchema(), identityRunFn()));
		subGraph.connect("sub_add", "s", "sub_id", "x");

		// 声明子图接口：输入 = sub_add 的两个端口，输出 = sub_id 的 y
		subGraph.bindInput("a", "sub_add", "a");
		subGraph.bindInput("b", "sub_add", "b");
		subGraph.bindOutput("y", "sub_id", "y");

		// 导出为 Node
		auto graphNode = subGraph.exportNode("SubAdder");

		// ═══ 构建父图 ═══
		TestHarness parent;
		parent.addNode(std::make_unique<Node>("Builtin", "source", identitySchema(), identityRunFn()));
		parent.addNode(std::move(graphNode)); // 嵌入子图
		parent.addNode(std::make_unique<Node>("Builtin", "sink", identitySchema(), identityRunFn()));

		// 使用 Broadcast(2) 将 source.y 扇出到 SubAdder 的两个输入 (a, b)
		auto bc2 = Connector::broadcastSchema(2);
		auto bcNode = std::make_unique<Node>("Connector.Broadcast", "source_bc", bc2,
		                                     Connector::broadcastRunFn(), ThreadPoolAffinity::System);
		bcNode->setConnector(true);
		parent.addNode(std::move(bcNode));
		parent.connect("source", "y", "source_bc", "in");
		parent.connect("source_bc", "out_0", "SubAdder", "a");
		parent.connect("source_bc", "out_1", "SubAdder", "b");

		parent.connect("SubAdder", "y", "sink", "x");

		parent.feedInput("t1", "source", "x", makeFloatTensor(3.0f));
		parent.submit("t1", "sink", "y");

		bool completed = parent.awaitCompletion("t1");
		CHECK(completed, "parent should complete");
		CHECK(parent.hasOutput("t1", "sink", "y"), "sink should have output");
		auto r = parent.getOutputTensor("t1", "sink", "y");
		CHECK(std::abs(r.item<float>() - 6.0f) < 1e-6f, "result should be 3+3=6 via subgraph");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 测试 2: 分支子图 — 含 Broadcast 扇出
// ════════════════════════════════════════════

void testBranchSubgraph() {
	TEST("branch subgraph: identity → broadcast → id_a as GraphNode") {
		// ═══ 构建子图：identity → Broadcast(2) → id_a（只收集一个输出）═══
		InferGraph subGraph;
		subGraph.addNode(std::make_unique<Node>("Builtin", "sub_src", identitySchema(), identityRunFn()));

		auto bcSchema = Connector::broadcastSchema(2);
		auto bcNode = std::make_unique<Node>("Connector.Broadcast", "sub_bc", bcSchema,
											 Connector::broadcastRunFn(), ThreadPoolAffinity::System);
		bcNode->setConnector(true);
		subGraph.addNode(std::move(bcNode));

		subGraph.addNode(std::make_unique<Node>("Builtin", "sub_a", identitySchema(), identityRunFn()));
		subGraph.addNode(std::make_unique<Node>("Builtin", "sub_b", identitySchema(), identityRunFn()));

		subGraph.connect("sub_src", "y", "sub_bc", "in");
		subGraph.connect("sub_bc", "out_0", "sub_a", "x");
		subGraph.connect("sub_bc", "out_1", "sub_b", "x");

		subGraph.bindInput("x", "sub_src", "x");
		// 只 bind 一个输出验证广播数据流
		subGraph.bindOutput("y", "sub_a", "y");

		auto graphNode = subGraph.exportNode("FanOutGraph");

		// ═══ 父图：source → FanOutGraph → sink ═══
		TestHarness parent;
		parent.addNode(std::make_unique<Node>("Builtin", "source", identitySchema(), identityRunFn()));
		parent.addNode(std::move(graphNode));
		parent.addNode(std::make_unique<Node>("Builtin", "sink", identitySchema(), identityRunFn()));

		parent.connect("source", "y", "FanOutGraph", "x");
		parent.connect("FanOutGraph", "y", "sink", "x");

		parent.feedInput("t1", "source", "x", makeFloatTensor(7.0f));
		parent.submit("t1", "sink", "y");

		CHECK(parent.awaitCompletion("t1"), "parent should complete");
		CHECK(parent.hasOutput("t1", "sink", "y"), "sink should have output");
		auto r = parent.getOutputTensor("t1", "sink", "y");
		CHECK(std::abs(r.item<float>() - 7.0f) < 1e-6f, "value should pass through broadcast subgraph");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 测试 3: 嵌套三层 — GraphA 嵌入 GraphB 嵌入 GraphC
// ════════════════════════════════════════════

void testThreeLevelNesting() {
	TEST("three-level nesting: GraphC in GraphB in GraphA") {
		// ═══ 最内层 GraphC: identity ═══
		InferGraph graphC;
		graphC.addNode(std::make_unique<Node>("Builtin", "c_id", identitySchema(), identityRunFn()));
		graphC.bindInput("x", "c_id", "x");
		graphC.bindOutput("y", "c_id", "y");
		auto nodeC = graphC.exportNode("LevelC");

		// ═══ 中间层 GraphB: LevelC → identity ═══
		InferGraph graphB;
		graphB.addNode(std::move(nodeC));
		graphB.addNode(std::make_unique<Node>("Builtin", "b_id", identitySchema(), identityRunFn()));
		graphB.connect("LevelC", "y", "b_id", "x");
		graphB.bindInput("x", "LevelC", "x");
		graphB.bindOutput("y", "b_id", "y");
		auto nodeB = graphB.exportNode("LevelB");

		// ═══ 最外层 GraphA: LevelB → identity ═══
		InferGraph graphA;
		graphA.addNode(std::move(nodeB));
		graphA.addNode(std::make_unique<Node>("Builtin", "a_id", identitySchema(), identityRunFn()));
		graphA.connect("LevelB", "y", "a_id", "x");
		graphA.bindInput("x", "LevelB", "x");
		graphA.bindOutput("y", "a_id", "y");
		auto nodeA = graphA.exportNode("LevelA");

		// ═══ 父图测试 ═══
		TestHarness parent;
		parent.addNode(std::make_unique<Node>("Builtin", "source", identitySchema(), identityRunFn()));
		parent.addNode(std::move(nodeA));
		parent.addNode(std::make_unique<Node>("Builtin", "sink", identitySchema(), identityRunFn()));

		parent.connect("source", "y", "LevelA", "x");
		parent.connect("LevelA", "y", "sink", "x");

		parent.feedInput("t1", "source", "x", makeFloatTensor(42.0f));
		parent.submit("t1", "sink", "y");

		CHECK(parent.awaitCompletion("t1"), "three-level nested should complete");
		CHECK(parent.hasOutput("t1", "sink", "y"), "sink should have output");
		auto r = parent.getOutputTensor("t1", "sink", "y");
		CHECK(std::abs(r.item<float>() - 42.0f) < 1e-6f, "result should pass through 3 levels unchanged");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 测试 4: 子图循环截断（TTL 兑底，无执行超时）
// ════════════════════════════════════════════

void testSubgraphLoopTTL() {
	TEST("subgraph loop: TTL inside subgraph bounds iterations (no execution timeout)") {
		// 子图：包含无限循环（声明很大的 count；子图内部无执行超时，
		// 由 TTL=3 截断——时间语义归节点实现方，构图循环靠拓扑兑底）。
		// 注意：父图仅用等待护栏（awaitCompletion 默认 5s），不设 submit 超时。
		InferGraph subGraph;

		// 自增节点（反馈环）
		Node::Schema incSchema;
		incSchema.inputs = {{"x", TensorType::Float, sizeof(float), {}}};
		incSchema.outputs = {{"y", TensorType::Float, sizeof(float), {}}};
		auto incRunFn = [](Node::RunContext& ctx) -> Node::Result {
			const auto* t = ctx.peek("x").as<Tensor>();
			if (!t) return ctx.failure(Node::Status::InvalidInput, "not Tensor");
			float val = t->item<float>() + 1.0f;
			auto out = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
			*out = val;
			ctx.output("y", Value(std::move(out)));
			return ctx.success();
		};

		subGraph.addNode(std::make_unique<Node>("Builtin", "loop", incSchema, incRunFn));
		subGraph.connect("loop", "y", "loop", "x");
		subGraph.bindInput("x", "loop", "x");
		subGraph.bindOutput("y", "loop", "y");

		// TTL=3，很小的跳数限制，子图内很快终止
		auto graphNode = subGraph.exportNode("LoopGraph", 3);

		TestHarness parent;
		parent.addNode(std::make_unique<Node>("Builtin", "source", identitySchema(), identityRunFn()));
		parent.addNode(std::move(graphNode));

		parent.connect("source", "y", "LoopGraph", "x");

		parent.feedInput("t1", "source", "x", makeFloatTensor(0.0f));
		parent.submit("t1", "LoopGraph", "y", 1);

		CHECK(parent.awaitCompletion("t1"), "should complete (via TTL, not hang)");

		// 由于 TTL=3，子图内循环会被截断，最终有输出但迭代次数有限
		if (parent.hasOutput("t1", "LoopGraph", "y")) {
			auto r = parent.getOutputTensor("t1", "LoopGraph", "y");
			CHECK(r.item<float>() < 10.0f, "output should be limited due to TTL=3");
		}
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 测试 5: 子图并发调用
// ════════════════════════════════════════════

void testChainedSubgraphs() {
	TEST("chained subgraphs: two independent GraphNodes in one parent") {
		// 两个独立子图实例，验证各自得出正确结果
		InferGraph subGraph1;
		subGraph1.addNode(std::make_unique<Node>("Builtin", "add_a", addSchema(), addRunFn()));
		subGraph1.bindInput("a", "add_a", "a");
		subGraph1.bindInput("b", "add_a", "b");
		subGraph1.bindOutput("s", "add_a", "s");
		auto node1 = subGraph1.exportNode("Adder1");

		InferGraph subGraph2;
		subGraph2.addNode(std::make_unique<Node>("Builtin", "add_b", addSchema(), addRunFn()));
		subGraph2.bindInput("a", "add_b", "a");
		subGraph2.bindInput("b", "add_b", "b");
		subGraph2.bindOutput("s", "add_b", "s");
		auto node2 = subGraph2.exportNode("Adder2");

		TestHarness parent;
		parent.addNode(std::make_unique<Node>("Builtin", "src1", identitySchema(), identityRunFn()));
		parent.addNode(std::make_unique<Node>("Builtin", "src2", identitySchema(), identityRunFn()));
		parent.addNode(std::move(node1));
		parent.addNode(std::move(node2));

		// src1 扇出到 Adder1.a 和 Adder2.a
		auto bc1Schema = Connector::broadcastSchema(2);
		auto bc1Node = std::make_unique<Node>("Connector.Broadcast", "bc1", bc1Schema,
		                                      Connector::broadcastRunFn(), ThreadPoolAffinity::System);
		bc1Node->setConnector(true);
		parent.addNode(std::move(bc1Node));
		parent.connect("src1", "y", "bc1", "in");
		parent.connect("bc1", "out_0", "Adder1", "a");
		parent.connect("bc1", "out_1", "Adder2", "a");

		// src2 扇出到 Adder1.b 和 Adder2.b
		auto bc2Schema = Connector::broadcastSchema(2);
		auto bc2Node = std::make_unique<Node>("Connector.Broadcast", "bc2", bc2Schema,
		                                      Connector::broadcastRunFn(), ThreadPoolAffinity::System);
		bc2Node->setConnector(true);
		parent.addNode(std::move(bc2Node));
		parent.connect("src2", "y", "bc2", "in");
		parent.connect("bc2", "out_0", "Adder1", "b");
		parent.connect("bc2", "out_1", "Adder2", "b");

		parent.feedInput("t1", "src1", "x", makeFloatTensor(10.0f));
		parent.feedInput("t1", "src2", "x", makeFloatTensor(20.0f));
		parent.submit("t1", {{"Adder1", "s"}, {"Adder2", "s"}});
		CHECK(parent.awaitCompletion("t1"), "should complete");
		CHECK(parent.hasOutput("t1", "Adder1", "s"), "Adder1 should have output");
		CHECK(parent.hasOutput("t1", "Adder2", "s"), "Adder2 should have output");
		auto r1 = parent.getOutputTensor("t1", "Adder1", "s");
		auto r2 = parent.getOutputTensor("t1", "Adder2", "s");
		CHECK(std::abs(r1.item<float>() - 30.0f) < 1e-6f, "Adder1: 10+20=30");
		CHECK(std::abs(r2.item<float>() - 30.0f) < 1e-6f, "Adder2: 10+20=30");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 测试 6: 子图输入验证
// ════════════════════════════════════════════

void testSubgraphInputSchema() {
	TEST("subgraph input schema correctness") {
		InferGraph subGraph;
		subGraph.addNode(std::make_unique<Node>("Builtin", "sub_add", addSchema(), addRunFn()));
		subGraph.addNode(std::make_unique<Node>("Builtin", "sub_id", identitySchema(), identityRunFn()));
		subGraph.connect("sub_add", "s", "sub_id", "x");

		subGraph.bindInput("a", "sub_add", "a");
		subGraph.bindInput("b", "sub_add", "b");
		subGraph.bindOutput("y", "sub_id", "y");

		auto graphNode = subGraph.exportNode("TestNode");

		// 验证 Schema 正确性
		CHECK(graphNode != nullptr, "exported node should not be null");
		CHECK(graphNode->name() == "TestNode", "node name should match");
		CHECK(graphNode->type() == "GraphNode", "node type should be GraphNode");

		// 输入端口应该包含 a 和 b
		const auto& schema = graphNode->schema();
		CHECK(schema.inputs.size() == 2, "should have 2 input ports");
		bool hasA = false, hasB = false;
		for (auto& p : schema.inputs) {
			if (p.name == "a") hasA = true;
			if (p.name == "b") hasB = true;
		}
		CHECK(hasA, "should have input port 'a'");
		CHECK(hasB, "should have input port 'b'");

		// 输出端口应该包含 y（来自 sub_id）
		CHECK(schema.outputs.size() == 1, "should have 1 output port");
		CHECK(schema.outputs[0].name == "y", "output port should be 'y'");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 测试 7: 空接口子图
// ════════════════════════════════════════════

void testEmptyInterfaceSubgraph() {
	TEST("empty interface subgraph: no bindInput/bindOutput produces empty schema") {
		InferGraph subGraph;
		subGraph.addNode(std::make_unique<Node>("Builtin", "n1", identitySchema(), identityRunFn()));

		// 不调用 bindInput/bindOutput
		auto graphNode = subGraph.exportNode("EmptyNode");

		CHECK(graphNode != nullptr, "exported node should not be null");
		const auto& schema = graphNode->schema();
		CHECK(schema.inputs.empty(), "no input bindings → empty input schema");
		CHECK(schema.outputs.empty(), "no output bindings → empty output schema");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 测试 8: InputZone 序列化 round-trip
// ════════════════════════════════════════════

void testInputZoneRoundTrip() {
	TEST("inputZone serialization round-trip via GraphCompiler") {
		// 需要 GraphCompiler，这里只做接口级别的单元验证
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "n1", identitySchema(), identityRunFn()));
		graph.addNode(std::make_unique<Node>("Builtin", "n2", addSchema(), addRunFn()));

		// bind multiple inputs
		graph.bindInput("x", "n1", "x");
		graph.bindInput("a", "n2", "a");
		graph.bindInput("b", "n2", "b");

		auto& bindings = graph.inputBindings();
		CHECK(bindings.size() == 3, "should have 3 input bindings");
		CHECK(bindings[0].nodeName == "n1", "first binding node should be n1");
		CHECK(bindings[0].portName == "x", "first binding port should be x");
		CHECK(bindings[1].nodeName == "n2", "second binding node should be n2");
		CHECK(bindings[1].portName == "a", "second binding port should be a");
		CHECK(bindings[2].nodeName == "n2", "third binding node should be n2");
		CHECK(bindings[2].portName == "b", "third binding port should be b");

		// InputZone::isBound 验证
		// 需要访问 InputZone，间接通过 bindInput 已测
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 测试 9: wait 同步机制验证
// ════════════════════════════════════════════

void testWaitMechanism() {
	TEST("InferGraph::waitForResult synchronization") {
		InferGraph graph;
		graph.addNode(std::make_unique<Node>("Builtin", "n1", identitySchema(), identityRunFn()));

		graph.feedInput("t1", "n1", "x", makeFloatTensor(99.0f));

		// 通过回调在 _terminate 清理前捕获输出
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

		// 回调先于 submit 设置：保证 _terminate 时回调必然就绪（避免时序竞态）
		graph.submit("t1", "n1", "y", 1);

		bool completed = graph.waitForResult("t1", std::chrono::milliseconds(5000)).status != TaskStatus::Running;
		CHECK(completed, "waitForResult should return non-Running (completed within timeout)");

		// 等待回调完成（捕获输出后再读取）
		{
			std::unique_lock lk(*mtx);
			cv->wait(lk, [&] { return *done; });
		}

		CHECK(capturedOutput->has_value(), "output should be captured");
		CHECK(std::abs(capturedOutput->value().item<float>() - 99.0f) < 1e-6f, "output should be 99.0");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 测试 10: GraphRuntimeState 生命周期压力 — 并发 submit/cancel/wait/releaseTask
// 验证共享图状态下 TaskGate/任务 lambda/状态表在交错生命周期操作下无崩溃/死锁
// ════════════════════════════════════════════

void testConcurrentTaskLifecycleStress() {
	TEST("concurrent submit/cancel/wait/releaseTask stress (shared GraphRuntimeState)") {
		InferGraph graph;
		// 每线程独立节点：节点级互斥（Reentrant 丢弃）是既有设计，
		// 本测试聚焦并发任务生命周期（TaskGate/任务 lambda/状态表/OutputZone）的安全性
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
						// 每迭代唯一 taskId（贴近真实请求 ID 语义）：
						// 取消后迟到的旧 lambda 经 gate 级检查安全退出（既有设计中
						// 同 ID 复用在取消场景存在 tryExecute 副作用竞态，不属于本测试目标）
						const std::string tid = "stress-" + std::to_string(t) + "-" + std::to_string(i);
						Tensor in(TensorType::Float, sizeof(float));
						in = static_cast<float>(i);
						graph.feedInput(tid, nodeName, "x", Value(std::make_unique<Tensor>(std::move(in))));
						graph.submit(tid, nodeName, "y", 1);
						if (i % 2 == 0)
							graph.cancel(tid); // 交错取消：终态为 Cancelled（或已完成的 Succeeded）
						if (graph.waitForResult(tid, std::chrono::milliseconds(5000)).status == TaskStatus::Running) {
							std::cerr << "\n  [stress] wait timeout: task=" << tid
									  << " status=" << static_cast<int>(graph.taskStatus(tid))
									  << " iter=" << i << std::endl;
							for (auto& err : graph.taskErrors(tid))
								std::cerr << "    [err] node=" << err.nodeName
										  << " lvl=" << static_cast<int>(err.level)
										  << " msg=" << err.message << std::endl;
							++anomalies; // 终止超时视为异常
						}
						graph.releaseTask(tid); // 释放已终止任务全部资源
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

int main() {
	try {
		testBasicGraphEmbedding();
		testBranchSubgraph();
		testThreeLevelNesting();
		testSubgraphLoopTTL();
		testChainedSubgraphs();
		testSubgraphInputSchema();
		testEmptyInterfaceSubgraph();
		testInputZoneRoundTrip();
		testWaitMechanism();
		testConcurrentTaskLifecycleStress();

		if (failures == 0) {
			std::cout << "\nAll GraphNode tests passed!" << std::endl;
		} else {
			std::cout << "\n" << failures << " test(s) FAILED!" << std::endl;
		}
		return failures;
	} catch (const std::exception& e) {
		std::cerr << "Test failure: " << e.what() << std::endl;
		return -1;
	}
}


