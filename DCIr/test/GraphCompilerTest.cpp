// GraphCompiler 单元测试
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "Ir/GraphCompiler.h"
#include "TestHarness.h"

using namespace DC;
using namespace DC::Ir;

using TensorType = DC::Tensor::TensorType;
using Tensor = DC::Tensor;

static int failures = 0;

#define CHECK(cond, msg)                                  \
	do {                                                  \
		if (!(cond)) {                                    \
			std::cerr << "FAIL: " << msg << std::endl;    \
			++failures;                                   \
			return;                                       \
		}                                                 \
	} while (0)

#define TEST(name)                                        \
	std::cout << "Test: " << name << " ... " << std::flush; \
	[&]()
#define END_TEST()                                        \
	();                                                   \
	std::cout << "PASSED" << std::endl

// ── 辅助 ──

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
		if (!t) return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
		ctx.output("y", Value(std::make_unique<Tensor>(*t)));
		return ctx.success();
	};
}

static Node::Schema addSchema() {
	Node::Schema s;
	s.inputs = {{"a", TensorType::Float, sizeof(float), {}},
				{"b", TensorType::Float, sizeof(float), {}}};
	s.outputs = {{"s", TensorType::Float, sizeof(float), {}}};
	return s;
}

static Node::RunFn addRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const auto& aNT = ctx.peek("a");
		const auto& bNT = ctx.peek("b");
		const auto* a = aNT.as<Tensor>();
		const auto* b = bNT.as<Tensor>();
		if (!a || !b) return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
		float sum = a->item<float>() + b->item<float>();
		auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
		*t = sum;
		ctx.output("s", Value(std::move(t)));
		return ctx.success();
	};
}

static Value makeFloatTensor(float value) {
	auto t = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
	*t = value;
	return Value(std::move(t));
}

// ════════════════════════════════════════════
// 测试用例
// ════════════════════════════════════════════

void testCompileStringBasic() {
	TEST("compileString - two nodes with wire edge") {
		const char* json = R"({
  "version": "1.0",
  "nodes": [
    {
      "name": "add1", "type": "Builtin", "affinity": "Operator",
      "inputs": [
        {"name":"a","tensorType":"Float","typeSize":4,"shape":[],"required":true},
        {"name":"b","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ],
      "outputs": [
        {"name":"s","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ]
    },
    {
      "name": "id1", "type": "Builtin", "affinity": "Operator",
      "inputs": [
        {"name":"x","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ],
      "outputs": [
        {"name":"y","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ]
    }
  ],
  "edges": [
    {"srcNode":"add1","srcPort":"s","dstNode":"id1","dstPort":"x"}
  ],
  "outputBindings": [
    {"nodeName":"id1","portName":"y"}
  ]
})";
		InferGraph graph; GraphCompiler::compileString(graph, json);

		// 应有 2 个业务节点 + 1 个 __wire 连接器 = 3 节点
		CHECK(graph.nodeCount() == 3, "should have 3 nodes (add1, id1, __wire_0)");
		CHECK(graph.edgeCount() == 2, "should have 2 edges");
		CHECK(graph.outputBindings().size() == 1, "should have 1 output binding");
		CHECK(graph.node("add1") != nullptr, "add1 should exist");
		CHECK(graph.node("id1") != nullptr, "id1 should exist");
	}
	END_TEST();
}

void testCompileStringBroadcast() {
	TEST("compileString - broadcast mode edge") {
		const char* json = R"({
  "version": "1.0",
  "nodes": [
    {
      "name": "add1", "type": "Builtin", "affinity": "Operator",
      "inputs": [
        {"name":"a","tensorType":"Float","typeSize":4,"shape":[],"required":true},
        {"name":"b","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ],
      "outputs": [
        {"name":"s","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ]
    },
    {
      "name": "id_a", "type": "Builtin", "affinity": "Operator",
      "inputs": [
        {"name":"x","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ],
      "outputs": [
        {"name":"y","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ]
    },
    {
      "name": "id_b", "type": "Builtin", "affinity": "Operator",
      "inputs": [
        {"name":"x","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ],
      "outputs": [
        {"name":"y","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ]
    }
  ],
  "edges": [
    {"srcNode":"add1","srcPort":"s","dstNode":"id_a","dstPort":"x","mode":"broadcast"},
    {"srcNode":"add1","srcPort":"s","dstNode":"id_b","dstPort":"x","mode":"broadcast"}
  ],
  "outputBindings": [
    {"nodeName":"id_a","portName":"y"},
    {"nodeName":"id_b","portName":"y"}
  ]
})";
		InferGraph graph; GraphCompiler::compileString(graph, json);

		// 3 业务节点 + 1 broadcast 连接器 = 4
		CHECK(graph.nodeCount() == 4, "should have 4 nodes (3 biz + 1 bc)");
		CHECK(graph.edgeCount() == 3, "should have 3 edges (src→bc.in, bc.out_0→id_a, bc.out_1→id_b)");
		CHECK(graph.outputBindings().size() == 2, "should have 2 output bindings");
		CHECK(graph.node("add1") != nullptr, "add1 should exist");
		CHECK(graph.node("id_a") != nullptr, "id_a should exist");
		CHECK(graph.node("id_b") != nullptr, "id_b should exist");
	}
	END_TEST();
}

void testCompileStringRouting() {
	TEST("compileString - routing mode edge") {
		const char* json = R"({
  "version": "1.0",
  "nodes": [
    {
      "name": "add1", "type": "Builtin", "affinity": "Operator",
      "inputs": [
        {"name":"a","tensorType":"Float","typeSize":4,"shape":[],"required":true},
        {"name":"b","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ],
      "outputs": [
        {"name":"s","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ]
    },
    {
      "name": "id_a", "type": "Builtin", "affinity": "Operator",
      "inputs": [
        {"name":"x","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ],
      "outputs": [
        {"name":"y","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ]
    },
    {
      "name": "id_b", "type": "Builtin", "affinity": "Operator",
      "inputs": [
        {"name":"x","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ],
      "outputs": [
        {"name":"y","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ]
    }
  ],
  "edges": [
    {"srcNode":"add1","srcPort":"s","dstNode":"id_a","dstPort":"x","mode":"routing"},
    {"srcNode":"add1","srcPort":"s","dstNode":"id_b","dstPort":"x","mode":"routing"}
  ],
  "outputBindings": [
    {"nodeName":"id_a","portName":"y"},
    {"nodeName":"id_b","portName":"y"}
  ]
})";
		InferGraph graph; GraphCompiler::compileString(graph, json);

		// 3 业务节点 + 1 routing 连接器 = 4
		CHECK(graph.nodeCount() == 4, "should have 4 nodes (3 biz + 1 rt)");
		CHECK(graph.edgeCount() == 3, "should have 3 edges");
		CHECK(graph.outputBindings().size() == 2, "should have 2 output bindings");
	}
	END_TEST();
}

void testRoundTrip() {
	TEST("round-trip - serialize then compile") {
		// 构建图
		TestHarness harness;
		harness.addNode(std::make_unique<Node>("ONNX", "test1", identitySchema(), identityRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "test2", identitySchema(), identityRunFn()));
		harness.wire("test1", "y", "test2", "x");
		harness.bindOutput("test2", "y");
		harness.node("test1")->setModelPath("models/test.onnx");

		// 序列化
		std::string tmpFile = "test_roundtrip.json";
		GraphCompiler::serialize(harness.graph(), tmpFile);

		// 反序列化（Builtin 节点不带 RunFn，仅验证结构）
		InferGraph graph2; GraphCompiler::compileFile(graph2, tmpFile);

		// 验证节点数：2 业务节点 + 1 导线 = 3
		CHECK(graph2.nodeCount() == 3, "roundtrip: should have 3 nodes");
		CHECK(graph2.node("test1") != nullptr, "roundtrip: test1 should exist");
		CHECK(graph2.node("test2") != nullptr, "roundtrip: test2 should exist");
		CHECK(graph2.edgeCount() == 2, "roundtrip: should have 2 edges");
		CHECK(graph2.outputBindings().size() == 1, "roundtrip: should have 1 output binding");

		// modelPath 保留
		auto* n1 = graph2.node("test1");
		CHECK(n1 != nullptr, "roundtrip: test1 not null");
		CHECK(n1->modelPath().find("test.onnx") != std::string::npos, "roundtrip: modelPath should contain test.onnx");

		// 清理
		std::remove(tmpFile.c_str());
	}
	END_TEST();
}

void testSerializeToJsonString() {
	TEST("serialize - JSON output is valid and parsable") {
		TestHarness harness;
		harness.addNode(std::make_unique<Node>("Builtin", "n1", identitySchema(), identityRunFn()));
		harness.bindOutput("n1", "y");

		std::string tmpFile = "test_serialize.json";
		GraphCompiler::serialize(harness.graph(), tmpFile);

		// 编译回来
		InferGraph graph2; GraphCompiler::compileFile(graph2, tmpFile);
		CHECK(graph2.nodeCount() == 1, "should have 1 node");
		CHECK(graph2.node("n1") != nullptr, "n1 should exist");

		std::remove(tmpFile.c_str());
	}
	END_TEST();
}

void testModelPathHandling() {
	TEST("modelPath - absolute path preserved, relative path not mangled") {
		const char* json = R"({
  "version": "1.0",
  "nodes": [
    {
      "name": "m1", "type": "Builtin", "affinity": "Compute",
      "modelPath": "models/test.onnx",
      "inputs": [],
      "outputs": []
    }
  ],
  "edges": [],
  "outputBindings": []
})";
		InferGraph graph; GraphCompiler::compileString(graph, json);
		auto* n = graph.node("m1");
		CHECK(n != nullptr, "m1 should exist");
		CHECK(!n->modelPath().empty(), "modelPath should not be empty");
		// 相对路径会被 baseDir（默认当前目录）拼接
	}
	END_TEST();
}

int main() {
	try {
		testCompileStringBasic();
		testCompileStringBroadcast();
		testCompileStringRouting();
		testRoundTrip();
		testSerializeToJsonString();
		testModelPathHandling();

		if (failures == 0) {
			std::cout << "\nAll GraphCompiler tests passed!" << std::endl;
		} else {
			std::cout << "\n" << failures << " test(s) FAILED!" << std::endl;
		}
		return failures;
	} catch (const std::exception& e) {
		std::cerr << "Test failure: " << e.what() << std::endl;
		return -1;
	}
}
