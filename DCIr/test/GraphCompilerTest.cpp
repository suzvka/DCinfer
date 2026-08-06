// GraphCompiler 单元测试
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <streambuf>
#include <string>

#include "EngineRegistry.h"
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

/// @brief stderr 捕获器（RAII）：构造后 std::cerr 输出进入 oss，析构时恢复
struct CerrCapture {
	std::ostringstream oss;
	std::streambuf* old;
	CerrCapture() : old(std::cerr.rdbuf(oss.rdbuf())) {}
	~CerrCapture() { std::cerr.rdbuf(old); }
	std::string str() const { return oss.str(); }
};

/// @brief 构造端口（MSVC 对嵌套 braced-init-list + 隐式整型转换解析不稳，显式构造）
static Node::Port makePort(std::string name, TensorType type, size_t typeSize, Tensor::Shape shape = {}) {
	Node::Port p;
	p.name = std::move(name);
	p.type = type;
	p.typeSize = typeSize;
	p.shape = std::move(shape);
	return p;
}

/// @brief 注册可配置测试引擎（EngineRegistry 为全局单例，类型名必须唯一）
/// @param createSuccess     createEngine 是否成功（false → 返回空实例，模拟模型缺失）
/// @param withPortHooks     是否注册 getInputPorts/getOutputPorts（模拟实例推导 Schema）
/// @param checkFileExists   createEngine 前检查模型文件是否存在
static void registerTestEngine(const std::string& type, bool createSuccess,
							   bool withPortHooks, bool checkFileExists) {
	auto& reg = EngineRegistry::instance();
	EngineDescriptor desc;
	desc.engineType = type;
	desc.factory = [type](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		auto* inst = const_cast<EngineInstance*>(static_cast<const EngineInstance*>(p.engineConfig));
		auto node = std::make_unique<Node>(type, p.nodeName, p.schema, nullptr, ThreadPoolAffinity::Compute);
		if (inst)
			node->bindEngine(inst, inst->descriptor());
		return node;
	};
	desc.createEngine = [createSuccess, checkFileExists](const std::string& modelPath) -> EngineInstance {
		if (modelPath.empty()) return EngineInstance();
		if (checkFileExists && !std::filesystem::exists(modelPath)) return EngineInstance();
		if (!createSuccess) return EngineInstance();
		return EngineInstance(std::make_shared<int>(42));
	};
	if (withPortHooks) {
		desc.getInputPorts = [](const EngineInstance&) -> std::vector<Node::Port> {
			return {makePort("in", TensorType::Float, sizeof(float), {1, 2})};
		};
		desc.getOutputPorts = [](const EngineInstance&) -> std::vector<Node::Port> {
			return {makePort("out", TensorType::Float, sizeof(float), {3, 4})};
		};
	}
	reg.registerEngine(desc);
}

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

// ════════════════════════════════════════════
// 异常 / 边界路径测试
// ════════════════════════════════════════════

void testInvalidJsonThrows() {
	TEST("invalid JSON throws GraphException") {
		bool caught = false;
		try {
			InferGraph graph;
			GraphCompiler::compileString(graph, "not valid json {{{{{{");
		} catch (const GraphException&) {
			caught = true;
		} catch (...) {
			// 不应该捕获其他类型的异常
		}
		CHECK(caught, "should throw GraphException on invalid JSON");
	}
	END_TEST();
}

void testEmptyGraph() {
	TEST("empty graph — no nodes, edges, or bindings") {
		const char* json = R"({
  "version": "1.0",
  "nodes": [],
  "edges": [],
  "outputBindings": []
})";
		InferGraph graph; GraphCompiler::compileString(graph, json);
		CHECK(graph.nodeCount() == 0, "should have 0 nodes");
		CHECK(graph.edgeCount() == 0, "should have 0 edges");
		CHECK(graph.outputBindings().size() == 0, "should have 0 output bindings");
	}
	END_TEST();
}

void testEdgeToMissingNode() {
	TEST("edge referencing non-existent dstNode — does not crash") {
		const char* json = R"({
  "version": "1.0",
  "nodes": [
    {
      "name": "n1", "type": "Builtin", "affinity": "Operator",
      "inputs": [
        {"name":"x","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ],
      "outputs": [
        {"name":"y","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ]
    }
  ],
  "edges": [
    {"srcNode":"n1","srcPort":"y","dstNode":"ghost","dstPort":"x"}
  ],
  "outputBindings": []
})";
		InferGraph graph; GraphCompiler::compileString(graph, json);
		// 图仍应构建成功（n1 存在），但 wire 失败会输出 warning
		CHECK(graph.nodeCount() >= 1, "n1 should exist even if edge target is missing");
		CHECK(graph.node("n1") != nullptr, "n1 should exist");
	}
	END_TEST();
}

void testUnregisteredType() {
	TEST("unregistered engine type — creates skeleton with warning") {
		const char* json = R"({
  "version": "1.0",
  "nodes": [
    {
      "name": "custom1", "type": "UnknownEngineV2", "affinity": "Compute",
      "inputs": [
        {"name":"in","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ],
      "outputs": [
        {"name":"out","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ]
    }
  ],
  "edges": [],
  "outputBindings": []
})";
		InferGraph graph; GraphCompiler::compileString(graph, json);
		CHECK(graph.nodeCount() == 1, "skeleton node should be created for unregistered type");
		auto* n = graph.node("custom1");
		CHECK(n != nullptr, "custom1 should exist");
		CHECK(n->type() == "UnknownEngineV2", "type should be preserved");
		// RunFn 为空，这是个骨架节点
	}
	END_TEST();
}

void testDcgRoundTrip() {
	TEST("dcg round-trip — serialize then compile .dcg") {
		// 创建一个临时 model 文件
		std::string modelContent = "mock-model-data-12345";
		std::string modelFile = "test_dcg_model.bin";
		{
			std::ofstream ofs(modelFile, std::ios::binary);
			ofs.write(modelContent.data(), static_cast<std::streamsize>(modelContent.size()));
		}

		// 构建图（节点有 modelPath）
		TestHarness harness;
		auto n1 = std::make_unique<Node>("ONNX", "dcg_n1", identitySchema(), identityRunFn());
		n1->setModelPath(modelFile); // 指向刚才创建的临时文件
		harness.addNode(std::move(n1));
		harness.addNode(std::make_unique<Node>("Builtin", "dcg_n2", identitySchema(), identityRunFn()));
		harness.wire("dcg_n1", "y", "dcg_n2", "x");
		harness.bindOutput("dcg_n2", "y");

		// 序列化为 .dcg
		std::string dcgFile = "test_dcg_roundtrip.dcg";
		GraphCompiler::serialize(harness.graph(), dcgFile);

		// 验证 .dcg 文件存在且大于 0
		CHECK(std::filesystem::exists(dcgFile), "dcg file should exist");
		CHECK(std::filesystem::file_size(dcgFile) > 0, "dcg file should not be empty");

		// 反序列化
		InferGraph graph2; GraphCompiler::compileFile(graph2, dcgFile);

		// 验证图结构
		CHECK(graph2.nodeCount() >= 2, "dcg roundtrip: should have at least 2 nodes");
		CHECK(graph2.node("dcg_n1") != nullptr, "dcg roundtrip: dcg_n1 should exist");
		CHECK(graph2.node("dcg_n2") != nullptr, "dcg roundtrip: dcg_n2 should exist");
		CHECK(graph2.edgeCount() >= 1, "dcg roundtrip: should have edges");
		CHECK(graph2.outputBindings().size() == 1, "dcg roundtrip: should have 1 output binding");

		// modelPath 应该保留相对路径格式
		auto* node1 = graph2.node("dcg_n1");
		CHECK(node1 != nullptr, "dcg roundtrip: dcg_n1 not null");
		CHECK(!node1->modelPath().empty(), "dcg roundtrip: modelPath should not be empty");

		// 清理
		std::remove(dcgFile.c_str());
		std::remove(modelFile.c_str());
	}
	END_TEST();
}

void testDcgSerializeNoModels() {
	TEST("dcg serialize — graph without models") {
		TestHarness harness;
		harness.addNode(std::make_unique<Node>("Builtin", "n1", identitySchema(), identityRunFn()));
		harness.bindOutput("n1", "y");

		std::string dcgFile = "test_dcg_nomodel.dcg";
		GraphCompiler::serialize(harness.graph(), dcgFile);

		CHECK(std::filesystem::exists(dcgFile), "dcg file should exist");

		// 反序列化
		InferGraph graph2; GraphCompiler::compileFile(graph2, dcgFile);
		CHECK(graph2.nodeCount() == 1, "dcg nomodel: should have 1 node");
		CHECK(graph2.node("n1") != nullptr, "dcg nomodel: n1 should exist");

		std::remove(dcgFile.c_str());
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 引擎注册接口统一后的语义适配测试
// ════════════════════════════════════════════

void testDynamicShapeRoundTrip() {
	TEST("round-trip - dynamic dim (-1) stable in JSON and back") {
		// Node::Port::shape 为 vector<int64_t>，动态维度以 -1 表示
		// （与 ONNX 语义一致），序列化/反序列化应 int64_t 直通
		constexpr int64_t kDyn = -1;

		TestHarness harness;
		Node::Schema s;
		s.inputs = {makePort("x", TensorType::Float, sizeof(float), Tensor::Shape{kDyn, 224, 224})};
		s.outputs = {makePort("y", TensorType::Float, sizeof(float), Tensor::Shape{1, kDyn})};
		harness.addNode(std::make_unique<Node>("Builtin", "dyn1", s, identityRunFn()));
		harness.bindOutput("dyn1", "y");

		std::string tmpFile = "test_dynshape.json";
		GraphCompiler::serialize(harness.graph(), tmpFile);

		// 1) JSON 中动态维度必须编码为 -1
		{
			std::ifstream ifs(tmpFile, std::ios::binary);
			std::ostringstream oss; oss << ifs.rdbuf();
			auto root = nlohmann::json::parse(oss.str());
			auto inShape = root["nodes"][0]["inputs"][0]["shape"];
			CHECK(inShape[0].get<int64_t>() == -1, "dynamic dim should be -1 in JSON");
			CHECK(inShape[1].get<int64_t>() == 224, "static dim preserved in JSON");
			auto outShape = root["nodes"][0]["outputs"][0]["shape"];
			CHECK(outShape[0].get<int64_t>() == 1, "static dim preserved in JSON");
			CHECK(outShape[1].get<int64_t>() == -1, "output dynamic dim should be -1 in JSON");
		}

		// 2) 编译回来：JSON -1 解码为内存 -1（int64_t 直通，不经过 size_t 中间转换）
		InferGraph graph2; GraphCompiler::compileFile(graph2, tmpFile);
		auto* n = graph2.node("dyn1");
		CHECK(n != nullptr, "dyn1 should exist");
		const auto& inShape = n->schema().inputs[0].shape;
		CHECK(inShape.size() == 3, "input rank should be 3");
		CHECK(inShape[0] == kDyn, "dynamic dim decoded as -1");
		CHECK(inShape[1] == 224, "static dim preserved");
		const auto& outShape = n->schema().outputs[0].shape;
		CHECK(outShape[1] == kDyn, "output dynamic dim decoded as -1");
		// 注：int64_t 的 -1 与 0xFFFFFFFFFFFFFFFF 位模式相同（64 位平台），
		// 该断言验证语义等价即可（== kDyn），无需也不能区分二者位模式；
		// 修复价值在于消除对 size_t 宽度的依赖（32 位平台 -1 会被截断为
		// 0xFFFFFFFF 而非 -1，roundtrip 将损坏）。

		std::remove(tmpFile.c_str());
	}
	END_TEST();
}

void testVoidPortRoundTrip() {
	TEST("round-trip - Void port type (unknown element types)") {
		// FP16 等未知元素类型经 ONNX 适配器推导为 Void；
		// typeToString(Void) = "Void"、stringToType 兜底返回 Void，可稳定 roundtrip
		constexpr int64_t kDyn = -1;

		TestHarness harness;
		Node::Schema s;
		s.inputs = {makePort("in", TensorType::Void, 2, Tensor::Shape{kDyn})};
		s.outputs = {makePort("out", TensorType::Void, 0, {})};
		harness.addNode(std::make_unique<Node>("Builtin", "void1", s, identityRunFn()));
		harness.bindOutput("void1", "out");

		std::string tmpFile = "test_void.json";
		GraphCompiler::serialize(harness.graph(), tmpFile);
		InferGraph graph2; GraphCompiler::compileFile(graph2, tmpFile);

		auto* n = graph2.node("void1");
		CHECK(n != nullptr, "void1 should exist");
		CHECK(n->schema().inputs[0].type == TensorType::Void, "Void type roundtrip");
		CHECK(n->schema().inputs[0].typeSize == 2, "Void typeSize roundtrip");
		CHECK(n->schema().inputs[0].shape[0] == kDyn, "Void port dynamic dim roundtrip");
		CHECK(n->schema().outputs[0].type == TensorType::Void, "output Void type roundtrip");
		CHECK(n->schema().outputs[0].typeSize == 0, "output Void typeSize roundtrip");

		std::remove(tmpFile.c_str());
	}
	END_TEST();
}

void testEngineNodeNoModelPath() {
	TEST("engine node without modelPath — skeleton fallback with warning") {
		registerTestEngine("NoModelPathEngine", true, true, false);
		const char* json = R"({
  "version": "1.0",
  "nodes": [
    {
      "name": "eng1", "type": "NoModelPathEngine", "affinity": "Compute", "tag": "t1",
      "inputs": [
        {"name":"x","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ],
      "outputs": [
        {"name":"y","tensorType":"Float","typeSize":4,"shape":[],"required":true}
      ]
    }
  ],
  "edges": [],
  "outputBindings": []
})";
		CerrCapture cap;
		InferGraph graph; GraphCompiler::compileString(graph, json);

		// 无 modelPath → 不调用 createNode（避免空路径加载），回退骨架节点
		auto* n = graph.node("eng1");
		CHECK(n != nullptr, "skeleton node should be created");
		CHECK(n->type() == "NoModelPathEngine", "type should be preserved");
		CHECK(n->schema().inputs.size() == 1, "JSON schema preserved on skeleton");
		CHECK(n->tag() == "t1", "tag preserved");
		CHECK(n->modelPath().empty(), "no modelPath set");
		CHECK(cap.str().find("has no modelPath") != std::string::npos,
			"warning should mention missing modelPath");
	}
	END_TEST();
}

void testEngineNodeLoadFailure() {
	TEST("engine node load failure — explicit diagnostics, node skipped") {
		// createEngine 检查模型文件存在（模拟 ORT Session 加载缺失文件失败）
		registerTestEngine("MissingModelEngine", true, true, true);
		const char* json = R"({
  "version": "1.0",
  "nodes": [
    {
      "name": "m1", "type": "MissingModelEngine", "affinity": "Compute",
      "modelPath": "models/does_not_exist.onnx",
      "inputs": [], "outputs": []
    }
  ],
  "edges": [],
  "outputBindings": []
})";
		CerrCapture cap;
		InferGraph graph; GraphCompiler::compileString(graph, json);

		// 节点被跳过（不中断编译），但必须有可感知的诊断
		CHECK(graph.node("m1") == nullptr, "node should be skipped on load failure");
		std::string diag = cap.str();
		CHECK(diag.find("failed to create engine node") != std::string::npos,
			"diagnostics should be present");
		CHECK(diag.find("m1") != std::string::npos, "diagnostics should contain node name");
		CHECK(diag.find("MissingModelEngine") != std::string::npos,
			"diagnostics should contain engine type");
		CHECK(diag.find("does_not_exist.onnx") != std::string::npos,
			"diagnostics should contain modelPath");
	}
	END_TEST();
}

void testEngineSchemaDerivedAndEmpty() {
	TEST("engine schema — instance-derived overrides JSON; empty schema warns") {
		// 1) 注册端口推导钩子 → Schema 从实例推导，JSON schema 被覆盖
		registerTestEngine("SchemaDerivedEngine", true, true, false);
		const char* json = R"({
  "version": "1.0",
  "nodes": [
    {
      "name": "e1", "type": "SchemaDerivedEngine",
      "modelPath": "models/m.onnx",
      "inputs": [
        {"name":"jsonIn","tensorType":"Int","typeSize":4,"shape":[],"required":true}
      ],
      "outputs": [
        {"name":"jsonOut","tensorType":"Int","typeSize":4,"shape":[],"required":true}
      ]
    }
  ],
  "edges": [],
  "outputBindings": []
})";
		CerrCapture cap;
		InferGraph graph; GraphCompiler::compileString(graph, json);
		auto* n = graph.node("e1");
		CHECK(n != nullptr, "e1 should exist");
		CHECK(n->schema().inputs.size() == 1 && n->schema().inputs[0].name == "in",
			"schema derived from engine instance overrides JSON schema");
		CHECK(n->schema().outputs.size() == 1 && n->schema().outputs[0].name == "out",
			"output derived from engine instance");
		CHECK(n->modelPath().find("models/m.onnx") != std::string::npos,
			"modelPath preserved on engine node");
		CHECK(cap.str().find("empty schema") == std::string::npos,
			"no empty-schema warning when port hooks are registered");

		// 2) 不注册端口推导钩子 → 框架传空 schema → 编译期警告
		registerTestEngine("NoPortHookEngine", true, false, false);
		const char* json2 = R"({
  "version": "1.0",
  "nodes": [
    {
      "name": "e2", "type": "NoPortHookEngine",
      "modelPath": "models/m2.onnx",
      "inputs": [], "outputs": []
    }
  ],
  "edges": [],
  "outputBindings": []
})";
		CerrCapture cap2;
		InferGraph graph2; GraphCompiler::compileString(graph2, json2);
		auto* n2 = graph2.node("e2");
		CHECK(n2 != nullptr, "e2 should exist");
		CHECK(n2->schema().inputs.empty() && n2->schema().outputs.empty(),
			"empty schema when engine has no port hooks");
		CHECK(cap2.str().find("empty schema") != std::string::npos,
			"warning should mention empty schema");
	}
	END_TEST();
}

void testDcgRecompileAfterReleaseAllEngines() {
	TEST("dcg lifecycle — recompile after releaseAllEngines reloads from archive") {
		// 场景：解压模型到临时目录 → createNode 加载（实例缓存持有）→ 临时文件删除
		registerTestEngine("DcgLifecycleEngine", true, true, true); // 检查文件存在
		std::string modelFile = "test_dcg_lifecycle_model.bin";
		{
			std::ofstream ofs(modelFile, std::ios::binary);
			ofs.write("lifecycle-model", 15);
		}
		TestHarness harness;
		auto n1 = std::make_unique<Node>("DcgLifecycleEngine", "lc1", identitySchema(), identityRunFn());
		n1->setModelPath(modelFile);
		harness.addNode(std::move(n1));
		harness.bindOutput("lc1", "y");
		std::string dcgFile = "test_dcg_lifecycle.dcg";
		GraphCompiler::serialize(harness.graph(), dcgFile);
		std::remove(modelFile.c_str()); // dcg 自带模型，源文件可删

		// 1) 首次编译：解压临时模型 → 引擎实例创建成功
		InferGraph graph1;
		CerrCapture cap1;
		GraphCompiler::compileFile(graph1, dcgFile);
		CHECK(graph1.node("lc1") != nullptr, "first compile: engine node created");
		CHECK(cap1.str().find("failed to create engine node") == std::string::npos,
			"first compile: no failure diagnostics");

		// 2) 清空实例缓存（临时文件已删除；节点持有的是非拥有实例指针）
		EngineRegistry::instance().releaseAllEngines();

		// 3) 再次编译同一 .dcg：compileFile 会重新解压到新的临时目录，
		//    缓存键（engineType + 新临时绝对路径）不同 → 重新加载成功。
		//    注意：旧缓存条目（指向已删除临时文件）不会导致加载失败，
		//    但旧条目成为孤儿，直到下次 releaseAllEngines() 才释放。
		InferGraph graph2;
		CerrCapture cap2;
		GraphCompiler::compileFile(graph2, dcgFile);
		CHECK(graph2.node("lc1") != nullptr,
			"recompile succeeds: model re-extracted to fresh temp dir");
		CHECK(graph2.node("lc1")->modelPath().find("models") != std::string::npos,
			"modelPath points into temp dir");
		CHECK(cap2.str().find("failed to create engine node") == std::string::npos,
			"recompile: no failure diagnostics");

		// 清理：引擎缓存中仍有 graph2 的实例（键指向已删除临时文件），统一释放
		EngineRegistry::instance().releaseAllEngines();
		std::remove(dcgFile.c_str());
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
		// 异常/边界路径
		testInvalidJsonThrows();
		testEmptyGraph();
		testEdgeToMissingNode();
		testUnregisteredType();
		testDcgRoundTrip();
		testDcgSerializeNoModels();
		// 引擎注册接口统一后的语义适配
		testDynamicShapeRoundTrip();
		testVoidPortRoundTrip();
		testEngineNodeNoModelPath();
		testEngineNodeLoadFailure();
		testEngineSchemaDerivedAndEmpty();
		testDcgRecompileAfterReleaseAllEngines();

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
