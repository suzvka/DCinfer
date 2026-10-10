// GraphCompiler 单元测试
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>

#include "EngineRegistry.h"
#include "Ir/DcgArchive.h"
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

/// @brief 构造 Void 端口（FP16 等未知元素类型：无 C++ 类型载体，
///        NodePort::in/out 工厂不适用；MSVC 对嵌套 braced-init-list +
///        隐式整型转换解析不稳，故保留显式构造辅助）
static Node::Port voidPort(std::string name, size_t typeSize, Tensor::Shape shape = {}) {
	Node::Port p;
	p.name = std::move(name);
	p.type = TensorType::Void;
	p.typeSize = typeSize;
	p.shape = std::move(shape);
	return p;
}

/// @brief 测试引擎调用计数（验证“编译期零引擎调用”）：
///        g_coreInitCalls = createEngineCore（引擎级初始化）调用次数；
///        g_modelLoadCalls = loadModel（模型级加载）调用次数
static std::atomic<int> g_coreInitCalls{0};
static std::atomic<int> g_modelLoadCalls{0};

/// @brief 注册可配置测试引擎（EngineRegistry 为全局单例，类型名必须唯一）
/// @param createSuccess     loadModel 是否成功（false → 返回空实例）
/// @param withPortHooks     是否注册实例端口推导钩子（验证延迟物化不触发推导）
/// @param checkFileExists   loadModel 前检查模型文件是否存在
/// @param throwOnCreate     loadModel 抛异常（模拟真实 ORT 加载失败行为，
///                          见 OnnxEngine.cpp：模型不可加载时抛 std::runtime_error）
static void registerTestEngine(const std::string& type, bool createSuccess,
							   bool withPortHooks, bool checkFileExists,
							   bool throwOnCreate = false) {
	auto& reg = EngineRegistry::instance();
	EngineDescriptor desc;
	desc.engineType = type;
	desc.factory = [type](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		// factory 提供真 RunFn：物化节点应可执行（未绑定实例时由引擎 RunFn 自行判错）
		auto node = std::make_unique<Node>(type, p.nodeName, p.schema,
										   [](Node::RunContext& ctx) { return ctx.success(); },
										   ResourceClass::Compute);
		if (p.engineInstance)
			node->bindEngine(p.engineInstance, p.engineInstance->descriptor());
		return node;
	};
	desc.createEngineCore = []() -> EngineCore {
		++g_coreInitCalls;
		return EngineCore(std::make_shared<int>(7)); // 引擎级占位对象（Env 语义）
	};
	desc.loadModel = [createSuccess, checkFileExists, throwOnCreate](
						 const EngineCore&, const std::string& modelPath) -> EngineInstance {
		++g_modelLoadCalls;
		if (throwOnCreate)
			throw std::runtime_error("simulated engine load failure for '" + modelPath + "'");
		if (modelPath.empty()) return EngineInstance();
		if (checkFileExists && !std::filesystem::exists(modelPath)) return EngineInstance();
		if (!createSuccess) return EngineInstance();
		return EngineInstance(std::make_shared<int>(42));
	};
	if (withPortHooks) {
		desc.getInputPorts = [](const EngineInstance&) -> std::vector<Node::Port> {
			return {Node::Port::in<float>("in", {1, 2})};
		};
		desc.getOutputPorts = [](const EngineInstance&) -> std::vector<Node::Port> {
			return {Node::Port::out<float>("out", {3, 4})};
		};
	}
	reg.registerEngine(desc);
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
		if (!t) return ctx.failure(Node::Status::InvalidInput, "not a Tensor");
		ctx.output("y", Value(std::make_unique<Tensor>(*t)));
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

		// 3 业务节点 + 1 broadcast 连接器 + 3 根包裹导线（重建经 connect 自动插入，
		// 序列化折叠后不可见，lowering 擦除后运行时视图不变）= 7
		CHECK(graph.nodeCount() == 7, "should have 7 nodes (3 biz + 1 bc + 3 wrapping wires)");
		CHECK(graph.edgeCount() == 6, "should have 6 edges (3 connects × 2 edges each)");
		CHECK(graph.outputBindings().size() == 2, "should have 2 output bindings");
		CHECK(graph.node("add1") != nullptr, "add1 should exist");
		CHECK(graph.node("id_a") != nullptr, "id_a should exist");
		CHECK(graph.node("id_b") != nullptr, "id_b should exist");
	}
	END_TEST();
}

void testCompileStringRoutingRejected() {
	TEST("compileString - routing mode edge is rejected with explicit error") {
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
		InferGraph graph;
		bool threw = false;
		try {
			GraphCompiler::compileString(graph, json);
		} catch (const GraphException&) {
			threw = true;
		}
		CHECK(threw, "routing mode should be rejected with GraphException");
	}
	END_TEST();
}

void testRoundTrip() {
	TEST("round-trip - serialize then compile") {
		// 构建图
		TestHarness harness;
		harness.addNode(std::make_unique<Node>("ONNX", "test1", identitySchema(), identityRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "test2", identitySchema(), identityRunFn()));
		harness.connect("test1", "y", "test2", "x");
		harness.bindOutput("y", "test2", "y");
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

		// modelPath 原样透传
		auto* n1 = graph2.node("test1");
		CHECK(n1 != nullptr, "roundtrip: test1 not null");
		CHECK(n1->modelPath() == "models/test.onnx", "roundtrip: modelPath must be preserved verbatim");

		// 清理
		std::remove(tmpFile.c_str());
	}
	END_TEST();
}

void testExpandedFanOutRoundTrip() {
	TEST("round-trip - auto-expanded fan-out: broadcast edges and re-serialization stability") {
		// 构建：同一输出口两次 connect → 导线自动扩容为广播扇出
		TestHarness harness;
		harness.addNode(std::make_unique<Node>("Builtin", "src", identitySchema(), identityRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "b", identitySchema(), identityRunFn()));
		harness.addNode(std::make_unique<Node>("Builtin", "c", identitySchema(), identityRunFn()));
		harness.connect("src", "y", "b", "x");
		harness.connect("src", "y", "c", "x"); // 自动扩容
		harness.bindOutput("y1", "b", "y");
		harness.bindOutput("y2", "c", "y");
		CHECK(harness.graph().nodeCount() == 4, "source: 3 biz + 1 expanded wire");

		const std::string f1 = "test_expanded_fanout_1.json";
		GraphCompiler::serialize(harness.graph(), f1);

		// 1) IR1：两条同源 mode=broadcast 逻辑边（连接器折叠）
		{
			std::ifstream ifs(f1, std::ios::binary);
			std::ostringstream oss;
			oss << ifs.rdbuf();
			auto root = nlohmann::json::parse(oss.str());
			CHECK(root["edges"].size() == 2, "IR1: 2 logical edges");
			for (auto& e : root["edges"]) {
				CHECK(e.value("mode", "") == "broadcast", "IR1: edge carries mode:broadcast");
				CHECK(e["srcNode"].get<std::string>() == "src", "IR1: edge source is the business node");
			}
		}

		// 2) 读回：重建为显式 Broadcast + 包裹导线（3 biz + 1 bc + 3 wires = 7）
		InferGraph graph2;
		GraphCompiler::compileFile(graph2, f1);
		CHECK(graph2.nodeCount() == 7, "rebuild: 3 biz + 1 bc + 3 wrapping wires");
		CHECK(graph2.edgeCount() == 6, "rebuild: 6 edges");

		// 3) 二次导出：穿透连接器链折叠，逻辑等价（round-trip 闭环）
		const std::string f2 = "test_expanded_fanout_2.json";
		GraphCompiler::serialize(graph2, f2);
		{
			std::ifstream ifs(f2, std::ios::binary);
			std::ostringstream oss;
			oss << ifs.rdbuf();
			auto root = nlohmann::json::parse(oss.str());
			CHECK(root["edges"].size() == 2, "IR2: 2 logical edges (chain folded through connectors)");
			for (auto& e : root["edges"]) {
				CHECK(e.value("mode", "") == "broadcast", "IR2: edge carries mode:broadcast");
				std::string dst = e["dstNode"].get<std::string>();
				CHECK(dst == "b" || dst == "c", "IR2: dst is a business node (no internal names leak)");
			}
		}

		std::remove(f1.c_str());
		std::remove(f2.c_str());
	}
	END_TEST();
}

void testSerializeToJsonString() {
	TEST("serialize - JSON output is valid and parsable") {
		TestHarness harness;
		harness.addNode(std::make_unique<Node>("Builtin", "n1", identitySchema(), identityRunFn()));
		harness.bindOutput("y", "n1", "y");

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
	TEST("modelPath - passed through verbatim (no path interpretation)") {
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
		CHECK(n->modelPath() == "models/test.onnx",
			"modelPath must be preserved verbatim (no baseDir concatenation)");
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
	TEST("edge referencing non-existent dstNode — fails fast with GraphException") {
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
		// IR-07：连接失败必须 fail-fast（stderr 告警 + 图残缺继续属静默错误），
		// 反序列化 fail-fast，不产生孤儿连接器
		bool rejected = false;
		try {
			InferGraph graph;
			GraphCompiler::compileString(graph, json);
		} catch (const GraphException&) {
			rejected = true;
		}
		CHECK(rejected, "edge to missing node must fail the compile (fail-fast)");
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
		harness.connect("dcg_n1", "y", "dcg_n2", "x");
		harness.bindOutput("y", "dcg_n2", "y");

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

		// modelPath 原样透传（归档内相对路径，不拼接临时目录）
		auto* node1 = graph2.node("dcg_n1");
		CHECK(node1 != nullptr, "dcg roundtrip: dcg_n1 not null");
		CHECK(node1->modelPath() == "models/test_dcg_model.bin",
			"dcg roundtrip: modelPath must stay archive-relative verbatim");

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
		harness.bindOutput("y", "n1", "y");

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
		s.inputs = {Node::Port::in<float>("x", Tensor::Shape{kDyn, 224, 224})};
		s.outputs = {Node::Port::out<float>("y", Tensor::Shape{1, kDyn})};
		harness.addNode(std::make_unique<Node>("Builtin", "dyn1", s, identityRunFn()));
		harness.bindOutput("y", "dyn1", "y");

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
		s.inputs = {voidPort("in", 2, Tensor::Shape{kDyn})};
		s.outputs = {voidPort("out", 0, {})};
		harness.addNode(std::make_unique<Node>("Builtin", "void1", s, identityRunFn()));
		harness.bindOutput("out", "void1", "out");

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

void testEngineNodeMaterialization() {
	TEST("engine node — declared schema + RunFn materialized without loading model") {
		g_coreInitCalls = 0;
		g_modelLoadCalls = 0;
		registerTestEngine("MaterializeEngine", true, true, false);
		const char* json = R"({
  "version": "1.0",
  "nodes": [
    {
      "name": "eng1", "type": "MaterializeEngine", "affinity": "Compute", "tag": "t1",
      "modelPath": "models/m.onnx",
      "inputs": [
        {"name":"declaredIn","tensorType":"Int","typeSize":4,"shape":[],"required":true}
      ],
      "outputs": [
        {"name":"declaredOut","tensorType":"Int","typeSize":4,"shape":[],"required":true}
      ]
    }
  ],
  "edges": [],
  "outputBindings": []
})";
		CerrCapture cap;
		InferGraph graph; GraphCompiler::compileString(graph, json);
		auto* n = graph.node("eng1");
		CHECK(n != nullptr, "engine node should be materialized");
		// 声明 schema 保留：不触发实例推导（引擎端口钩子产出 in/out，声明为 declaredIn/Out）
		CHECK(n->schema().inputs.size() == 1 && n->schema().inputs[0].name == "declaredIn",
			"JSON declared schema must be preserved (no instance derivation)");
		CHECK(n->schema().outputs.size() == 1 && n->schema().outputs[0].name == "declaredOut",
			"JSON declared output schema must be preserved");
		// 工厂提供的引擎 RunFn 已注入：可执行物化节点（非骨架）
		CHECK(static_cast<bool>(n->runFn()), "factory-provided RunFn must be present");
		// modelPath 原样透传
		CHECK(n->modelPath() == "models/m.onnx", "modelPath must be passed through verbatim");
		CHECK(n->tag() == "t1", "tag preserved");
		CHECK(cap.str().find("empty declared schema") == std::string::npos,
			"no empty-schema warning for non-empty declaration");
		// 编译期零引擎调用：createEngineCore / loadModel 从未被调用
		CHECK(g_coreInitCalls == 0, "createEngineCore must not be invoked at compile time");
		CHECK(g_modelLoadCalls == 0, "loadModel must not be invoked at compile time");
	}
	END_TEST();
}

void testEngineNodeNoModelPath() {
	TEST("engine node without modelPath — materialized as declarative node, no warning") {
		g_coreInitCalls = 0;
		g_modelLoadCalls = 0;
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
		auto* n = graph.node("eng1");
		CHECK(n != nullptr, "node should be materialized (no skeleton fallback)");
		CHECK(n->type() == "NoModelPathEngine", "type preserved");
		CHECK(n->schema().inputs.size() == 1, "declared schema preserved");
		CHECK(n->tag() == "t1", "tag preserved");
		CHECK(n->modelPath().empty(), "no modelPath set");
		CHECK(static_cast<bool>(n->runFn()), "RunFn present (executable materialization)");
		CHECK(cap.str().find("warning") == std::string::npos, "no warnings for a valid declaration");
		CHECK(g_coreInitCalls == 0, "createEngineCore must not be invoked at compile time");
		CHECK(g_modelLoadCalls == 0, "loadModel must not be invoked at compile time");
	}
	END_TEST();
}

void testCompileNeverInvokesCreateEngine() {
	TEST("compile never invokes createEngineCore / loadModel — URL modelPath (issue scenario)") {
		// URL modelPath 不可加载：编译期一旦触发加载，真实 ORT 引擎会抛异常并
		// 中断整个编译——加载必须完全推迟到宿主绑定/执行期。
		g_coreInitCalls = 0;
		g_modelLoadCalls = 0;
		registerTestEngine("ThrowOnLoadEngine", true, true, false, /*throwOnCreate=*/true);
		const char* json = R"({
  "version": "1.0",
  "nodes": [
    {
      "name": "m", "type": "ThrowOnLoadEngine", "affinity": "Compute",
      "modelPath": "https://obj.example.com/mnist-12.onnx",
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
		InferGraph graph;
		bool threw = false;
		try {
			GraphCompiler::compileString(graph, json);
		} catch (const std::exception&) {
			threw = true;
		}
		CHECK(!threw, "compile must not trigger engine load");
		auto* n = graph.node("m");
		CHECK(n != nullptr, "URL-modelPath engine node must be materialized");
		CHECK(n->modelPath() == "https://obj.example.com/mnist-12.onnx",
			"URL must be passed through verbatim (no baseDir concatenation)");
		CHECK(g_coreInitCalls == 0, "createEngineCore must never be invoked at compile time");
		CHECK(g_modelLoadCalls == 0, "loadModel must never be invoked at compile time");
	}
	END_TEST();
}

void testEngineDeclaredSchemaEmptyWarns() {
	TEST("engine node — empty declared schema warns") {
		registerTestEngine("NoPortDeclEngine", true, false, false);
		const char* json = R"({
  "version": "1.0",
  "nodes": [
    {
      "name": "e2", "type": "NoPortDeclEngine", "affinity": "Compute",
      "modelPath": "models/m2.onnx",
      "inputs": [], "outputs": []
    }
  ],
  "edges": [],
  "outputBindings": []
})";
		CerrCapture cap;
		InferGraph graph2; GraphCompiler::compileString(graph2, json);
		auto* n2 = graph2.node("e2");
		CHECK(n2 != nullptr, "e2 should exist");
		CHECK(n2->schema().inputs.empty() && n2->schema().outputs.empty(),
			"empty declared schema stays empty (no derivation)");
		CHECK(cap.str().find("empty declared schema") != std::string::npos,
			"warning should mention empty declared schema");
	}
	END_TEST();
}

void testDcgCompileDefersModelResolution() {
	TEST("dcg compile — no extraction, no engine load; modelPath stays archive-relative") {
		g_coreInitCalls = 0;
		g_modelLoadCalls = 0;
		registerTestEngine("DcgDeferEngine", true, true, false);
		std::string modelFile = "test_dcg_defer_model.bin";
		{
			std::ofstream ofs(modelFile, std::ios::binary);
			ofs.write("defer-model", 11);
		}
		TestHarness harness;
		auto n1 = std::make_unique<Node>("DcgDeferEngine", "dc_n1", identitySchema(), identityRunFn());
		n1->setModelPath(modelFile);
		harness.addNode(std::move(n1));
		harness.bindOutput("y", "dc_n1", "y");
		std::string dcgFile = "test_dcg_defer.dcg";
		GraphCompiler::serialize(harness.graph(), dcgFile);
		std::remove(modelFile.c_str()); // 模型随归档分发，源文件删除不影响编译

		InferGraph graph;
		GraphCompiler::compileFile(graph, dcgFile);
		auto* n = graph.node("dc_n1");
		CHECK(n != nullptr, "engine node must be materialized from .dcg");
		CHECK(static_cast<bool>(n->runFn()), "materialized node carries engine RunFn");
		CHECK(n->modelPath() == "models/test_dcg_defer_model.bin",
			"modelPath must stay archive-relative (no temp-dir rewriting)");
		CHECK(g_coreInitCalls == 0, "no engine core init at compile time");
		CHECK(g_modelLoadCalls == 0, "no engine load at compile time");

		std::remove(dcgFile.c_str());
	}
	END_TEST();
}

// ════════════════════════════════════════════
// IR-01：共享模型文件的多节点 .dcg 序列化
// ════════════════════════════════════════════

void testSharedModelDcgRoundTrip() {
	TEST("IR-01: two nodes sharing one model file round-trip through .dcg") {
		std::string modelFile = "test_shared_model.bin";
		{
			std::ofstream ofs(modelFile, std::ios::binary);
			ofs.write("shared-weights-payload", 21);
		}

		// 两个节点引用同一模型文件（共享权重场景）
		TestHarness harness;
		auto na = std::make_unique<Node>("ONNX", "shared_a", identitySchema(), identityRunFn());
		na->setModelPath(modelFile);
		auto nb = std::make_unique<Node>("ONNX", "shared_b", identitySchema(), identityRunFn());
		nb->setModelPath(modelFile);
		harness.addNode(std::move(na));
		harness.addNode(std::move(nb));

		std::string dcgFile = "test_shared_model.dcg";
		GraphCompiler::serialize(harness.graph(), dcgFile);

		// 反序列化必须成功（重复条目二次改名覆盖记录 → 首节点引用悬空、编译必然失败）
		InferGraph graph2;
		GraphCompiler::compileFile(graph2, dcgFile);
		auto* a = graph2.node("shared_a");
		auto* b = graph2.node("shared_b");
		CHECK(a != nullptr && b != nullptr, "both shared-model nodes must exist");
		CHECK(!a->modelPath().empty() && !b->modelPath().empty(), "both modelPaths must resolve");
		CHECK(a->modelPath() == "models/test_shared_model.bin" && b->modelPath() == a->modelPath(),
			"shared nodes must keep the same archive-relative path verbatim");

		std::remove(dcgFile.c_str());
		std::remove(modelFile.c_str());
	}
	END_TEST();
}

// ════════════════════════════════════════════
// IR-02：.dcg 反序列化安全不变量
// ════════════════════════════════════════════

void testDcgObjectShapedNodesRejected() {
	TEST("IR-02: object-shaped 'nodes' in .dcg is rejected") {
		std::string dcgFile = "test_object_nodes.dcg";
		{
			auto w = DC::Ir::DcgArchive::openWrite(dcgFile);
			// 对象形状 nodes：曾整体绕过 modelPath 校验与解压
			w->writeGraphJson(R"({"version":"1.0","nodes":{"a":{"name":"a","type":"Builtin","affinity":"Compute","inputs":[],"outputs":[]}},"edges":[],"outputBindings":[]})");
			w->finalize();
		}

		bool rejected = false;
		try {
			InferGraph graph;
			GraphCompiler::compileFile(graph, dcgFile);
		} catch (const GraphException&) {
			rejected = true;
		}
		CHECK(rejected, "object-shaped nodes must be rejected");

		std::remove(dcgFile.c_str());
	}
	END_TEST();
}

void testDcgModelPathPassThrough() {
	TEST("IR-02': .dcg modelPath passed through verbatim at compile (landing防御 at extractOne)") {
		// 编译期不落盘→不做路径校验，modelPath 为不透明字符串原样保留。
		// 实际落盘防御（越界/ADS/符号链接/预算）由 DcgArchive::extractOne 承担
		// （DcgArchiveSecurityTest 覆盖），宿主在绑定期/执行期消费时仍受保护。
		auto makeDcg = [](const std::string& dcgFile, const std::string& mp) {
			auto w = DC::Ir::DcgArchive::openWrite(dcgFile);
			std::string json = R"({"version":"1.0","nodes":[{"name":"n1","type":"Builtin","affinity":"Compute","modelPath":")"
				+ mp + R"(","inputs":[],"outputs":[]}],"edges":[],"outputBindings":[]})";
			w->writeGraphJson(json);
			w->finalize();
		};

		// 引号/反斜杠不落入 JSON 语法：用正斜杠形式覆盖原有用例
		const std::string cases[] = {"../evil.onnx", "/tmp/evil.onnx", "C:/tmp/evil.onnx"};
		constexpr size_t kCaseCount = 3;
		for (size_t i = 0; i < kCaseCount; ++i) {
			std::string dcgFile = "test_unsafe_path_" + std::to_string(i) + ".dcg";
			makeDcg(dcgFile, cases[i]);
			InferGraph graph;
			GraphCompiler::compileFile(graph, dcgFile);
			auto* n = graph.node("n1");
			CHECK(n != nullptr && n->modelPath() == cases[i],
				"modelPath must pass through verbatim (no compile-time rejection)");
			std::remove(dcgFile.c_str());
		}
	}
	END_TEST();
}

// ════════════════════════════════════════════
// IR-07/08：fail-fast 与 typeSize 校验
// ════════════════════════════════════════════

void testInvalidEdgeFailFast() {
	TEST("IR-07: invalid edge port fails the compile (no orphan connector)") {
		const char* json = R"({
  "version": "1.0",
  "nodes": [
    {
      "name": "n1", "type": "Builtin", "affinity": "Operator",
      "inputs": [{"name":"x","tensorType":"Float","typeSize":4,"shape":[],"required":true}],
      "outputs": [{"name":"y","tensorType":"Float","typeSize":4,"shape":[],"required":true}]
    }
  ],
  "edges": [
    {"srcNode":"n1","srcPort":"bogus","dstNode":"n1","dstPort":"x"}
  ],
  "outputBindings": []
})";
		bool rejected = false;
		try {
			InferGraph graph;
			GraphCompiler::compileString(graph, json);
		} catch (const GraphException&) {
			rejected = true;
		}
		CHECK(rejected, "edge with invalid port must fail fast");
	}
	END_TEST();
}

void testTypeSizeNegativeRejected() {
	TEST("IR-08: negative port typeSize is rejected at compile time") {
		const char* json = R"({
  "version": "1.0",
  "nodes": [
    {
      "name": "n1", "type": "Builtin", "affinity": "Operator",
      "inputs": [{"name":"x","tensorType":"Float","typeSize":-5,"shape":[],"required":true}],
      "outputs": []
    }
  ],
  "edges": [],
  "outputBindings": []
})";
		bool rejected = false;
		try {
			InferGraph graph;
			GraphCompiler::compileString(graph, json);
		} catch (const GraphException&) {
			rejected = true;
		}
		CHECK(rejected, "typeSize:-5 must be rejected (would otherwise wrap to SIZE_MAX)");
	}
	END_TEST();
}

int main() {
	try {
		testCompileStringBasic();
		testCompileStringBroadcast();
		testCompileStringRoutingRejected();
		testRoundTrip();
		testExpandedFanOutRoundTrip();
		testSerializeToJsonString();
		testModelPathHandling();
		// 异常/边界路径
		testInvalidJsonThrows();
		testEmptyGraph();
		testEdgeToMissingNode();
		testUnregisteredType();
		testDcgRoundTrip();
		testDcgSerializeNoModels();
		// 引擎注册接口统一后的语义适配（延迟物化：编译期零加载）
		testDynamicShapeRoundTrip();
		testVoidPortRoundTrip();
		testEngineNodeMaterialization();
		testEngineNodeNoModelPath();
		testCompileNeverInvokesCreateEngine();
		testEngineDeclaredSchemaEmptyWarns();
		testDcgCompileDefersModelResolution();
		// v0.5.2 修复项回归（IR-01/02/07/08）
		testSharedModelDcgRoundTrip();
		testDcgObjectShapedNodesRejected();
		testDcgModelPathPassThrough();
		testInvalidEdgeFailFast();
		testTypeSizeNegativeRejected();

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
