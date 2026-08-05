// OnnxEngineTest - ONNX Runtime 引擎适配器集成测试
//
// 测试流程：
//   1. 在运行时生成一个小型 Add 模型字节流（X + Y → Z, float32 [1,4]）
//   2. 注册 OnnxRuntime 引擎
//   3. 通过 EngineRegistry::createNode 从模型路径创建节点（自动推导 Schema）
//   4. 构建单节点推理图，注入数据，提交执行
//   5. 校验输出数值与形状
//
// 覆盖点：
//   - registerOnnxEngine 注册流程
//   - 模型加载 + 端口 Schema 自动推导
//   - RunFn 中 DC::Tensor → Ort::Value → DC::Tensor 的往返转换
//   - InferGraph 异步调度下的引擎节点执行
//
// 设计说明：测试模型不依赖 onnx/protobuf 库生成，而是直接内嵌手工编码的
// 最小 ONNX protobuf 字节流（ONNX 文件格式即 protobuf 序列化）。
// 这避免了 onnxruntime.dll 内嵌的 ONNX 描述符与外部 onnx 静态库重复注册
// 导致的 protobuf "File already exists in database" 崩溃。

#include "TestHarness.h"
#include "Tensor.hpp"
#include "DCEngine/OnnxEngine.h"

#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

// ── 手写 protobuf varint 编码 ──
void encodeVarint(std::vector<std::byte>& out, uint64_t value) {
	while (value >= 0x80) {
		out.push_back(static_cast<std::byte>((value & 0x7F) | 0x80));
		value >>= 7;
	}
	out.push_back(static_cast<std::byte>(value));
}

void encodeTag(std::vector<std::byte>& out, uint32_t fieldNumber, uint32_t wireType) {
	encodeVarint(out, (static_cast<uint64_t>(fieldNumber) << 3) | wireType);
}

// 变长字段（wire type 2）：tag + length + payload
void encodeLengthDelimited(std::vector<std::byte>& out, uint32_t fieldNumber,
						   const std::vector<std::byte>& payload) {
	encodeTag(out, fieldNumber, 2);
	encodeVarint(out, payload.size());
	out.insert(out.end(), payload.begin(), payload.end());
}

void encodeString(std::vector<std::byte>& out, uint32_t fieldNumber, const std::string& str) {
	std::vector<std::byte> payload(str.size());
	std::memcpy(payload.data(), str.data(), str.size());
	encodeLengthDelimited(out, fieldNumber, payload);
}

void encodeVarintField(std::vector<std::byte>& out, uint32_t fieldNumber, uint64_t value) {
	encodeTag(out, fieldNumber, 0);
	encodeVarint(out, value);
}

// TensorShapeProto.Dimension { dim_value = 1 }
std::vector<std::byte> encodeDim(int64_t value) {
	std::vector<std::byte> dim;
	encodeVarintField(dim, 1, static_cast<uint64_t>(value));
	return dim;
}

// TensorShapeProto { dim = 1 (repeated) }
std::vector<std::byte> encodeShape(std::initializer_list<int64_t> dims) {
	std::vector<std::byte> shape;
	for (auto d : dims)
		encodeLengthDelimited(shape, 1, encodeDim(d));
	return shape;
}

// TypeProto.Tensor { elem_type = 1, shape = 2 }
std::vector<std::byte> encodeTensorType(int32_t elemType, std::initializer_list<int64_t> dims) {
	std::vector<std::byte> tt;
	encodeVarintField(tt, 1, static_cast<uint64_t>(elemType));
	encodeLengthDelimited(tt, 2, encodeShape(dims));
	return tt;
}

// TypeProto { tensor_type = 1 }（oneof value，字段号 1）
std::vector<std::byte> encodeTypeProto(int32_t elemType, std::initializer_list<int64_t> dims) {
	std::vector<std::byte> tp;
	encodeLengthDelimited(tp, 1, encodeTensorType(elemType, dims));
	return tp;
}

// ValueInfoProto { name = 1, type = 2 }
std::vector<std::byte> encodeValueInfo(const std::string& name, int32_t elemType,
										 std::initializer_list<int64_t> dims) {
	std::vector<std::byte> vi;
	encodeString(vi, 1, name);
	encodeLengthDelimited(vi, 2, encodeTypeProto(elemType, dims));
	return vi;
}

// NodeProto { input = 1 (repeated), output = 2 (repeated), name = 3, op_type = 4 }
std::vector<std::byte> encodeNode(const std::string& name, const std::string& opType,
								  std::initializer_list<std::string> inputs,
								  std::initializer_list<std::string> outputs) {
	std::vector<std::byte> node;
	for (const auto& in : inputs)
		encodeString(node, 1, in);
	for (const auto& out : outputs)
		encodeString(node, 2, out);
	encodeString(node, 3, name);
	encodeString(node, 4, opType);
	return node;
}

// OperatorSetIdProto { domain = 1, version = 2 }
std::vector<std::byte> encodeOpsetImport(const std::string& domain, int64_t version) {
	std::vector<std::byte> opset;
	encodeString(opset, 1, domain);
	encodeVarintField(opset, 2, static_cast<uint64_t>(version));
	return opset;
}

// ── 生成小型 ONNX 模型字节流：Z = X + Y，float32 [1,4]，opset 13 ──
// ModelProto 字段：ir_version=1, graph=7, opset_import=8
// GraphProto 字段：node=1, name=2, input=11, output=12
// TensorProto_DataType_FLOAT = 1
std::vector<std::byte> buildAddModelBytes() {
	std::vector<std::byte> graph;
	encodeLengthDelimited(graph, 11, encodeValueInfo("X", 1, {1, 4})); // GraphProto.input
	encodeLengthDelimited(graph, 11, encodeValueInfo("Y", 1, {1, 4})); // GraphProto.input
	encodeLengthDelimited(graph, 1, encodeNode("add0", "Add", {"X", "Y"}, {"Z"})); // GraphProto.node
	encodeLengthDelimited(graph, 12, encodeValueInfo("Z", 1, {1, 4})); // GraphProto.output
	encodeString(graph, 2, "dcinfer_test_add");                        // GraphProto.name

	std::vector<std::byte> model;
	encodeVarintField(model, 1, 8);                          // ir_version = 8
	encodeLengthDelimited(model, 8, encodeOpsetImport("", 13)); // opset_import
	encodeLengthDelimited(model, 7, graph);                  // graph
	return model;
}

// 将字节流写入临时文件（Session 创建时 ORT 会自行校验模型合法性）
std::string generateAddModel() {
	auto bytes = buildAddModelBytes();

	auto path = std::filesystem::temp_directory_path() / "dcinfer_test_add.onnx";
	std::ofstream out(path, std::ios::binary);
	out.write(reinterpret_cast<const char*>(bytes.data()),
			  static_cast<std::streamsize>(bytes.size()));
	if (!out) {
		std::cerr << "Failed to write test model" << std::endl;
		return {};
	}
	return path.string();
}

// ── 构造 float32 [1,4] 张量 ──
DC::Tensor makeFloatTensor(const float (&values)[4]) {
	DC::Tensor::DataBlock block(sizeof(values));
	std::memcpy(block.data(), values, sizeof(values));
	return DC::Tensor::Create<float>({1, 4}, std::move(block));
}

int fail(const std::string& msg) {
	std::cerr << "[FAIL] " << msg << std::endl;
	return 1;
}

} // namespace

int main() {
	try {
		// ── 1. 生成测试模型 ──
		auto modelPath = generateAddModel();
		if (modelPath.empty())
			return fail("model generation failed");
		std::cout << "Test model: " << modelPath << std::endl;

		// ── 2. 注册 OnnxRuntime 引擎 ──
		DC::Onnx::registerOnnxEngine();
		auto& reg = DC::EngineRegistry::instance();
		if (!reg.hasEngine("OnnxRuntime"))
			return fail("OnnxRuntime engine not registered");

		// ── 3. 从模型路径创建节点（自动推导 Schema）──
		auto node = reg.createNode("OnnxRuntime", "onnx_add", modelPath);
		if (!node)
			return fail("createNode returned null");

		// 校验推导出的端口
		const auto& schema = node->schema();
		if (schema.inputs.size() != 2)
			return fail("expected 2 inputs, got " + std::to_string(schema.inputs.size()));
		if (schema.outputs.size() != 1)
			return fail("expected 1 output, got " + std::to_string(schema.outputs.size()));
		if (schema.inputs[0].name != "X" || schema.inputs[1].name != "Y")
			return fail("unexpected input port names: " + schema.inputs[0].name + ", " +
						schema.inputs[1].name);
		if (schema.outputs[0].name != "Z")
			return fail("unexpected output port name: " + schema.outputs[0].name);
		if (schema.inputs[0].type != DC::Tensor::TensorType::Float)
			return fail("input port type should be Float");
		std::cout << "Schema derivation OK: X[1,4] + Y[1,4] -> Z" << std::endl;

		// ── 4. 构建推理图并执行 ──
		// 使用 TestHarness：task 完成回调在输出清理前触发，能安全取到结果
		DC::TestHarness harness;
		harness.addNode(std::move(node));
		harness.bindOutput("onnx_add", "Z");

		const float xData[4] = {1.0f, 2.0f, 3.0f, 4.0f};
		const float yData[4] = {10.0f, 20.0f, 30.0f, 40.0f};
		harness.feedInput("task1", "onnx_add", "X", makeFloatTensor(xData));
		harness.feedInput("task1", "onnx_add", "Y", makeFloatTensor(yData));

		harness.submit("task1", "onnx_add", "Z");

		if (!harness.awaitCompletion("task1")) {
			for (const auto& err : harness.taskErrors("task1"))
				std::cerr << "  " << err.nodeName << ": " << err.message << std::endl;
			return fail("task timed out or failed");
		}
		if (harness.hasErrors()) {
			for (const auto& err : harness.taskErrors("task1"))
				std::cerr << "  " << err.nodeName << ": " << err.message << std::endl;
			return fail("task completed with errors");
		}
		if (!harness.hasOutput("task1", "onnx_add", "Z"))
			return fail("no output captured at onnx_add.Z");

		// ── 5. 校验输出 ──
		auto result = harness.getOutputTensor("task1", "onnx_add", "Z");
		auto data = result.data<float>();
		if (data.size() != 4)
			return fail("expected 4 output elements, got " + std::to_string(data.size()));

		const float expected[4] = {11.0f, 22.0f, 33.0f, 44.0f};
		for (size_t i = 0; i < 4; ++i) {
			if (data[i] != expected[i])
				return fail("output[" + std::to_string(i) + "] = " + std::to_string(data[i]) +
							", expected " + std::to_string(expected[i]));
		}

		auto outShape = result.shape();
		if (outShape.size() != 2 || outShape[0] != 1 || outShape[1] != 4)
			return fail("unexpected output shape");

		std::cout << "[PASS] OnnxEngineTest: X + Y = Z verified via ONNX Runtime" << std::endl;

		// 清理：释放引擎实例与临时模型
		reg.releaseAllEngines();
		std::filesystem::remove(modelPath);
		return 0;
	} catch (const Ort::Exception& e) {
		return fail(std::string("Ort::Exception: ") + e.what());
	} catch (const std::exception& e) {
		return fail(std::string("std::exception: ") + e.what());
	}
}
