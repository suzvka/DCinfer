// OnnxEngineTest - ONNX Runtime 引擎适配器集成测试。
// 测试模型为手工编码的最小 ONNX protobuf 字节流，不依赖 onnx/protobuf 库：
// 避免 onnxruntime.dll 内嵌描述符与外部 onnx 静态库重复注册导致
// protobuf "File already exists in database" 崩溃。

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

// 变长字段即 wire type 2：tag + length + payload
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

// TypeProto { tensor_type = 1 }，oneof 字段号 1
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

// 生成 ONNX 模型字节流：Z = X + Y，opset 13，元素类型与形状参数化
// ModelProto 字段：ir_version=1, graph=7, opset_import=8
// GraphProto 字段：node=1, name=2, input=11, output=12；TensorProto 类型：FLOAT=1, FLOAT16=10
std::vector<std::byte> buildAddModelBytes(int elemType, std::initializer_list<int64_t> dims) {
	std::vector<std::byte> graph;
	encodeLengthDelimited(graph, 11, encodeValueInfo("X", elemType, dims));
	encodeLengthDelimited(graph, 11, encodeValueInfo("Y", elemType, dims));
	encodeLengthDelimited(graph, 1, encodeNode("add0", "Add", {"X", "Y"}, {"Z"}));
	encodeLengthDelimited(graph, 12, encodeValueInfo("Z", elemType, dims));
	encodeString(graph, 2, "dcinfer_test_add");

	std::vector<std::byte> model;
	encodeVarintField(model, 1, 8);
	encodeLengthDelimited(model, 8, encodeOpsetImport("", 13));
	encodeLengthDelimited(model, 7, graph);
	return model;
}

// 将字节流写入临时文件；ORT 在 Session 创建时自行校验模型合法性
std::string generateAddModel(const std::string& fileName, int elemType,
							 std::initializer_list<int64_t> dims) {
	auto bytes = buildAddModelBytes(elemType, dims);

	auto path = std::filesystem::temp_directory_path() / fileName;
	std::ofstream out(path, std::ios::binary);
	out.write(reinterpret_cast<const char*>(bytes.data()),
			  static_cast<std::streamsize>(bytes.size()));
	if (!out) {
		std::cerr << "Failed to write test model" << std::endl;
		return {};
	}
	return path.string();
}

DC::Tensor makeFloatTensor(const float (&values)[4]) {
	DC::Tensor::DataBlock block(sizeof(values));
	std::memcpy(block.data(), values, sizeof(values));
	return DC::Tensor::Create<float>({1, 4}, std::move(block));
}

// fp32 转 fp16 位模式；测试值均为 fp16 精确可表示，截断转换即得精确结果
static uint16_t fp32ToFp16Bits(float value) {
	uint32_t bits = 0;
	std::memcpy(&bits, &value, sizeof(bits));
	const uint32_t sign = (bits >> 16) & 0x8000u;
	const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
	const uint32_t mant = (bits >> 13) & 0x3FFu;
	if (exp >= 0x1F)
		return static_cast<uint16_t>(sign | 0x7C00u); // ±Inf
	if (exp <= 0)
		return static_cast<uint16_t>(sign);           // 0 或 subnormal，测试值不涉及
	return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

int fail(const std::string& msg) {
	std::cerr << "[FAIL] " << msg << std::endl;
	return 1;
}

} // namespace

int main() {
	try {
		auto modelPath = generateAddModel("dcinfer_test_add.onnx", 1, {1, 4});
		if (modelPath.empty())
			return fail("model generation failed");
		std::cout << "Test model: " << modelPath << std::endl;

		DC::Onnx::registerOnnxEngine();
		auto& reg = DC::EngineRegistry::instance();
		if (!reg.hasEngine("OnnxRuntime"))
			return fail("OnnxRuntime engine not registered");

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

		// TestHarness：task 完成回调在输出清理前触发，能安全取到结果
		DC::TestHarness harness;
		harness.addNode(std::move(node));
		harness.bindOutput("Z", "onnx_add", "Z");

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

		// TensorConverter 契约直测
		{
			const auto* desc = reg.find("OnnxRuntime");
			if (!desc || !desc->converter.toNative || !desc->converter.toDC)
				return fail("OnnxRuntime descriptor converter not configured");

			const float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
			// 注意：Ort::Value 是外部内存零拷贝视图，源 Tensor 必须具名存活至
			// 使用结束，否则临时对象析构后 GetTensorData 读到悬垂指针。
			DC::Tensor srcTensor = makeFloatTensor(src);
			auto native = desc->converter.toNative(srcTensor);
			auto* ortVal = native.as<Ort::Value>();
			if (!ortVal)
				return fail("toNative returned empty Value");
			auto tinfo = ortVal->GetTensorTypeAndShapeInfo();
			if (tinfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
				return fail("toNative element type mismatch");
			auto nshape = tinfo.GetShape();
			if (nshape.size() != 2 || nshape[0] != 1 || nshape[1] != 4)
				return fail("toNative shape mismatch");
			const float* ndata = ortVal->GetTensorData<float>();
			if (!ndata || ndata[0] != 1.0f || ndata[3] != 4.0f)
				return fail("toNative data mismatch");

			// Ort::Value 转 DC::Tensor，深拷贝
			DC::Tensor back = desc->converter.toDC(ortVal);
			auto bdata = back.data<float>();
			if (bdata.size() != 4 || bdata[0] != 1.0f || bdata[3] != 4.0f)
				return fail("toDC round-trip mismatch");
			auto bshape = back.shape();
			if (bshape.size() != 2 || bshape[0] != 1 || bshape[1] != 4)
				return fail("toDC shape mismatch");

			// 无法映射的类型使 toNative 返回空，显式失败而非静默降级
			DC::Tensor unsupported(DC::Tensor::TensorType::Void, 4);
			if (desc->converter.toNative(unsupported))
				return fail("toNative should return empty for unmappable type");
		}
		std::cout << "[PASS] converter round-trip: DC::Tensor <-> Ort::Value" << std::endl;

		// FP16 模型：挂 Float 族 typeSize=2 并推理；BF16 仍降级 Void。
		// CPU EP 的 fp16/bf16 内核支持因 ORT 构建而异：内核缺失在 Session 构造期抛，
		// 经 createNode 暴露；createNode 失败、推理失败、成功都是合法路径。
		{
			auto fp16Path = generateAddModel("dcinfer_test_add_fp16.onnx", 10, {1, 4});
			if (fp16Path.empty())
				return fail("FP16 model generation failed");
			try {
				auto node = reg.createNode("OnnxRuntime", "onnx_fp16", fp16Path);
				if (!node)
					return fail("FP16 createNode returned null");
				const auto& fp16Schema = node->schema();
				if (fp16Schema.inputs.empty() || fp16Schema.inputs[0].type != DC::Tensor::TensorType::Float)
					return fail("FP16 port should map to Float family");
				if (fp16Schema.inputs[0].typeSize != 2)
					return fail("FP16 port typeSize should be 2");

				DC::TestHarness harness;
				harness.addNode(std::move(node));
				harness.bindOutput("Z", "onnx_fp16", "Z");
				const float xData[4] = {1.0f, 2.0f, 3.0f, 4.0f};
				const float yData[4] = {10.0f, 20.0f, 30.0f, 40.0f};
				auto makeFp16 = [](const float (&vals)[4]) {
					DC::Tensor::DataBlock block(4 * sizeof(uint16_t));
					uint16_t tmp[4];
					for (size_t i = 0; i < 4; ++i)
						tmp[i] = fp32ToFp16Bits(vals[i]);
					std::memcpy(block.data(), tmp, sizeof(tmp));
					return DC::Tensor(DC::Tensor::TensorType::Float, sizeof(uint16_t), {1, 4}, std::move(block));
				};
				harness.feedInput("task_fp16", "onnx_fp16", "X", makeFp16(xData));
				harness.feedInput("task_fp16", "onnx_fp16", "Y", makeFp16(yData));
				harness.submit("task_fp16", "onnx_fp16", "Z");
				if (!harness.awaitCompletion("task_fp16"))
					return fail("FP16 task timed out");
				if (harness.hasErrors()) {
					// 推理期内核缺失：错误经引擎 RunFn 归一化进 taskErrors
					bool kernelErrorSeen = false;
					for (const auto& err : harness.taskErrors("task_fp16")) {
						std::cout << "  " << err.nodeName << ": " << err.message << std::endl;
						if (err.message.find("implementation") != std::string::npos)
							kernelErrorSeen = true;
					}
					if (!kernelErrorSeen)
						return fail("FP16 failure should surface the missing-kernel Ort error");
				} else {
					// 内核可用：校验 fp16 位模式推理结果
					auto result = harness.getOutputTensor("task_fp16", "onnx_fp16", "Z");
					if (result.type() != DC::Tensor::TensorType::Float || result.typeSize() != 2)
						return fail("FP16 output should be Float typeSize=2");
					auto zData = result.data<uint16_t>();
					const float expected[4] = {11.0f, 22.0f, 33.0f, 44.0f};
					if (zData.size() != 4)
						return fail("FP16 expected 4 output elements, got " + std::to_string(zData.size()));
					for (size_t i = 0; i < 4; ++i) {
						if (zData[i] != fp32ToFp16Bits(expected[i]))
							return fail("FP16 output[" + std::to_string(i) + "] bits mismatch");
					}
				}
			} catch (const std::exception& e) {
				// Session 构造期内核缺失，如 x64-linux 静态构建：引擎归一化后上抛
				std::cout << "  fp16 createNode rejected: " << e.what() << std::endl;
				if (std::string(e.what()).find("implementation") == std::string::npos)
					return fail(std::string("FP16 createNode failure should be missing-kernel: ") + e.what());
			}
			std::filesystem::remove(fp16Path);

			// BF16 与 FP16 同为 2 字节、反向映射歧义：显式降级 Void；无内核时构造期失败，属合法路径
			auto bf16Path = generateAddModel("dcinfer_test_add_bf16.onnx", 16, {1, 4});
			if (bf16Path.empty())
				return fail("BF16 model generation failed");
			try {
				auto bf16Node = reg.createNode("OnnxRuntime", "onnx_bf16", bf16Path);
				if (!bf16Node)
					return fail("BF16 createNode returned null");
				const auto& bf16Schema = bf16Node->schema();
				if (bf16Schema.inputs.empty() || bf16Schema.inputs[0].type != DC::Tensor::TensorType::Void)
					return fail("BF16 port should be explicitly mapped to Void");
				if (bf16Schema.inputs[0].typeSize != 2)
					return fail("BF16 port typeSize should be 2");
			} catch (const std::exception& e) {
				std::cout << "  bf16 createNode rejected: " << e.what() << std::endl;
				if (std::string(e.what()).find("implementation") == std::string::npos)
					return fail(std::string("BF16 createNode failure should be missing-kernel: ") + e.what());
			}
			std::filesystem::remove(bf16Path);
		}
		std::cout << "[PASS] FP16 model: Float family (typeSize=2), inference or missing-kernel path verified; BF16 still Void" << std::endl;

		// 动态 shape 模型 dim=-1：推导保留 -1 且执行通过
		{
			auto dynPath = generateAddModel("dcinfer_test_add_dyn.onnx", 1, {-1, 4});
			if (dynPath.empty())
				return fail("dynamic-shape model generation failed");
			auto node = reg.createNode("OnnxRuntime", "onnx_dyn", dynPath);
			if (!node)
				return fail("dynamic-shape createNode returned null");
			const auto& dynSchema = node->schema();
			if (dynSchema.inputs.size() != 2 || dynSchema.inputs[0].shape.size() != 2
				|| dynSchema.inputs[0].shape[0] != -1)
				return fail("dynamic dim (-1) should be preserved in derived schema");

			DC::TestHarness harness;
			harness.addNode(std::move(node));
			harness.bindOutput("Z", "onnx_dyn", "Z");
			const float xData[4] = {5.0f, 6.0f, 7.0f, 8.0f};
			const float yData[4] = {0.5f, 1.5f, 2.5f, 3.5f};
			harness.feedInput("task_dyn", "onnx_dyn", "X", makeFloatTensor(xData));
			harness.feedInput("task_dyn", "onnx_dyn", "Y", makeFloatTensor(yData));
			harness.submit("task_dyn", "onnx_dyn", "Z");
			if (!harness.awaitCompletion("task_dyn"))
				return fail("dynamic-shape task timed out or failed");
			if (harness.hasErrors()) {
				for (const auto& err : harness.taskErrors("task_dyn"))
					std::cerr << "  " << err.nodeName << ": " << err.message << std::endl;
				return fail("dynamic-shape task completed with errors");
			}
			auto result = harness.getOutputTensor("task_dyn", "onnx_dyn", "Z");
			auto data = result.data<float>();
			if (data.size() != 4 || data[0] != 5.5f || data[3] != 11.5f)
				return fail("dynamic-shape output mismatch");
			std::filesystem::remove(dynPath);
		}
		std::cout << "[PASS] dynamic-shape model: -1 dim preserved, execution OK" << std::endl;

		reg.releaseAllEngines();
		std::filesystem::remove(modelPath);
		return 0;
	} catch (const Ort::Exception& e) {
		return fail(std::string("Ort::Exception: ") + e.what());
	} catch (const std::exception& e) {
		return fail(std::string("std::exception: ") + e.what());
	}
}
