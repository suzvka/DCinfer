#include "DCEngine/OnnxEngine.h"
#include "Tensor.hpp"
#include "Node.h"

#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace DC::Onnx {

// Windows 下 Ort::Session 仅接受 wchar_t* 路径
static std::wstring toWide(const std::string& path) {
	return std::filesystem::path(path).wstring();
}

// ════════════════════════════════════════════
// ONNX Runtime 环境单例
// ════════════════════════════════════════════

static Ort::Env& sharedEnv() {
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DCinfer");
	return env;
}

// ════════════════════════════════════════════
// ONNX 数据类型 → DC::Tensor::TensorType 映射
// ════════════════════════════════════════════

static Tensor::TensorType onnxTypeToTensorType(ONNXTensorElementDataType type) {
	switch (type) {
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return Tensor::TensorType::Float;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return Tensor::TensorType::Float;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return Tensor::TensorType::Int;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return Tensor::TensorType::Uint;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return Tensor::TensorType::Bool;
	default:                                    return Tensor::TensorType::Void;
	}
}

static size_t onnxTypeSize(ONNXTensorElementDataType type) {
	switch (type) {
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return sizeof(float);
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return sizeof(double);
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return 1;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return 2;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return 4;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return 8;
	default:                                    return 0;
	}
}

// DC::Tensor 类型（类型族 + 元素字节数）→ ONNX 元素类型
// 返回 true 表示映射成功
static bool tensorTypeToOnnxType(Tensor::TensorType type, size_t typeSize,
								 ONNXTensorElementDataType& out) {
	switch (type) {
	case Tensor::TensorType::Float:
		if (typeSize == 4) { out = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;  return true; }
		if (typeSize == 8) { out = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE; return true; }
		return false;
	case Tensor::TensorType::Int:
		if (typeSize == 1) { out = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;  return true; }
		if (typeSize == 2) { out = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16; return true; }
		if (typeSize == 4) { out = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; return true; }
		if (typeSize == 8) { out = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; return true; }
		return false;
	case Tensor::TensorType::Uint:
		if (typeSize == 1) { out = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;  return true; }
		if (typeSize == 2) { out = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16; return true; }
		if (typeSize == 4) { out = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32; return true; }
		if (typeSize == 8) { out = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64; return true; }
		return false;
	case Tensor::TensorType::Bool:
		out = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
		return true;
	default:
		return false;
	}
}

// ════════════════════════════════════════════
// 从 Ort::Session 推导输入/输出端口 Schema
// ════════════════════════════════════════════

static std::vector<Node::Port> getPortsFromSession(const Ort::Session& session, bool isInput) {
	std::vector<Node::Port> ports;

	size_t count = isInput ? session.GetInputCount() : session.GetOutputCount();
	ports.reserve(count);

	for (size_t i = 0; i < count; ++i) {
		Ort::AllocatorWithDefaultOptions allocator;

		// 获取端口名称
		auto namePtr = isInput
			? session.GetInputNameAllocated(i, allocator)
			: session.GetOutputNameAllocated(i, allocator);
		std::string name(namePtr.get());

		// 获取类型和形状信息
		auto typeInfo = isInput
			? session.GetInputTypeInfo(i)
			: session.GetOutputTypeInfo(i);
		auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
		auto elementType = tensorInfo.GetElementType();
		auto onnxShape = tensorInfo.GetShape();

		// 构造 Node::Port
		Node::Port port;
		port.name = std::move(name);
		port.type = onnxTypeToTensorType(elementType);
		port.typeSize = onnxTypeSize(elementType);
		port.required = true;

		// ONNX shape 中的 -1 表示动态维度，直接保留（DC::Tensor::Shape 支持 -1）
		port.shape = Tensor::Shape(onnxShape.begin(), onnxShape.end());

		ports.push_back(std::move(port));
	}

	return ports;
}

// ════════════════════════════════════════════
// 推理执行 RunFn
// ════════════════════════════════════════════

static Node::RunFn onnxRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		auto* engine = ctx.engine();
		if (!engine)
			return ctx.failure(Node::Status::ExecutionFailed, "OnnxRuntime: no engine instance");

		auto& session = *static_cast<Ort::Session*>(engine);
		Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

		// ── 1. 收集输入 ──
		const auto& schema = ctx.schema();
		std::vector<const char*> inputNames;
		std::vector<Ort::Value> inputValues;
		inputNames.reserve(schema.inputs.size());
		inputValues.reserve(schema.inputs.size());

		for (const auto& port : schema.inputs) {
			const auto& val = ctx.peek(port.name);
			const auto* tensor = val.as<Tensor>();
			if (!tensor)
				return ctx.failure(Node::Status::InvalidInput,
								   "OnnxRuntime: input '" + port.name + "' is not a DC::Tensor");

			auto shape = tensor->shape();
			auto bytes = tensor->bytes();

			ONNXTensorElementDataType onnxType{};
			if (!tensorTypeToOnnxType(tensor->type(), tensor->typeSize(), onnxType))
				return ctx.failure(Node::Status::InvalidInput,
								   "OnnxRuntime: input '" + port.name + "' has unsupported type");

			inputNames.push_back(port.name.c_str());

			// 创建 Ort::Value（外部内存，不拥有数据——tensor 在 Run 期间有效）
			auto ortVal = Ort::Value::CreateTensor(
				memInfo,
				const_cast<void*>(static_cast<const void*>(bytes.data())),
				bytes.size(),
				shape.data(),
				shape.size(),
				onnxType
			);
			inputValues.push_back(std::move(ortVal));
		}

		// ── 2. 收集输出名称 ──
		std::vector<const char*> outputNames;
		outputNames.reserve(schema.outputs.size());
		for (const auto& port : schema.outputs) {
			outputNames.push_back(port.name.c_str());
		}

		// ── 3. 执行推理 ──
		std::vector<Ort::Value> outputs;
		try {
			outputs = session.Run(
				Ort::RunOptions{nullptr},
				inputNames.data(),
				inputValues.data(),
				inputValues.size(),
				outputNames.data(),
				outputNames.size()
			);
		} catch (const Ort::Exception& e) {
			return ctx.failure(Node::Status::ExecutionFailed,
							   std::string("OnnxRuntime inference failed: ") + e.what());
		}

		// ── 4. 收集输出：Ort::Value → DC::Tensor ──
		for (size_t i = 0; i < outputs.size() && i < outputNames.size(); ++i) {
			auto& ortVal = outputs[i];
			auto info = ortVal.GetTensorTypeAndShapeInfo();
			auto elementType = info.GetElementType();
			auto typeSize = onnxTypeSize(elementType);
			auto elementCount = info.GetElementCount();
			auto byteSize = elementCount * typeSize;
			auto onnxShape = info.GetShape();

			// 从 Ort::Value 拷贝数据到 DC::Tensor
			const void* data = ortVal.GetTensorData<void>();
			Tensor::DataBlock block(byteSize);
			if (byteSize > 0)
				std::memcpy(block.data(), data, byteSize);

			Tensor::Shape shape(onnxShape.begin(), onnxShape.end());
			auto t = std::make_unique<Tensor>(
				onnxTypeToTensorType(elementType), typeSize, shape, std::move(block));
			ctx.output(outputNames[i], Value(std::move(t)));
		}

		return ctx.success();
	};
}

// ════════════════════════════════════════════
// 引擎注册入口
// ════════════════════════════════════════════

void registerOnnxEngine(EngineRegistry& reg) {
	EngineDescriptor desc;
	desc.engineType = "OnnxRuntime";

	// ── TensorConverter ──
	desc.converter = {
		// toNative: DC::Tensor → Value（简单包装，实际转换在 RunFn 中处理）
		[](const Tensor& t) -> Value {
			return Value(std::make_unique<Tensor>(t));
		},
		// toDC: 引擎原生指针 → DC::Tensor（RunFn 中自行转换，此处留空）
		[](const void* /*native*/) -> Tensor {
			return Tensor();
		}
	};

	// ── loadModel: 从路径加载模型，返回类型擦除的模型句柄 ──
	desc.loadModel = [](const std::string& path) -> ModelHandle {
		Ort::SessionOptions opts;
		opts.SetIntraOpNumThreads(1); // 算子内部单线程，与 DCinfer 图级并行模型一致
		auto session = std::make_shared<Ort::Session>(sharedEnv(), toWide(path).c_str(), opts);
		return ModelHandle(std::move(session));
	};

	// ── getInputPorts: 从已加载模型推导输入端口 ──
	desc.getInputPorts = [](ModelHandle handle) -> std::vector<Node::Port> {
		auto* session = static_cast<Ort::Session*>(handle.get());
		if (!session)
			return {};
		return getPortsFromSession(*session, true);
	};

	// ── getOutputPorts: 从已加载模型推导输出端口 ──
	desc.getOutputPorts = [](ModelHandle handle) -> std::vector<Node::Port> {
		auto* session = static_cast<Ort::Session*>(handle.get());
		if (!session)
			return {};
		return getPortsFromSession(*session, false);
	};

	// ── createEngine: 从模型路径创建引擎实例 ──
	// 通过 engineType 从 Registry 查找已注册的 EngineDescriptor 指针
	desc.createEngine = [engineType = desc.engineType](const std::string& modelPath) -> EngineInstance {
		Ort::SessionOptions opts;
		opts.SetIntraOpNumThreads(1);
		auto session = std::make_shared<Ort::Session>(sharedEnv(), toWide(modelPath).c_str(), opts);

		const EngineDescriptor* d = EngineRegistry::instance().find(engineType);
		return EngineInstance(std::move(session), d);
	};

	// ── synchronize: ONNX Runtime 的 Run 是同步的，无需额外同步 ──
	desc.synchronize = [](void* /*engine*/) {
		// no-op: Ort::Session::Run() blocks until completion
	};

	// ── factory: 从 EngineInstance 创建 Node ──
	// 从 session 推导 Schema，绑定引擎实例
	// 注意：EngineRegistry::createNode(engineType, name, modelPath) 传入的
	// engineConfig 是 EngineInstance* 本身（见 EngineRegistry.cpp 注释），
	// 直接转型即可，不可二次解引用
	desc.factory = [](std::string name, const void* engineConfig) -> std::unique_ptr<Node> {
		auto* engineInstance = const_cast<EngineInstance*>(static_cast<const EngineInstance*>(engineConfig));

		Node::Schema schema;
		if (engineInstance) {
			auto* session = static_cast<Ort::Session*>(engineInstance->get());
			schema.inputs = getPortsFromSession(*session, true);
			schema.outputs = getPortsFromSession(*session, false);
		}

		auto node = std::make_unique<Node>(
			"OnnxRuntime", std::move(name), schema, onnxRunFn(),
			ThreadPoolAffinity::Compute);

		if (engineInstance)
			node->bindEngine(engineInstance, engineInstance->descriptor());

		return node;
	};

	// ── 可选钩子（当前留空）──
	desc.preRun = nullptr;
	desc.postRun = nullptr;
	desc.releaseEngine = nullptr; // shared_ptr 自动释放
	desc.onError = nullptr;

	reg.registerEngine(desc);
}

} // namespace DC::Onnx
