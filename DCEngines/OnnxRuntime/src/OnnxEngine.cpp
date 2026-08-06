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

// Windows 下 Ort::Session 仅接受 wchar_t* 路径；其他平台 ORTCHAR_T 即 char
static std::basic_string<ORTCHAR_T> toNativePath(const std::string& path) {
#ifdef _WIN32
	return std::filesystem::path(path).wstring();
#else
	return path;
#endif
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
	default:
		// FLOAT16/BFLOAT16/STRING/UNDEFINED 等：DC 类型系统无对应族，
		// 显式降级并告警，避免静默错误（下游经端口类型 Void + typeSize 感知）
		std::cerr << "[OnnxRuntime] warning: ONNX element type " << static_cast<int>(type)
				  << " has no DC::TensorType mapping; port mapped to Void" << std::endl;
		return Tensor::TensorType::Void;
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
// TensorConverter 契约：DC::Tensor ↔ Ort::Value
// ════════════════════════════════════════════

// DC::Tensor → Value(Ort::Value)：外部内存视图（零拷贝）。
// 返回的 Value 与输入 tensor 共享数据区，调用方必须保证 tensor 在
// Ort::Value 使用期间存活（RunFn 内满足：tensor 由 ctx 的 Value 持有）。
static Value onnxToNative(const Tensor& dc) {
	ONNXTensorElementDataType onnxType{};
	if (!tensorTypeToOnnxType(dc.type(), dc.typeSize(), onnxType))
		return {}; // 无法映射 → 空 Value，由 RunFn 上报 InvalidInput

	auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	auto shape = dc.shape();
	auto* ortVal = new Ort::Value(Ort::Value::CreateTensor(
		memInfo,
		const_cast<void*>(static_cast<const void*>(dc.bytes().data())),
		dc.bytes().size(),
		shape.data(),
		shape.size(),
		onnxType));
	// 仅析构 Ort::Value 外壳，不触碰 tensor 数据区（其所有权仍属调用方）
	return Value(ortVal, [](Ort::Value* v) { delete v; });
}

// Ort::Value* → DC::Tensor（深拷贝：device 侧数据不可长期引用时安全）
static Tensor onnxToDC(const void* native) {
	auto* ortVal = static_cast<const Ort::Value*>(native);
	if (!ortVal)
		return Tensor();

	auto info = ortVal->GetTensorTypeAndShapeInfo();
	auto elementType = info.GetElementType();
	auto typeSize = onnxTypeSize(elementType);
	auto elementCount = info.GetElementCount();
	auto byteSize = elementCount * typeSize;
	auto onnxShape = info.GetShape();

	const void* data = ortVal->GetTensorData<void>();
	Tensor::DataBlock block(byteSize);
	if (byteSize > 0)
		std::memcpy(block.data(), data, byteSize);

	Tensor::Shape shape(onnxShape.begin(), onnxShape.end());
	return Tensor(onnxTypeToTensorType(elementType), typeSize, shape, std::move(block));
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

		const auto* converter = ctx.converter();
		if (!converter || !converter->toNative || !converter->toDC)
			return ctx.failure(Node::Status::ExecutionFailed,
							   "OnnxRuntime: TensorConverter not configured on engine");

		// ── 1. 收集输入：DC::Tensor → Ort::Value（经 converter，零拷贝外部内存视图）──
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

			auto native = converter->toNative(*tensor);
			auto* ortVal = native.as<Ort::Value>();
			if (!ortVal)
				return ctx.failure(Node::Status::InvalidInput,
								   "OnnxRuntime: input '" + port.name
									   + "' type not supported by converter");

			inputNames.push_back(port.name.c_str());
			// Ort::Value 内部已持有外部内存指针（tensor 数据），move 仅转移外壳，
			// 数据区在 Run 期间由 ctx 的 Value 保证存活
			inputValues.push_back(std::move(*ortVal));
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

		// ── 4. 输出数量校验：缺失输出必须显式失败 ──
		if (outputs.size() < schema.outputs.size()) {
			return ctx.failure(Node::Status::ExecutionFailed,
							   "OnnxRuntime: expected " + std::to_string(schema.outputs.size())
								   + " outputs, got " + std::to_string(outputs.size()));
		}

		// ── 5. 收集输出：Ort::Value → DC::Tensor（经 converter 深拷贝）──
		for (size_t i = 0; i < schema.outputs.size(); ++i) {
			Tensor t = converter->toDC(&outputs[i]);
			ctx.output(schema.outputs[i].name, Value(std::make_unique<Tensor>(std::move(t))));
		}

		return ctx.success();
	};
}

// ════════════════════════════════════════════
// 引擎注册入口
// ════════════════════════════════════════════

void registerOnnxEngine(EngineRegistry& reg, const OnnxOptions& opts) {
	EngineDescriptor desc;
	desc.engineType = "OnnxRuntime";

	// ── TensorConverter：DC::Tensor ↔ Ort::Value ──
	desc.converter = {onnxToNative, onnxToDC};

	// ── createEngine: 从模型路径创建引擎实例（含模型加载与资源分配）──
	// 实例被 Registry 以 modelPath 为 key 缓存（建图仅加载一次）；
	// 所属描述符由框架在缓存时注入，无需依赖全局单例
	desc.createEngine = [opts](const std::string& modelPath) -> EngineInstance {
		Ort::SessionOptions sessionOpts;
		sessionOpts.SetIntraOpNumThreads(opts.intraOpThreads); // 与 DCinfer 图级并行模型一致
		if (opts.sessionCustomizer)
			opts.sessionCustomizer(&sessionOpts); // 追加 EP（CUDA/DML 等）或调整 SessionOptions

		auto session = std::make_shared<Ort::Session>(sharedEnv(), toNativePath(modelPath).c_str(), sessionOpts);
		return EngineInstance(std::move(session));
	};

	// ── getInputPorts/getOutputPorts: 从引擎实例推导端口 Schema ──
	desc.getInputPorts = [](const EngineInstance& inst) -> std::vector<Node::Port> {
		auto* session = static_cast<Ort::Session*>(const_cast<void*>(inst.get()));
		if (!session)
			return {};
		return getPortsFromSession(*session, true);
	};

	desc.getOutputPorts = [](const EngineInstance& inst) -> std::vector<Node::Port> {
		auto* session = static_cast<Ort::Session*>(const_cast<void*>(inst.get()));
		if (!session)
			return {};
		return getPortsFromSession(*session, false);
	};

	// ── factory: 使用框架推导的 Schema 构造节点并绑定引擎实例 ──
	// createNode(engineType, name, modelPath) 传入的 engineConfig 是
	// EngineInstance* 本身（框架已创建并缓存），直接转型即可
	desc.factory = [](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		auto* engineInstance = const_cast<EngineInstance*>(static_cast<const EngineInstance*>(p.engineConfig));

		auto node = std::make_unique<Node>(
			"OnnxRuntime", p.nodeName, p.schema, onnxRunFn(),
			ThreadPoolAffinity::Compute);

		if (engineInstance)
			node->bindEngine(engineInstance, engineInstance->descriptor());

		return node;
	};

	// ── 运行时钩子 ──
	desc.synchronize = [](void* /*engine*/) {
		// no-op: Ort::Session::Run() blocks until completion
	};
	desc.preRun = nullptr;
	desc.postRun = nullptr;
	desc.releaseEngine = nullptr; // shared_ptr 自动释放
	desc.onError = nullptr;

	reg.registerEngine(desc);
}

} // namespace DC::Onnx
