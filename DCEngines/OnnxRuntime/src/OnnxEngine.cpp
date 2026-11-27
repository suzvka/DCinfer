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

#ifdef DCINFER_ORT_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace DC::Onnx {

struct OrtCore {
	std::shared_ptr<Ort::Env> env;
#ifdef DCINFER_ORT_ENABLE_CUDA
	cudaStream_t stream = nullptr;
	int deviceId = 0;

	~OrtCore() {
		if (!stream)
			return;
		auto status = cudaSetDevice(deviceId);
		if (status == cudaSuccess)
			status = cudaStreamDestroy(stream);
		if (status != cudaSuccess)
			std::cerr << "[OnnxRuntime] failed to release CUDA stream: "
					  << cudaGetErrorString(status) << std::endl;
	}
#endif
};

// Windows 下 Ort::Session 仅接受 wchar_t*；其他平台 ORTCHAR_T 即 char
static std::basic_string<ORTCHAR_T> toNativePath(const std::string& path) {
#ifdef _WIN32
	return std::filesystem::path(path).wstring();
#else
	return path;
#endif
}

static Tensor::TensorType onnxTypeToTensorType(ONNXTensorElementDataType type) {
	switch (type) {
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return Tensor::TensorType::Float;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return Tensor::TensorType::Float;
	// FP16 挂 Float 族，typeSize=2，数据黑盒传递；反向映射见 tensorTypeToOnnxType
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return Tensor::TensorType::Float;
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
		// 无对应族的类型如 BFLOAT16、STRING：显式降级为 Void 并告警，避免静默错误；
		// BFLOAT16 与 FLOAT16 同为 2 字节，挂 Float 会使反向映射歧义
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

// DC::Tensor 的类型族与字节数映射到 ONNX 元素类型；true 表示映射成功
static bool tensorTypeToOnnxType(Tensor::TensorType type, size_t typeSize,
								 ONNXTensorElementDataType& out) {
	switch (type) {
	case Tensor::TensorType::Float:
		if (typeSize == 2) { out = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; return true; }
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

#ifdef DCINFER_ORT_ENABLE_CUDA
static void checkCuda(cudaError_t status, const char* operation) {
	if (status != cudaSuccess)
		throw std::runtime_error(std::string("CUDA ") + operation + ": " + cudaGetErrorString(status));
}

static Ort::MemoryInfo cudaMemoryInfo(int deviceId) {
	return Ort::MemoryInfo("Cuda", OrtDeviceAllocator, deviceId, OrtMemTypeDefault);
}
#endif

static std::vector<Node::Port> getPortsFromSession(const Ort::Session& session, bool isInput) {
	std::vector<Node::Port> ports;

	size_t count = isInput ? session.GetInputCount() : session.GetOutputCount();
	ports.reserve(count);

	for (size_t i = 0; i < count; ++i) {
		Ort::AllocatorWithDefaultOptions allocator;

		auto namePtr = isInput
			? session.GetInputNameAllocated(i, allocator)
			: session.GetOutputNameAllocated(i, allocator);
		std::string name(namePtr.get());

		auto typeInfo = isInput
			? session.GetInputTypeInfo(i)
			: session.GetOutputTypeInfo(i);
		auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
		auto elementType = tensorInfo.GetElementType();
		auto onnxShape = tensorInfo.GetShape();

		Node::Port port;
		port.name = std::move(name);
		port.type = onnxTypeToTensorType(elementType);
		port.typeSize = onnxTypeSize(elementType);
		port.required = true;

		// ONNX -1 动态维直接保留，DC::Tensor::Shape 支持 -1
		port.shape = Tensor::Shape(onnxShape.begin(), onnxShape.end());

		ports.push_back(std::move(port));
	}

	return ports;
}

// DC::Tensor 转 Value(Ort::Value)：零拷贝外部内存视图；调用方必须保证
// tensor 在 Ort::Value 使用期间存活，RunFn 内由 ctx 的 Value 保证。
static Value onnxToNative(const Tensor& dc) {
	ONNXTensorElementDataType onnxType{};
	if (!tensorTypeToOnnxType(dc.type(), dc.typeSize(), onnxType))
		return {}; // 无法映射则空 Value，RunFn 上报 InvalidInput

	auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	auto shape = dc.shape();
	auto* ortVal = new Ort::Value(Ort::Value::CreateTensor(
		memInfo,
		const_cast<void*>(static_cast<const void*>(dc.bytes().data())),
		dc.bytes().size(),
		shape.data(),
		shape.size(),
		onnxType));
	// 仅析构外壳，不触碰 tensor 数据区，所有权属调用方
	return Value(ortVal, [](Ort::Value* v) { delete v; });
}

// Ort::Value* 转 DC::Tensor，深拷贝，device 侧数据不可长期引用
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

#ifdef DCINFER_ORT_ENABLE_CUDA
static Tensor onnxToExternal(Ort::Value&& native, const EngineCoreHandle& coreHandle, OrtCore* core) {
	auto value = std::make_shared<Ort::Value>(std::move(native));
	auto info = value->GetTensorTypeAndShapeInfo();
	auto elementType = info.GetElementType();
	auto typeSize = onnxTypeSize(elementType);
	if (typeSize == 0)
		throw std::runtime_error("OnnxRuntime: unsupported non-tensor output type for external backing");
	auto byteSize = info.GetElementCount() * typeSize;
	auto onnxShape = info.GetShape();
	Tensor::Shape shape(onnxShape.begin(), onnxShape.end());

	ExternalBacking backing;
	backing.producerEngine = "OnnxRuntime";
	backing.coreToken = std::shared_ptr<const void>(coreHandle, core);
	backing.handle = value;
	backing.exportToHost = [value, core, byteSize]() {
		std::vector<std::byte> host(byteSize);
		if (byteSize == 0)
			return host;
		checkCuda(cudaSetDevice(core->deviceId), "cudaSetDevice");
		checkCuda(cudaMemcpyAsync(
			host.data(),
			value->GetTensorMutableData<void>(),
			byteSize,
			cudaMemcpyDeviceToHost,
			core->stream),
			"cudaMemcpyAsync");
		checkCuda(cudaStreamSynchronize(core->stream), "cudaStreamSynchronize");
		return host;
	};
	return Tensor::fromExternal(onnxTypeToTensorType(elementType), typeSize, shape, std::move(backing));
}
#endif

static Node::RunFn onnxRunFn(bool deviceDirect) {
	return [deviceDirect](Node::RunContext& ctx) -> Node::Result {
		auto* engine = ctx.engine();
		if (!engine)
			return ctx.failure(Node::Status::ExecutionFailed, "OnnxRuntime: no engine instance");
		auto& session = *static_cast<Ort::Session*>(engine);

		const auto* instance = ctx.engineInstance();
		EngineCoreHandle coreHandle = instance ? instance->core() : EngineCoreHandle{};
		if (!coreHandle)
			return ctx.failure(Node::Status::ExecutionFailed, "OnnxRuntime: no engine core");
		auto* core = static_cast<OrtCore*>(coreHandle->get());

		const auto* converter = ctx.converter();
		if (!converter || !converter->toNative || !converter->toDC)
			return ctx.failure(Node::Status::ExecutionFailed,
							   "OnnxRuntime: TensorConverter not configured on engine");

		// 收集输入：同核心 external backing 可包装为 CUDA 指针视图，否则走 host converter。
		const auto& schema = ctx.schema();
		std::vector<const char*> inputNames;
		std::vector<Ort::Value> inputValues;
		inputNames.reserve(schema.inputs.size());
		inputValues.reserve(schema.inputs.size());

		for (const auto& port : schema.inputs) {
			const auto* tensor = ctx.input<Tensor>(port.name);
			if (!tensor)
				return ctx.failure(Node::Status::InvalidInput,
								   "OnnxRuntime: input '" + port.name + "' is not a DC::Tensor");

			Value native;
			try {
#ifdef DCINFER_ORT_ENABLE_CUDA
				const auto* backing = tensor->externalBacking();
				if (deviceDirect && backing && backing->coreToken.get() == core) {
					checkCuda(cudaSetDevice(core->deviceId), "cudaSetDevice");
					auto* source = static_cast<Ort::Value*>(backing->handle.get());
					auto info = source->GetTensorTypeAndShapeInfo();
					ONNXTensorElementDataType onnxType{};
					if (!tensorTypeToOnnxType(tensor->type(), tensor->typeSize(), onnxType))
						return ctx.failure(Node::Status::InvalidInput,
										   "OnnxRuntime: external input type is not supported");
					auto shape = tensor->shape();
					if (info.GetElementType() != onnxType || info.GetShape() != shape)
						return ctx.failure(Node::Status::InvalidInput,
										   "OnnxRuntime: external input handle metadata mismatch");
					auto memInfo = cudaMemoryInfo(core->deviceId);
					auto* direct = new Ort::Value(Ort::Value::CreateTensor(
						memInfo,
						source->GetTensorMutableData<void>(),
						info.GetElementCount() * tensor->typeSize(),
						shape.data(),
						shape.size(),
						onnxType));
					native = Value(direct, [](Ort::Value* v) { delete v; });
				} else
#endif
				{
					native = converter->toNative(*tensor);
				}
			} catch (const Ort::Exception& e) {
				return ctx.failure(Node::Status::ExecutionFailed,
								   std::string("OnnxRuntime input conversion failed: ") + e.what());
			} catch (const std::exception& e) {
				return ctx.failure(Node::Status::ExecutionFailed,
								   std::string("OnnxRuntime input conversion failed: ") + e.what());
			}
			auto* ortVal = native.as<Ort::Value>();
			if (!ortVal)
				return ctx.failure(Node::Status::InvalidInput,
								   "OnnxRuntime: input '" + port.name
									   + "' type not supported by converter");

			inputNames.push_back(port.name.c_str());
			// move 仅转移外壳；数据区在 Run 期间由 ctx 的 Value 保证存活
			inputValues.push_back(std::move(*ortVal));
		}

		std::vector<const char*> outputNames;
		outputNames.reserve(schema.outputs.size());
		for (const auto& port : schema.outputs) {
			outputNames.push_back(port.name.c_str());
		}

		std::vector<Ort::Value> outputs;
		try {
#ifdef DCINFER_ORT_ENABLE_CUDA
			if (deviceDirect) {
				checkCuda(cudaSetDevice(core->deviceId), "cudaSetDevice");
				Ort::IoBinding binding(session);
				for (size_t i = 0; i < inputNames.size(); ++i)
					binding.BindInput(inputNames[i], inputValues[i]);
				auto memInfo = cudaMemoryInfo(core->deviceId);
				for (const auto* name : outputNames)
					binding.BindOutput(name, memInfo);
				session.Run(Ort::RunOptions{nullptr}, binding);
				outputs = binding.GetOutputValues();
			} else
#endif
			{
				outputs = session.Run(
					Ort::RunOptions{nullptr},
					inputNames.data(),
					inputValues.data(),
					inputValues.size(),
					outputNames.data(),
					outputNames.size()
				);
			}
		} catch (const Ort::Exception& e) {
			return ctx.failure(Node::Status::ExecutionFailed,
							   std::string("OnnxRuntime inference failed: ") + e.what());
		} catch (const std::exception& e) {
			return ctx.failure(Node::Status::ExecutionFailed,
							   std::string("OnnxRuntime inference failed: ") + e.what());
		}

		// 缺失输出必须显式失败
		if (outputs.size() < schema.outputs.size()) {
			return ctx.failure(Node::Status::ExecutionFailed,
							   "OnnxRuntime: expected " + std::to_string(schema.outputs.size())
								   + " outputs, got " + std::to_string(outputs.size()));
		}

		// 收集输出：CUDA 直传模式持有 Ort::Value 以保活 arena 内存；其余走 host 深拷贝。
		for (size_t i = 0; i < schema.outputs.size(); ++i) {
#ifdef DCINFER_ORT_ENABLE_CUDA
			if (deviceDirect) {
				Tensor t;
				try {
					t = onnxToExternal(std::move(outputs[i]), coreHandle, core);
				} catch (const Ort::Exception& e) {
					return ctx.failure(Node::Status::ExecutionFailed,
									   std::string("OnnxRuntime output conversion failed: ") + e.what());
				} catch (const std::exception& e) {
					return ctx.failure(Node::Status::ExecutionFailed,
									   std::string("OnnxRuntime output conversion failed: ") + e.what());
				}
				ctx.output(schema.outputs[i].name, Value(std::make_unique<Tensor>(std::move(t))));
				continue;
			}
#endif
			Tensor t = converter->toDC(&outputs[i]);
			ctx.output(schema.outputs[i].name, Value(std::make_unique<Tensor>(std::move(t))));
		}

		return ctx.success();
	};
}

void registerOnnxEngine(EngineRegistry& reg, const OnnxOptions& opts) {
	EngineDescriptor desc;
	desc.engineType = "OnnxRuntime";

	desc.converter = {onnxToNative, onnxToDC};

	// createEngineCore：共享 Ort::Env；CUDA 直传模式下每个引擎核心拥有一条共享流。
	desc.createEngineCore = [opts]() -> EngineCore {
		auto core = std::make_shared<OrtCore>();
		core->env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DCinfer");
#ifdef DCINFER_ORT_ENABLE_CUDA
		if (opts.deviceDirect) {
			checkCuda(cudaSetDevice(core->deviceId), "cudaSetDevice");
			checkCuda(cudaStreamCreateWithFlags(&core->stream, cudaStreamNonBlocking),
					  "cudaStreamCreateWithFlags");
		}
#endif
		return EngineCore(std::move(core));
	};

	// loadModel：加载 Session，缓存键为 engineType:modelPath，每组合一次；Env 来自
	// 引擎核心，Session 不得超出 Env 生命周期，实例共享持有核心句柄由框架保证
	desc.loadModel = [opts](const EngineCore& core, const std::string& modelPath) -> EngineInstance {
		auto& ortCore = *static_cast<OrtCore*>(const_cast<void*>(core.get()));
		auto& env = *ortCore.env;
#ifdef DCINFER_ORT_ENABLE_CUDA
		if (opts.deviceDirect)
			checkCuda(cudaSetDevice(ortCore.deviceId), "cudaSetDevice");
#endif

		Ort::SessionOptions sessionOpts;
		sessionOpts.SetIntraOpNumThreads(opts.intraOpThreads); // 与 DCinfer 图级并行模型一致

		// 编译期默认 EP 由构建选项 DCINFER_ORT_EP 决定：先于 sessionCustomizer 追加以保证优先
#ifdef DCINFER_ORT_ENABLE_CUDA
		OrtCUDAProviderOptions cudaOpts{};
		if (opts.deviceDirect) {
			cudaOpts.has_user_compute_stream = 1;
			cudaOpts.user_compute_stream = ortCore.stream;
		}
		sessionOpts.AppendExecutionProvider_CUDA(cudaOpts);
#endif
#ifdef DCINFER_ORT_ENABLE_TENSORRT
		OrtTensorRTProviderOptions trtOpts{};
		sessionOpts.AppendExecutionProvider_TensorRT(trtOpts);
#endif
#ifdef DCINFER_ORT_ENABLE_OPENVINO
		OrtOpenVINOProviderOptions ovoOpts{};
		sessionOpts.AppendExecutionProvider_OpenVINO(ovoOpts);
#endif

		if (opts.sessionCustomizer)
			opts.sessionCustomizer(&sessionOpts);

		std::shared_ptr<Ort::Session> session;
		try {
			session = std::make_shared<Ort::Session>(env, toNativePath(modelPath).c_str(), sessionOpts);
		} catch (const Ort::Exception& e) {
			// 适配器封装契约：宿主不含 onnxruntime 头，引擎期异常统一转
			// std::runtime_error 上抛，如模型损坏、内核缺失、EP 不可用
			throw std::runtime_error(std::string("OnnxRuntime: failed to load model '")
												 + modelPath + "': " + e.what());
		}
		return EngineInstance(std::move(session));
	};

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

	// factory：用框架推导的 Schema 构造节点并绑定引擎实例；engineInstance 为
	// 框架缓存的共享句柄，引擎存活期覆盖节点存活期
	desc.factory = [opts](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		auto node = std::make_unique<Node>(
			"OnnxRuntime", p.nodeName, p.schema, onnxRunFn(opts.deviceDirect),
			ResourceClass::Compute);

		if (p.engineInstance)
			node->bindEngine(p.engineInstance, p.engineInstance->descriptor());

		return node;
	};

	// 同步引擎：仅 no-op synchronize，其余相位留空，逻辑内联 RunFn
	desc.phases.synchronize = [](void* /*engine*/) {
		// no-op: Ort::Session::Run() blocks until completion
	};

	reg.registerEngine(desc);
}

} // namespace DC::Onnx
