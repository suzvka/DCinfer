#include "InferOrt.h"
#include "tensor.h"
#include "tool.h"

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <stdexcept>

namespace DC {
	InferOrt::InferOrt(const std::filesystem::path& modelPath, size_t maxParallelCount) {
		setTypeMap();
		_maxParallelCount = maxParallelCount;

		std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
		if (!file.is_open()) {
			ready = false;
			errorMessage = "Failed to open ONNX model file: " + modelPath.string();
			return;
		}
		auto fileSize = file.tellg();
		file.seekg(0, std::ios::beg);
		std::vector<std::byte> buffer(fileSize);
		if (!file.read(reinterpret_cast<char*>(buffer.data()), fileSize)) {
			ready = false;
			errorMessage = "Failed to read ONNX model file: " + modelPath.string();
			return;
		}
		parseONNX(buffer);
	}

	InferOrt::InferOrt(const std::vector<std::byte>& modelData, size_t maxParallelCount) {
		setTypeMap();
		addEngine("ONNXRuntime", [](std::vector<std::byte> data, size_t maxParallelCount) { return new InferOrt(data, maxParallelCount); });
		_maxParallelCount = maxParallelCount;
		parseONNX(modelData);
	}

	Infer::Task InferOrt::Run(Infer::Task& inputs){
		if (!ready) {
			errorMessage = "InferOrt instance is not ready. Check errorMessage for details.";
			return {};
		}

		std::unique_lock<std::mutex> lock(_mutex);
		_condition.wait(lock, [this]() { return _maxParallelCount == 0 || _activeSessions < _maxParallelCount; });
		++_activeSessions;
		lock.unlock();

		auto results = runInfer(const_cast<Infer::Task&>(inputs));
		lock.lock();
		--_activeSessions;
		_condition.notify_one();
		return results.get();
	}

	void InferOrt::setTypeMap() {
		registerType<float>    (ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
		registerType<uint8_t>  (ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
		registerType<int8_t>   (ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);
		registerType<uint16_t> (ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);
		registerType<int16_t>  (ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16);
		registerType<int32_t>  (ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
		registerType<int64_t>  (ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
		registerType<bool>     (ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
		registerType<double>   (ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);
		registerType<uint32_t> (ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32);
		registerType<uint64_t> (ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64);
	}

	bool InferOrt::parseONNX(const std::vector<std::byte>& onnxData) {
		try {
			_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "DC-Infer");
			_options = Ort::SessionOptions();
			_options.SetIntraOpNumThreads(1);
			_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

			_session = std::make_unique<Ort::Session>(
				_env,
				onnxData.data(),
				onnxData.size(),
				_options
			);
			Ort::AllocatorWithDefaultOptions allocator;

			size_t numInputNodes = _session->GetInputCount();
			for (size_t i = 0; i < numInputNodes; i++) {
				auto* inputName = _session->GetInputNameAllocated(i, allocator).get();
				auto typeInfo = _session->GetInputTypeInfo(i);
				auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
				inputList[inputName] = createTensorSlot(inputName, tensorInfo);
				inputNames.push_back(inputName);
				allocator.Free(inputName);
			}

			size_t numOutputNodes = _session->GetOutputCount();
			for (size_t i = 0; i < numOutputNodes; i++) {
				auto* outputName = _session->GetOutputNameAllocated(i, allocator).get();
				auto typeInfo = _session->GetOutputTypeInfo(i);
				auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
				outputList[outputName] = createTensorSlot(outputName, tensorInfo);
				outputNames.push_back(outputName);
				allocator.Free(outputName);
			}

			ready = true;
			return true;
		}
		catch (const Ort::Exception& e) {
			errorMessage = "ONNX Runtime error: " + std::string(e.what());
			ready = false;
			return ready;
		}
		catch (const std::exception& e) {
			errorMessage = "Standard exception: " + std::string(e.what());
			ready = false;
			return ready;
		}
		catch (...) {
			errorMessage = "Unknown error occurred while parsing ONNX model.";
			ready = false;
			return ready;
		}

	}
	
	TensorSlot InferOrt::createTensorSlot(std::string name, const Ort::ConstTensorTypeAndShapeInfo& tensorInfo) {
		const auto& type = getType<ONNXTensorElementDataType>(tensorInfo.GetElementType());
		if (
			type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
			type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
			) {
			return TensorSlot(
				name,
				TensorMeta::TensorType::Float,
				"Float",
				tensorInfo.GetShape()
			);
		}
		else if (
			type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
			type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
			type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 ||
			type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
			) {
			return TensorSlot(
				name,
				TensorMeta::TensorType::Int,
				"Int",
				tensorInfo.GetShape()
			);
		}
		else if (
			type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 ||
			type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 ||
			type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 ||
			type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
			) {
			return TensorSlot(
				name,
				TensorMeta::TensorType::Uint,
				"Uint",
				tensorInfo.GetShape()
			);
		}
		else if (
			type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
			) {
			return TensorSlot(
				name,
				TensorMeta::TensorType::Bool,
				"Bool",
				tensorInfo.GetShape()
			);
		}
		else {
			throw std::runtime_error("Unsupported tensor type: " + std::to_string(tensorInfo.GetElementType()));
		}
	}

	InferBase::Results InferOrt::runInfer(Infer::Task& inputs){
		if (!prepareInputs(inputs)) return false;

		_inputTensors.reserve(inputList.size());
		for (auto& [name, slot] : inputList) {
			auto tensor = dynamic_cast<TensorOrt*>(&slot.getTensor());
			
			if (tensor) {
				auto tensor = dynamic_cast<TensorOrt*>(&slot.getTensor());
				_inputTensors.emplace_back(tensor->getValue());
			}
			else {
				auto ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
				if (slot.typeName() == "Float")			{ ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;	}
				else if (slot.typeName() == "Int64")	{ ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;	}
				else if (slot.typeName() == "Uint64")	{ ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;	}
				else if (slot.typeName() == "Bool")		{ ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;		}
				else {
					errorMessage = "Unsupported tensor type for input: " + slot.typeName();
					return false;
				}

				*tensor = TensorOrt(
					std::move(slot.getTensor()),
					ortType
				);

				_inputTensors.emplace_back(tensor->getValue());
			}
		}

		// 进行推理
		NameList inputNamesCStr, outputNamesCStr;

		inputNamesCStr.reserve(inputNames.size());
		for (const auto& name : inputNames) {
			inputNamesCStr.push_back(name.c_str());
		}
		outputNamesCStr.reserve(outputNames.size());
		for (const auto& name : outputNames) {
			outputNamesCStr.push_back(name.c_str());
		}

		try {
			_outputTensors = _session->Run(
				Ort::RunOptions{ nullptr },
				inputNamesCStr.data(),
				_inputTensors.data(),
				_inputTensors.size(),
				outputNamesCStr.data(),
				outputNamesCStr.size()
			);

			_inputTensors.clear();
		}
		catch (const Ort::Exception& e) {
			errorMessage = "ONNX Runtime error during inference: " + std::string(e.what());
			return false;
		}
		catch (const std::exception& e) {
			errorMessage = "Standard exception during inference: " + std::string(e.what());
			return false;
		}
		catch (...) {
			errorMessage = "Unknown error occurred during inference.";
			return false;
		}

		// 解析输出
		std::unordered_map<std::string, Tensor> results; // 不指定大小，让它为空初始化

		for (size_t i = 0; i < _outputTensors.size(); ++i) {
			results.emplace(outputNames[i], TensorOrt(
				outputList[outputNames[i]].name(),
				std::move(_outputTensors[i])
			));
		}
		
		_outputTensors.clear();
		return results;
	}
}
	