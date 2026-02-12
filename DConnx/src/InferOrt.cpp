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
		DC::Type::registerType<float>    (ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
		DC::Type::registerType<uint8_t>  (ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
		DC::Type::registerType<int8_t>   (ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);
		DC::Type::registerType<uint16_t> (ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);
		DC::Type::registerType<int16_t>  (ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16);
		DC::Type::registerType<int32_t>  (ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
		DC::Type::registerType<int64_t>  (ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
		DC::Type::registerType<bool>     (ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
		DC::Type::registerType<double>   (ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);
		DC::Type::registerType<uint32_t> (ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32);
		DC::Type::registerType<uint64_t> (ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64);
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
		const auto ortElemType = tensorInfo.GetElementType();

		const auto& type = Type::getType<ONNXTensorElementDataType>(ortElemType);

		switch (ortElemType) {
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: 
			return CreateSlot<float>(name, tensorInfo.GetShape());
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
			return CreateSlot<double>(name, tensorInfo.GetShape());
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
			return CreateSlot<int8_t>(name, tensorInfo.GetShape());
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
			return CreateSlot<int16_t>(name, tensorInfo.GetShape());
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			return CreateSlot<int32_t>(name, tensorInfo.GetShape());
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			return CreateSlot<int64_t>(name, tensorInfo.GetShape());
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
			return CreateSlot<uint8_t>(name, tensorInfo.GetShape());
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
			return CreateSlot<uint16_t>(name, tensorInfo.GetShape());
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
			return CreateSlot<uint32_t>(name, tensorInfo.GetShape());
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
			return CreateSlot<uint64_t>(name, tensorInfo.GetShape());
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
			return CreateSlot<bool>(name, tensorInfo.GetShape());
		default:
			throw std::runtime_error("Unsupported tensor type: " + std::to_string(ortElemType));
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