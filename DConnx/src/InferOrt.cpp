#include "InferOrt.h"
#include "tensor.h"
#include "tool.h"

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <stdexcept>

namespace DC {
	TypeManager<ONNXTensorElementDataType> InferOrt::_typeMap;

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
		_maxParallelCount = maxParallelCount;
		parseONNX(modelData);
	}

	Infer::Results InferOrt::Run(const std::vector<Tensor>& inputs){
		if (!ready) {
			errorMessage = "InferOrt instance is not ready. Check errorMessage for details.";
			return false;
		}

		std::unique_lock<std::mutex> lock(_mutex);
		_condition.wait(lock, [this]() { return _maxParallelCount == 0 || _activeSessions < _maxParallelCount; });
		++_activeSessions;
		lock.unlock();

		auto results = runInfer(const_cast<std::vector<Tensor>&>(inputs));
		lock.lock();
		--_activeSessions;
		_condition.notify_one();
		return results;
	}

	void InferOrt::setTypeMap() {
		_typeMap.registerType<float>    (ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT    , "Float"   );
		_typeMap.registerType<uint8_t>  (ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8    , "UInt8"   );
		_typeMap.registerType<int8_t>   (ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8     , "Int8"    );
		_typeMap.registerType<uint16_t> (ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16   , "UInt16"  );
		_typeMap.registerType<int16_t>  (ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16    , "Int16"   );
		_typeMap.registerType<int32_t>  (ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32    , "Int32"   );
		_typeMap.registerType<int64_t>  (ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64    , "Int64"   );
		_typeMap.registerType<bool>     (ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL     , "Bool"    );
		_typeMap.registerType<double>   (ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE   , "Double"  );
		_typeMap.registerType<uint32_t> (ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32   , "UInt32"  );
		_typeMap.registerType<uint64_t> (ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64   , "UInt64"  );
	}

	Infer::Promise InferOrt::parseONNX(const std::vector<std::byte>& onnxData) {
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
			return ErrorCode::ERROR_TENSOR;
		}
		catch (const std::exception& e) {
			errorMessage = "Standard exception: " + std::string(e.what());
			ready = false;
			return ErrorCode::ERROR_TENSOR;
		}
		catch (...) {
			errorMessage = "Unknown error occurred while parsing ONNX model.";
			ready = false;
			return ErrorCode::ERROR_TENSOR;
		}

	}
	
	TensorSlot InferOrt::createTensorSlot(std::string name, const Ort::ConstTensorTypeAndShapeInfo& tensorInfo) {
		auto& typeName = _typeMap.fromEnum(tensorInfo.GetElementType()).getTypeName();
		if (
			typeName == "Float" ||
			typeName == "Double"
			) {
			return TensorSlot(
				name,
				TensorMeta::TensorType::Float,
				typeName,
				tensorInfo.GetShape()
			);
		}
		else if (
			typeName == "Int64" ||
			typeName == "Int32" ||
			typeName == "Int16" ||
			typeName == "Int8"
			) {
			return TensorSlot(
				name,
				TensorMeta::TensorType::Int,
				typeName,
				tensorInfo.GetShape()
			);
		}
		else if (
			typeName == "UInt64" ||
			typeName == "UInt32" ||
			typeName == "UInt16" ||
			typeName == "UInt8"
			) {
			return TensorSlot(
				name,
				TensorMeta::TensorType::Uint,
				typeName,
				tensorInfo.GetShape()
			);
		}
		else if (
			typeName == "Bool"
			) {
			return TensorSlot(
				name,
				TensorMeta::TensorType::Bool,
				typeName,
				tensorInfo.GetShape()
			);
		}
		else {
			throw std::runtime_error("Unsupported tensor type: " + std::to_string(tensorInfo.GetElementType()));
		}
	}

	Infer::Results InferOrt::runInfer(std::vector<Tensor>& inputs){
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
				if (slot.type() == "Float")			{ ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;	}
				else if (slot.type() == "Int64")	{ ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;	}
				else if (slot.type() == "Uint64")	{ ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;	}
				else if (slot.type() == "Bool")		{ ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;		}
				else if (slot.type() == "Int32")	{ ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;	}
				else if (slot.type() == "Uint32")	{ ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;	}
				else if (slot.type() == "Int16")	{ ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;	}
				else if (slot.type() == "Uint16")	{ ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;	}
				else if (slot.type() == "Int8")		{ ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;		}
				else if (slot.type() == "Uint8")	{ ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;	}
				else if (slot.type() == "Double")	{ ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;	}
				else {
					errorMessage = "Unsupported tensor type for input: " + slot.type();
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
		std::unordered_map<std::string, Tensor> results(_outputTensors.size());

		for (size_t i = 0; i < _outputTensors.size(); ++i) {
			results[outputNames[i]] = TensorOrt(
				outputList[outputNames[i]].name(),
				std::move(_outputTensors[i])
			);
		}
		
		_outputTensors.clear();
		return results;
	}
}
	