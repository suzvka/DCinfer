//#pragma once
//#include "Tensor.h"
//
//#if defined(DCONNX_ENABLE_ORT)
//#include <onnxruntime_cxx_api.h>
//#endif
//
//namespace DC{
//	#if defined(DCONNX_ENABLE_ORT)
//	class TensorOrt : public Tensor {
//	public:
//		using OrtType = ONNXTensorElementDataType;
//		TensorOrt(
//			const std::string& name,				// - 张量名称
//			const OrtType& type,					// - 张量类型
//			const std::vector<int64_t>& shape = {},	// - 张量形状
//			std::vector<char>&& data = {}			// - 张量数据
//		) : Tensor(name, TensorType::Void, shape, std::move(data)) {}
//
//		TensorOrt(
//			Tensor&& tensor,
//			const OrtType& type
//		) : Tensor(std::move(tensor)) {
//			_type = type;
//		}
//
//		TensorOrt(
//			const std::string& name,
//			Ort::Value&& value
//		) :TensorOrt(
//			name,
//			value.GetTensorTypeAndShapeInfo().GetElementType(),
//			value.GetTensorTypeAndShapeInfo().GetShape()
//		) {
//			// 直接接收为稠密数据，避免重复进行数据块登记
//			auto info = value.GetTensorTypeAndShapeInfo();
//			auto shp = info.GetShape();
//			const auto elemType = info.GetElementType();
//			size_t elemSize = 0;
//			switch (elemType) {
//				case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: elemSize = sizeof(float); break;
//				case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: elemSize = sizeof(double); break;
//				case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: elemSize = sizeof(int8_t); break;
//				case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: elemSize = sizeof(int16_t); break;
//				case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: elemSize = sizeof(int32_t); break;
//				case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: elemSize = sizeof(int64_t); break;
//				case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: elemSize = sizeof(uint8_t); break;
//				case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: elemSize = sizeof(uint16_t); break;
//				case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: elemSize = sizeof(uint32_t); break;
//				case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: elemSize = sizeof(uint64_t); break;
//				case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: elemSize = sizeof(uint8_t); break;
//				default: elemSize = 0; break;
//			}
//
//			size_t elemCount = 1;
//			for (auto d : shp) {
//				elemCount *= static_cast<size_t>(d > 0 ? d : 0);
//			}
//
//			std::vector<char> bytes;
//			if (elemSize > 0 && elemCount > 0) {
//				bytes.resize(elemCount * elemSize);
//				std::memcpy(bytes.data(), value.GetTensorRawData(), bytes.size());
//				setDense(std::move(bytes), shp);
//			}
//
//			_ortValue = std::make_unique<Ort::Value>(std::move(value));
//		}
//
//		template<typename T>
//		bool load() {
//			try {
//				const auto& bytes = getBytes();
//				if (bytes.empty()) {
//					_ortValue.reset();
//					return false;
//				}
//
//				auto* ptr = reinterpret_cast<T*>(const_cast<char*>(bytes.data()));
//				const size_t count = bytes.size() / sizeof(T);
//
//				_ortValue = std::make_unique<Ort::Value>(
//					Ort::Value::CreateTensor<T>(
//						Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
//						ptr,
//						count,
//						shape().data(),
//						shape().size()
//					)
//				);
//				return true;
//			}
//			catch (...) {
//				return false;
//			}
//		}
//
//		template<>
//		bool load<bool>() {
//			try {
//				auto vecBool = getData<bool>();
//				std::vector<uint8_t> data;
//				data.reserve(vecBool.size());
//				for (bool b : vecBool) {
//					data.push_back(static_cast<uint8_t>(b));
//				}
//
//				_ortValue = std::make_unique<Ort::Value>(
//					Ort::Value::CreateTensor(
//						Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
//						data.data(),
//						data.size(),
//						shape().data(),
//						shape().size(),
//						ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
//					)
//				);
//
//				return true;
//			}
//			catch (...) {
//				return false;
//			}
//		}
//
//		Ort::Value getValue() {
//			if (!_ortValue) {
//				switch (ortType()) {
//					case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT :{
//						load<float>();
//						break;
//					}
//					case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 :{
//						load<int64_t>();
//						break;
//					}
//					case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 :{
//						load<uint64_t>();
//						break;
//					}
//					case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL :{
//						load<bool>();
//						break;
//					}
//					default: {
//						throw std::runtime_error("Unsupported tensor type for Ort::Value: " + std::to_string(ortType()));
//						break;
//					}
//				}
//			}
//			return std::move(*_ortValue);
//		}
//
//		ONNXTensorElementDataType ortType() const {
//			return _type;
//		}
//
//	private:
//		std::unique_ptr<Ort::Value> _ortValue = nullptr;
//		ONNXTensorElementDataType _type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
//
//		void setTypeMap() {
//			DC::Type::registerType<float>			(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
//			DC::Type::registerType<uint8_t>		(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
//			DC::Type::registerType<int8_t>		(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);
//			DC::Type::registerType<uint16_t>		(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);
//			DC::Type::registerType<int16_t>		(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16);
//			DC::Type::registerType<int32_t>		(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
//			DC::Type::registerType<int64_t>		(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
//			DC::Type::registerType<bool>			(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
//			DC::Type::registerType<double>		(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);
//			DC::Type::registerType<uint32_t>		(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32);
//			DC::Type::registerType<uint64_t>		(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64);
//		}
//	};
//	#endif
//}
