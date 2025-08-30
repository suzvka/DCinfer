#pragma once
#include "Tensor.h"

#include <onnxruntime_cxx_api.h>

namespace DC{
	class TensorOrt : public Tensor {
	public:
		using OrtType = ONNXTensorElementDataType;
		TensorOrt(
			const std::string& name,				// - 张量名称
			const OrtType& type,					// - 张量类型
			const std::vector<int64_t>& shape = {},	// - 张量形状
			std::vector<char>&& data = {}			// - 张量数据
		) : Tensor(name, TensorType::Void, shape, std::move(data)) {}

		TensorOrt(
			Tensor&& tensor,
			const OrtType& type
		) : Tensor(std::move(tensor)) {
			_type = type;
		}

		TensorOrt(
			const std::string& name,
			Ort::Value&& value
		) :TensorOrt(
			name,
			value.GetTensorTypeAndShapeInfo().GetElementType(),
			value.GetTensorTypeAndShapeInfo().GetShape()
		) {
			_ortValue = std::make_unique<Ort::Value>(std::move(value));
		}

		template<typename T>
		bool load() {
			try {
				auto data = getData<T>();

				_ortValue = std::make_unique<Ort::Value>(
					Ort::Value::CreateTensor<T>(
						Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
						data.data(),
						data.size(),
						shape().data(),
						shape().size()
					)
				);
			}
			catch (...) {
				return false;
			}
		}

		template<>
		bool load<bool>() {
			try {
				auto vecBool = getData<bool>();
				std::vector<uint8_t> data;
				data.reserve(vecBool.size());
				for (bool b : vecBool) {
					data.push_back(static_cast<uint8_t>(b));
				}

				_ortValue = std::make_unique<Ort::Value>(
					Ort::Value::CreateTensor(
						Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
						data.data(),
						data.size(),
						shape().data(),
						shape().size(),
						ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
					)
				);

				return true;
			}
			catch (...) {
				return false;
			}
		}

		Ort::Value getValue() {
			if (!_ortValue) {
				switch (ortType()) {
					case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT :{
						load<float>();
						break;
					}
					case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 :{
						load<int64_t>();
						break;
					}
					case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 :{
						load<uint64_t>();
						break;
					}
					case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL :{
						load<bool>();
						break;
					}
					default: {
						throw std::runtime_error("Unsupported tensor type for Ort::Value: " + std::to_string(ortType()));
						break;
					}
				}
			}
			return std::move(*_ortValue);
		}

		ONNXTensorElementDataType ortType() const {
			return _type;
		}

	private:
		std::unique_ptr<Ort::Value> _ortValue = nullptr;
		ONNXTensorElementDataType _type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
		static TypeManager<ONNXTensorElementDataType> _typeMap;

		void setTypeMap() {
			_typeMap.registerType<float>	(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT	, "Float"	);
			_typeMap.registerType<uint8_t>	(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8	, "UInt8"	);
			_typeMap.registerType<int8_t>	(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8		, "Int8"	);
			_typeMap.registerType<uint16_t>	(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16	, "UInt16"	);
			_typeMap.registerType<int16_t>	(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16	, "Int16"	);
			_typeMap.registerType<int32_t>	(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32	, "Int32"	);
			_typeMap.registerType<int64_t>	(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64	, "Int64"	);
			_typeMap.registerType<bool>		(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL		, "Bool"	);
			_typeMap.registerType<double>	(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE	, "Double"	);
			_typeMap.registerType<uint32_t>	(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32	, "UInt32"	);
			_typeMap.registerType<uint64_t>	(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64	, "UInt64"	);
		}
	};
}
