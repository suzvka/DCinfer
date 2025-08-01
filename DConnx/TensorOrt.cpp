#pragma once
#include <onnxruntime_cxx_api.h>
#include "tensor.h"

// onnxruntime 支持
// 继承 DC::Tensor，可以直接当作 Ort::Value 使用

namespace DC {
	class TensorOrt : public DC::Tensor {
	public:
		// 构造函数
		template<typename T>
		TensorOrt(const std::string& name, Ort::Value& value)
			: DC::Tensor(name, value.GetTensorTypeAndShapeInfo().GetShape()){
			_value = std::move(value);
		}

		// 隐式类型转换
		// 返回 Ort::Value 引用，与onnxruntime 保持一致
		operator Ort::Value& () {
			return _value;
		}

	private:
		Ort::Value _value;
	};
}