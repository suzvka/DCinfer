#include "tensor.h"
#include "Value.h"

namespace DC {
    Tensor::Tensor(){
		value.reset(new Value());
    }
    Tensor::Tensor(const std::string& name, const std::string& type, const std::vector<int64_t>& shape)
		:_name(name), _type(type), _shape(shape), _tensorData(type), value(new Value()){
		// 找不到 float 和 int64
		// 这是为什么呢
		auto it = getName.find(type);
		if (it != getName.end() && it->second != getName.at(type)) {
			throw("不支持的张量类型：" + type);
		}
		value.reset(new Value());
	}
	Tensor::Tensor(const std::string& name,const Ort::Value& newValue):value(new Value())
	{
		value->load(newValue);
		_name = name;
		_type = value->getType();
		_shape = value->getShape();
		_tensorData = TensorEdit(_type);
		_tensorData.set(_shape, value->getData());
	}
	Ort::Value Tensor::getValue() const  {
        return value->getValue();
    }

}