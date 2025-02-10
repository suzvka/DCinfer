#include "Value.h"

namespace DC {
    // 从 Ort::Value 中直接加载数据，一般用来接受输出
    Value::Value(Ort::Value& ortValue){
        load(ortValue);
    }
    // 从输入数据中加载
    template<typename T>
    Value::Value(const std::vector<T>& value) {
        load(value);
    }
    // 重新加载数据结构
	void Value::load(const Ort::Value& inOrtValue) {
		// 忽略空 Ort::Value
		if (!inOrtValue) {
			return;
		}
        auto typeInfo = inOrtValue.GetTensorTypeAndShapeInfo();
        auto shape = typeInfo.GetShape();
        auto onnxType = typeInfo.GetElementType();
        
        parseEnum(inOrtValue); // 解析数据
    }
    template<typename T>
    Value::Value(const std::vector<T>& value, const std::vector<int64_t>& shape){
        load<T>(value, shape);
    }
    
    Ort::Value Value::getValue() const {
        auto onnxType = findType.at(_type);          // 转换类型格式
        return Ort::Value::CreateTensor(
            *memoryInfo,
            const_cast<char*>(_data.data()),
            calculateTotalSize(_shape, onnxType),
            _shape.data(),
            _shape.size(),
            onnxType
        );
    }

    const std::vector<char> Value::getData() const{
        return _data;
    }

    const std::string Value::getType() const{
        return _type;
    }

    const std::vector<int64_t> Value::getShape() const{
        return _shape;
    }

    // 初始化数据结构
    void Value::init(){
        // 准备内存
        memoryInfo.reset(
            new Ort::MemoryInfo(
                Ort::MemoryInfo::CreateCpu(
                    OrtArenaAllocator,          // 使用内存池
                    OrtMemTypeDefault           // 默认内存分配方式，一般会分在显存上
                )
            )
        );
        _data = std::vector<char>();  // 初始化为空的字节容器
    }

    Value& Value::copy(const Ort::Value& inValue) const{
        auto typeInfo = inValue.GetTensorTypeAndShapeInfo();
        auto shape = typeInfo.GetShape();

        return *new Value(_data, shape);
    }

    // 从 Ort::Value 方向解析数据
    void Value::parseEnum(const Ort::Value& inOrtValue){
        //auto typeInfo = inOrtValue
        //    .GetTypeInfo()
        //    .GetTensorTypeAndShapeInfo();
        const char* OrtData = inOrtValue.GetTensorData<char>();
        size_t elementCount = inOrtValue.GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementCount();
        // 将数据提取出来
        auto elemenSize = typeSize.at(inOrtValue.GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementType());
        _data = std::vector<char>(OrtData, OrtData + elementCount * elemenSize);
        _shape = inOrtValue.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
        _type = findEnum.at(inOrtValue.GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementType());
    }

    // 计算总数据大小
    size_t Value::calculateTotalSize(const std::vector<int64_t>& shape, ONNXTensorElementDataType onnxType) {
        size_t totalSize = 1;
        for (const auto& dim : shape) {
            totalSize *= dim;
        }
        return totalSize * typeSize.at(onnxType);
    }
}