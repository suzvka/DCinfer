#pragma once
#include <memory>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include "tool.h"

namespace DC {
    // DC::Value
    // 在内存中保存原始张量数据
    // 并且可以随意复制、修改
    // 需要时，使用这里面的方法生成相应推理引擎的张量
    // 输入与输出都是一维化的二进制流
    // 需要提供、可以提供这些数据的形状
    // 
    // 支持
    // - onnxruntime
    // -- Ort::Value
    class Value {
    public:
        // 默认构造，不做任何事
        // 当然，也就不消耗性能
        // 你想先创建一个对象再赋值时有用
        Value() {}
        // 从 Ort::Value 中构造
        // - Ort::Value 对象
        Value(
            Ort::Value& ortValue                  // Ort::Value 对象
        );
        // 直接提供输入数据来构造
        // - 张量数据
        template<typename T> Value(
            const std::vector<T>& value                 // 张量数据
        );
        // 直接提供输入数据来构造
        // - 张量数据
        // - 张量形状
        template<typename T> Value(
            const std::vector<T>& value,                // 张量数据
            const std::vector<int64_t>& shape           // 张量形状
        );

        // 重新加载张量
        // - Ort::Value 对象
        // - 重新分配显存
        void load(
            const Ort::Value& ortValue                  // Ort::Value 对象
        );
        // 重新加载张量
        // - 张量数据
        template<typename T> void load(
            const std::vector<T>& value,                // 张量数据
            const std::vector<int64_t>& shape           // 张量形状
        ) {
            init();                                     // 初始化
            std::string type = typeid(value[0]).name();
            auto onnxType = findType.at(type);          // 转换类型格式
            std::vector<char> byteStream;
            // 遍历原始向量中的每个元素
            for (const T& element : value) {
                const char* data_ptr = reinterpret_cast<const char*>(&element);
                for (size_t i = 0; i < sizeof(T); ++i) {
                    byteStream.push_back(data_ptr[i]);
                }
            }
            _data = byteStream;
            _shape = shape;
            _type = type;
        }

        Ort::Value getValue()const;                     // 获取张量
        const std::vector<char> getData() const;        // 获取数据
        const std::string getType() const;              // 获取类型
        const std::vector<int64_t> getShape() const;    // 获取形状

    private:
        std::unique_ptr<Ort::MemoryInfo> memoryInfo;    // Ort 张量依赖的显存
        std::string _type;                              // 张量类型
        std::vector<char> _data;                        // 原始张量数据
        std::vector<int64_t> _shape;                    // 张量形状

        // 初始化
        void init();

        // 拷贝 Ort::Value
        Value& copy(const Ort::Value& inValue) const;

        // 解析 Ort 张量
        void parseEnum(const Ort::Value& inOrtValue);

        // 计算张量数据大小
        static size_t calculateTotalSize(
            const std::vector<int64_t>& shape, 
            ONNXTensorElementDataType onnxType
        );
        
        //张量扁平化为字节流================================================================
        // 更精确的模板会优先匹配
        template <typename T>
        const std::vector<char>& toOneDim(std::vector<char>& result, const std::vector<T>& nestedVec) {
            for (const auto& elem : nestedVec) {
                lowerDim(result, elem);
            }
            return result;
        }
        // 递归终止条件：如果当前不是std::vector，则将其值追加到结果中---------
        template <typename T>
        const std::vector<char>& toOneDim(std::vector<char>& result, const T& value) {
            lowerDim(result, value);
            return result;
        }
        // 降低维度------------------------------------------------------------
        // 辅助函数，用于将任意类型的值追加到std::vector<char>中
        template <typename T>
        void lowerDim(std::vector<char>& result, const T& value) {
            static_assert(std::is_trivially_copyable_v<T>, "Value type must be trivially copyable");
            const char* data = reinterpret_cast<const char*>(&value);
            result.insert(result.end(), data, data + sizeof(T));
        }
        // 降低维度------------------------------------------------------------
        // 特化处理bool类型
        template <>
        void lowerDim<bool>(std::vector<char>& result, const bool& value) {
            char boolAsChar = value ? 1 : 0; // 将bool转换为1或0
            result.push_back(boolAsChar);
        }
        // 降低维度------------------------------------------------------------
        // 特化处理std::string类型
        template <>
        void lowerDim<std::string>(std::vector<char>& result, const std::string& value) {
            // 1. 存储字符串长度
            size_t length = value.size();
            const char* lengthData = reinterpret_cast<const char*>(&length);
            result.insert(result.end(), lengthData, lengthData + sizeof(size_t));

            // 2. 存储字符串内容
            result.insert(result.end(), value.begin(), value.end());
        }
        //================================================================张量扁平化为字节流
    };
}