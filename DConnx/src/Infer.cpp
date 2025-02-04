#include "Infer.h"
#include "tensor.h"
#include <onnxruntime_cxx_api.h>
#include <Value.h>
#include <iostream>
namespace DC {
    Infer::Infer(const std::vector<char>& onnxData, Ort::Env* env){
        try {
            Ort::SessionOptions session_options;
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0)); // 使用 CUDA 提供者

            if (env == nullptr) {
                this->env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "ONNXModel");
            }
            else {
                this->env.reset(env);
            }
            
            session = std::make_unique<Ort::Session>(*env, onnxData.data(), onnxData.size(), session_options);
            parseInputTensorInfo(); // 初始化输入张量信息
            ready = true;
        }
        catch (const Ort::Exception& e) {
            errorMessage = "ONNX Runtime Error: " + std::string(e.what());
        }
        catch (const std::exception& e) {
            errorMessage = "Standard Error: " + std::string(e.what());
        }
        catch (...) {
            errorMessage = "Unknown Error occurred during worker initialization.";
        }
    }

    void Infer::parseInputTensorInfo() {
        size_t numInputs = session->GetInputCount();
        for (size_t i = 0; i < numInputs; ++i) {
            auto tensorNameCstr = session->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            std::string tensorName = tensorNameCstr.get();
            auto typeInfo = session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();

            ONNXTensorElementDataType type = typeInfo.GetElementType();
            std::vector<int64_t> shape = typeInfo.GetShape();
            recordTensor(
                tensorName, 
                getName.at(findEnum.at(type)),
                shape,
                true
            );
        }
        size_t numOutputs = session->GetOutputCount();
        for (size_t i = 0; i < numOutputs; ++i) {
            auto tensorNameCstr = session->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            std::string tensorName = tensorNameCstr.get();
            auto typeInfo = session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();

            ONNXTensorElementDataType type = typeInfo.GetElementType();
            std::vector<int64_t> shape = typeInfo.GetShape();
            recordTensor(
                tensorName,
                findEnum.at(type),
                shape,
                false
            );
        }
    }

    void Infer::recordTensor(std::string name, std::string type, std::vector<int64_t> shape, bool IOtype){
        tensorList[name] = Tensor(name, type, shape);
        tensorInfo tInfo;
        tInfo.name = tensorList[name].name();
        tInfo.type = tensorList[name].type();
        tInfo.shape = tensorList[name].shape();
        tInfo.IOtype = IOtype;
        inputTensorInfo[name] = tInfo;
    }

    const std::map<std::string, tensorInfo>& Infer::getInfo() const {
        return inputTensorInfo;
    }

    Infer& Infer::config(){
        return *this;
    }

    int Infer::check(const std::vector<Tensor>& inputs) {
        for (const auto& atensor : inputs) {
            // 检查输入张量是否存在
            auto it = inputTensorInfo.find(atensor.name());
            if (it == inputTensorInfo.end()) {
                return MISSING_TENSOR;
            }

            // 检查输入值是否符合定义
            //auto& t = it->second;
            //if (t.shape != value.type().name()) {
            //    return SHAPE_MISMATCH;
            //}
        }
        return VALIDATION_SUCCESS;
    }

    std::vector<Ort::Value> Infer::prepareInputs(
        const std::vector<Tensor>& inputs
    ) {
        std::vector<Ort::Value> preparedInputs;
        std::unordered_set<std::string> missingInputs; // 用于记录缺失的输入张量名
        for (auto& [name, tensor] : tensorList) {
            bool found = false;
            for (auto& atensor : inputs) {
                if (atensor.name() == name && atensor.type() == tensor.type()) {
                    tensor.copy(atensor)
                        .load();
                    if (!tensor.getValue().HasValue()) {
                        throw std::runtime_error("张量加载失败：" + name);
                    }
                    preparedInputs.push_back(tensor.getValue());
                    found = true;
                    break;
                }
            }
            if (!found && inputTensorInfo[name].IOtype) {
                // 如果没有输入，尝试在默认值中找
                auto it = defaultList.find(name);
                if (it != defaultList.end() && it->second.type() == tensor.type()) {
                    preparedInputs.push_back(it->second.getValue());
                }
                else {
                    missingInputs.insert(name); // 如果没有找到该输入张量，记录它
                }
            }
        }
        // 如果 missingInputs 中有缺失的输入，抛出异常并列出缺失的张量名
        if (!missingInputs.empty()) {
            std::string missingNames;
            for (const auto& missingName : missingInputs) {
                missingNames += missingName + " ";
            }
            throw std::runtime_error("缺少输入: " + missingNames);
        }
        return preparedInputs;
    }

    std::unordered_map<std::string, Tensor> Infer::Run(
        const std::vector<Tensor>& inputs
    ){
        if (!ready) {
            throw std::runtime_error("Worker is not ready for inference.");
        }

        // 准备输入张量
        alignas(8) auto preparedInputs = prepareInputs(inputs);

        // 获取输入和输出的名称数组
        alignas(8) std::vector<const char*> inputNames;
        for (const auto& [name, _] : tensorList) {
            if (!inputTensorInfo[name].IOtype) {
                continue;
            }
            inputNames.push_back(name.c_str());
        }

        size_t numOutputs = session->GetOutputCount();
        alignas(8) std::vector<const char*> outputNames(numOutputs);
        for (size_t i = 0; i < numOutputs; ++i) {
            auto outputName = session->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            outputNames[i] = _strdup(outputName.get()); // 由于获取的是引用，因此需要显式复制，避免生命周期导致的悬空
        }

        // 调用 Run 方法进行推理
        auto outputValues = session->Run(
            Ort::RunOptions{ nullptr },         // 默认运行选项
            inputNames.data(),                  // 输入张量名称数组
            preparedInputs.data(),              // 输入张量数组
            inputNames.size(),                  // 输入张量数量
            outputNames.data(),                 // 输出张量名称数组
            numOutputs                          // 输出张量数量
        );

        // 定义用于存储推理结果的字典
        std::unordered_map<std::string, Tensor> resultMap;

        // 遍历输出张量
        for (size_t i = 0; i < numOutputs; ++i) {
            // 获取输出张量的数据指针和大小
            auto& outputTensor = outputValues[i];
            // 将数据存储到结果字典中，键为输出张量名称
            resultMap[outputNames[i]] = Tensor(outputNames[i], outputTensor);
        }
        for (const char* name : outputNames) {
            delete (const_cast<char*>(name));  // 释放 _strdup 分配的内存
        }
        // 返回推理结果字典
        return resultMap;
    }
}