#include "Infer.h"
#include "tensor.h"
#include <onnxruntime_cxx_api.h>
#include <Value.h>
#include <iostream>
namespace DC {
    Infer::Infer(const std::vector<char>& onnxData, Ort::Env* env){
        try {
            this->_options = std::make_unique<Ort::SessionOptions>();
            //this->_options->EnableMemPattern();
            //this->_options->EnableCpuMemArena();
            this->_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(*_options, 0)); // 使用 CUDA 提供者
            this->_options->AddConfigEntry("trt_fp16_enable", "1");

            if (env == nullptr) {
                this->env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "ONNXModel");
            }
            else {
                this->env.reset(env);
            }
            
            session = std::make_unique<Ort::Session>(*env, onnxData.data(), onnxData.size(), *_options);
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
    ) {
        if (!ready) {
            throw std::runtime_error("Worker is not ready for inference.");
        }

        // 准备输入张量
        alignas(8) auto preparedInputs = prepareInputs(inputs);

        // 获取输入和输出名称
        alignas(8) std::vector<const char*> inputNames;
        for (const auto& [name, _] : tensorList) {
            if (inputTensorInfo[name].IOtype) {
                inputNames.push_back(name.c_str());
            }
        }

        size_t numOutputs = session->GetOutputCount();
        alignas(8) std::vector<const char*> outputNames(numOutputs);
        for (size_t i = 0; i < numOutputs; ++i) {
            auto outputName = session->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            outputNames[i] = _strdup(outputName.get());
        }

        // 执行推理
        auto outputValues = session->Run(
            Ort::RunOptions{ nullptr },
            inputNames.data(),
            preparedInputs.data(),
            inputNames.size(),
            outputNames.data(),
            numOutputs
        );

        // 构建结果字典并过滤无效输出
        std::unordered_map<std::string, Tensor> resultMap;
        for (size_t i = 0; i < numOutputs; ++i) {
            const char* name = outputNames[i];
            auto& ortValue = outputValues[i];

            // 检查是否为有效张量
            if (!ortValue.IsTensor()) {
                continue; // 非张量类型，跳过
            }

            try {
                auto tensorInfo = ortValue.GetTensorTypeAndShapeInfo();
                size_t elementCount = tensorInfo.GetElementCount();
                float* dataPtr = ortValue.GetTensorMutableData<float>();

                // 验证元素数量和数据指针
                if (elementCount == 0 || dataPtr == nullptr || tensorInfo.GetShape().empty()) {
                    continue; // 空张量或无效数据，跳过
                }

                // 转换为自定义 Tensor
                resultMap[name] = Tensor(name, ortValue);
            }
            catch (const Ort::Exception& e) {
                // 处理可能的异常（例如无效张量结构）
                std::cerr << "Error processing output '" << name << "': " << e.what() << std::endl;
                continue;
            }
        }

        // 清理动态分配的 outputNames
        for (const char* name : outputNames) {
            delete[] name; // 使用 _strdup 分配需用 delete[] 释放
        }

        return resultMap;
    }
}