#include <cuda_runtime.h>
#include <iostream>
#include <info.h>
#include <onnxruntime_cxx_api.h>

namespace DC {
    std::vector<GPUInfo> getInfo_GPU() {
        std::vector<GPUInfo> gpuInfos;
        int device_count = 0;

        // 获取可用的GPU设备数
        cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
        if (cuda_status != cudaSuccess) {
            std::cerr << "Error: Unable to get device count: " << cudaGetErrorString(cuda_status) << std::endl;
            return gpuInfos; // 返回空向量
        }

        for (int i = 0; i < device_count; i++) {
            cudaDeviceProp device_prop;
            cuda_status = cudaGetDeviceProperties(&device_prop, i);

            if (cuda_status != cudaSuccess) {
                std::cerr << "Error: Unable to get device properties for device " << i << ": " << cudaGetErrorString(cuda_status) << std::endl;
                continue;
            }

            GPUInfo info;
            info.name = device_prop.name;
            info.cudaCapability = std::to_string(device_prop.major) + "." + std::to_string(device_prop.minor);
            info.totalGlobalMemMB = device_prop.totalGlobalMem / (1024 * 1024);
            info.sharedMemPerBlockKB = device_prop.sharedMemPerBlock / 1024;
            info.regsPerBlock = device_prop.regsPerBlock;
            info.warpSize = device_prop.warpSize;
            info.maxThreadsPerBlock = device_prop.maxThreadsPerBlock;
            info.maxThreadsDim = { device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2] };
            info.maxGridSize = { device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2] };
            info.memoryClockRateMHz = device_prop.memoryClockRate / 1000;
            info.memoryBusWidthBits = device_prop.memoryBusWidth;
            info.peakMemoryBandwidthGBs = 2.0 * device_prop.memoryClockRate * (device_prop.memoryBusWidth / 8) / 1.0e6;

            gpuInfos.push_back(info);
        }

        return gpuInfos;
    }

    std::map<std::string, tensorInfo> getInfo_ONNX(const std::vector<char>& onnxModelData) {
        std::map<std::string, tensorInfo> tensorInfoMap;

        try {
            Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "ONNXModelInfo");
            Ort::SessionOptions session_options;
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0)); // 使用 CUDA 提供者

            Ort::Session session(env, onnxModelData.data(), onnxModelData.size(), session_options);

            size_t num_inputs = session.GetInputCount();

            for (size_t i = 0; i < num_inputs; ++i) {
                // 获取输入名称
                auto input_name_cstr = session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
                auto input_name = input_name_cstr.get();

                // 获取输入类型信息
                Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

                // 获取数据类型
                ONNXTensorElementDataType type = tensor_info.GetElementType();
                std::string type_str;
                switch (type) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: type_str = "float"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: type_str = "uint8"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: type_str = "int8"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: type_str = "uint16"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: type_str = "int16"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: type_str = "int32"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: type_str = "int64"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: type_str = "string"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: type_str = "bool"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: type_str = "float16"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: type_str = "double"; break;
                    // 根据需要添加更多类型
                default: type_str = "unknown"; break;
                }

                // 获取张量形状
                std::vector<int64_t> input_shape = tensor_info.GetShape();

                // 填充ONNXTensorInfo结构体
                tensorInfo info;
                info.type = type_str;
                info.shape = input_shape;

                tensorInfoMap[input_name] = info;
            }
        }
        catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
            // 返回空字典
        }
        catch (const std::exception& e) {
            std::cerr << "Standard Error: " << e.what() << std::endl;
            // 返回空字典
        }

        return tensorInfoMap;
    }
}