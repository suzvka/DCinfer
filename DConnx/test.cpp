// main.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include "Infer.h"
#include "tensor.h"

#include <windows.h>

extern void test1();
// 读取 ONNX 文件到 std::vector<char>
std::vector<char> LoadONNXModel(const std::string& model_path) {
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open ONNX model file: " + model_path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read ONNX model file: " + model_path);
    }

    return buffer;
}

int main() {
    try {
        // 打印 CUDA 设备信息
        std::cout << "========== CUDA Device Information ==========" << std::endl;
        std::vector<DC::GPUInfo> gpus = DC::getInfo_GPU();
        for (size_t i = 0; i < gpus.size(); ++i) {
            const DC::GPUInfo& gpu = gpus[i];
            std::cout << "Device " << i << ": " << gpu.name << std::endl;
            std::cout << "  CUDA Capability: " << gpu.cudaCapability << std::endl;
            std::cout << "  Total Global Memory: " << gpu.totalGlobalMemMB << " MB" << std::endl;
            std::cout << "  Shared Memory per Block: " << gpu.sharedMemPerBlockKB << " KB" << std::endl;
            std::cout << "  Registers per Block: " << gpu.regsPerBlock << std::endl;
            std::cout << "  Warp Size: " << gpu.warpSize << std::endl;
            std::cout << "  Max Threads per Block: " << gpu.maxThreadsPerBlock << std::endl;
            std::cout << "  Max Threads Dimension (x,y,z): ("
                << gpu.maxThreadsDim[0] << ", "
                << gpu.maxThreadsDim[1] << ", "
                << gpu.maxThreadsDim[2] << ")" << std::endl;
            std::cout << "  Max Grid Size (x,y,z): ("
                << gpu.maxGridSize[0] << ", "
                << gpu.maxGridSize[1] << ", "
                << gpu.maxGridSize[2] << ")" << std::endl;
            std::cout << "  Memory Clock Rate: " << gpu.memoryClockRateMHz << " MHz" << std::endl;
            std::cout << "  Memory Bus Width: " << gpu.memoryBusWidthBits << " bits" << std::endl;
            std::cout << "  Peak Memory Bandwidth: " << gpu.peakMemoryBandwidthGBs << " GB/s" << std::endl;
            std::cout << "---------------------------------------------" << std::endl;
        }
        std::cout << "=============================================" << std::endl;

        // 加载 ONNX 模型
        const std::string model_path = "C:/Users/东风谷早苗/Desktop/DiffSinger_Yoko/yoko.onnx"; // 替换为你的 ONNX 模型路径
        std::vector<char> onnxModelData = LoadONNXModel(model_path);

        // 使用 worker 对象
        DC::Infer inferenceWorker(onnxModelData);
        if (!inferenceWorker.isReady()) {
            std::cerr << "Failed to initialize inference worker: "
                << inferenceWorker.getErrorMessage() << std::endl;
            return -1;
        }

        // 获取并打印 ONNX 模型的输入张量信息
        const auto& onnxInputs = inferenceWorker.getInfo();
        std::cout << "\n========== ONNX Model Input Tensors ==========" << std::endl;
        for (const auto& [tensorName, tensorInfo] : onnxInputs) {
            if (!tensorInfo.IOtype) {
                continue;
            }
            std::cout << "Tensor Name: " << tensorName << std::endl;
            std::cout << "  Type: " << tensorInfo.type << std::endl;
            std::cout << "  Shape: [";
            for (size_t i = 0; i < tensorInfo.shape.size(); ++i) {
                std::cout << tensorInfo.shape[i];
                if (i < tensorInfo.shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "---------------------------------------------" << std::endl;
        }
        std::cout << "=============================================" << std::endl;
        std::cout << "\n========== ONNX Model Output Tensors ==========" << std::endl;
        for (const auto& [tensorName, tensorInfo] : onnxInputs) {
            if (tensorInfo.IOtype) {
                continue;
            }
            std::cout << "Tensor Name: " << tensorName << std::endl;
            std::cout << "  Type: " << tensorInfo.type << std::endl;
            std::cout << "  Shape: [";
            for (size_t i = 0; i < tensorInfo.shape.size(); ++i) {
                std::cout << tensorInfo.shape[i];
                if (i < tensorInfo.shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "---------------------------------------------" << std::endl;
        }
        std::cout << "=============================================" << std::endl;
        // 示例用法
        if (!gpus.empty()) {
            std::cout << "\n示例: 首个GPU的Warp Size: " << gpus[0].warpSize << std::endl;
        }

        if (!onnxInputs.empty()) {
            const std::string sampleTensor = "tension"; // 替换为实际的张量名
            auto it = onnxInputs.find(sampleTensor);
            if (it != onnxInputs.end()) {
                std::cout << "示例: 张量 \"" << sampleTensor << "\" 的类型: " << it->second.type << std::endl;
            }
            else {
                std::cout << "示例: 张量 \"" << sampleTensor << "\" 不存在。" << std::endl;
            }
        }
        
        test1();
        Sleep(10000);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Error: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        return -1;
    }
    
    
    return 0;
}
