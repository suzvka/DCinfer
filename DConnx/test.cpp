// main.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include "InferOrt.h"
#include "tensor.h"
#include "Info.h"
#include <filesystem>

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
        auto gpus = DC::getInfo_GPU();
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

        DC::Tensor a = DC::Tensor::Create<float>("test", { 2,3 }, { 'a','a' ,'a' ,'a' ,'a' ,'a' });

        // 加载 ONNX 模型
        const std::string model_path = "C:/Users/东风谷早苗/Desktop/acoustic.onnx"; // 替换为你的 ONNX 模型路径
		std::wstring w_model_path = std::filesystem::path(model_path).wstring();

		Ort::Env test_env(ORT_LOGGING_LEVEL_WARNING, "test");
		Ort::SessionOptions test_options;
		test_options.SetIntraOpNumThreads(1);
		test_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		Ort::Session test_session(test_env, w_model_path.c_str(), test_options);

		DC::InferOrt worker(model_path, 1);
        
        // test1();
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
