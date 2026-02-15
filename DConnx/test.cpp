// main.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include "InferOrt.h"
#include "tensor.h"
#include "Info.h"
#include <filesystem>
#include <cstring>

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

		std::vector<float> testData = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
		std::vector<char> testBytes(testData.size() * sizeof(float));
		std::memcpy(testBytes.data(), testData.data(), testBytes.size());

        DC::Tensor a = DC::Tensor::Create<float>({ 2,3 }, std::move(testBytes));
		std::cout << "Tensor test typeSize = " << DC::Type::getSize(DC::TensorMeta::TensorType::Float) << std::endl;

		// Dense passthrough -> explicit editable start -> sparse write should not lose original dense content.
		{
			std::vector<char> denseBytes(sizeof(float) * 6);
			std::memcpy(denseBytes.data(), testData.data(), denseBytes.size());

			DC::Tensor t = DC::Tensor::Create<float>();
			t.setDense(std::move(denseBytes), { 2, 3 });
			const auto before = t.getData<float>();
			if (before != testData) {
				throw std::runtime_error("Dense passthrough getData<float>() mismatch");
			}

			t[0] = std::vector<float>{10.0f, 20.0f, 30.0f};
			// Multi-level chained indexing should work without excessive allocations and preserve semantics.
			t[1][2] = 99.0f;
			const float scalar = t[1][2];
			auto scalarVec = t.getData<float>();
			if (scalar != 99.0f) {
				throw std::runtime_error("Chained scalar write/read mismatch");
			}

			const auto after = t.getData<float>();
			std::vector<float> expected = { 10.0f, 20.0f, 30.0f, 4.0f, 5.0f, 99.0f };
			if (after != expected) {
				throw std::runtime_error("Editable materialization or write mismatch");
			}
		}

        Sleep(10000);
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
