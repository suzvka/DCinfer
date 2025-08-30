#pragma once
#include <string>
#include <vector>
#include <map>


namespace DC {
	// GPU 信息
	struct GPUInfo {
		std::string name;				// GPU型号
		std::string cudaCapability;		// CUDA能力
		size_t totalGlobalMemMB;		// 显存大小（MB）
		size_t sharedMemPerBlockKB;		// 每个块的共享内存（KB）
		int regsPerBlock;				// 每个块的寄存器数
		int warpSize;					// warp大小
		int maxThreadsPerBlock;			// 每个块的最大线程数
		std::vector<int> maxThreadsDim;	// 每个维度的最大线程数
		std::vector<int> maxGridSize;	// 每个维度的最大网格大小
		size_t memoryClockRateMHz;		// 内存时钟速率（MHz）
		size_t memoryBusWidthBits;		// 内存总线宽度（位）
		double peakMemoryBandwidthGBs;	// 峰值内存带宽（GB/s）
	};

	// 张量信息
	struct tensorInfo {
		bool IOtype;                    // 1 输入 0 输出
		std::string name;               // 张量名称
		std::string typeName;           // 张量类型
		std::vector<int64_t> shape;     // 张量形状
	};

	std::vector<GPUInfo> getInfo_GPU();
	std::map<std::string, tensorInfo> getInfo_ONNX(const std::vector<char>& onnxModelData);

	
}