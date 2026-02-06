#include "Info.h"
#include <cuda_runtime.h>

namespace DC {
	std::vector<GPUInfo> getInfo_GPU() {
		std::vector<GPUInfo> gpuInfos;
		int deviceCount = 0;
		cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

		if (error_id != cudaSuccess) {
			// 如果CUDA初始化失败或没有找到CUDA设备，可以返回一个空列表或进行错误处理
			// std::cerr << "cudaGetDeviceCount failed with error: " << cudaGetErrorString(error_id) << std::endl;
			return gpuInfos;
		}

		for (int i = 0; i < deviceCount; ++i) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);
			GPUInfo info;

			info.name = prop.name;
			info.cudaCapability = std::to_string(prop.major) + "." + std::to_string(prop.minor);
			info.totalGlobalMemMB = prop.totalGlobalMem / (1024 * 1024);
			info.sharedMemPerBlockKB = prop.sharedMemPerBlock / 1024;
			info.regsPerBlock = prop.regsPerBlock;
			info.warpSize = prop.warpSize;
			info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
			info.maxThreadsDim = { prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] };
			info.maxGridSize = { prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] };
			info.memoryClockRateMHz = prop.memoryClockRate / 1000;
			info.memoryBusWidthBits = prop.memoryBusWidth;
			info.peakMemoryBandwidthGBs = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / 1.0e6;

			gpuInfos.push_back(info);
		}

		return gpuInfos;
	}
}