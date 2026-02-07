// main.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include "InferOrt.h"
#include "tensor.h"
#include "Info.h"
#include <filesystem>

#include <windows.h>
#include <sstream>
#include <iomanip>
#include <wincrypt.h>

#pragma comment(lib, "Crypt32.lib")

static bool TryGetFileSha256Hex(const std::wstring& path, std::string& outHex) {
	outHex.clear();
	HANDLE hFile = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (hFile == INVALID_HANDLE_VALUE) return false;

	HCRYPTPROV hProv = 0;
	HCRYPTHASH hHash = 0;
	bool ok = false;

	if (!CryptAcquireContextW(&hProv, nullptr, nullptr, PROV_RSA_AES, CRYPT_VERIFYCONTEXT)) goto cleanup;
	if (!CryptCreateHash(hProv, CALG_SHA_256, 0, 0, &hHash)) goto cleanup;

	{
		BYTE buffer[64 * 1024];
		DWORD bytesRead = 0;
		while (ReadFile(hFile, buffer, (DWORD)sizeof(buffer), &bytesRead, nullptr) && bytesRead != 0) {
			if (!CryptHashData(hHash, buffer, bytesRead, 0)) goto cleanup;
		}
	}

	{
		DWORD hashLen = 0;
		DWORD lenSize = sizeof(hashLen);
		if (!CryptGetHashParam(hHash, HP_HASHSIZE, (BYTE*)&hashLen, &lenSize, 0)) goto cleanup;
		std::vector<BYTE> hash(hashLen);
		DWORD cbHash = hashLen;
		if (!CryptGetHashParam(hHash, HP_HASHVAL, hash.data(), &cbHash, 0)) goto cleanup;

		std::ostringstream os;
		os << std::hex << std::setfill('0');
		for (BYTE b : hash) os << std::setw(2) << (int)b;
		outHex = os.str();
		ok = true;
	}

cleanup:
	if (hHash) CryptDestroyHash(hHash);
	if (hProv) CryptReleaseContext(hProv, 0);
	CloseHandle(hFile);
	return ok;
}

static void DebugPrintA(const std::string& s) {
	OutputDebugStringA(s.c_str());
	std::cout << s;
}

static std::string WideToUtf8(const std::wstring& ws) {
	if (ws.empty()) return {};
	int size_needed = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), nullptr, 0, nullptr, nullptr);
	std::string str(size_needed, 0);
	WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), str.data(), size_needed, nullptr, nullptr);
	return str;
}

static std::wstring GetModulePathW(HMODULE h) {
	wchar_t buf[MAX_PATH] = { 0 };
	DWORD len = GetModuleFileNameW(h, buf, MAX_PATH);
	return std::wstring(buf, buf + len);
}

static void PrintEnvironmentDiagnostics() {
	std::ostringstream os;
	os << "==== Diagnostics ====" << std::endl;
	os << "PointerSize=" << sizeof(void*) * 8 << "-bit" << std::endl;

	wchar_t cwd[MAX_PATH] = { 0 };
	GetCurrentDirectoryW(MAX_PATH, cwd);
	os << "CWD=" << WideToUtf8(cwd) << std::endl;

	os << "EXE=" << WideToUtf8(GetModulePathW(GetModuleHandleW(nullptr))) << std::endl;
	{
		std::string sha;
		auto exePath = GetModulePathW(GetModuleHandleW(nullptr));
		if (TryGetFileSha256Hex(exePath, sha)) {
			os << "EXE_SHA256=" << sha << std::endl;
		}
	}

	// Force-load onnxruntime.dll from current directory first to avoid PATH pollution.
	{
		auto localOrt = std::filesystem::path(cwd) / L"onnxruntime.dll";
		if (std::filesystem::exists(localOrt)) {
			HMODULE h = LoadLibraryW(localOrt.c_str());
			os << "LoadLibrary(local onnxruntime.dll)=" << (h ? "OK" : "FAIL") << " path=" << WideToUtf8(localOrt.wstring()) << std::endl;
		}
		else {
			os << "local onnxruntime.dll not found at " << WideToUtf8(localOrt.wstring()) << std::endl;
		}
	}

	HMODULE hOrt = GetModuleHandleW(L"onnxruntime.dll");
	if (hOrt) {
		auto ortPath = GetModulePathW(hOrt);
		os << "onnxruntime.dll loaded from: " << WideToUtf8(ortPath) << std::endl;
		std::string sha;
		if (TryGetFileSha256Hex(ortPath, sha)) {
			os << "onnxruntime.dll_SHA256=" << sha << std::endl;
		}
	}
	else {
		os << "onnxruntime.dll not loaded yet" << std::endl;
	}

	DWORD pathLen = GetEnvironmentVariableW(L"PATH", nullptr, 0);
	if (pathLen > 0) {
		std::wstring pathW(pathLen, L'\0');
		GetEnvironmentVariableW(L"PATH", pathW.data(), pathLen);
		os << "PATH=" << WideToUtf8(pathW) << std::endl;
	}

	os << "=====================" << std::endl;
	DebugPrintA(os.str());
}

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
		PrintEnvironmentDiagnostics();
		DebugPrintA("[main] before GPU info\n");
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
		const std::wstring model_path_w = L"C:/Users/东风谷早苗/Desktop/acoustic.onnx"; // 临时测试文件
		const std::filesystem::path model_path(model_path_w);		

		DebugPrintA("[main] before InferOrt ctor\n");
		DC::InferOrt worker(model_path, 1);
		DebugPrintA("[main] after InferOrt ctor\n");
		
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
