#include "Ir/DcgArchive.h"

#include "GraphException.h"

#include <minizip/unzip.h>
#include <minizip/zip.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

namespace DC::Ir {

// ════════════════════════════════════════════
// 工厂方法
// ════════════════════════════════════════════

std::unique_ptr<DcgArchive> DcgArchive::openRead(const std::filesystem::path& path) {
	auto archive = std::unique_ptr<DcgArchive>(new DcgArchive());
	archive->_archivePath = path;

	auto pathStr = path.string();
	archive->_readHandle = ::unzOpen64(pathStr.c_str());
	if (!archive->_readHandle) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::openRead",
			"cannot open archive: " + pathStr);
	}

	// 创建临时目录
	auto tmpBase = std::filesystem::temp_directory_path();
	auto now = std::chrono::system_clock::now().time_since_epoch().count();
	auto tmpDir = tmpBase / ("dcg_" + path.stem().string() + "_" + std::to_string(now));
	std::filesystem::create_directories(tmpDir);
	archive->_tempDir = tmpDir;

	return archive;
}

std::unique_ptr<DcgArchive> DcgArchive::openWrite(const std::filesystem::path& path) {
	auto archive = std::unique_ptr<DcgArchive>(new DcgArchive());
	archive->_archivePath = path;

	auto pathStr = path.string();
	archive->_writeHandle = ::zipOpen64(pathStr.c_str(), APPEND_STATUS_CREATE);
	if (!archive->_writeHandle) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::openWrite",
			"cannot create archive: " + pathStr);
	}

	return archive;
}

DcgArchive::~DcgArchive() {
	// 写入模式：自动 finalize
	if (_writeHandle && !_finalized) {
		try {
			finalize();
		} catch (...) {
			// 析构中忽略异常
		}
	}

	// 关闭读取句柄
	if (_readHandle) {
		::unzClose(_readHandle);
	}

	// 清理临时目录
	if (!_tempDir.empty()) {
		std::error_code ec;
		std::filesystem::remove_all(_tempDir, ec);
	}
}

// ════════════════════════════════════════════
// 读取：从 ZIP 条目解压到内存
// ════════════════════════════════════════════

static std::vector<char> readEntryToMemory(unzFile handle, const std::string& entryName) {
	// 定位条目
	int ret = ::unzLocateFile(handle, entryName.c_str(), 2); // case-insensitive
	if (ret != UNZ_OK) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive",
			"entry not found in archive: " + entryName);
	}

	// 获取文件信息
	unz_file_info64 info{};
	char filenameBuf[256]{};
	ret = ::unzGetCurrentFileInfo64(handle, &info, filenameBuf, sizeof(filenameBuf),
									nullptr, 0, nullptr, 0);
	if (ret != UNZ_OK) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive",
			"failed to get info for: " + entryName);
	}

	// 打开条目
	ret = ::unzOpenCurrentFile(handle);
	if (ret != UNZ_OK) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive",
			"failed to open entry: " + entryName);
	}

	// 读取全部数据
	std::vector<char> buffer(static_cast<size_t>(info.uncompressed_size));
	int bytesRead = ::unzReadCurrentFile(handle, buffer.data(), static_cast<unsigned>(buffer.size()));

	::unzCloseCurrentFile(handle);

	if (bytesRead < 0) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive",
			"read error for: " + entryName);
	}

	buffer.resize(static_cast<size_t>(bytesRead));
	return buffer;
}

// ════════════════════════════════════════════
// 公开读取接口
// ════════════════════════════════════════════

std::string DcgArchive::readGraphJson() {
	auto data = readEntryToMemory(_readHandle, "graph.json");
	return std::string(data.data(), data.size());
}

std::filesystem::path DcgArchive::extractOne(const std::string& archivePath) {
	// 定位条目
	int ret = ::unzLocateFile(_readHandle, archivePath.c_str(), 2);
	if (ret != UNZ_OK) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::extractOne",
			"entry not found in archive: " + archivePath);
	}

	// 获取文件信息
	unz_file_info64 info{};
	ret = ::unzGetCurrentFileInfo64(_readHandle, &info, nullptr, 0, nullptr, 0, nullptr, 0);
	if (ret != UNZ_OK) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::extractOne",
			"failed to get info for: " + archivePath);
	}

	// 确定输出路径
	auto tmpPath = _tempDir / archivePath;
	std::filesystem::create_directories(tmpPath.parent_path());

	// 打开条目
	ret = ::unzOpenCurrentFile(_readHandle);
	if (ret != UNZ_OK) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::extractOne",
			"failed to open entry: " + archivePath);
	}

	// 读取并写入临时文件
	std::vector<char> buffer(static_cast<size_t>(info.uncompressed_size));
	int bytesRead = ::unzReadCurrentFile(_readHandle, buffer.data(), static_cast<unsigned>(buffer.size()));
	::unzCloseCurrentFile(_readHandle);

	if (bytesRead < 0) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::extractOne",
			"read error for: " + archivePath);
	}

	std::ofstream ofs(tmpPath, std::ios::binary);
	if (!ofs.is_open()) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::extractOne",
			"cannot create temp file: " + tmpPath.string());
	}
	ofs.write(buffer.data(), bytesRead);
	if (!ofs) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::extractOne",
			"write error for: " + tmpPath.string());
	}

	return tmpPath;
}

void DcgArchive::cleanup(const std::filesystem::path& tempPath) {
	std::error_code ec;
	std::filesystem::remove(tempPath, ec);
}

// ════════════════════════════════════════════
// 写入
// ════════════════════════════════════════════

void DcgArchive::writeGraphJson(std::string_view json) {
	int ret = ::zipOpenNewFileInZip64(_writeHandle, "graph.json",
		nullptr, nullptr, 0, nullptr, 0, nullptr,
		Z_DEFLATED, Z_DEFAULT_COMPRESSION, 0);
	if (ret != ZIP_OK) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::writeGraphJson",
			"failed to open graph.json entry");
	}

	ret = ::zipWriteInFileInZip(_writeHandle, json.data(), static_cast<unsigned>(json.size()));
	if (ret != ZIP_OK) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::writeGraphJson",
			"failed to write graph.json data");
	}

	ret = ::zipCloseFileInZip(_writeHandle);
	if (ret != ZIP_OK) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::writeGraphJson",
			"failed to close graph.json entry");
	}
}

void DcgArchive::addModelFile(const std::string& archivePath, const std::filesystem::path& diskPath) {
	// 读取整个文件
	std::ifstream ifs(diskPath, std::ios::binary | std::ios::ate);
	if (!ifs.is_open()) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::addModelFile",
			"cannot open model file: " + diskPath.string());
	}
	auto fileSize = static_cast<size_t>(ifs.tellg());
	ifs.seekg(0);

	std::vector<char> buf(fileSize);
	ifs.read(buf.data(), static_cast<std::streamsize>(fileSize));
	if (!ifs) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::addModelFile",
			"failed to read model file: " + diskPath.string());
	}

	// 写入 ZIP（store 模式，因为模型文件通常已经压缩）
	int ret = ::zipOpenNewFileInZip64(_writeHandle, archivePath.c_str(),
		nullptr, nullptr, 0, nullptr, 0, nullptr,
		0, 0, 0); // method=0 → store
	if (ret != ZIP_OK) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::addModelFile",
			"failed to open entry: " + archivePath);
	}

	ret = ::zipWriteInFileInZip(_writeHandle, buf.data(), static_cast<unsigned>(buf.size()));
	if (ret != ZIP_OK) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::addModelFile",
			"failed to write data for: " + archivePath);
	}

	ret = ::zipCloseFileInZip(_writeHandle);
	if (ret != ZIP_OK) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::addModelFile",
			"failed to close entry: " + archivePath);
	}
}

void DcgArchive::finalize() {
	if (_finalized || !_writeHandle) return;
	_finalized = true;

	int ret = ::zipClose(_writeHandle, nullptr);
	_writeHandle = nullptr;

	if (ret != ZIP_OK) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::finalize",
			"failed to close archive");
	}
}

} // namespace DC::Ir
