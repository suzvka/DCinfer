#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

// minizip opaque types
typedef void* unzFile;
typedef void* zipFile;

namespace DC::Ir {

/// @brief 轻量 ZIP 容器读写器（基于 minizip）
///
/// .dcg 格式 = ZIP 容器，内含 graph.json + models/* 模型文件。
///
/// 读取流程：
///   1. openRead() 打开 .dcg，创建临时目录
///   2. readGraphJson() 从 ZIP 中解压 graph.json 到内存
///   3. extractOne("models/x.onnx") 解压单个模型到临时目录
///   4. 引擎加载模型后调用 cleanup() 删除临时文件
///   5. 析构时自动清理残留临时目录
///
/// 写入流程：
///   1. openWrite() 创建 .dcg
///   2. writeGraphJson() 写入图描述（deflate 压缩）
///   3. addModelFile() 逐个添加模型文件（store 模式）
///   4. finalize() 关闭 ZIP（析构时自动调用）
class DcgArchive {
public:
	// ── 工厂方法 ──

	/// @brief 打开 .dcg 文件用于读取，创建临时目录
	static std::unique_ptr<DcgArchive> openRead(const std::filesystem::path& path);

	/// @brief 创建 .dcg 文件用于写入
	static std::unique_ptr<DcgArchive> openWrite(const std::filesystem::path& path);

	~DcgArchive();

	DcgArchive(const DcgArchive&) = delete;
	DcgArchive& operator=(const DcgArchive&) = delete;
	DcgArchive(DcgArchive&&) = delete;
	DcgArchive& operator=(DcgArchive&&) = delete;

	// ── 读取接口 ──

	/// @brief 从 ZIP 中读取并解压 graph.json，返回内容字符串
	std::string readGraphJson();

	/// @brief 解压 archive 内的单个文件到临时目录，返回绝对路径
	/// @param archivePath ZIP 内路径，如 "models/resnet.onnx"
	/// @return 临时目录下的绝对路径
	std::filesystem::path extractOne(const std::string& archivePath);

	/// @brief 删除 extractOne 产生的临时文件
	void cleanup(const std::filesystem::path& tempPath);

	/// @brief 临时目录路径（反序列化时作为模型路径的 baseDir）
	const std::filesystem::path& tempDir() const { return _tempDir; }

	// ── 写入接口 ──

	/// @brief 将 graph.json 写入 ZIP（deflate 压缩）
	void writeGraphJson(std::string_view json);

	/// @brief 将磁盘上的模型文件添加到 ZIP（store 模式，不压缩）
	/// @param archivePath ZIP 内路径，如 "models/resnet.onnx"
	/// @param diskPath   磁盘上的模型文件路径
	void addModelFile(const std::string& archivePath, const std::filesystem::path& diskPath);

	/// @brief 写入 Central Directory + EOCD，关闭文件
	void finalize();

private:
	DcgArchive() = default;

	// ── minizip 句柄 ──
	unzFile _readHandle = nullptr;   // unzFile
	zipFile _writeHandle = nullptr;  // zipFile
	std::filesystem::path _tempDir;
	std::filesystem::path _archivePath;
	bool _finalized = false;
};

} // namespace DC::Ir
