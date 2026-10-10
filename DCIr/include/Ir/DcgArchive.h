#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

// minizip opaque types
typedef void* unzFile;
typedef void* zipFile;

namespace DC::Ir {

namespace detail { class SecureExtraction; }

/// @brief 轻量 ZIP 容器读写器（基于 minizip）
///
/// .dcg 格式 = ZIP 容器，内含 graph.json + models/* 模型文件。
/// 读取：openRead 创建临时目录，extractOne 解压模型；析构自动清理残留。
/// 写入：graph.json 走 deflate；模型文件走 store；finalize 关闭 ZIP（析构自动调用）。
class DcgArchive {
public:
	/// @brief 打开 .dcg 读取，创建临时解压目录
	static std::unique_ptr<DcgArchive> openRead(const std::filesystem::path& path);

	static std::unique_ptr<DcgArchive> openWrite(const std::filesystem::path& path);

	~DcgArchive();

	DcgArchive(const DcgArchive&) = delete;
	DcgArchive& operator=(const DcgArchive&) = delete;
	DcgArchive(DcgArchive&&) = delete;
	DcgArchive& operator=(DcgArchive&&) = delete;

	// ── 读取接口 ──

	std::string readGraphJson();

	/// @brief 解压 archive 内的单个文件到临时目录，返回绝对路径
	/// @param archivePath ZIP 内路径（如 "models/resnet.onnx"）；必须为安全相对路径，否则拒绝
	/// @throws GraphException(Other) 路径不安全、超预算（单条目/压缩比/条目数/总量）、CRC 或写盘失败
	std::filesystem::path extractOne(const std::string& archivePath);

	void cleanup(const std::filesystem::path& tempPath);

	/// @brief 临时目录路径（反序列化时作为模型路径的 baseDir）
	const std::filesystem::path& tempDir() const { return _tempDir; }

	// ── 写入接口 ──

	/// @brief 将 graph.json 写入 ZIP（deflate 压缩）
	void writeGraphJson(std::string_view json);

	/// @brief 添加磁盘模型文件到 ZIP（store 模式，不压缩）
	/// @param archivePath ZIP 内路径，如 "models/resnet.onnx"
	void addModelFile(const std::string& archivePath, const std::filesystem::path& diskPath);

	/// @brief 写入 Central Directory + EOCD，关闭文件
	void finalize();

private:
	DcgArchive();
	std::unique_ptr<detail::SecureExtraction> _extraction;

	// ── minizip 句柄 ──
	unzFile _readHandle = nullptr;
	zipFile _writeHandle = nullptr;
	std::filesystem::path _tempDir;
	std::filesystem::path _archivePath;
	bool _finalized = false;

	// 解包预算聚合：extractOne 累计解压字节与条目计数
	uint64_t _extractTotalBytes = 0;
	std::size_t _extractEntries = 0;
};

} // namespace DC::Ir
