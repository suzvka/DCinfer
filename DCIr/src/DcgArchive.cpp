#include "Ir/DcgArchive.h"

#include "GraphException.h"
#include "PathGuard.h"

#include <minizip/unzip.h>
#include <minizip/zip.h>

#include <chrono>
#include <cstdint>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace DC::Ir {

namespace {

// ── 解包安全与预算（审查 F01 / 归档健壮性）──

/// 单条目未压缩体积上限（防大文件/损坏归档耗尽内存或磁盘）
constexpr uint64_t kMaxEntryBytes = 1ull << 30; // 1 GiB
/// graph.json 专用单条目上限（图描述文件；收紧于通用 1 GiB，限制 DOM 解析内存放大）
constexpr uint64_t kMaxGraphJsonBytes = 64ull << 20; // 64 MiB
/// 一次性解包总预算（graph.json 之外的 extractOne 累计；防多条目聚合耗尽磁盘）
constexpr uint64_t kMaxExtractTotalBytes = 4ull << 30; // 4 GiB
/// extractOne 条目数上限（防多条目磁盘耗尽与 O(M×N) 定位 CPU 放大）
constexpr std::size_t kMaxExtractEntries = 256;
/// 归档全局条目数上限（openRead 校验，防海量条目拖慢逐次定位）
constexpr uint64_t kMaxArchiveEntries = 4096;
/// 压缩比上限（防 zip bomb：低熵膨胀条目在读取前拒绝；真实模型/JSON 远低于此）
constexpr uint64_t kMaxCompressionRatio = 200;
/// 流式读取块大小
constexpr std::size_t kReadChunkBytes = 64 * 1024;

/// 读取前预算校验：体积 + 压缩比（只依赖 ZIP 目录声明，不解压）
void ensureEntryWithinBudget(const unz_file_info64& info, const std::string& entry) {
	if (info.uncompressed_size > kMaxEntryBytes) {
		throw GraphException(GraphException::ErrorType::Other, "DcgArchive",
			"entry exceeds size budget (" + std::to_string(info.uncompressed_size) + " > "
				+ std::to_string(kMaxEntryBytes) + " bytes): " + entry);
	}
	if (info.compressed_size > 0
		&& info.uncompressed_size / kMaxCompressionRatio > info.compressed_size) {
		throw GraphException(GraphException::ErrorType::Other, "DcgArchive",
			"entry rejected: suspicious compression ratio: " + entry);
	}
}

/// 符号链接防御：baseDir 到 target 之间已存在的路径组件不得是符号链接
/// （防归档借解包目录中既有 symlink 将写入重定向到目录之外）
void ensureNoSymlinkAncestor(const std::filesystem::path& baseDir, const std::filesystem::path& target,
							 const std::string& entry) {
	const auto rel = target.lexically_normal().lexically_relative(baseDir.lexically_normal());
	if (rel.empty())
		return;
	std::filesystem::path cur = baseDir;
	for (const auto& comp : rel) {
		if (comp == "..") {
			throw GraphException(GraphException::ErrorType::Other, "DcgArchive",
				"extraction target escapes base directory: " + entry);
		}
		cur /= comp;
		std::error_code ec;
		const auto st = std::filesystem::symlink_status(cur, ec);
		if (!ec && std::filesystem::is_symlink(st)) {
			throw GraphException(GraphException::ErrorType::Other, "DcgArchive",
				"symlink in extraction path is not allowed: " + cur.string());
		}
	}
}

/// 当前打开条目的 RAII 关闭（异常路径防句柄泄漏；CRC 检查后置 closed 标志）
struct CurrentEntryGuard {
	unzFile handle;
	bool closed = false;

	~CurrentEntryGuard() {
		if (!closed && handle)
			::unzCloseCurrentFile(handle);
	}
};

/// 流式读取当前打开条目：分块循环至 EOF，边读边交给 sink；
/// 累计字节与声明体积比对（防截断/超读），CRC 由调用方经
/// closeCurrentEntryWithCrcCheck 收尾校验。
template <typename Sink>
void streamCurrentEntry(unzFile handle, uint64_t expectedSize, const std::string& entry, Sink&& sink) {
	std::vector<char> buf(kReadChunkBytes);
	uint64_t total = 0;
	for (;;) {
		const int n = ::unzReadCurrentFile(handle, buf.data(), static_cast<unsigned>(buf.size()));
		if (n < 0) {
			throw GraphException(GraphException::ErrorType::Other, "DcgArchive",
				"read error for: " + entry);
		}
		if (n == 0)
			break; // EOF
		total += static_cast<uint64_t>(n);
		if (total > expectedSize) {
			throw GraphException(GraphException::ErrorType::Other, "DcgArchive",
				"entry larger than declared size: " + entry);
		}
		sink(buf.data(), static_cast<std::size_t>(n));
	}
	if (total != expectedSize) {
		throw GraphException(GraphException::ErrorType::Other, "DcgArchive",
			"truncated entry: expected " + std::to_string(expectedSize) + " got "
				+ std::to_string(total) + " bytes: " + entry);
	}
}

/// CRC 校验收尾：unzCloseCurrentFile 返回非 OK（如 UNZ_CRCERROR）时拒绝
void closeCurrentEntryWithCrcCheck(unzFile handle, CurrentEntryGuard& guard, const std::string& entry) {
	const int ret = ::unzCloseCurrentFile(handle);
	guard.closed = true;
	if (ret != UNZ_OK) {
		throw GraphException(GraphException::ErrorType::Other, "DcgArchive",
			"CRC/integrity verification failed for: " + entry);
	}
}

} // namespace

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

	// 全局条目数上限（IR-04）：海量条目的归档会使每次 unzLocateFile 线性定位
	// 变得昂贵（O(M×N) CPU 放大）；超限直接拒绝打开。
	unz_global_info64 globalInfo{};
	if (::unzGetGlobalInfo64(archive->_readHandle, &globalInfo) == UNZ_OK
		&& globalInfo.number_entry > kMaxArchiveEntries) {
		throw GraphException(GraphException::ErrorType::Other, "DcgArchive::openRead",
			"archive has too many entries (" + std::to_string(globalInfo.number_entry) + " > "
				+ std::to_string(kMaxArchiveEntries) + ")");
	}

	// 创建唯一且私有的临时目录（随机后缀 + 独占创建；冲突则换名重试）
	auto tmpBase = std::filesystem::temp_directory_path();
	auto now = std::chrono::system_clock::now().time_since_epoch().count();
	std::random_device rd;
	std::uniform_int_distribution<uint64_t> dist;
	std::filesystem::path tmpDir;
	bool created = false;
	for (int attempt = 0; attempt < 16; ++attempt) {
		tmpDir = tmpBase / ("dcg_" + path.stem().string() + "_" + std::to_string(now) + "_"
							+ std::to_string(dist(rd)));
		std::error_code ec;
		const bool made = std::filesystem::create_directory(tmpDir, ec);
		if (made && !ec) {
			created = true;
			break;
		}
		if (ec) {
			throw GraphException(GraphException::ErrorType::Other,
				"DcgArchive::openRead",
				"cannot create temp dir: " + tmpDir.string() + ": " + ec.message());
		}
		// made=false 且无错误：同名目录已存在（极小概率冲突）→ 换随机名重试
	}
	if (!created) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::openRead",
			"cannot create unique temp dir under: " + tmpBase.string());
	}
	// 尽力收紧目录权限（POSIX 0700；Windows 无权限位模型，忽略失败）
	std::error_code pec;
	std::filesystem::permissions(tmpDir, std::filesystem::perms::owner_all,
		std::filesystem::perm_options::replace, pec);
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

	// 读取前预算校验（体积/压缩比）；graph.json 走更严的专用预算（IR-05：
	// 限制图描述文件的内存驻留与 DOM 解析放大）
	if (entryName == "graph.json" && info.uncompressed_size > kMaxGraphJsonBytes) {
		throw GraphException(GraphException::ErrorType::Other, "DcgArchive",
			"graph.json exceeds size budget (" + std::to_string(info.uncompressed_size) + " > "
				+ std::to_string(kMaxGraphJsonBytes) + " bytes)");
	}
	ensureEntryWithinBudget(info, entryName);

	// 打开条目
	ret = ::unzOpenCurrentFile(handle);
	if (ret != UNZ_OK) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive",
			"failed to open entry: " + entryName);
	}
	CurrentEntryGuard guard{handle};

	// 流式读取全部数据（分块循环）。预分配不再信任 ZIP 声明的体积
	// （声明 1 GiB 实际 1 KiB 也会立即提交 1 GiB —— IR-05）：小量起步、按需增长。
	std::vector<char> buffer;
	buffer.reserve(static_cast<std::size_t>(std::min<uint64_t>(info.uncompressed_size, 1ull << 20)));
	streamCurrentEntry(handle, info.uncompressed_size, entryName,
		[&buffer](const char* p, std::size_t n) { buffer.insert(buffer.end(), p, p + n); });

	// CRC/完整性校验收尾（unzCloseCurrentFile 返回值）
	closeCurrentEntryWithCrcCheck(handle, guard, entryName);
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
	// 路径安全校验（F01）：拒绝空/内嵌 NUL/绝对路径/盘符/父目录跳转/归一化越界
	std::string reason;
	if (!detail::isSafeArchiveRelPath(archivePath, _tempDir, &reason)) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::extractOne",
			"unsafe archive path '" + archivePath + "': " + reason);
	}

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

	// 读取前预算校验（体积/压缩比）
	ensureEntryWithinBudget(info, archivePath);

	// 聚合预算（IR-04）：条目数 + 累计解压量——单条目合规不代表聚合合规，
	// 多条目同样能耗尽磁盘；逐次 unzLocateFile 线性扫描还存在 O(M×N) CPU 放大。
	if (++_extractEntries > kMaxExtractEntries) {
		throw GraphException(GraphException::ErrorType::Other, "DcgArchive::extractOne",
			"too many extracted entries (limit " + std::to_string(kMaxExtractEntries) + ")");
	}
	if (info.uncompressed_size > kMaxExtractTotalBytes - _extractTotalBytes) {
		throw GraphException(GraphException::ErrorType::Other, "DcgArchive::extractOne",
			"extract total budget exceeded (" + std::to_string(_extractTotalBytes) + " + "
				+ std::to_string(info.uncompressed_size) + " > " + std::to_string(kMaxExtractTotalBytes)
				+ " bytes)");
	}
	_extractTotalBytes += info.uncompressed_size;

	// 确定输出路径（校验后归一化）并防御既有符号链接组件
	const auto tmpPath = (_tempDir / archivePath).lexically_normal();
	ensureNoSymlinkAncestor(_tempDir, tmpPath, archivePath);
	std::filesystem::create_directories(tmpPath.parent_path());

	// 打开条目
	ret = ::unzOpenCurrentFile(_readHandle);
	if (ret != UNZ_OK) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::extractOne",
			"failed to open entry: " + archivePath);
	}
	CurrentEntryGuard guard{_readHandle};

	// 流式读取写盘（分块）：完整性/CRC 校验失败或写盘失败时清理部分文件后重抛
	try {
		{
			std::ofstream ofs(tmpPath, std::ios::binary | std::ios::trunc);
			if (!ofs.is_open()) {
				throw GraphException(GraphException::ErrorType::Other,
					"DcgArchive::extractOne",
					"cannot create temp file: " + tmpPath.string());
			}
			streamCurrentEntry(_readHandle, info.uncompressed_size, archivePath,
				[&ofs](const char* p, std::size_t n) {
					ofs.write(p, static_cast<std::streamsize>(n));
					if (!ofs) {
						throw GraphException(GraphException::ErrorType::Other,
							"DcgArchive::extractOne", "write error to extraction target");
					}
				});
		} // ofs 关闭（异常清理前先释放文件句柄）
		closeCurrentEntryWithCrcCheck(_readHandle, guard, archivePath);
	} catch (...) {
		std::error_code ec;
		std::filesystem::remove(tmpPath, ec);
		throw;
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

	// 防御：graph.json 实际规模受序列化侧约束，此处兜底 unsigned 写入上限
	if (json.size() > std::numeric_limits<unsigned>::max()) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::writeGraphJson",
			"graph.json exceeds the single-entry write limit");
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
	const std::streamoff endPos = ifs.tellg();
	if (endPos < 0) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::addModelFile",
			"cannot determine size of model file: " + diskPath.string());
	}
	// minizip 单次写入长度参数为 unsigned，且本写入路径未启用 zip64 条目：
	// >4 GiB 的文件此前会静默截断（只写低 32 位且无报错）——现显式拒绝（IR-06）。
	if (static_cast<uint64_t>(endPos) > std::numeric_limits<unsigned>::max()) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::addModelFile",
			"model file exceeds the 4 GiB single-entry limit: " + diskPath.string());
	}
	ifs.seekg(0);

	// 写入 ZIP（store 模式，因为模型文件通常已经压缩）
	int ret = ::zipOpenNewFileInZip64(_writeHandle, archivePath.c_str(),
		nullptr, nullptr, 0, nullptr, 0, nullptr,
		0, 0, 0); // method=0 → store
	if (ret != ZIP_OK) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::addModelFile",
			"failed to open entry: " + archivePath);
	}

	// 分块流式写入（64 KiB）：不再把整模型读入内存（IR-05）；每块长度
	// 转换 unsigned 安全（块大小远小于 4 GiB 上限）。
	std::vector<char> buf(kReadChunkBytes);
	while (ifs) {
		ifs.read(buf.data(), static_cast<std::streamsize>(buf.size()));
		const auto got = static_cast<std::size_t>(ifs.gcount());
		if (got == 0)
			break;
		ret = ::zipWriteInFileInZip(_writeHandle, buf.data(), static_cast<unsigned>(got));
		if (ret != ZIP_OK) {
			throw GraphException(GraphException::ErrorType::Other,
				"DcgArchive::addModelFile",
				"failed to write data for: " + archivePath);
		}
	}
	if (!ifs.eof()) {
		throw GraphException(GraphException::ErrorType::Other,
			"DcgArchive::addModelFile",
			"failed to read model file: " + diskPath.string());
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
