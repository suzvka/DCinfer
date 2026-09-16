// DcgArchive 路径安全与完整性 回归测试（发布前审查 F01 + 归档健壮性 + v0.5.2 IR-03/04/05）
//
// 覆盖：
//   - isSafeArchiveRelPath：空/内嵌 NUL/绝对路径/盘符/UNC/父目录跳转拒绝，正常相对路径放行；
//     IR-03：Windows 保留设备名 / ADS 冒号 / 尾点尾空格拒绝
//   - extractOne 端到端：../ 条目拒绝且无文件逃逸解包目录
//   - 符号链接祖先目录拒绝（无 symlink 权限的环境自动跳过）
//   - 截断归档明确报错；高压缩比（zip bomb 形态）条目按预算拒绝
//   - IR-04/05：解包条目数聚合预算、归档全局条目数上限、graph.json 专用体积预算
//   - 正常归档读取不被安全校验误伤（graph.json 往返 + 模型解压内容比对）
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "Ir/DcgArchive.h"
#include "GraphException.h"
#include "PathGuard.h"

using namespace DC;
using namespace DC::Ir;

static int failures = 0;

#define CHECK(cond, msg)                                                                                               \
	do {                                                                                                               \
		if (!(cond)) {                                                                                                 \
			std::cerr << "FAIL: " << msg << std::endl;                                                                 \
			++failures;                                                                                                \
			return;                                                                                                    \
		}                                                                                                              \
	} while (0)

#define TEST(name)                                                                                                     \
	std::cout << "Test: " << name << " ... " << std::flush;                                                            \
	[&]()
#define END_TEST()                                                                                                     \
	();                                                                                                                \
	std::cout << "PASSED" << std::endl

namespace {

std::filesystem::path makeWorkDir(const char* tag) {
	const auto stamp = std::chrono::system_clock::now().time_since_epoch().count();
	auto dir = std::filesystem::temp_directory_path() / (std::string("dcg_sec_") + tag + "_" + std::to_string(stamp));
	std::filesystem::remove_all(dir);
	std::filesystem::create_directories(dir);
	return dir;
}

void writePayload(const std::filesystem::path& path, const std::string& content) {
	std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
	ofs << content;
}

std::string readFile(const std::filesystem::path& path) {
	std::ifstream ifs(path, std::ios::binary);
	std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	return content;
}

} // namespace

// ════════════════════════════════════════════
// 路径校验单元（跨平台一致规则）
// ════════════════════════════════════════════

static void testSafePathValidator() {
	TEST("path guard: reject traversal/absolute/drive/UNC; accept safe relative") {
		const auto base = std::filesystem::temp_directory_path() / "dcg_guard_base";
		std::string reason;

		CHECK(!detail::isSafeArchiveRelPath("", base, &reason), "empty path must be rejected");
		CHECK(!detail::isSafeArchiveRelPath("../escaped.bin", base, &reason), "'..' must be rejected");
		CHECK(!detail::isSafeArchiveRelPath("models/../../x.bin", base, &reason), "nested '..' must be rejected");
		CHECK(!detail::isSafeArchiveRelPath("..\\win.bin", base, &reason), "backslash '..' must be rejected");
		CHECK(!detail::isSafeArchiveRelPath("/abs.bin", base, &reason), "unix-absolute path must be rejected");
		CHECK(!detail::isSafeArchiveRelPath("//server/share/x.bin", base, &reason), "UNC-style path must be rejected");
		CHECK(!detail::isSafeArchiveRelPath("C:/abs.bin", base, &reason), "drive path must be rejected");
		CHECK(!detail::isSafeArchiveRelPath("c:relative.bin", base, &reason), "drive-relative path must be rejected");
		CHECK(!detail::isSafeArchiveRelPath(std::string("models/x") + '\0' + "y.bin", base, &reason),
			  "embedded NUL must be rejected");

		CHECK(detail::isSafeArchiveRelPath("graph.json", base, &reason), "top-level relative must be accepted");
		CHECK(detail::isSafeArchiveRelPath("models/resnet.onnx", base, &reason), "nested relative must be accepted");
		CHECK(detail::isSafeArchiveRelPath("models/./x.bin", base, &reason), "dot component must be accepted");

		// IR-03：Windows 罪名字形（保留设备名 / ADS 冒号 / 尾点尾空格）
		CHECK(!detail::isSafeArchiveRelPath("models/CON", base, &reason), "CON device name must be rejected");
		CHECK(!detail::isSafeArchiveRelPath("models/con.txt", base, &reason),
			  "case-insensitive device name with extension must be rejected");
		CHECK(!detail::isSafeArchiveRelPath("models/AUX.onnx", base, &reason), "AUX device name must be rejected");
		CHECK(!detail::isSafeArchiveRelPath("models/COM1", base, &reason), "COM1 device name must be rejected");
		CHECK(!detail::isSafeArchiveRelPath("models/LPT9.bin", base, &reason), "LPT9 device name must be rejected");
		CHECK(!detail::isSafeArchiveRelPath("models/data.txt:ads", base, &reason), "ADS colon must be rejected");
		CHECK(!detail::isSafeArchiveRelPath("models/trailing.", base, &reason), "trailing dot must be rejected");
		CHECK(!detail::isSafeArchiveRelPath("models/trailing ", base, &reason), "trailing space must be rejected");
		// 合法名不被误伤
		CHECK(detail::isSafeArchiveRelPath("models/console.onnx", base, &reason), "'console' must be accepted");
		CHECK(detail::isSafeArchiveRelPath("models/com10.onnx", base, &reason), "'com10' must be accepted");
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 端到端：../ 条目拒绝且无逃逸写入
// ════════════════════════════════════════════

static void testExtractOneRejectsTraversal() {
	TEST("extractOne: traversal entry rejected, no file escapes extraction dir") {
		const auto workDir = makeWorkDir("traversal");
		const auto dcgPath = workDir / "traversal.dcg";
		const auto escapedName = "escaped-model-" +
								 std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".bin";
		const auto escapedPath = std::filesystem::temp_directory_path() / escapedName;

		writePayload(workDir / "payload.bin", "review-only marker");
		{
			auto w = DcgArchive::openWrite(dcgPath);
			w->writeGraphJson("{}");
			w->addModelFile("../" + escapedName, workDir / "payload.bin");
			w->finalize();
		}

		auto r = DcgArchive::openRead(dcgPath);
		bool rejected = false;
		try {
			r->extractOne("../" + escapedName);
		} catch (const GraphException&) {
			rejected = true;
		}
		CHECK(rejected, "traversal entry must be rejected");
		CHECK(!std::filesystem::exists(escapedPath), "no file may escape the extraction directory");

		r.reset(); // 释放归档读句柄后再清理
		std::error_code cleanupEc;
		std::filesystem::remove_all(workDir, cleanupEc);
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 符号链接祖先目录拒绝（环境无权限时跳过）
// ════════════════════════════════════════════

static void testSymlinkAncestorRejected() {
	TEST("extractOne: symlinked ancestor directory is rejected") {
		const auto workDir = makeWorkDir("symlink");
		const auto dcgPath = workDir / "symlink.dcg";
		const auto outside = workDir / "outside";
		std::filesystem::create_directories(outside);

		writePayload(workDir / "payload.bin", "symlink-escape probe");
		{
			auto w = DcgArchive::openWrite(dcgPath);
			w->writeGraphJson("{}");
			w->addModelFile("models/link.bin", workDir / "payload.bin");
			w->finalize();
		}

		auto r = DcgArchive::openRead(dcgPath);
		std::error_code ec;
		std::filesystem::create_directory_symlink(outside, r->tempDir() / "models", ec);
		if (ec) {
			std::cout << "SKIP (symlink privilege unavailable: " << ec.message() << ") ";
			r.reset();
			std::error_code cleanupEc;
			std::filesystem::remove_all(workDir, cleanupEc);
			return;
		}

		bool rejected = false;
		try {
			r->extractOne("models/link.bin");
		} catch (const GraphException&) {
			rejected = true;
		}
		CHECK(rejected, "symlinked ancestor directory must be rejected");
		CHECK(!std::filesystem::exists(outside / "link.bin"), "nothing may be written through the symlink");

		r.reset(); // 释放归档读句柄后再清理
		std::error_code cleanupEc;
		std::filesystem::remove_all(workDir, cleanupEc);
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 完整性/预算：截断与高压缩比拒绝
// ════════════════════════════════════════════

static void testTruncatedArchiveRejected() {
	TEST("truncated archive must be rejected with a clear error") {
		const auto workDir = makeWorkDir("truncated");
		const auto dcgPath = workDir / "truncated.dcg";

		writePayload(workDir / "payload.bin", std::string(4096, 'x'));
		{
			auto w = DcgArchive::openWrite(dcgPath);
			w->writeGraphJson("{\"nodes\":[]}");
			w->addModelFile("models/pad.bin", workDir / "payload.bin");
			w->finalize();
		}
		const auto size = std::filesystem::file_size(dcgPath);
		std::filesystem::resize_file(dcgPath, size / 2);

		bool rejected = false;
		try {
			auto r = DcgArchive::openRead(dcgPath);
			r->readGraphJson();
		} catch (const GraphException&) {
			rejected = true;
		}
		CHECK(rejected, "truncated archive must raise a clear GraphException");

		std::error_code cleanupEc;
		std::filesystem::remove_all(workDir, cleanupEc);
	}
	END_TEST();
}

static void testCompressionRatioBombRejected() {
	TEST("high-compression-ratio entry (zip-bomb shape) rejected by budget") {
		const auto workDir = makeWorkDir("bomb");
		const auto dcgPath = workDir / "bomb.dcg";

		// 低熵大体积条目：deflate 后极小，压缩比远超预算阈值
		const std::string bigJson = "{\"pad\":\"" + std::string(16u << 20, 'a') + "\"}";
		{
			auto w = DcgArchive::openWrite(dcgPath);
			w->writeGraphJson(bigJson);
			w->finalize();
		}

		auto r = DcgArchive::openRead(dcgPath);
		bool rejected = false;
		try {
			r->readGraphJson();
		} catch (const GraphException& e) {
			rejected = std::string(e.what()).find("compression ratio") != std::string::npos;
		}
		CHECK(rejected, "suspicious compression ratio must be rejected before decompression");

		r.reset(); // 释放归档读句柄后再清理
		std::error_code cleanupEc;
		std::filesystem::remove_all(workDir, cleanupEc);
	}
	END_TEST();
}

// ════════════════════════════════════════════
// 不误伤：正常归档读取/解压照常
// ════════════════════════════════════════════

static void testNormalRoundTripStillWorks() {
	TEST("normal archive (graph.json + models/x) reads/extracts intact") {
		const auto workDir = makeWorkDir("roundtrip");
		const auto dcgPath = workDir / "normal.dcg";

		writePayload(workDir / "tiny.bin", "tiny-model-bytes");
		{
			auto w = DcgArchive::openWrite(dcgPath);
			w->writeGraphJson("{\"nodes\":[]}");
			w->addModelFile("models/tiny.bin", workDir / "tiny.bin");
			w->finalize();
		}

		auto r = DcgArchive::openRead(dcgPath);
		CHECK(r->readGraphJson() == "{\"nodes\":[]}", "graph.json roundtrip intact");
		const auto p = r->extractOne("models/tiny.bin");
		CHECK(std::filesystem::exists(p), "normal extraction path must work");
		CHECK(readFile(p) == std::string("tiny-model-bytes"), "extracted content must match");

		r.reset(); // 释放归档读句柄后再清理
		std::error_code cleanupEc;
		std::filesystem::remove_all(workDir, cleanupEc);
	}
	END_TEST();
}

// ════════════════════════════════════════════
// IR-04/05：聚合预算（条目数 / 全局条目 / graph.json 体积）
// ════════════════════════════════════════════

static void testExtractEntryCountBudgetRejected() {
	TEST("IR-04: extract entry-count budget rejects excessive extraction") {
		const auto workDir = makeWorkDir("entrycount");
		const auto dcgPath = workDir / "entrycount.dcg";

		writePayload(workDir / "payload.bin", "x");
		{
			auto w = DcgArchive::openWrite(dcgPath);
			w->writeGraphJson("{\"nodes\":[]}");
			// 超过 kMaxExtractEntries(256) 条解包请求：路径各异、内容相同
			for (int i = 0; i < 257; ++i)
				w->addModelFile("models/entry_" + std::to_string(i) + ".bin", workDir / "payload.bin");
			w->finalize();
		}

		auto r = DcgArchive::openRead(dcgPath);
		bool rejected = false;
		try {
			for (int i = 0; i < 257; ++i)
				r->extractOne("models/entry_" + std::to_string(i) + ".bin");
		} catch (const GraphException&) {
			rejected = true;
		}
		CHECK(rejected, "257th extractOne must hit the entry-count budget");

		r.reset(); // 释放归档读句柄后再清理
		std::error_code cleanupEc;
		std::filesystem::remove_all(workDir, cleanupEc);
	}
	END_TEST();
}

static void testArchiveGlobalEntryBudgetRejected() {
	TEST("IR-04: archive with too many entries is rejected on open") {
		const auto workDir = makeWorkDir("globalentries");
		const auto dcgPath = workDir / "globalentries.dcg";

		writePayload(workDir / "payload.bin", "x");
		{
			auto w = DcgArchive::openWrite(dcgPath);
			w->writeGraphJson("{\"nodes\":[]}");
			// 4097 + graph.json = 4098 > kMaxArchiveEntries(4096)
			for (int i = 0; i < 4097; ++i)
				w->addModelFile("models/e" + std::to_string(i) + ".bin", workDir / "payload.bin");
			w->finalize();
		}

		bool rejected = false;
		try {
			auto r = DcgArchive::openRead(dcgPath);
			(void)r;
		} catch (const GraphException&) {
			rejected = true;
		}
		CHECK(rejected, "openRead must reject archives above the global entry budget");

		std::error_code cleanupEc;
		std::filesystem::remove_all(workDir, cleanupEc);
	}
	END_TEST();
}

static void testGraphJsonSizeBudgetRejected() {
	TEST("IR-05: graph.json above the dedicated size budget is rejected") {
		const auto workDir = makeWorkDir("bigjson");
		const auto dcgPath = workDir / "bigjson.dcg";

		// > 64 MiB 的 graph.json：预算检查在读入前触发（不依赖压缩比检查）
		const std::string bigJson = "{\"pad\":\"" + std::string(65u << 20, 'a') + "\"}";
		{
			auto w = DcgArchive::openWrite(dcgPath);
			w->writeGraphJson(bigJson);
			w->finalize();
		}

		auto r = DcgArchive::openRead(dcgPath);
		bool rejected = false;
		try {
			r->readGraphJson();
		} catch (const GraphException& e) {
			rejected = std::string(e.what()).find("graph.json exceeds size budget") != std::string::npos;
		}
		CHECK(rejected, "oversized graph.json must be rejected by the dedicated budget");

		r.reset(); // 释放归档读句柄后再清理
		std::error_code cleanupEc;
		std::filesystem::remove_all(workDir, cleanupEc);
	}
	END_TEST();
}

static void testMultiChunkModelRoundTrip() {
	TEST("IR-06: multi-chunk (streamed) model file round-trips intact") {
		const auto workDir = makeWorkDir("multichunk");
		const auto dcgPath = workDir / "multichunk.dcg";

		// > 64 KiB 分块：跨多次循环迭代校验流式写入正确性（全程不整读入内存）
		std::string payload;
		payload.reserve((256u << 10) + 64);
		for (int i = 0; payload.size() < (256u << 10); ++i)
			payload += "chunk-" + std::to_string(i) + ";";
		writePayload(workDir / "payload.bin", payload);

		{
			auto w = DcgArchive::openWrite(dcgPath);
			w->writeGraphJson("{\"nodes\":[]}");
			w->addModelFile("models/multi.bin", workDir / "payload.bin");
			w->finalize();
		}

		auto r = DcgArchive::openRead(dcgPath);
		const auto p = r->extractOne("models/multi.bin");
		CHECK(readFile(p) == payload, "multi-chunk model content must round-trip intact");

		r.reset(); // 释放归档读句柄后再清理
		std::error_code cleanupEc;
		std::filesystem::remove_all(workDir, cleanupEc);
	}
	END_TEST();
}

// ════════════════════════════════════════════

int main() {
	try {
		testSafePathValidator();
		testExtractOneRejectsTraversal();
		testSymlinkAncestorRejected();
		testTruncatedArchiveRejected();
		testCompressionRatioBombRejected();
		testMultiChunkModelRoundTrip();
		testExtractEntryCountBudgetRejected();
		testArchiveGlobalEntryBudgetRejected();
		testGraphJsonSizeBudgetRejected();
		testNormalRoundTripStillWorks();
	} catch (const std::exception& e) {
		std::cerr << "UNEXPECTED EXCEPTION: " << e.what() << std::endl;
		return 1;
	}

	if (failures != 0) {
		std::cerr << failures << " test(s) failed" << std::endl;
		return 1;
	}
	std::cout << "All DcgArchive security tests passed" << std::endl;
	return 0;
}
