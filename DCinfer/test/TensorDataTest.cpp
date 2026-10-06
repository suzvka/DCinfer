#include "Tensor.hpp"
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <iostream>

// Detailed low-level TensorData tests exercising scalar, block, cache, and view paths.
static void runTensorDataDetailedTests() {
	using namespace DC;

	// 1) Scalar write/read with float then overwrite with smaller integer
	{
		TensorData td;
		td.write({}, 3.5f);
		auto f = td.readElement<float>({});
		if (std::abs(f - 3.5f) > 1e-6f)
			throw std::runtime_error("TensorData scalar float mismatch");

		// Overwrite with int16_t: should keep only lower bytes and zero rest
		td.write({}, static_cast<int16_t>(42));
		auto i16 = td.readElement<int16_t>({});
		if (i16 != 42)
			throw std::runtime_error("TensorData scalar int16 overwrite mismatch");
	}

	// 2) 1-D block write then element overwrite with smaller type
	{
		TensorData td2;
		// create a 1-D block of two floats (this will create a block at path {})
		td2.write({}, std::vector<float>{1.0f, 2.0f});
		// overwrite element index 1 with a 16-bit integer
		td2.write(std::vector<size_t>{1}, static_cast<int16_t>(123));
		auto got = td2.readElement<int16_t>(std::vector<size_t>{1});
		if (got != 123)
			throw std::runtime_error("TensorData element int16 write mismatch");
	}

	// 3) Multi-dim blocks: create 2x2 blocks with last-dim size 3, test cache build and view materialize
	{
		TensorData td3;
		// write two blocks at paths {0} and {1}, each block has 3 floats
		td3.write(std::vector<size_t>{0}, std::vector<float>{1.0f, 2.0f, 3.0f});
		td3.write(std::vector<size_t>{1}, std::vector<float>{4.0f, 5.0f, 6.0f});

		// reading as float span via cache forces cache build
		auto span = td3.data<float>();
		if (span.size() != 6)
			throw std::runtime_error("TensorData multi-block dense size mismatch");

		// edit an element via view write (smaller type)
		td3.write(std::vector<size_t>{1, 2}, static_cast<int16_t>(321));
		// ensure view read returns overwritten value when interpreted as int16 at element index
		auto val = td3.readElement<int16_t>(std::vector<size_t>{1, 2});
		if (val != 321)
			throw std::runtime_error("TensorData multi-dim overwrite mismatch");
	}

	// 4) writeCache path: create dense payload, then use writeCache to overwrite a block
	{
		// prepare dense shape {2,3} with float elements
		TensorData td4;
		std::vector<std::byte> dense(2 * 3 * sizeof(float));
		std::fill(dense.begin(), dense.end(), std::byte(0));
		td4.loadData(std::vector<size_t>{2, 3}, sizeof(float), std::move(dense));

		// write a full block (path {0}) with floats
		std::vector<float> block0 = {1.5f, 2.5f, 3.5f};
		if (!td4.writeCache(std::vector<size_t>{0}, block0))
			throw std::runtime_error("TensorData writeCache block failed");

		// overwrite single element via writeCache element form
		if (!td4.writeCacheElement(std::vector<size_t>{1, 2}, static_cast<int16_t>(77)))
			throw std::runtime_error("TensorData writeCache element failed");

		auto gotf = td4.readElement<float>(std::vector<size_t>{0, 1});
		// previously untouched element should be 2.5
		if (std::abs(gotf - 2.5f) > 1e-6f)
			throw std::runtime_error("TensorData writeCache readback mismatch");
	}

	// 5) getData Test
	{
		TensorData td5;
		std::vector<std::byte> dense(4 * sizeof(float), std::byte(1));
		std::vector<std::byte> sample = dense;
		td5.loadData(std::vector<size_t>{4}, sizeof(float), std::move(dense));
		auto dataSpan = td5.getData();

		if (dataSpan != sample)
			throw std::runtime_error("TensorData getVector mismatch");
		if (td5.hasCache())
			throw std::runtime_error("TensorData getVector did not clear cache");
	}

	std::cout << "TensorData detailed tests passed" << std::endl;
}

// Negative/exception tests for TensorData: verifies error handling and boundary checks.
static void runTensorDataExceptionTests() {
	using namespace DC;

	auto fail = [](const std::string& msg) { throw std::runtime_error("TensorData exception test failed: " + msg); };

	auto expectInvalidArgument = [&](auto&& fn, const char* caseName) {
		try {
			fn();
			fail(std::string(caseName) + " expected std::invalid_argument, but no exception thrown");
		} catch (const std::invalid_argument&) {
			// OK
		} catch (const std::exception& e) {
			fail(std::string(caseName) + " threw unexpected std::exception: " + e.what());
		} catch (...) {
			fail(std::string(caseName) + " threw non-std exception");
		}
	};

	auto expectOutOfRange = [&](auto&& fn, const char* caseName) {
		try {
			fn();
			fail(std::string(caseName) + " expected std::out_of_range, but no exception thrown");
		} catch (const std::out_of_range&) {
			// OK
		} catch (const std::exception& e) {
			fail(std::string(caseName) + " threw unexpected std::exception: " + e.what());
		} catch (...) {
			fail(std::string(caseName) + " threw non-std exception");
		}
	};

	auto expectRuntimeError = [&](auto&& fn, const char* caseName) {
		try {
			fn();
			fail(std::string(caseName) + " expected std::runtime_error, but no exception thrown");
		} catch (const std::runtime_error&) {
			// OK
		} catch (const std::exception& e) {
			fail(std::string(caseName) + " threw unexpected std::exception: " + e.what());
		} catch (...) {
			fail(std::string(caseName) + " threw non-std exception");
		}
	};

	// 1) setTypeSize(0) should throw invalid_argument
	{
		TensorData td;
		expectInvalidArgument([&] { td.setTypeSize(0); }, "setTypeSize(0)");
	}

	// 2) Ctor with denseBytes size mismatch should throw invalid_argument
	{
		TensorData::Shape shape{2, 3};
		TensorData::DataBlock wrongBytes(2 * 3 * sizeof(float) - 1, std::byte(0));
		expectInvalidArgument([&] { TensorData td(shape, sizeof(float), std::move(wrongBytes)); },
							  "TensorData(shape, typeSize, denseBytes size mismatch)");
	}

	// Prepare a valid dense tensor with cache for subsequent tests: shape {2,3}, float slot (typeSize=4)
	auto makeDenseFloat23 = []() {
		TensorData td;
		std::vector<std::byte> dense(2 * 3 * sizeof(float), std::byte(0));
		td.loadData(std::vector<size_t>{2, 3}, sizeof(float), std::move(dense));
		return td;
	};

	// 3) writeCache: path rank exceeds tensor rank -> out_of_range
	{
		auto td = makeDenseFloat23();
		expectOutOfRange([&] { td.writeCache(std::vector<size_t>{0, 0, 0}, std::vector<float>{1.0f}); },
						 "writeCache rank exceeds");
	}

	// 4) writeCache: unsupported slice form (neither full element nor block) -> invalid_argument
	{
		auto td = makeDenseFloat23();
		expectInvalidArgument(
			[&] {
				td.writeCache(std::vector<size_t>{}, std::vector<float>{1.0f}); // {} is not supported for rank-2 cache
			},
			"writeCache unsupported slice form");
	}

	// 5) writeCache: index out of range leads to computed write exceeding cache size -> out_of_range
	{
		auto td = makeDenseFloat23();
		expectOutOfRange(
			[&] {
				// shape {2,3}: valid indices are [0..1]x[0..2], so {2,0} should overflow range check
				td.writeCache(std::vector<size_t>{2, 0}, std::vector<float>{1.0f});
			},
			"writeCache write exceeds cache size");
	}

	// 6) writeCache: data size larger than target region -> invalid_argument
	{
		auto td = makeDenseFloat23();
		expectInvalidArgument(
			[&] {
				// block write at path {0} targets 3 floats; provide 4 floats (too large)
				td.writeCache(std::vector<size_t>{0}, std::vector<float>{1.f, 2.f, 3.f, 4.f});
			},
			"writeCache data size mismatch (too large)");
	}

	// 7) read: type mismatch (typeSize not divisible by sizeof(T)) -> invalid_argument
	{
		auto td = makeDenseFloat23();
		expectInvalidArgument([&] { (void)td.read<double>(std::vector<size_t>{0, 0}); },
							  "read<double> type size mismatch");
	}

	// 8) read: path rank exceeds -> out_of_range
	{
		auto td = makeDenseFloat23();
		expectOutOfRange([&] { (void)td.read<float>(std::vector<size_t>{0, 0, 0}); }, "read path rank exceeds");
	}

	// 9) read: element index out of range -> out_of_range
	{
		auto td = makeDenseFloat23();
		expectOutOfRange([&] { (void)td.read<float>(std::vector<size_t>{2, 0}); }, "read element index out of range");
	}

	// 10) write(element): incompatible incoming type size -> invalid_argument
	{
		TensorData td;
		td.write(std::vector<size_t>{0}, std::vector<float>{1.0f, 2.0f, 3.0f}); // establishes typeSize=4 (float slots)
		expectInvalidArgument(
			[&] {
				// try to write a double into float-slot tensor (4 % 8 != 0)
				td.write(std::vector<size_t>{0, 0}, 1.0); // double literal
			},
			"write(element) type size mismatch");
	}

	// 11) editMode on a completely empty TensorData should throw runtime_error (invalid state)
	{
		TensorData td;
		expectRuntimeError([&] { td.editMode(); }, "editMode on empty");
	}

	// 12) 双参构造（shape, data）除零防护（CORE-03）：shape 含 0 维 → invalid_argument
	{
		TensorData::Shape shape{2, 0};
		TensorData::DataBlock bytes(2 * sizeof(float), std::byte(0));
		expectInvalidArgument([&] { TensorData td(shape, std::move(bytes)); },
							  "TensorData(shape{2,0}, data) division-by-zero guard");
	}

	// 12b) 双参构造正常路径：typeSize = data.size() / 形状乘积
	{
		TensorData::Shape shape{2, 3};
		TensorData::DataBlock bytes(2 * 3 * sizeof(float), std::byte(0));
		TensorData td(shape, std::move(bytes));
		if (td.typeSize() != sizeof(float))
			fail("TensorData(shape{2,3}, data) typeSize inference mismatch");
	}

	// 13) expand（#8-5）：秩不匹配/空 shape/降维目标拒绝；1D/标量目标行为正确
	{
		TensorData td;
		td.write(std::vector<size_t>{0}, std::vector<float>{1.0f, 2.0f, 3.0f}); // 2D 数据（catalog {0} + 3 元素）
		// 目标秩低于当前 → 拒绝（原实现越界读 UB）
		expectInvalidArgument([&] { td.expand(std::vector<size_t>{2}, 0.0f); }, "expand rank mismatch (target lower)");
		// 目标秩高于当前 → 拒绝
		expectInvalidArgument([&] { td.expand(std::vector<size_t>{2, 1, 1}, 0.0f); },
							  "expand rank mismatch (target higher)");
		// 空 shape 目标 → 拒绝（原实现对空 shape 的 back() 为 UB）
		expectInvalidArgument([&] { td.expand(std::vector<size_t>{}, 0.0f); }, "expand empty target shape");
	}

	// 13b) expand 2-D 正常路径：缺失块以 fillData 填充（回归保护）
	{
		TensorData td;
		td.write(std::vector<size_t>{0}, std::vector<float>{1.0f, 2.0f, 3.0f}); // block {0}
		td.expand(std::vector<size_t>{2, 3}, 9.0f); // 目标 {2,3}：块 {1} 缺失 → 填充 9.0
		auto span = td.data<float>();
		if (span.size() != 6)
			fail("expand 2-D dense size mismatch");
		if (std::abs(span[0] - 1.0f) > 1e-6f || std::abs(span[5] - 9.0f) > 1e-6f)
			fail("expand 2-D must fill missing block with fillData");
	}

	// 13c) expand 1-D 目标：已有 root 块扩容（旧值保留 + 新区域 fillData；
	//      修复后存储与形状同表 targetShape，不再停留旧块长）
	{
		TensorData td;
		td.write({}, std::vector<float>{1.0f, 2.0f, 3.0f}); // 1-D root block
		td.expand(std::vector<size_t>{5}, 7.0f); // 目标 {5} ≥ {3}
		auto span = td.data<float>();
		if (span.size() != 5)
			fail("expand 1-D must enlarge storage to target block length");
		if (std::abs(span[0] - 1.0f) > 1e-6f || std::abs(span[1] - 2.0f) > 1e-6f
			|| std::abs(span[2] - 3.0f) > 1e-6f)
			fail("expand 1-D must keep existing data");
		if (std::abs(span[3] - 7.0f) > 1e-6f || std::abs(span[4] - 7.0f) > 1e-6f)
			fail("expand 1-D must fill the new region with fillData");
		if (td.getCurrentShape() != TensorData::Shape{5})
			fail("expand 1-D must update current shape to target");
	}

	// 13d) expand 标量目标：等 shape 提前返回（no-op）
	{
		TensorData td;
		td.write({}, 3.5f);
		td.expand(std::vector<size_t>{}, 9.99f);
		auto v = td.readElement<float>({});
		if (std::abs(v - 3.5f) > 1e-6f)
			fail("expand scalar target must be a no-op");
	}

	// 14) loadData：shape/typeSize/bytes 声明自洽性校验（载荷本身不被解释）
	{
		expectInvalidArgument(
			[&] {
				TensorData td;
				td.loadData(std::vector<size_t>{2}, sizeof(float),
							TensorData::DataBlock(sizeof(float), std::byte(0))); // 少一半
			},
			"loadData bytes smaller than shape product");
		expectInvalidArgument(
			[&] {
				TensorData td;
				td.loadData(std::vector<size_t>{2}, sizeof(float),
							TensorData::DataBlock(3 * sizeof(float), std::byte(0))); // 多一半
			},
			"loadData bytes larger than shape product");
		expectInvalidArgument(
			[&] {
				TensorData td;
				td.loadData(std::vector<size_t>{2, 0}, sizeof(float), TensorData::DataBlock(0));
			},
			"loadData zero dimension rejected");
		expectInvalidArgument(
			[&] {
				TensorData td;
				td.loadData(std::vector<size_t>{2}, 0, TensorData::DataBlock(0));
			},
			"loadData zero typeSize rejected");
		expectInvalidArgument(
			[&] {
				TensorData td;
				td.loadData(std::vector<size_t>{std::numeric_limits<size_t>::max(), 2}, 1,
							TensorData::DataBlock(0));
			},
			"loadData shape product overflow rejected");
		expectInvalidArgument(
			[&] {
				TensorData td(std::vector<size_t>{std::numeric_limits<size_t>::max(), 2},
							 1, TensorData::DataBlock(0));
			},
			"TensorData constructor shape product overflow rejected");
		// 零维（0 元素张量）合法语义：空文本/空集合的既有表示（OpenAI 空响应、
		// wire 空文本帧）——metadata-only 空载荷构造成功；带载荷则 mismatch。
		{
			TensorData td(std::vector<size_t>{0}, 1, TensorData::DataBlock(0));
			if (td.typeSize() != 1 || !td.data().empty())
				fail("zero-dim metadata-only TensorData must keep typeSize and empty payload");
		}
		expectInvalidArgument(
			[&] {
				TensorData td(std::vector<size_t>{0}, 1, TensorData::DataBlock(3));
			},
			"TensorData zero-dim with non-empty payload rejected");
		// 合法路径行为不变：字节载荷原样进入 cache，不解释内容
		TensorData td;
		TensorData::DataBlock bytes(2 * sizeof(float));
		const float payload[2] = {42.0f, -1.5f};
		std::memcpy(bytes.data(), payload, sizeof(payload));
		td.loadData(std::vector<size_t>{2}, sizeof(float), std::move(bytes));
		auto span = td.data<float>();
		if (span.size() != 2 || span[0] != 42.0f || span[1] != -1.5f)
			fail("loadData must pass payload bytes through uninterpreted");
	}

	// 15) write(element)：溢出边界与回绕负索引在污染 catalog 前拒绝
	{
		TensorData td;
		td.write(std::vector<size_t>{0}, std::vector<float>{1.0f, 2.0f, 3.0f}); // typeSize=4
		constexpr size_t maxElems = std::numeric_limits<size_t>::max() / sizeof(float);
		expectOutOfRange(
			[&] {
				// 边界值 elementIndex == SIZE_MAX/typeSize：(elementIndex+1)*typeSize
				// 恰好回绕，旧条件 ">" 放行后 memcpy 越界写
				td.write(std::vector<size_t>{0, maxElems}, 1.0f);
			},
			"write(element) at SIZE_MAX/typeSize boundary");
		expectOutOfRange(
			[&] {
				// int64 负索引经 static_cast<size_t> 回绕为巨大值
				td.write(std::vector<size_t>{0, static_cast<size_t>(-1)}, 1.0f);
			},
			"write(element) with wrapped negative index");
		// catalog 未被污染：拒绝后正常写入仍有效
		td.write(std::vector<size_t>{0, 1}, 9.0f);
		if (td.readElement<float>(std::vector<size_t>{0, 1}) != 9.0f)
			fail("write(element) after rejected oversized index must still work");
	}

	// 16) crop：多维前缀裁剪（修复前 {2,3}→{2,2} 得 1,2,3,4）
	{
		TensorData td;
		std::vector<std::byte> dense(2 * 3 * sizeof(float));
		const float vals[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
		std::memcpy(dense.data(), vals, sizeof(vals));
		td.loadData(std::vector<size_t>{2, 3}, sizeof(float), std::move(dense));
		td.crop(std::vector<size_t>{2, 2});
		auto span = td.data<float>();
		if (span.size() != 4)
			fail("crop 2-D must shrink storage to new element count");
		const float expected[4] = {1.0f, 2.0f, 4.0f, 5.0f}; // [[1,2],[4,5]]
		for (size_t i = 0; i < 4; ++i)
			if (std::abs(span[i] - expected[i]) > 1e-6f)
				fail("crop 2-D must preserve per-dimension prefix elements");
		// 1-D 回归：扁平前缀截断行为不变
		TensorData td1d;
		std::vector<std::byte> dense1d(4 * sizeof(float));
		const float vals1d[4] = {1.0f, 2.0f, 3.0f, 4.0f};
		std::memcpy(dense1d.data(), vals1d, sizeof(vals1d));
		td1d.loadData(std::vector<size_t>{4}, sizeof(float), std::move(dense1d));
		td1d.crop(std::vector<size_t>{2});
		auto span1d = td1d.data<float>();
		if (span1d.size() != 2 || std::abs(span1d[0] - 1.0f) > 1e-6f
			|| std::abs(span1d[1] - 2.0f) > 1e-6f)
			fail("crop 1-D must keep the flat prefix");
	}

	// 17) expand 扩已有块：{2,2}→{2,4}（修复前已有行块不扩容，形状仍停留 {2,2}）
	{
		TensorData td;
		td.write(std::vector<size_t>{0}, std::vector<float>{1.0f, 2.0f}); // 块{0}
		td.write(std::vector<size_t>{1}, std::vector<float>{3.0f, 4.0f}); // 块{1}
		td.expand(std::vector<size_t>{2, 4}, 9.0f);
		auto span = td.data<float>();
		if (span.size() != 8)
			fail("expand 2-D must enlarge existing blocks to new block length");
		const float expected[8] = {1.0f, 2.0f, 9.0f, 9.0f, 3.0f, 4.0f, 9.0f, 9.0f};
		for (size_t i = 0; i < 8; ++i)
			if (std::abs(span[i] - expected[i]) > 1e-6f)
				fail("expand 2-D must keep old values and fill only the new region");
		if (td.getCurrentShape() != TensorData::Shape({2, 4}))
			fail("expand 2-D must update current shape to target");
	}

	std::cout << "TensorData exception tests passed" << std::endl;
}

int main() {
	try {
		runTensorDataDetailedTests();
		runTensorDataExceptionTests();
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Test failed: " << e.what() << std::endl;
		return 1;
	}
}