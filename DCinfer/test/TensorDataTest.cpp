#include "Tensor.hpp"
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <iostream>

static void runTensorDataDetailedTests() {
	using namespace DC;

	{
		TensorData td;
		td.write({}, 3.5f);
		auto f = td.readElement<float>({});
		if (std::abs(f - 3.5f) > 1e-6f)
			throw std::runtime_error("TensorData scalar float mismatch");

		td.write({}, static_cast<int16_t>(42));
		auto i16 = td.readElement<int16_t>({});
		if (i16 != 42)
			throw std::runtime_error("TensorData scalar int16 overwrite mismatch");
	}

	{
		TensorData td2;
		td2.write({}, std::vector<float>{1.0f, 2.0f});
		td2.write(std::vector<size_t>{1}, static_cast<int16_t>(123));
		auto got = td2.readElement<int16_t>(std::vector<size_t>{1});
		if (got != 123)
			throw std::runtime_error("TensorData element int16 write mismatch");
	}

	{
		TensorData td3;
		td3.write(std::vector<size_t>{0}, std::vector<float>{1.0f, 2.0f, 3.0f});
		td3.write(std::vector<size_t>{1}, std::vector<float>{4.0f, 5.0f, 6.0f});

		auto span = td3.data<float>();
		if (span.size() != 6)
			throw std::runtime_error("TensorData multi-block dense size mismatch");

		td3.write(std::vector<size_t>{1, 2}, static_cast<int16_t>(321));
		auto val = td3.readElement<int16_t>(std::vector<size_t>{1, 2});
		if (val != 321)
			throw std::runtime_error("TensorData multi-dim overwrite mismatch");
	}

	{
		TensorData td4;
		std::vector<std::byte> dense(2 * 3 * sizeof(float));
		std::fill(dense.begin(), dense.end(), std::byte(0));
		td4.loadData(std::vector<size_t>{2, 3}, sizeof(float), std::move(dense));

		std::vector<float> block0 = {1.5f, 2.5f, 3.5f};
		if (!td4.writeCache(std::vector<size_t>{0}, block0))
			throw std::runtime_error("TensorData writeCache block failed");

		if (!td4.writeCacheElement(std::vector<size_t>{1, 2}, static_cast<int16_t>(77)))
			throw std::runtime_error("TensorData writeCache element failed");

		auto gotf = td4.readElement<float>(std::vector<size_t>{0, 1});
		if (std::abs(gotf - 2.5f) > 1e-6f)
			throw std::runtime_error("TensorData writeCache readback mismatch");
	}

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

static void runTensorDataExceptionTests() {
	using namespace DC;

	auto fail = [](const std::string& msg) { throw std::runtime_error("TensorData exception test failed: " + msg); };

	auto expectInvalidArgument = [&](auto&& fn, const char* caseName) {
		try {
			fn();
			fail(std::string(caseName) + " expected std::invalid_argument, but no exception thrown");
		} catch (const std::invalid_argument&) {
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
		} catch (const std::exception& e) {
			fail(std::string(caseName) + " threw unexpected std::exception: " + e.what());
		} catch (...) {
			fail(std::string(caseName) + " threw non-std exception");
		}
	};

	{
		TensorData td;
		expectInvalidArgument([&] { td.setTypeSize(0); }, "setTypeSize(0)");
	}

	{
		TensorData::Shape shape{2, 3};
		TensorData::DataBlock wrongBytes(2 * 3 * sizeof(float) - 1, std::byte(0));
		expectInvalidArgument([&] { TensorData td(shape, sizeof(float), std::move(wrongBytes)); },
							  "TensorData(shape, typeSize, denseBytes size mismatch)");
	}

	auto makeDenseFloat23 = []() {
		TensorData td;
		std::vector<std::byte> dense(2 * 3 * sizeof(float), std::byte(0));
		td.loadData(std::vector<size_t>{2, 3}, sizeof(float), std::move(dense));
		return td;
	};

	{
		auto td = makeDenseFloat23();
		expectOutOfRange([&] { td.writeCache(std::vector<size_t>{0, 0, 0}, std::vector<float>{1.0f}); },
						 "writeCache rank exceeds");
	}

	{
		auto td = makeDenseFloat23();
		expectInvalidArgument(
			[&] {
				td.writeCache(std::vector<size_t>{}, std::vector<float>{1.0f});
			},
			"writeCache unsupported slice form");
	}

	{
		auto td = makeDenseFloat23();
		expectOutOfRange(
			[&] {
				td.writeCache(std::vector<size_t>{2, 0}, std::vector<float>{1.0f});
			},
			"writeCache write exceeds cache size");
	}

	{
		auto td = makeDenseFloat23();
		expectInvalidArgument(
			[&] {
				td.writeCache(std::vector<size_t>{0}, std::vector<float>{1.f, 2.f, 3.f, 4.f});
			},
			"writeCache data size mismatch (too large)");
	}

	{
		auto td = makeDenseFloat23();
		expectInvalidArgument([&] { (void)td.read<double>(std::vector<size_t>{0, 0}); },
							  "read<double> type size mismatch");
	}

	{
		auto td = makeDenseFloat23();
		expectOutOfRange([&] { (void)td.read<float>(std::vector<size_t>{0, 0, 0}); }, "read path rank exceeds");
	}

	{
		auto td = makeDenseFloat23();
		expectOutOfRange([&] { (void)td.read<float>(std::vector<size_t>{2, 0}); }, "read element index out of range");
	}

	{
		TensorData td;
		td.write(std::vector<size_t>{0}, std::vector<float>{1.0f, 2.0f, 3.0f});
		expectInvalidArgument(
			[&] {
				td.write(std::vector<size_t>{0, 0}, 1.0);
			},
			"write(element) type size mismatch");
	}

	{
		TensorData td;
		expectRuntimeError([&] { td.editMode(); }, "editMode on empty");
	}

	{
		TensorData::Shape shape{2, 0};
		TensorData::DataBlock bytes(2 * sizeof(float), std::byte(0));
		expectInvalidArgument([&] { TensorData td(shape, std::move(bytes)); },
							  "TensorData(shape{2,0}, data) division-by-zero guard");
	}

	{
		TensorData::Shape shape{2, 3};
		TensorData::DataBlock bytes(2 * 3 * sizeof(float), std::byte(0));
		TensorData td(shape, std::move(bytes));
		if (td.typeSize() != sizeof(float))
			fail("TensorData(shape{2,3}, data) typeSize inference mismatch");
	}

	{
		TensorData td;
		td.write(std::vector<size_t>{0}, std::vector<float>{1.0f, 2.0f, 3.0f});
		expectInvalidArgument([&] { td.expand(std::vector<size_t>{2}, 0.0f); }, "expand rank mismatch (target lower)");
		expectInvalidArgument([&] { td.expand(std::vector<size_t>{2, 1, 1}, 0.0f); },
							  "expand rank mismatch (target higher)");
		expectInvalidArgument([&] { td.expand(std::vector<size_t>{}, 0.0f); }, "expand empty target shape");
	}

	{
		TensorData td;
		td.write(std::vector<size_t>{0}, std::vector<float>{1.0f, 2.0f, 3.0f});
		td.expand(std::vector<size_t>{2, 3}, 9.0f);
		auto span = td.data<float>();
		if (span.size() != 6)
			fail("expand 2-D dense size mismatch");
		if (std::abs(span[0] - 1.0f) > 1e-6f || std::abs(span[5] - 9.0f) > 1e-6f)
			fail("expand 2-D must fill missing block with fillData");
	}

	{
		TensorData td;
		td.write({}, std::vector<float>{1.0f, 2.0f, 3.0f});
		td.expand(std::vector<size_t>{5}, 7.0f);
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

	{
		TensorData td;
		td.write({}, 3.5f);
		td.expand(std::vector<size_t>{}, 9.99f);
		auto v = td.readElement<float>({});
		if (std::abs(v - 3.5f) > 1e-6f)
			fail("expand scalar target must be a no-op");
	}

	{
		expectInvalidArgument(
			[&] {
				TensorData td;
				td.loadData(std::vector<size_t>{2}, sizeof(float),
							TensorData::DataBlock(sizeof(float), std::byte(0)));
			},
			"loadData bytes smaller than shape product");
		expectInvalidArgument(
			[&] {
				TensorData td;
				td.loadData(std::vector<size_t>{2}, sizeof(float),
							TensorData::DataBlock(3 * sizeof(float), std::byte(0)));
			},
			"loadData bytes larger than shape product");
		expectInvalidArgument(
			[&] {
				TensorData td;
				td.loadData(std::vector<size_t>{2, 0}, sizeof(float), TensorData::DataBlock(1));
			},
			"loadData zero dimension with non-empty payload rejected");
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
		// 零元素是有效空稠密张量；带载荷仍必须 mismatch。
		{
			TensorData td(std::vector<size_t>{0}, 1, TensorData::DataBlock(0));
			if (td.typeSize() != 1 || !td.valid() || !td.hasCache()
				|| td.getCurrentShape() != TensorData::Shape{0} || !td.data().empty())
				fail("zero-element TensorData must keep shape, valid cache and empty payload");
		}
		expectInvalidArgument(
			[&] {
				TensorData td(std::vector<size_t>{0}, 1, TensorData::DataBlock(3));
			},
			"TensorData zero-dim with non-empty payload rejected");
		TensorData td;
		TensorData::DataBlock bytes(2 * sizeof(float));
		const float payload[2] = {42.0f, -1.5f};
		std::memcpy(bytes.data(), payload, sizeof(payload));
		td.loadData(std::vector<size_t>{2}, sizeof(float), std::move(bytes));
		auto span = td.data<float>();
		if (span.size() != 2 || span[0] != 42.0f || span[1] != -1.5f)
			fail("loadData must pass payload bytes through uninterpreted");
	}

	{
		TensorData td;
		td.write(std::vector<size_t>{0}, std::vector<float>{1.0f, 2.0f, 3.0f});
		constexpr size_t maxElems = std::numeric_limits<size_t>::max() / sizeof(float);
		expectOutOfRange(
			[&] {
				// elementIndex == SIZE_MAX/typeSize：(elementIndex+1)*typeSize 回绕，必须拒绝
				td.write(std::vector<size_t>{0, maxElems}, 1.0f);
			},
			"write(element) at SIZE_MAX/typeSize boundary");
		expectOutOfRange(
			[&] {
				td.write(std::vector<size_t>{0, static_cast<size_t>(-1)}, 1.0f);
			},
			"write(element) with wrapped negative index");
		td.write(std::vector<size_t>{0, 1}, 9.0f);
		if (td.readElement<float>(std::vector<size_t>{0, 1}) != 9.0f)
			fail("write(element) after rejected oversized index must still work");
	}

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
		const float expected[4] = {1.0f, 2.0f, 4.0f, 5.0f};
		for (size_t i = 0; i < 4; ++i)
			if (std::abs(span[i] - expected[i]) > 1e-6f)
				fail("crop 2-D must preserve per-dimension prefix elements");
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

	{
		TensorData td;
		td.write(std::vector<size_t>{0}, std::vector<float>{1.0f, 2.0f});
		td.write(std::vector<size_t>{1}, std::vector<float>{3.0f, 4.0f});
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

static void runTensorDataZeroElementTests() {
	using namespace DC;
	auto fail = [](const char* msg) { throw std::runtime_error(msg); };
	auto checkEmptyDense = [&](const TensorData& td, const TensorData::Shape& shape) {
		if (!td.valid() || !td.hasCache() || !td.empty() || td.isScalar()
			|| td.getCurrentShape() != shape || td.typeSize() != sizeof(float)
			|| td.size() != 0 || !td.data().empty() || !td.data<float>().empty())
			fail("zero-element dense tensor lost shape, type or valid empty cache");
	};
	const size_t maxDim = std::numeric_limits<size_t>::max();
	for (const auto& shape : std::vector<TensorData::Shape>{
			 {0}, {0, 3}, {2, 0}, {2, 0, 3}, {maxDim, 2, 0}, {0, maxDim}}) {
		TensorData constructed(shape, sizeof(float), TensorData::DataBlock{});
		checkEmptyDense(constructed, shape);
		// 从非空旧状态装入，确保 cache/view/scalar 都不会残留。
		TensorData loaded;
		loaded.write({}, 7.0f);
		loaded.loadData(shape, sizeof(float), TensorData::DataBlock{});
		checkEmptyDense(loaded, shape);
		for (auto* td : {&constructed, &loaded}) {
			td->editMode();
			td->editMode();
			checkEmptyDense(*td, shape);
			td->freeze();
			td->freeze();
			checkEmptyDense(*td, shape);
			if (!td->isFrozen())
				fail("zero-element dense tensor did not freeze");
			bool rejected = false;
			try {
				td->editMode();
			} catch (const TensorException&) {
				rejected = true;
			}
			if (!rejected)
				fail("frozen zero-element editMode must reject mutation");
		}
		TensorData copied(constructed);
		checkEmptyDense(copied, shape);
		if (copied.isFrozen())
			fail("zero-element copy construction must unfreeze");
		TensorData assigned;
		assigned = loaded;
		checkEmptyDense(assigned, shape);
		if (assigned.isFrozen())
			fail("zero-element copy assignment must unfreeze");
		TensorData moved(std::move(constructed));
		checkEmptyDense(moved, shape);
		if (!moved.isFrozen())
			fail("zero-element move construction must retain freeze");
		TensorData moveAssigned;
		moveAssigned = std::move(loaded);
		checkEmptyDense(moveAssigned, shape);
		if (!moveAssigned.isFrozen())
			fail("zero-element move assignment must retain freeze");
		copied.editMode();
		assigned.editMode();
		checkEmptyDense(copied, shape);
		checkEmptyDense(assigned, shape);

		bool rejected = false;
		try {
			TensorData bad(shape, sizeof(float), TensorData::DataBlock(1));
		} catch (const std::invalid_argument&) {
			rejected = true;
		}
		if (!rejected)
			fail("zero-element constructor must reject non-empty payload");
		rejected = false;
		try {
			copied.loadData(shape, sizeof(float), TensorData::DataBlock(1));
		} catch (const std::invalid_argument&) {
			rejected = true;
		}
		if (!rejected)
			fail("zero-element loadData must reject non-empty payload");
		checkEmptyDense(copied, shape); // 校验失败不得清除原有状态。
	}
	// 非零形状空载荷的构造仍只登记类型，不物化或保存传入形状。
	for (const auto& shape : std::vector<TensorData::Shape>{{2, 3}, {}}) {
		TensorData metadata(shape, sizeof(float), TensorData::DataBlock{});
		if (metadata.typeSize() != sizeof(float) || metadata.valid() || metadata.hasCache()
			|| metadata.hasView() || !metadata.getCurrentShape().empty()
			|| metadata.isScalar() != shape.empty() || !metadata.data().empty())
			fail("non-zero shape empty payload constructor metadata-only compatibility changed");
		bool rejected = false;
		try {
			metadata.loadData(shape, sizeof(float), TensorData::DataBlock{});
		} catch (const std::invalid_argument&) {
			rejected = true;
		}
		if (!rejected || metadata.valid() || metadata.typeSize() != sizeof(float))
			fail("non-zero shape empty payload loadData must still reject without mutation");
	}
	std::cout << "TensorData zero-element tests passed" << std::endl;
}

int main() {
	try {
		runTensorDataDetailedTests();
		runTensorDataExceptionTests();
		runTensorDataZeroElementTests();
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Test failed: " << e.what() << std::endl;
		return 1;
	}
}