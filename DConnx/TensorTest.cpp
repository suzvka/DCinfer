// Lightweight tests for Tensor behaviors introduced in refactor.
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include "Tensor.hpp"
#include "TensorData.h"

// Detailed low-level TensorData tests exercising scalar, block, cache, and view paths.
static void runTensorDataDetailedTests() {
    using namespace DC;

    // 1) Scalar write/read with float then overwrite with smaller integer
    {
        TensorData td;
        td.write({}, 3.5f);
        auto f = td.readElement<float>({});
        if (std::abs(f - 3.5f) > 1e-6f) throw std::runtime_error("TensorData scalar float mismatch");

        // Overwrite with int16_t: should keep only lower bytes and zero rest
        td.write({}, static_cast<int16_t>(42));
        auto i16 = td.readElement<int16_t>({});
        if (i16 != 42) throw std::runtime_error("TensorData scalar int16 overwrite mismatch");
    }

    // 2) 1-D block write then element overwrite with smaller type
    {
        TensorData td2;
        // create a 1-D block of two floats (this will create a block at path {})
        td2.write({}, std::vector<float>{1.0f, 2.0f});
        // overwrite element index 1 with a 16-bit integer
        td2.write(std::vector<size_t>{1}, static_cast<int16_t>(123));
        auto got = td2.readElement<int16_t>(std::vector<size_t>{1});
        if (got != 123) throw std::runtime_error("TensorData element int16 write mismatch");
    }

    // 3) Multi-dim blocks: create 2x2 blocks with last-dim size 3, test cache build and view materialize
    {
        TensorData td3;
        // write two blocks at paths {0} and {1}, each block has 3 floats
        td3.write(std::vector<size_t>{0}, std::vector<float>{1.0f, 2.0f, 3.0f});
        td3.write(std::vector<size_t>{1}, std::vector<float>{4.0f, 5.0f, 6.0f});

        // reading as float span via cache forces cache build
        auto span = td3.data<float>();
        if (span.size() != 6) throw std::runtime_error("TensorData multi-block dense size mismatch");

        // edit an element via view write (smaller type)
        td3.write(std::vector<size_t>{1, 2}, static_cast<int16_t>(321));
        // ensure view read returns overwritten value when interpreted as int16 at element index
        auto val = td3.readElement<int16_t>(std::vector<size_t>{1,2});
        if (val != 321) throw std::runtime_error("TensorData multi-dim overwrite mismatch");
    }

    // 4) writeCache path: create dense payload, then use writeCache to overwrite a block
    {
        // prepare dense shape {2,3} with float elements
        TensorData td4;
        std::vector<std::byte> dense(2 * 3 * sizeof(float));
        std::fill(dense.begin(), dense.end(), std::byte(0));
        td4.loadData(std::vector<size_t>{2,3}, sizeof(float), std::move(dense));

        // write a full block (path {0}) with floats
        std::vector<float> block0 = {1.5f, 2.5f, 3.5f};
        if (!td4.writeCache(std::vector<size_t>{0}, block0)) throw std::runtime_error("TensorData writeCache block failed");

        // overwrite single element via writeCache element form
        if (!td4.writeCacheElement(std::vector<size_t>{1,2}, static_cast<int16_t>(77))) throw std::runtime_error("TensorData writeCache element failed");

        auto gotf = td4.readElement<float>(std::vector<size_t>{0,1});
        // previously untouched element should be 2.5
        if (std::abs(gotf - 2.5f) > 1e-6f) throw std::runtime_error("TensorData writeCache readback mismatch");
    }

	std::cout << "TensorData detailed tests passed" << std::endl;
}

// Negative/exception tests for TensorData: verifies error handling and boundary checks.
static void runTensorDataExceptionTests() {
    using namespace DC;

    auto fail = [](const std::string& msg) {
        throw std::runtime_error("TensorData exception test failed: " + msg);
        };

    auto expectInvalidArgument = [&](auto&& fn, const char* caseName) {
        try {
            fn();
            fail(std::string(caseName) + " expected std::invalid_argument, but no exception thrown");
        }
        catch (const std::invalid_argument&) {
            // OK
        }
        catch (const std::exception& e) {
            fail(std::string(caseName) + " threw unexpected std::exception: " + e.what());
        }
        catch (...) {
            fail(std::string(caseName) + " threw non-std exception");
        }
        };

    auto expectOutOfRange = [&](auto&& fn, const char* caseName) {
        try {
            fn();
            fail(std::string(caseName) + " expected std::out_of_range, but no exception thrown");
        }
        catch (const std::out_of_range&) {
            // OK
        }
        catch (const std::exception& e) {
            fail(std::string(caseName) + " threw unexpected std::exception: " + e.what());
        }
        catch (...) {
            fail(std::string(caseName) + " threw non-std exception");
        }
        };

    auto expectRuntimeError = [&](auto&& fn, const char* caseName) {
        try {
            fn();
            fail(std::string(caseName) + " expected std::runtime_error, but no exception thrown");
        }
        catch (const std::runtime_error&) {
            // OK
        }
        catch (const std::exception& e) {
            fail(std::string(caseName) + " threw unexpected std::exception: " + e.what());
        }
        catch (...) {
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
        TensorData::Shape shape{ 2, 3 };
        TensorData::DataBlock wrongBytes(2 * 3 * sizeof(float) - 1, std::byte(0));
        expectInvalidArgument([&] {
            TensorData td(shape, sizeof(float), std::move(wrongBytes));
            }, "TensorData(shape, typeSize, denseBytes size mismatch)");
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
        expectOutOfRange([&] {
            td.writeCache(std::vector<size_t>{0, 0, 0}, std::vector<float>{1.0f});
            }, "writeCache rank exceeds");
    }

    // 4) writeCache: unsupported slice form (neither full element nor block) -> invalid_argument
    {
        auto td = makeDenseFloat23();
        expectInvalidArgument([&] {
            td.writeCache(std::vector<size_t>{}, std::vector<float>{1.0f}); // {} is not supported for rank-2 cache
            }, "writeCache unsupported slice form");
    }

    // 5) writeCache: index out of range leads to computed write exceeding cache size -> out_of_range
    {
        auto td = makeDenseFloat23();
        expectOutOfRange([&] {
            // shape {2,3}: valid indices are [0..1]x[0..2], so {2,0} should overflow range check
            td.writeCache(std::vector<size_t>{2, 0}, std::vector<float>{1.0f});
            }, "writeCache write exceeds cache size");
    }

    // 6) writeCache: data size larger than target region -> invalid_argument
    {
        auto td = makeDenseFloat23();
        expectInvalidArgument([&] {
            // block write at path {0} targets 3 floats; provide 4 floats (too large)
            td.writeCache(std::vector<size_t>{0}, std::vector<float>{1.f, 2.f, 3.f, 4.f});
            }, "writeCache data size mismatch (too large)");
    }

    // 7) read: type mismatch (typeSize not divisible by sizeof(T)) -> invalid_argument
    {
        auto td = makeDenseFloat23();
        expectInvalidArgument([&] {
            (void)td.read<double>(std::vector<size_t>{0, 0});
            }, "read<double> type size mismatch");
    }

    // 8) read: path rank exceeds -> out_of_range
    {
        auto td = makeDenseFloat23();
        expectOutOfRange([&] {
            (void)td.read<float>(std::vector<size_t>{0, 0, 0});
            }, "read path rank exceeds");
    }

    // 9) read: element index out of range -> out_of_range
    {
        auto td = makeDenseFloat23();
        expectOutOfRange([&] {
            (void)td.read<float>(std::vector<size_t>{2, 0});
            }, "read element index out of range");
    }

    // 10) write(element): incompatible incoming type size -> invalid_argument
    {
        TensorData td;
        td.write(std::vector<size_t>{0}, std::vector<float>{1.0f, 2.0f, 3.0f}); // establishes typeSize=4 (float slots)
        expectInvalidArgument([&] {
            // try to write a double into float-slot tensor (4 % 8 != 0)
            td.write(std::vector<size_t>{0, 0}, 1.0); // double literal
            }, "write(element) type size mismatch");
    }

    // 11) editMode on a completely empty TensorData should throw runtime_error (invalid state)
    {
        TensorData td;
        expectRuntimeError([&] {
            td.editMode();
            }, "editMode on empty");
    }

    std::cout << "TensorData exception tests passed" << std::endl;
}

static void runTensorDetailedTests() {
    using namespace DC;

    // Prepare test data: 2x3 float tensor
    std::vector<float> src = { 1,2,3,4,5,6 };
    std::vector<std::byte> bytes(src.size() * sizeof(float));
    std::memcpy(bytes.data(), src.data(), bytes.size());

    // Create tensor from dense bytes and verify data() view
    Tensor t = Tensor::Create<float>({ 2,3 }, std::move(bytes));
    auto span = t.data<float>();
    std::vector<float> got(span.begin(), span.end());
    if (got != src) throw std::runtime_error("create/read mismatch");

    // Test dense fast-path editing via view: replace row 0
    t[0].set(std::vector<float>{10.0f, 20.0f, 30.0f});
    t[1][2].set<float>(99.0f);
    auto after = t.data<float>();
    std::vector<float> expected = { 10,20,30,4,5,99 };
    if (!std::equal(after.begin(), after.end(), expected.begin())) throw std::runtime_error("dense edit mismatch");

    // Test scalar (0-D) assignment and item()
    Tensor s = Tensor::Create<float>();
    s = 3.14f;
    if (s.item<float>() != 3.14f) throw std::runtime_error("scalar assign/item mismatch");

    std::cout << "Tensor detailed tests passed" << std::endl;
}

int main() {
    try {
        // Detailed TensorData tests
        runTensorDataDetailedTests();

		// Exception tests for TensorData
		runTensorDataExceptionTests();

		// Detailed Tensor tests
		runTensorDetailedTests();

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failure: " << e.what() << std::endl;
        return -1;
    }
}
