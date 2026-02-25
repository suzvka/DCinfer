// Expanded tests for DC::Tensor to increase coverage and validate view/path semantics.
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "Tensor.hpp"

static void runTensorTests() {
    using namespace DC;

    // prepare dense test data 2x3 floats
    std::vector<float> src = { 1,2,3,4,5,6 };
    std::vector<std::byte> bytes(src.size() * sizeof(float));
    std::memcpy(bytes.data(), src.data(), bytes.size());

    // 1) Create from dense bytes and verify data<T>()
    Tensor t = Tensor::Create<float>({ 2,3 }, std::move(bytes));
    auto span = t.data<float>();
    std::vector<float> got(span.begin(), span.end());
    if (got != src) throw std::runtime_error("create/read mismatch");

    // 2) Dense fast-path editing via view: replace row 0 and an element
	t[0] = std::vector<float>{ 10, 20, 30 }; // whole row replace
    t[1][2].set<float>(99.0f);
    auto after = t.data<float>();
    std::vector<float> expected = { 10,20,30,4,5,99 };
    if (!std::equal(after.begin(), after.end(), expected.begin())) throw std::runtime_error("dense edit mismatch");

    // 3) Scalar (0-D) assignment and item()
    Tensor s = Tensor::Create<float>();
    s = 3.14f;
    if (std::abs(s.item<float>() - 3.14f) > 1e-6f) throw std::runtime_error("scalar assign/item mismatch");

    // 4) Copy constructor and copy semantics
    Tensor copy = t; // copy should preserve data
    auto copySpan = copy.data<float>();
    if (!std::equal(copySpan.begin(), copySpan.end(), after.begin())) throw std::runtime_error("copy mismatch");

    // 5) Move constructor: move the copy and ensure moved-from is valid to destroy
    Tensor moved = std::move(copy);
    auto movedSpan = moved.data<float>();
    if (!std::equal(movedSpan.begin(), movedSpan.end(), after.begin())) throw std::runtime_error("move mismatch");

    // 6) fill() on an int tensor
    std::vector<int32_t> srcI = { 1,2,3,4 };
    std::vector<std::byte> bytesI(srcI.size() * sizeof(int32_t));
    std::memcpy(bytesI.data(), srcI.data(), bytesI.size());
    Tensor ti = Tensor::Create<int32_t>({2,2}, std::move(bytesI));
    ti.fill<int32_t>(7);
    auto afterI = ti.data<int32_t>();
    for (auto v : afterI) if (v != 7) throw std::runtime_error("fill mismatch");

    // 7) Negative indexing write (last row) and verify
    moved[-1][2].set<float>(101.0f); // -1 refers to last first-dim
    if (std::abs(moved.data<float>()[ (static_cast<size_t>(1) * 3) + 2 ] - 101.0f) > 1e-6f) throw std::runtime_error("negative index write mismatch");

    // 8) Const view reads
    const Tensor& ct = moved;
    auto row1 = ct[1].read<float>();
    if (row1.size() != 3) throw std::runtime_error("const view read size mismatch");

    // 9) bytes() view size check
    auto b = moved.bytes();
    if (b.size() != moved.data<float>().size() * sizeof(float)) throw std::runtime_error("bytes size mismatch");

    // 10) Error cases: type mismatch on scalar assign
    Tensor tf = Tensor::Create<float>();
    bool gotTypeMismatch = false;
    try {
        tf = 1.0; // double into float tensor -> should throw TensorException
    }
    catch (const TensorException& e) {
        if (e.getErrorType() == TensorException::ErrorType::TypeMismatch) gotTypeMismatch = true;
    }
    if (!gotTypeMismatch) throw std::runtime_error("expected type mismatch on scalar assign");

    // 11) View.item() on non-scalar view should throw NotAScalar
    bool gotNotScalar = false;
    try {
        moved[0].item<float>(); // row with 3 elements
    }
    catch (const TensorException& e) {
        if (e.getErrorType() == TensorException::ErrorType::NotAScalar) gotNotScalar = true;
    }
    if (!gotNotScalar) throw std::runtime_error("expected NotAScalar on view.item()");

    // 12) fast read
    try {
		std::vector<double> sample = { 1.1, 2.2, 3.3, 4.4 };
		Tensor t2 = Tensor::Create<double>();
        t2[0] = sample;
		auto spanD = t2.data<double>();
		auto readSample = t2.getData<double>();
		if (readSample != sample) throw std::runtime_error("fast read mismatch");
        if (t2.hasCache()) throw std::runtime_error("fast read should not have cache");
    }
    catch (const TensorException& e) {
        throw std::runtime_error("fast read failed");
	}

    std::cout << "Tensor tests passed" << std::endl;
}

int main() {
    try {
        runTensorTests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failure: " << e.what() << std::endl;
        return -1;
    }
}
