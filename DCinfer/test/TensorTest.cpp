// DC::Tensor 视图/路径语义测试
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "Tensor.hpp"

static void runTensorTests() {
	using namespace DC;

	std::vector<float> src = {1, 2, 3, 4, 5, 6};
	std::vector<std::byte> bytes(src.size() * sizeof(float));
	std::memcpy(bytes.data(), src.data(), bytes.size());

	Tensor t = Tensor::Create<float>({2, 3}, std::move(bytes));
	auto span = t.data<float>();
	std::vector<float> got(span.begin(), span.end());
	if (got != src)
		throw std::runtime_error("create/read mismatch");

	t[0] = std::vector<float>{10, 20, 30};
	t[1][2].set<float>(99.0f);
	auto after = t.data<float>();
	std::vector<float> expected = {10, 20, 30, 4, 5, 99};
	if (!std::equal(after.begin(), after.end(), expected.begin()))
		throw std::runtime_error("dense edit mismatch");

	Tensor s = Tensor::Create<float>();
	s = 3.14f;
	if (std::abs(s.item<float>() - 3.14f) > 1e-6f)
		throw std::runtime_error("scalar assign/item mismatch");

	Tensor copy = t;
	auto copySpan = copy.data<float>();
	if (!std::equal(copySpan.begin(), copySpan.end(), after.begin()))
		throw std::runtime_error("copy mismatch");

	Tensor moved = std::move(copy);
	auto movedSpan = moved.data<float>();
	if (!std::equal(movedSpan.begin(), movedSpan.end(), after.begin()))
		throw std::runtime_error("move mismatch");

	std::vector<int32_t> srcI = {1, 2, 3, 4};
	std::vector<std::byte> bytesI(srcI.size() * sizeof(int32_t));
	std::memcpy(bytesI.data(), srcI.data(), bytesI.size());
	Tensor ti = Tensor::Create<int32_t>({2, 2}, std::move(bytesI));
	ti.fill<int32_t>(7);
	auto afterI = ti.data<int32_t>();
	for (auto v : afterI)
		if (v != 7)
			throw std::runtime_error("fill mismatch");

	moved[-1][2].set<float>(101.0f);
	if (std::abs(moved.data<float>()[(static_cast<size_t>(1) * 3) + 2] - 101.0f) > 1e-6f)
		throw std::runtime_error("negative index write mismatch");

	const Tensor& ct = moved;
	auto row1 = ct[1].read<float>();
	if (row1.size() != 3)
		throw std::runtime_error("const view read size mismatch");

	float c00 = ct[0][0].readScalar<float>();
	if (std::abs(c00 - 10.0f) > 1e-6f)
		throw std::runtime_error("const chained scalar read mismatch");

	auto b = moved.bytes();
	if (b.size() != moved.data<float>().size() * sizeof(float))
		throw std::runtime_error("bytes size mismatch");

	Tensor tf = Tensor::Create<float>();
	bool gotTypeMismatch = false;
	try {
		tf = 1.0;
	} catch (const TensorException& e) {
		if (e.getErrorType() == TensorException::ErrorType::TypeMismatch)
			gotTypeMismatch = true;
	}
	if (!gotTypeMismatch)
		throw std::runtime_error("expected type mismatch on scalar assign");

	try {
		std::vector<double> sample = {1.1, 2.2, 3.3, 4.4};
		Tensor t2 = Tensor::Create<double>();
		t2[0] = sample;
		auto spanD = t2.data<double>();
		auto readSample = t2.getData<double>();
		if (readSample != sample)
			throw std::runtime_error("fast read mismatch");
		if (t2.hasCache())
			throw std::runtime_error("fast read should not have cache");
	} catch (const TensorException& e) {
		throw std::runtime_error("fast read failed");
	}

	// getData<T>：无类型校验，按 sizeof(T) 重解释原始字节，不足整除的尾部零填充
	{
		std::vector<float> fsrc = {1.5f, -2.25f, 3.125f};
		std::vector<std::byte> fbytes(fsrc.size() * sizeof(float));
		std::memcpy(fbytes.data(), fsrc.data(), fbytes.size());

		Tensor tu8 = Tensor::Create<float>({3}, std::vector<std::byte>(fbytes));
		auto byteOut = tu8.getData<uint8_t>();
		if (byteOut.size() != fbytes.size()
			|| std::memcmp(byteOut.data(), fbytes.data(), fbytes.size()) != 0)
			throw std::runtime_error("getData<uint8_t> byte reinterpretation mismatch");
		if (tu8.hasCache())
			throw std::runtime_error("getData should consume cache");

		Tensor td = Tensor::Create<float>({3}, std::vector<std::byte>(fbytes));
		auto doubleOut = td.getData<double>();
		if (doubleOut.size() != 2) // ceil(12 / 8)
			throw std::runtime_error("getData<double> ceil element count mismatch");
		const auto* outBytes = reinterpret_cast<const std::byte*>(doubleOut.data());
		if (std::memcmp(outBytes, fbytes.data(), fbytes.size()) != 0)
			throw std::runtime_error("getData<double> leading bytes mismatch");
		for (size_t i = fbytes.size(); i < doubleOut.size() * sizeof(double); ++i)
			if (outBytes[i] != std::byte{0})
				throw std::runtime_error("getData<double> tail not zero-filled");

		Tensor tf2 = Tensor::Create<float>({3}, std::vector<std::byte>(fbytes));
		auto floatOut = tf2.getData<float>();
		if (floatOut != fsrc)
			throw std::runtime_error("getData<float> same-type mismatch");

		Tensor emptyT = Tensor::Create<float>();
		if (!emptyT.getData<int32_t>().empty())
			throw std::runtime_error("getData on empty tensor should return empty vector");
	}

	// 视图分叉：同一命名视图多次派生应各自拥有含前缀的独立路径
	{
		Tensor tf = Tensor::Create<float>({3, 4});
		auto row = tf[0];
		auto a = row[1];
		auto b = row[2];
		if (a._shape != Tensor::Shape{0, 1})
			throw std::runtime_error("view fork: a path corrupted");
		if (b._shape != Tensor::Shape{0, 2})
			throw std::runtime_error("view fork: b path corrupted (prefix lost)");
		a.set<float>(11.0f);
		b.set<float>(22.0f);
		auto row0 = tf.data<float>();
		if (std::abs(row0[1] - 11.0f) > 1e-6f || std::abs(row0[2] - 22.0f) > 1e-6f)
			throw std::runtime_error("view fork: writes landed at wrong offsets");

		auto row1 = tf[1];
		for (int i = 0; i < 4; ++i) {
			row1[i].set<float>(static_cast<float>(100 + i));
		}
		auto row1span = tf.data<float>();
		for (int i = 0; i < 4; ++i) {
			if (std::abs(row1span[4 + i] - static_cast<float>(100 + i)) > 1e-6f)
				throw std::runtime_error("view fork in loop: wrong element written");
		}

		const Tensor& ctf = tf;
		auto crow = ctf[0];
		auto ca = crow[1];
		auto cb = crow[2];
		if (ca._shape != Tensor::Shape{0, 1} || cb._shape != Tensor::Shape{0, 2})
			throw std::runtime_error("const view fork: path corrupted");
		if (std::abs(ca.readScalar<float>() - 11.0f) > 1e-6f ||
			std::abs(cb.readScalar<float>() - 22.0f) > 1e-6f)
			throw std::runtime_error("const view fork: reads returned wrong values");
	}

	std::cout << "Tensor tests passed" << std::endl;
}

int main() {
	try {
		runTensorTests();
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Test failure: " << e.what() << std::endl;
		return -1;
	}
}
