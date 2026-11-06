// BuiltinTensorOps 集成测试：单节点图运行 6 个张量算子，对拍 ONNX-13 参考值。
#include "DCEngine/BuiltinOps.h"
#include "EngineRegistry.h"
#include "InferGraph.h"
#include "TaskStatus.h"
#include "Tensor.hpp"
#include "Value.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace DC;

static int failures = 0;

#define CHECK(cond, msg)                                                                                               \
	do {                                                                                                               \
		if (!(cond)) {                                                                                                 \
			std::cerr << "FAIL: " << msg << std::endl;                                                                 \
			++failures;                                                                                                \
			return;                                                                                                    \
		}                                                                                                              \
	} while (0)

#define TEST(name) \
	do { \
		const int failuresBefore = failures; \
		std::cout << "Test: " << name << " ... " << std::flush; \
		[&]()
#define END_TEST() \
		(); \
		std::cout << (failures == failuresBefore ? "PASSED" : "FAILED") << std::endl; \
	} while (0)

// ── 构造输入 Value（张量按字节装入，任意 dtype/shape）──
static std::vector<std::byte> toBytes(const void* p, size_t bytes) {
	if (bytes == 0)
		return {};
	const auto* b = static_cast<const std::byte*>(p);
	return std::vector<std::byte>(b, b + bytes);
}

static Value vFloat(const Tensor::Shape& dims, const std::vector<float>& v) {
	auto t = Tensor::Create<float>(dims, toBytes(v.data(), v.size() * sizeof(float)));
	return Value(std::make_unique<Tensor>(std::move(t)));
}

static Value vInt32(const Tensor::Shape& dims, const std::vector<int32_t>& v) {
	auto t = Tensor::Create<int32_t>(dims, toBytes(v.data(), v.size() * sizeof(int32_t)));
	return Value(std::make_unique<Tensor>(std::move(t)));
}

static Value vInt64(const Tensor::Shape& dims, const std::vector<int64_t>& v) {
	auto t = Tensor::Create<int64_t>(dims, toBytes(v.data(), v.size() * sizeof(int64_t)));
	return Value(std::make_unique<Tensor>(std::move(t)));
}

// ── 运行单算子节点并取回输出张量；参数以 (端口名, 值) 交替变参传入 ──
using Inputs = std::vector<std::pair<std::string, Value>>;

static void collect(Inputs&) {}
template <typename... Rest>
static void collect(Inputs& v, const char* name, Value val, Rest&&... rest) {
	v.emplace_back(std::string(name), std::move(val));
	collect(v, std::forward<Rest>(rest)...);
}

template <typename... Args>
static Tensor runOp(const std::string& op, const std::string& outPort, Args&&... args) {
	Inputs in;
	collect(in, std::forward<Args>(args)...);

	InferGraph g;
	auto node = EngineRegistry::instance().createOperator(op, op + "_node");
	if (!node)
		throw std::runtime_error("createOperator failed for " + op);
	const std::string name = node->name();
	g.addNode(std::move(node));
	for (auto& kv : in)
		g.feedInput("t1", name, kv.first, std::move(kv.second));
	g.submit("t1", name, outPort, 1);
	const auto res = g.waitForResult("t1", std::chrono::milliseconds(5000));
	if (res.status != TaskStatus::Succeeded) {
		std::string why = res.errors.empty() ? std::string("(no diagnostic)") : res.errors.front().message;
		throw std::runtime_error(op + " task did not succeed: " + why);
	}
	if (!g.hasOutput("t1", name, outPort))
		throw std::runtime_error(op + " produced no output");
	return g.takeOutputTensor("t1", name, outPort);
}

static std::vector<float> readF(const Tensor& t) {
	auto s = t.data<float>();
	return std::vector<float>(s.begin(), s.end());
}
static std::vector<int32_t> readI32(const Tensor& t) {
	auto s = t.data<int32_t>();
	return std::vector<int32_t>(s.begin(), s.end());
}
static std::vector<int64_t> readI64(const Tensor& t) {
	auto s = t.data<int64_t>();
	return std::vector<int64_t>(s.begin(), s.end());
}
static bool near(const std::vector<float>& a, const std::vector<float>& b) {
	if (a.size() != b.size())
		return false;
	for (size_t i = 0; i < a.size(); ++i)
		if (std::abs(a[i] - b[i]) > 1e-5f)
			return false;
	return true;
}

// ── Shape ──
static void testShape() {
	TEST("Shape: {2,3} -> [2,3]") {
		auto out = runOp("Shape", "shape", "data", vFloat({2, 3}, {1, 2, 3, 4, 5, 6}));
		CHECK(out.shape() == Tensor::Shape({2}), "shape output is 1-D of length 2 (input rank)");
		auto v = readI64(out);
		CHECK((v == std::vector<int64_t>{2, 3}), "Shape values [2,3]");
	}
	END_TEST();
	TEST("Shape: {1,4,2} -> [1,4,2]") {
		auto out = runOp("Shape", "shape", "data", vInt32({1, 4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}));
		auto v = readI64(out);
		CHECK((v == std::vector<int64_t>{1, 4, 2}), "Shape values [1,4,2]");
	}
	END_TEST();
}

// ── Sub ──
static void testSub() {
	TEST("Sub: same-shape float") {
		auto out = runOp("Sub", "diff", "a", vFloat({3}, {1, 2, 3}), "b", vFloat({3}, {0, 1, 1}));
		CHECK(near(readF(out), {1, 1, 2}), "1-0,2-1,3-1");
	}
	END_TEST();
	TEST("Sub: broadcast {2,3} - {3}") {
		auto out = runOp("Sub", "diff", "a", vFloat({2, 3}, {1, 2, 3, 4, 5, 6}), "b", vFloat({3}, {1, 2, 3}));
		CHECK(out.shape() == Tensor::Shape({2, 3}), "broadcast shape {2,3}");
		CHECK(near(readF(out), {0, 0, 0, 3, 3, 3}), "row-wise broadcast subtract");
	}
	END_TEST();
	TEST("Sub: int32 elementwise") {
		auto out = runOp("Sub", "diff", "a", vInt32({4}, {5, 7, 9, 11}), "b", vInt32({4}, {1, 1, 1, 1}));
		CHECK(out.type() == Tensor::TensorType::Int, "int result stays Int");
		CHECK((readI32(out) == std::vector<int32_t>{4, 6, 8, 10}), "int subtract");
	}
	END_TEST();
	TEST("Sub: broadcast scalar to tensor") {
		auto out = runOp("Sub", "diff", "a", vFloat({2, 2}, {10, 20, 30, 40}), "b", vFloat({}, {5}));
		CHECK(near(readF(out), {5, 15, 25, 35}), "scalar broadcast");
	}
	END_TEST();
}

// ── Unsqueeze ──
static void testUnsqueeze() {
	TEST("Unsqueeze: axes [0,2] on {2,3} -> {1,2,1,3}") {
		auto out = runOp("Unsqueeze", "expanded", "data", vFloat({2, 3}, {1, 2, 3, 4, 5, 6}),
						 "axes", vInt64({2}, {0, 2}));
		CHECK(out.shape() == Tensor::Shape({1, 2, 1, 3}), "rank 4 with two size-1 dims");
		CHECK(near(readF(out), {1, 2, 3, 4, 5, 6}), "data unchanged");
	}
	END_TEST();
	TEST("Unsqueeze: negative axis [-1] on {2,3} -> {2,3,1}") {
		auto out = runOp("Unsqueeze", "expanded", "data", vFloat({2, 3}, {1, 2, 3, 4, 5, 6}),
						 "axes", vInt64({1}, {-1}));
		CHECK(out.shape() == Tensor::Shape({2, 3, 1}), "append trailing dim");
	}
	END_TEST();
}

// ── Slice ──
static void testSlice() {
	TEST("Slice: 2-D starts/ends/axes") {
		auto out = runOp("Slice", "sliced", "data", vFloat({3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
						 "starts", vInt64({2}, {0, 1}), "ends", vInt64({2}, {2, 3}), "axes", vInt64({2}, {0, 1}));
		CHECK(out.shape() == Tensor::Shape({2, 2}), "slice shape {2,2}");
		CHECK(near(readF(out), {2, 3, 6, 7}), "rows0-1 cols1-2");
	}
	END_TEST();
	TEST("Slice: steps=2 on {5}") {
		auto out = runOp("Slice", "sliced", "data", vFloat({5}, {1, 2, 3, 4, 5}), "starts", vInt64({1}, {0}),
						 "ends", vInt64({1}, {5}), "steps", vInt64({1}, {2}));
		CHECK(near(readF(out), {1, 3, 5}), "step 2 -> every other");
	}
	END_TEST();
	TEST("Slice: negative indices on {5}") {
		auto out = runOp("Slice", "sliced", "data", vFloat({5}, {1, 2, 3, 4, 5}), "starts", vInt64({1}, {-3}),
						 "ends", vInt64({1}, {-1}));
		CHECK(near(readF(out), {3, 4}), "[-3,-1) -> [3,4]");
	}
	END_TEST();
}

static void testSliceBoundaries() {
	TEST("Slice: reverse start clamps to zero") {
		auto out = runOp("Slice", "sliced", "data", vFloat({5}, {1, 2, 3, 4, 5}),
			"starts", vInt64({1}, {-10}), "ends", vInt64({1}, {-6}), "steps", vInt64({1}, {-1}));
		CHECK(out.shape() == Tensor::Shape({1}) && near(readF(out), {1}), "reverse clamped first element");
	} END_TEST();
	TEST("Slice: extreme steps and reverse sentinels") {
		const auto hi = std::numeric_limits<int64_t>::max();
		const auto lo = std::numeric_limits<int64_t>::min();
		for (auto step : {hi, -hi, lo}) {
			auto out = runOp("Slice", "sliced", "data", vFloat({5}, {1, 2, 3, 4, 5}),
				"starts", vInt64({1}, {step > 0 ? 0 : hi}),
				"ends", vInt64({1}, {step > 0 ? hi : lo}), "steps", vInt64({1}, {step}));
			CHECK(out.shape() == Tensor::Shape({1}), "extreme step keeps exactly one element");
			CHECK(near(readF(out), {step > 0 ? 1.f : 5.f}), "extreme step value");
		}
		auto reversed = runOp("Slice", "sliced", "data", vFloat({5}, {1, 2, 3, 4, 5}),
			"starts", vInt64({1}, {hi}), "ends", vInt64({1}, {lo}), "steps", vInt64({1}, {-1}));
		CHECK(near(readF(reversed), {5, 4, 3, 2, 1}), "reverse full axis with sentinels");
	} END_TEST();
	TEST("Slice: empty results retain shape and can be consumed") {
		auto out = runOp("Slice", "sliced", "data", vFloat({5}, {1, 2, 3, 4, 5}),
			"starts", vInt64({1}, {3}), "ends", vInt64({1}, {2}));
		CHECK(out.valid() && out.empty() && out.shape() == Tensor::Shape({0}), "valid empty 1-D result");
		auto shape = runOp("Shape", "shape", "data", Value(std::make_unique<Tensor>(out)));
		CHECK(readI64(shape) == std::vector<int64_t>({0}), "downstream Shape sees empty axis");
		auto multi = runOp("Slice", "sliced", "data", vFloat({2, 3}, {1, 2, 3, 4, 5, 6}),
			"starts", vInt64({1}, {2}), "ends", vInt64({1}, {1}), "axes", vInt64({1}, {1}));
		CHECK(multi.valid() && multi.shape() == Tensor::Shape({2, 0}), "retain nonempty dimensions");
		auto again = runOp("Slice", "sliced", "data", Value(std::make_unique<Tensor>(out)),
			"starts", vInt64({1}, {0}), "ends", vInt64({1}, {-1}), "steps", vInt64({1}, {-1}));
		CHECK(again.valid() && again.shape() == Tensor::Shape({0}), "reverse empty axis stays empty");
	} END_TEST();
	TEST("Shape: scalar produces valid empty shape vector") {
		auto out = runOp("Shape", "shape", "data", vFloat({}, {7}));
		CHECK(out.valid() && out.shape() == Tensor::Shape({0}) && readI64(out).empty(), "rank-zero shape vector");
	} END_TEST();
}

// ── Gather ──
static void testGather() {
	TEST("Gather: axis 0 default") {
		auto out = runOp("Gather", "gathered", "data", vFloat({3, 2}, {1, 2, 3, 4, 5, 6}),
						 "indices", vInt64({2}, {2, 0}));
		CHECK(out.shape() == Tensor::Shape({2, 2}), "gather shape {2,2}");
		CHECK(near(readF(out), {5, 6, 1, 2}), "rows [2,0]");
	}
	END_TEST();
	TEST("Gather: axis 1") {
		auto out = runOp("Gather", "gathered", "data", vFloat({2, 3}, {1, 2, 3, 4, 5, 6}), "indices",
						 vInt64({2}, {0, 2}), "axis", vInt64({}, {1}));
		CHECK(out.shape() == Tensor::Shape({2, 2}), "gather cols shape {2,2}");
		CHECK(near(readF(out), {1, 3, 4, 6}), "cols [0,2]");
	}
	END_TEST();
	TEST("Gather: negative index") {
		auto out = runOp("Gather", "gathered", "data", vFloat({3, 2}, {1, 2, 3, 4, 5, 6}),
						 "indices", vInt64({1}, {-1}));
		CHECK(near(readF(out), {5, 6}), "index -1 -> last row");
	}
	END_TEST();
}

// ── Pad ──
static void testPad() {
	TEST("Pad: constant 0 on {2,2}, pads [0,1,0,1]") {
		auto out = runOp("Pad", "padded", "data", vFloat({2, 2}, {1, 2, 3, 4}), "pads", vInt64({4}, {0, 1, 0, 1}));
		CHECK(out.shape() == Tensor::Shape({2, 4}), "padded shape {2,4}");
		CHECK(near(readF(out), {0, 1, 2, 0, 0, 3, 4, 0}), "left/right pad col, zero fill");
	}
	END_TEST();
	TEST("Pad: constant_value 9") {
		auto out = runOp("Pad", "padded", "data", vFloat({2, 2}, {1, 2, 3, 4}), "pads",
						 vInt64({4}, {0, 1, 0, 1}), "constant_value", vFloat({}, {9}));
		CHECK(near(readF(out), {9, 1, 2, 9, 9, 3, 4, 9}), "fill value 9");
	}
	END_TEST();
}

static void testPadInvalidConstants() {
	TEST("Pad: reject wrong dtype, width and non-scalar constants") {
		std::vector<Value> constants;
		constants.push_back(vInt32({}, {1}));
		constants.push_back(vInt64({}, {1}));
		constants.push_back(vFloat({1}, {1}));
		constants.push_back(vFloat({2}, {1, 2}));
		constants.push_back(vFloat({0}, {}));
		for (auto& constant : constants) {
			InferGraph g;
			auto node = EngineRegistry::instance().createOperator("Pad", "pad_invalid");
			g.addNode(std::move(node));
			g.feedInput("bad", "pad_invalid", "data", vFloat({1}, {2}));
			g.feedInput("bad", "pad_invalid", "pads", vInt64({2}, {1, 0}));
			g.feedInput("bad", "pad_invalid", "constant_value", std::move(constant));
			g.submit("bad", "pad_invalid", "padded", 1);
			auto result = g.waitForResult("bad", std::chrono::milliseconds(5000));
			CHECK(result.status == TaskStatus::Failed, "bad fill must fail instead of silently reinterpreting");
			CHECK(!result.errors.empty() && result.errors.front().message.find("constant_value") != std::string::npos,
				"failure must identify constant_value");
			CHECK(!g.hasOutput("bad", "pad_invalid", "padded"), "invalid fill produces no output");
		}
	} END_TEST();
}

int main() {
	try {
		Builtin::registerBuiltinOperators();
		testShape();
		testSub();
		testUnsqueeze();
		testSlice();
		testSliceBoundaries();
		testGather();
		testPad();
		testPadInvalidConstants();
	} catch (const std::exception& e) {
		std::cerr << "UNEXPECTED EXCEPTION: " << e.what() << std::endl;
		return 1;
	}

	if (failures != 0) {
		std::cerr << failures << " test(s) failed" << std::endl;
		return 1;
	}
	std::cout << "All Builtin tensor-op tests passed" << std::endl;
	return 0;
}
