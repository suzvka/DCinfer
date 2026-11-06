// BuiltinTensorOps - 内置 CPU 张量算子（ONNX-13 语义）
//
// 覆盖 Shape / Gather / Unsqueeze / Slice / Sub / Pad。除 Sub 需逐元素算术外，
// 其余五个只搬运字节，故以 typeSize 为单位做 memcpy，天然支持任意元素类型。
// 参数（axis/axes/starts/ends/steps/pads/constant_value）一律走 ONNX-13 的
// “参数即张量输入”风格；端口声明为 Void，令框架入口放行任意 dtype/shape
// （见 TaskBuffer::drainInputsTo：仅当 port.type != Void 才做类型校验）。
#include "DCEngine/BuiltinOps.h"
#include "Tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <vector>

namespace DC::Builtin {
namespace {

using Shape = std::vector<int64_t>;
using Bytes = Tensor::DataBlock; // std::vector<std::byte>

// ── Void 端口工厂：type=Void 让框架跳过 type/typeSize/shape 强校验 ──
Node::Port voidIn(std::string name, bool required = true) {
	Node::Port p;
	p.name = std::move(name);
	p.type = Tensor::TensorType::Void;
	p.typeSize = 0;
	p.required = required;
	return p;
}

Node::Port voidOut(std::string name) {
	Node::Port p;
	p.name = std::move(name);
	p.type = Tensor::TensorType::Void;
	p.typeSize = 0;
	p.required = true;
	return p;
}

// ── 形状/索引工具 ──
size_t elemCount(const Shape& dims) {
	size_t n = 1;
	for (auto d : dims)
		n *= static_cast<size_t>(d < 0 ? 0 : d);
	return n;
}

// 行主序元素 stride
std::vector<size_t> rowStrides(const Shape& dims) {
	std::vector<size_t> s(dims.size(), 1);
	for (size_t i = dims.size(); i-- > 0;) {
		if (i + 1 < dims.size())
			s[i] = s[i + 1] * static_cast<size_t>(dims[i + 1]);
	}
	return s;
}

void linToCoord(size_t lin, const Shape& dims, const std::vector<size_t>& st, std::vector<size_t>& coord) {
	for (size_t i = 0; i < dims.size(); ++i) {
		size_t d = static_cast<size_t>(dims[i] < 0 ? 0 : dims[i]);
		coord[i] = d ? ((lin / st[i]) % d) : 0;
	}
}

size_t coordToLin(const std::vector<size_t>& coord, const std::vector<size_t>& st) {
	size_t s = 0;
	for (size_t i = 0; i < coord.size(); ++i)
		s += coord[i] * st[i];
	return s;
}

// 广播形状推断（numpy 右对齐）；不兼容返回 nullopt
std::optional<Shape> broadcastShape(const Shape& a, const Shape& b) {
	size_t ra = a.size(), rb = b.size();
	size_t R = std::max(ra, rb);
	Shape out(R);
	for (size_t i = 0; i < R; ++i) {
		int64_t da = (i + ra >= R) ? a[i + ra - R] : 1;
		int64_t db = (i + rb >= R) ? b[i + rb - R] : 1;
		if (da == db)
			out[i] = da;
		else if (da == 1)
			out[i] = db;
		else if (db == 1)
			out[i] = da;
		else
			return std::nullopt;
	}
	return out;
}

// 操作数在广播输出坐标下的 per-out-dim 元素 stride（广播/缺失维为 0）
std::vector<size_t> bcastStrides(const Shape& op, const Shape& out) {
	size_t R = out.size(), ro = op.size();
	auto od = rowStrides(op);
	std::vector<size_t> stride(R, 0);
	for (size_t i = 0; i < R; ++i) {
		if (i + ro < R) // 前导缺失维 → 广播
			continue;
		size_t j = i + ro - R;
		if (op[j] > 1)
			stride[i] = od[j];
	}
	return stride;
}

// 从整数张量原始字节按 typeSize 读出 int64 向量（支持 1/2/4/8 字节，含负值）
std::vector<int64_t> readIntVec(const Tensor& t) {
	std::vector<int64_t> out;
	const size_t es = t.typeSize();
	auto b = t.bytes();
	if (es == 0 || b.empty())
		return out;
	size_t n = b.size() / es;
	out.reserve(n);
	for (size_t i = 0; i < n; ++i) {
		const std::byte* p = b.data() + i * es;
		int64_t v = 0;
		switch (es) {
		case 1: { int8_t x; std::memcpy(&x, p, 1); v = x; } break;
		case 2: { int16_t x; std::memcpy(&x, p, 2); v = x; } break;
		case 4: { int32_t x; std::memcpy(&x, p, 4); v = x; } break;
		case 8: { int64_t x; std::memcpy(&x, p, 8); v = x; } break;
		default: std::memcpy(&v, p, std::min<size_t>(es, 8)); break;
		}
		out.push_back(v);
	}
	return out;
}

// 便捷取输入张量；未喂入的可选端口 peek 会抛异常，此处归一为 nullptr（缺省端口）。
// 必选端口在执行前已由框架保证存在，异常不会误吞其真实错误。
const Tensor* input(const Node::RunContext& ctx, const char* name) {
	try {
		return ctx.peek(name).as<Tensor>();
	} catch (const std::exception&) {
		return nullptr;
	}
}

} // namespace

// ── Shape：输出输入张量的形状，1-D int64 ──
static Node::Schema shapeSchema() {
	Node::Schema s;
	s.inputs = {voidIn("data")};
	s.outputs = {voidOut("shape")};
	return s;
}

static Node::RunFn shapeRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const Tensor* d = input(ctx, "data");
		if (!d || !d->valid())
			return ctx.failure(Node::Status::InvalidInput, "Shape: data must be a valid DC::Tensor");

		Shape dims = d->shape();
		const size_t rank = dims.size();
		Bytes out(rank * sizeof(int64_t));
		for (size_t i = 0; i < rank; ++i) {
			int64_t v = dims[i];
			std::memcpy(out.data() + i * sizeof(int64_t), &v, sizeof(int64_t));
		}
		auto t = std::make_unique<Tensor>(Tensor::TensorType::Int, sizeof(int64_t), Shape{static_cast<int64_t>(rank)},
										   std::move(out));
		ctx.output("shape", Value(std::move(t)));
		return ctx.success();
	};
}

// ── Sub：逐元素 a-b，支持广播；本版覆盖 Float(4/8) 与 Int(1/2/4/8) ──
static Node::Schema subSchema() {
	Node::Schema s;
	s.inputs = {voidIn("a"), voidIn("b")};
	s.outputs = {voidOut("diff")};
	return s;
}

static Node::RunFn subRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const Tensor* a = input(ctx, "a");
		const Tensor* b = input(ctx, "b");
		if (!a || !b || !a->valid() || !b->valid())
			return ctx.failure(Node::Status::InvalidInput, "Sub: inputs must be valid DC::Tensor");
		if (a->type() != b->type() || a->typeSize() != b->typeSize())
			return ctx.failure(Node::Status::InvalidInput, "Sub: a and b must share element type and width");

		auto outShape = broadcastShape(a->shape(), b->shape());
		if (!outShape)
			return ctx.failure(Node::Status::InvalidInput, "Sub: operand shapes are not broadcastable");

		const size_t es = a->typeSize();
		const auto type = a->type();
		const size_t n = elemCount(*outShape);
		const size_t R = outShape->size();

		auto ab = a->bytes();
		auto bb = b->bytes();
		auto sa = bcastStrides(a->shape(), *outShape);
		auto sb = bcastStrides(b->shape(), *outShape);
		auto stOut = rowStrides(*outShape);

		Bytes out(n * es);
		std::vector<size_t> coord(R, 0);

		auto elemwise = [&](auto tag) {
			using T = decltype(tag);
			for (size_t lin = 0; lin < n; ++lin) {
				linToCoord(lin, *outShape, stOut, coord);
				size_t ai = coordToLin(coord, sa);
				size_t bi = coordToLin(coord, sb);
				T x, y;
				std::memcpy(&x, ab.data() + ai * sizeof(T), sizeof(T));
				std::memcpy(&y, bb.data() + bi * sizeof(T), sizeof(T));
				T r = static_cast<T>(x - y);
				std::memcpy(out.data() + lin * sizeof(T), &r, sizeof(T));
			}
		};

		if (type == Tensor::TensorType::Float) {
			if (es == sizeof(float))
				elemwise(float{});
			else if (es == sizeof(double))
				elemwise(double{});
			else
				return ctx.failure(Node::Status::InvalidInput, "Sub: unsupported float width");
		} else if (type == Tensor::TensorType::Int) {
			if (es == 1)
				elemwise(int8_t{});
			else if (es == 2)
				elemwise(int16_t{});
			else if (es == 4)
				elemwise(int32_t{});
			else if (es == 8)
				elemwise(int64_t{});
			else
				return ctx.failure(Node::Status::InvalidInput, "Sub: unsupported int width");
		} else {
			return ctx.failure(Node::Status::InvalidInput, "Sub: only Float and Int element types are supported");
		}

		auto t = std::make_unique<Tensor>(type, es, *outShape, std::move(out));
		ctx.output("diff", Value(std::move(t)));
		return ctx.success();
	};
}

// ── Unsqueeze：按 axes 插入 size-1 维度；数据不变，仅重排形状 ──
static Node::Schema unsqueezeSchema() {
	Node::Schema s;
	s.inputs = {voidIn("data"), voidIn("axes")};
	s.outputs = {voidOut("expanded")};
	return s;
}

static Node::RunFn unsqueezeRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const Tensor* d = input(ctx, "data");
		const Tensor* ax = input(ctx, "axes");
		if (!d || !d->valid())
			return ctx.failure(Node::Status::InvalidInput, "Unsqueeze: data must be a valid DC::Tensor");
		if (!ax || !ax->valid())
			return ctx.failure(Node::Status::InvalidInput, "Unsqueeze: axes input is required (opset 13)");

		Shape in = d->shape();
		auto axes = readIntVec(*ax);
		const size_t inRank = in.size();
		const size_t outRank = inRank + axes.size();

		std::vector<size_t> norm;
		norm.reserve(axes.size());
		for (auto v : axes) {
			int64_t p = v < 0 ? v + static_cast<int64_t>(outRank) : v;
			if (p < 0 || p >= static_cast<int64_t>(outRank))
				return ctx.failure(Node::Status::InvalidInput, "Unsqueeze: axis out of range");
			if (std::find(norm.begin(), norm.end(), static_cast<size_t>(p)) != norm.end())
				return ctx.failure(Node::Status::InvalidInput, "Unsqueeze: duplicate axis");
			norm.push_back(static_cast<size_t>(p));
		}
		std::sort(norm.begin(), norm.end());

		std::vector<bool> isNew(outRank, false);
		for (auto p : norm)
			isNew[p] = true;

		Shape outDims(outRank, 1);
		size_t inPos = 0;
		for (size_t i = 0; i < outRank; ++i)
			if (!isNew[i])
				outDims[i] = in[inPos++];

		auto src = d->bytes();
		Bytes out(src.begin(), src.end());
		auto t = std::make_unique<Tensor>(d->type(), d->typeSize(), outDims, std::move(out));
		ctx.output("expanded", Value(std::move(t)));
		return ctx.success();
	};
}

// ── Slice：starts/ends(必选) + axes/steps(可选)，含负索引与钳位 ──
static Node::Schema sliceSchema() {
	Node::Schema s;
	s.inputs = {voidIn("data"), voidIn("starts"), voidIn("ends"), voidIn("axes", false), voidIn("steps", false)};
	s.outputs = {voidOut("sliced")};
	return s;
}

static Node::RunFn sliceRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const Tensor* d = input(ctx, "data");
		const Tensor* stT = input(ctx, "starts");
		const Tensor* enT = input(ctx, "ends");
		if (!d || !d->valid() || !stT || !enT)
			return ctx.failure(Node::Status::InvalidInput, "Slice: data/starts/ends are required");

		Shape dims = d->shape();
		const size_t R = dims.size();
		auto starts = readIntVec(*stT);
		auto ends = readIntVec(*enT);
		const size_t L = starts.size();
		if (ends.size() != L)
			return ctx.failure(Node::Status::InvalidInput, "Slice: starts/ends length mismatch");

		std::vector<int64_t> axes;
		const Tensor* axT = input(ctx, "axes");
		if (axT && axT->valid()) {
			axes = readIntVec(*axT);
			if (axes.size() != L)
				return ctx.failure(Node::Status::InvalidInput, "Slice: axes length mismatch");
		} else {
			for (size_t i = 0; i < L; ++i)
				axes.push_back(static_cast<int64_t>(i));
		}

		std::vector<int64_t> steps(L, 1);
		const Tensor* spT = input(ctx, "steps");
		if (spT && spT->valid()) {
			auto sp = readIntVec(*spT);
			if (sp.size() != L)
				return ctx.failure(Node::Status::InvalidInput, "Slice: steps length mismatch");
			steps = sp;
		}

		auto clamp = [](int64_t v, int64_t lo, int64_t hi) {
			return v < lo ? lo : (v > hi ? hi : v);
		};

		// 各输入维默认整段保留
		std::vector<int64_t> start(R, 0), stop(R), step(R, 1);
		for (size_t i = 0; i < R; ++i)
			stop[i] = dims[i];

		for (size_t k = 0; k < L; ++k) {
			int64_t axis = axes[k];
			if (axis < 0)
				axis += static_cast<int64_t>(R);
			if (axis < 0 || axis >= static_cast<int64_t>(R))
				return ctx.failure(Node::Status::InvalidInput, "Slice: axis out of range");
			int64_t sv = steps[k];
			if (sv == 0)
				return ctx.failure(Node::Status::InvalidInput, "Slice: step must not be zero");
			int64_t dim = dims[axis];
			int64_t s = starts[k], e = ends[k];
			if (sv > 0) {
				if (s < 0) s += dim;
				if (e < 0) e += dim;
				start[axis] = clamp(s, 0, dim);
				stop[axis] = clamp(e, 0, dim);
			} else {
				if (s < 0) s += dim;
				if (e < 0) e += dim;
				// ONNX: reverse starts clamp to [0, dim-1], ends to [-1, dim-1].
				// Empty axes have no valid starting element and must stay empty.
				start[axis] = dim == 0 ? -1 : clamp(s, 0, dim - 1);
				stop[axis] = clamp(e, -1, dim - 1);
			}
			step[axis] = sv;
		}

		auto computeLen = [](int64_t s, int64_t e, int64_t st) -> int64_t {
			// Divide before rounding up; never add the step to the distance or
			// negate INT64_MIN in signed arithmetic.
			if (st > 0)
				return e <= s ? 0 : 1 + (e - s - 1) / st;
			if (s <= e)
				return 0;
			const uint64_t magnitude = uint64_t{0} - static_cast<uint64_t>(st);
			return static_cast<int64_t>(1 + static_cast<uint64_t>(s - e - 1) / magnitude);
		};

		Shape outDims(R);
		for (size_t i = 0; i < R; ++i)
			outDims[i] = computeLen(start[i], stop[i], step[i]);

		const size_t es = d->typeSize();
		const size_t n = elemCount(outDims);
		auto src = d->bytes();
		auto stOut = rowStrides(outDims);
		auto stIn = rowStrides(dims);
		std::vector<size_t> cOut(R, 0), cIn(R, 0);

		Bytes out(n * es);
		for (size_t lin = 0; lin < n; ++lin) {
			linToCoord(lin, outDims, stOut, cOut);
			for (size_t i = 0; i < R; ++i)
				cIn[i] = static_cast<size_t>(start[i] + static_cast<int64_t>(cOut[i]) * step[i]);
			size_t inLin = coordToLin(cIn, stIn);
			std::memcpy(out.data() + lin * es, src.data() + inLin * es, es);
		}

		auto t = std::make_unique<Tensor>(d->type(), es, outDims, std::move(out));
		ctx.output("sliced", Value(std::move(t)));
		return ctx.success();
	};
}

// ── Gather：沿 axis 用 indices 取值；axis 可选默认 0，支持负索引 ──
static Node::Schema gatherSchema() {
	Node::Schema s;
	s.inputs = {voidIn("data"), voidIn("indices"), voidIn("axis", false)};
	s.outputs = {voidOut("gathered")};
	return s;
}

static Node::RunFn gatherRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const Tensor* d = input(ctx, "data");
		const Tensor* ix = input(ctx, "indices");
		if (!d || !d->valid() || !ix || !ix->valid())
			return ctx.failure(Node::Status::InvalidInput, "Gather: data and indices are required");

		Shape dims = d->shape();
		const size_t R = dims.size();
		if (R == 0)
			return ctx.failure(Node::Status::InvalidInput, "Gather: data must have rank >= 1");

		int64_t axis = 0;
		const Tensor* axT = input(ctx, "axis");
		if (axT && axT->valid()) {
			auto av = readIntVec(*axT);
			if (!av.empty())
				axis = av[0];
		}
		if (axis < 0)
			axis += static_cast<int64_t>(R);
		if (axis < 0 || axis >= static_cast<int64_t>(R))
			return ctx.failure(Node::Status::InvalidInput, "Gather: axis out of range");

		auto indices = readIntVec(*ix);
		Shape idxShape = ix->shape();
		const size_t axisDim = static_cast<size_t>(dims[axis]);

		size_t outer = 1;
		for (size_t i = 0; i < static_cast<size_t>(axis); ++i)
			outer *= static_cast<size_t>(dims[i]);
		size_t inner = 1;
		for (size_t i = static_cast<size_t>(axis) + 1; i < R; ++i)
			inner *= static_cast<size_t>(dims[i]);
		const size_t M = indices.size();

		Shape outDims;
		for (size_t i = 0; i < static_cast<size_t>(axis); ++i)
			outDims.push_back(dims[i]);
		for (auto v : idxShape)
			outDims.push_back(v);
		for (size_t i = static_cast<size_t>(axis) + 1; i < R; ++i)
			outDims.push_back(dims[i]);

		const size_t es = d->typeSize();
		auto src = d->bytes();
		Bytes out(outer * M * inner * es);

		for (size_t o = 0; o < outer; ++o) {
			for (size_t m = 0; m < M; ++m) {
				int64_t idx = indices[m];
				if (idx < 0)
					idx += static_cast<int64_t>(axisDim);
				if (idx < 0 || idx >= static_cast<int64_t>(axisDim)) {
					std::string msg = "Gather: index " + std::to_string(indices[m]) + " out of range for axis size " +
									  std::to_string(axisDim);
					return ctx.failure(Node::Status::InvalidInput, msg);
				}
				const std::byte* srcRow = src.data() + (o * axisDim + static_cast<size_t>(idx)) * inner * es;
				std::byte* dstRow = out.data() + (o * M + m) * inner * es;
				std::memcpy(dstRow, srcRow, inner * es);
			}
		}

		auto t = std::make_unique<Tensor>(d->type(), es, outDims, std::move(out));
		ctx.output("gathered", Value(std::move(t)));
		return ctx.success();
	};
}

// ── Pad：constant 模式；pads 长 2*rank，constant_value 可选（默认 0）──
static Node::Schema padSchema() {
	Node::Schema s;
	s.inputs = {voidIn("data"), voidIn("pads"), voidIn("constant_value", false)};
	s.outputs = {voidOut("padded")};
	return s;
}

static Node::RunFn padRunFn() {
	return [](Node::RunContext& ctx) -> Node::Result {
		const Tensor* d = input(ctx, "data");
		const Tensor* pT = input(ctx, "pads");
		if (!d || !d->valid() || !pT || !pT->valid())
			return ctx.failure(Node::Status::InvalidInput, "Pad: data and pads are required");

		Shape dims = d->shape();
		const size_t R = dims.size();
		auto pads = readIntVec(*pT);
		if (pads.size() != 2 * R)
			return ctx.failure(Node::Status::InvalidInput, "Pad: pads must have length 2 * rank");

		const size_t es = d->typeSize();
		if (es == 0)
			return ctx.failure(Node::Status::InvalidInput, "Pad: element width is zero");

		for (size_t i = 0; i < 2 * R; ++i)
			if (pads[i] < 0)
				return ctx.failure(Node::Status::InvalidInput, "Pad: negative pads (crop) are not supported");

		Shape outDims(R);
		for (size_t i = 0; i < R; ++i)
			outDims[i] = dims[i] + pads[i] + pads[R + i];

		// 填充值必须是同 dtype、同宽度的有效 0-D 标量，不能仅按宽度重解释字节。
		Bytes pattern(es, std::byte(0));
		const Tensor* cv = input(ctx, "constant_value");
		if (cv) {
			if (!cv->valid() || cv->type() != d->type() || cv->typeSize() != es || !cv->shape().empty())
				return ctx.failure(Node::Status::InvalidInput,
					"Pad: constant_value must be a valid scalar with the same type and width as data");
			auto cb = cv->bytes();
			if (cb.size() != es)
				return ctx.failure(Node::Status::InvalidInput, "Pad: constant_value must contain exactly one element");
			std::memcpy(pattern.data(), cb.data(), es);
		}

		const size_t n = elemCount(outDims);
		Bytes out(n * es);
		for (size_t off = 0; off < out.size(); off += es)
			std::memcpy(out.data() + off, pattern.data(), es);

		auto src = d->bytes();
		auto stOut = rowStrides(outDims);
		auto stIn = rowStrides(dims);
		const size_t inCount = elemCount(dims);
		std::vector<size_t> cIn(R, 0), cOut(R, 0);
		for (size_t lin = 0; lin < inCount; ++lin) {
			linToCoord(lin, dims, stIn, cIn);
			for (size_t i = 0; i < R; ++i)
				cOut[i] = cIn[i] + static_cast<size_t>(pads[i]);
			size_t outLin = coordToLin(cOut, stOut);
			std::memcpy(out.data() + outLin * es, src.data() + lin * es, es);
		}

		auto t = std::make_unique<Tensor>(d->type(), es, outDims, std::move(out));
		ctx.output("padded", Value(std::move(t)));
		return ctx.success();
	};
}

void registerBuiltinTensorOps(EngineRegistry& reg) {
	reg.registerOperator("Shape", shapeSchema(), shapeRunFn());
	reg.registerOperator("Sub", subSchema(), subRunFn());
	reg.registerOperator("Unsqueeze", unsqueezeSchema(), unsqueezeRunFn());
	reg.registerOperator("Slice", sliceSchema(), sliceRunFn());
	reg.registerOperator("Gather", gatherSchema(), gatherRunFn());
	reg.registerOperator("Pad", padSchema(), padRunFn());
}

} // namespace DC::Builtin
