// 张量 JSON codec：DCNet v1 线上格式（DESIGN.md §4 内置适配器）
//   {"dtype":"float32","shape":[1,1,28,28],"data":"<base64>"}

#include "DCNet/DcNetHttp.h"
#include "DCNet/NetError.h"
#include "DCNet/NetTransport.h"
#include "Tensor.hpp"

#include "NetBase64.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace DC::Net {

namespace {

std::string dtypeToString(Tensor::TensorType type, size_t typeSize) {
	switch (type) {
	case Tensor::TensorType::Float:
		return typeSize == 2 ? "float16" : (typeSize == 8 ? "float64" : "float32");
	case Tensor::TensorType::Int:
		return typeSize == 1 ? "int8" : (typeSize == 2 ? "int16" : (typeSize == 8 ? "int64" : "int32"));
	case Tensor::TensorType::Uint:
		return typeSize == 1 ? "uint8" : (typeSize == 2 ? "uint16" : (typeSize == 8 ? "uint64" : "uint32"));
	case Tensor::TensorType::Bool:
		return "bool";
	default:
		return "unknown";
	}
}

bool dtypeFromString(const std::string& s, Tensor::TensorType& type, size_t& typeSize) {
	if (s == "float32") { type = Tensor::TensorType::Float; typeSize = 4; return true; }
	if (s == "float64") { type = Tensor::TensorType::Float; typeSize = 8; return true; }
	if (s == "float16") { type = Tensor::TensorType::Float; typeSize = 2; return true; }
	if (s == "int8")    { type = Tensor::TensorType::Int;   typeSize = 1; return true; }
	if (s == "int16")   { type = Tensor::TensorType::Int;   typeSize = 2; return true; }
	if (s == "int32")   { type = Tensor::TensorType::Int;   typeSize = 4; return true; }
	if (s == "int64")   { type = Tensor::TensorType::Int;   typeSize = 8; return true; }
	if (s == "uint8")   { type = Tensor::TensorType::Uint;  typeSize = 1; return true; }
	if (s == "uint16")  { type = Tensor::TensorType::Uint;  typeSize = 2; return true; }
	if (s == "uint32")  { type = Tensor::TensorType::Uint;  typeSize = 4; return true; }
	if (s == "uint64")  { type = Tensor::TensorType::Uint;  typeSize = 8; return true; }
	if (s == "bool")    { type = Tensor::TensorType::Bool;  typeSize = 1; return true; }
	return false;
}

} // namespace

class TensorJsonCodec : public DcNetCodec {
public:
	Node::Schema schema() const override {
		Node::Schema s;
		s.inputs = {NodePort::in<float>("data")};
		s.outputs = {NodePort::out<float>("result")};
		return s;
	}

	std::string requestPath() const override { return "/infer"; }

	Payload encodeRequest(const Node::RunContext& ctx) override {
		const auto& v = ctx.peek("data");
		const auto* t = v.as<Tensor>();
		if (!t)
			return {};
		nlohmann::json j;
		j["dtype"] = dtypeToString(t->type(), t->typeSize());
		j["shape"] = t->shape();
		auto bytes = t->bytes();
		j["data"] = detail::base64Encode(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
		return j.dump();
	}

	void decodeResponse(Payload& payload, Node::RunContext& ctx) override {
		const auto j = nlohmann::json::parse(payload);
		Tensor::TensorType type = Tensor::TensorType::Void;
		size_t typeSize = 0;
		const std::string dtype = j.value("dtype", "unknown");
		if (!dtypeFromString(dtype, type, typeSize))
			throw std::runtime_error("tensor codec: unknown dtype '" + dtype + "'");
		Tensor::Shape shape = j.value("shape", Tensor::Shape{});
		const std::string b64 = j.value("data", "");
		const std::string bytes = detail::base64Decode(b64);
		Tensor::DataBlock block(bytes.size());
		if (!bytes.empty())
			std::memcpy(block.data(), bytes.data(), bytes.size());
		ctx.output("result",
				   Value(std::make_unique<Tensor>(type, typeSize, std::move(shape), std::move(block))));
	}
};

std::shared_ptr<DcNetCodec> makeTensorJsonCodec() {
	return std::make_shared<TensorJsonCodec>();
}

} // namespace DC::Net
