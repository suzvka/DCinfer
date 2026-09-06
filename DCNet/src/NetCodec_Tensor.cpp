// 张量 JSON codec：DCNet v1 线上格式（DESIGN.md §4 内置数据格式）
//   数值：{"dtype":"float32","shape":[1,1,28,28],"data":"<base64>"}
//   文本：{"dtype":"text","shape":[N],"data":"<utf-8>"}

#include "DCNet/NetCodec_Tensor.h"
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
#include <unordered_map>
#include <utility>

namespace DC::Net {

namespace {

// dtype 字符串 ↔ Tensor 类型映射（数值 base64；Data 文本 UTF-8 直传）
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
	case Tensor::TensorType::Data:
		return "text";
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
	if (s == "text")    { type = Tensor::TensorType::Data;  typeSize = 1; return true; }
	return false;
}

/// Tensor → JSON（Data 文本 UTF-8 直传；数值 base64）
nlohmann::json encodeTensor(const Tensor& t) {
	nlohmann::json j;
	j["dtype"] = dtypeToString(t.type(), t.typeSize());
	j["shape"] = t.shape();
	auto bytes = t.bytes();
	if (t.type() == Tensor::TensorType::Data) {
		j["data"] = std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
	} else {
		j["data"] = detail::base64Encode(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
	}
	return j;
}

/// JSON → Tensor（支持 "text" 与全部数值 dtype）
Tensor decodeTensor(const nlohmann::json& j) {
	Tensor::TensorType type = Tensor::TensorType::Void;
	size_t typeSize = 0;
	const std::string dtype = j.value("dtype", "unknown");
	if (!dtypeFromString(dtype, type, typeSize))
		throw std::runtime_error("tensor codec: unknown dtype '" + dtype + "'");
	Tensor::Shape shape = j.value("shape", Tensor::Shape{});
	if (type == Tensor::TensorType::Data) {
		const std::string text = j.value("data", "");
		Tensor::DataBlock block(text.size());
		if (!text.empty())
			std::memcpy(block.data(), text.data(), text.size());
		return Tensor(type, typeSize, std::move(shape), std::move(block));
	}
	const std::string b64 = j.value("data", "");
	const std::string bytes = detail::base64Decode(b64);
	Tensor::DataBlock block(bytes.size());
	if (!bytes.empty())
		std::memcpy(block.data(), bytes.data(), bytes.size());
	return Tensor(type, typeSize, std::move(shape), std::move(block));
}

} // namespace

/// 数值张量端口（in "data" Float → out "result" Float）
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
		return encodeTensor(*t).dump();
	}

	void decodeResponse(Payload& payload, Node::RunContext& ctx) override {
		const auto j = nlohmann::json::parse(payload);
		ctx.output("result", Value(std::make_unique<Tensor>(decodeTensor(j))));
	}
};

/// Data 文本端口（in "text" Data → out "result" Data）
class TextJsonCodec : public DcNetCodec {
public:
	Node::Schema schema() const override {
		Node::Schema s;
		s.inputs = {NodePort::in<std::vector<char>>("text")};
		s.outputs = {NodePort::out<std::vector<char>>("result")};
		return s;
	}

	std::string requestPath() const override { return "/infer"; }

	Payload encodeRequest(const Node::RunContext& ctx) override {
		const auto& v = ctx.peek("text");
		const auto* t = v.as<Tensor>();
		if (!t)
			return {};
		return encodeTensor(*t).dump();
	}

	void decodeResponse(Payload& payload, Node::RunContext& ctx) override {
		const auto j = nlohmann::json::parse(payload);
		ctx.output("result", Value(std::make_unique<Tensor>(decodeTensor(j))));
	}
};

std::shared_ptr<DcNetCodec> makeTensorJsonCodec() {
	return std::make_shared<TensorJsonCodec>();
}

std::shared_ptr<DcNetCodec> makeTextJsonCodec() {
	return std::make_shared<TextJsonCodec>();
}

// ── 服务端镜像 codec（M-server；DESIGN.md §3.6）──

/// 单张量进出的服务端 codec：与出站 tensor/text codec 共用同一 v1 报文格式。
class TensorJsonServerCodec final : public DcNetServerCodec {
public:
	TensorJsonServerCodec(std::string inputPort, std::string outputPort)
		: _inputPort(std::move(inputPort)), _outputPort(std::move(outputPort)) {}

	std::unordered_map<std::string, Tensor> decodeRequest(const Payload& request) override {
		const auto j = nlohmann::json::parse(request); // 解析失败 → 异常 → 监听器 415
		return {{_inputPort, decodeTensor(j)}};        // 未知 dtype → 异常 → 415
	}

	Payload encodeResponse(const std::unordered_map<std::string, Tensor>& outputs) override {
		const auto it = outputs.find(_outputPort);
		if (it == outputs.end())
			throw std::runtime_error("tensor server codec: output port '" + _outputPort + "' missing");
		return encodeTensor(it->second).dump();
	}

	std::string requestPath() const override { return "/infer"; }

private:
	std::string _inputPort;
	std::string _outputPort;
};

std::shared_ptr<DcNetServerCodec> makeTensorJsonServerCodec(std::string inputPort, std::string outputPort) {
	return std::make_shared<TensorJsonServerCodec>(std::move(inputPort), std::move(outputPort));
}

} // namespace DC::Net
