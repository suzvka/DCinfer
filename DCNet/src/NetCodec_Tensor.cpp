// 张量 JSON codec：v1 线上格式（数值 base64 / 文本 UTF-8 直传）。

#include "DCNet/NetCodec_Tensor.h"
#include "DCNet/NetError.h"
#include "DCNet/NetTransport.h"
#include "Tensor.hpp"

#include "NetBase64.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace DC::Net {

namespace {

// Defensive limits for untrusted remote tensor frames.  These are wire-level
// limits and deliberately do not alter the public Tensor API.
constexpr std::size_t kMaxTensorRank = 64;
constexpr std::size_t kMaxTensorBytes = std::size_t{1} << 30; // 1 GiB

// dtype 字符串 ↔ Tensor 类型映射
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
	if (!j.is_object())
		throw std::runtime_error("tensor codec: frame must be an object");
	Tensor::TensorType type = Tensor::TensorType::Void;
	size_t typeSize = 0;
	if (!j.contains("dtype") || !j["dtype"].is_string())
		throw std::runtime_error("tensor codec: dtype must be a string");
	const std::string dtype = j["dtype"].get<std::string>();
	if (!dtypeFromString(dtype, type, typeSize))
		throw std::runtime_error("tensor codec: unknown dtype '" + dtype + "'");
	if (!j.contains("shape") || !j["shape"].is_array() || j["shape"].size() > kMaxTensorRank)
		throw std::runtime_error("tensor codec: invalid or oversized shape rank");
	Tensor::Shape shape;
	shape.reserve(j["shape"].size());
	std::size_t elements = 1;
	for (const auto& dim : j["shape"]) {
		if (!dim.is_number_integer())
			throw std::runtime_error("tensor codec: shape dimensions must be integers");
		const auto d = dim.get<std::int64_t>();
		// 负数拒绝（无符号转换回绕）；零维允许——0 元素张量为合法空载荷表示
		if (d < 0)
			throw std::runtime_error("tensor codec: shape dimensions must be non-negative");
		const auto ud = static_cast<std::size_t>(d);
		if (ud == 0) {
			// 零维置零后乘法恒 0 无溢出，同时避开下方对 ud 的除法（整数除零）
			elements = 0;
			shape.push_back(d);
			continue;
		}
		if (elements > std::numeric_limits<std::size_t>::max() / ud)
			throw std::runtime_error("tensor codec: shape element count overflows");
		elements *= ud;
		shape.push_back(d);
	}
	if (elements > kMaxTensorBytes / typeSize)
		throw std::runtime_error("tensor codec: tensor exceeds maximum size");
	if (!j.contains("data"))
		throw std::runtime_error("tensor codec: missing data");
	std::string payload;
	if (type == Tensor::TensorType::Data) {
		if (!j["data"].is_string())
			throw std::runtime_error("tensor codec: text data must be a string");
		payload = j["data"].get<std::string>();
	} else {
		if (!j["data"].is_string())
			throw std::runtime_error("tensor codec: numeric data must be base64 string");
		payload = detail::base64Decode(j["data"].get<std::string>());
	}
	const std::size_t expected = elements * typeSize;
	if (payload.size() != expected)
		throw std::runtime_error("tensor codec: data byte size does not match shape and dtype");
	Tensor::DataBlock block(payload.size());
	if (!payload.empty())
		std::memcpy(block.data(), payload.data(), payload.size());
	return Tensor(type, typeSize, std::move(shape), std::move(block));
}

/// 远端响应帧解码：解析 / dtype / 字段异常统一转抛 DcCodecRemoteError
/// （映射为 RemoteMalformed 诊断）；本地端口写入不在此处，避免形状异常误分类。
Tensor decodeRemoteTensorFrame(const Payload& payload, const char* context) {
	try {
		return decodeTensor(nlohmann::json::parse(payload));
	} catch (const DcCodecRemoteError&) {
		throw; // 契约类型，不二次包装
	} catch (const std::exception& e) {
		throw DcCodecRemoteError(std::string(context) + e.what());
	}
}

} // namespace

/// 数值张量端口
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
		Tensor t = decodeRemoteTensorFrame(payload, "tensor response is not a valid frame: ");
		ctx.output("result", Value(std::make_unique<Tensor>(std::move(t))));
	}
};

/// Data 文本端口
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
		Tensor t = decodeRemoteTensorFrame(payload, "text response is not a valid frame: ");
		ctx.output("result", Value(std::make_unique<Tensor>(std::move(t))));
	}
};

std::shared_ptr<DcNetCodec> makeTensorJsonCodec() {
	return std::make_shared<TensorJsonCodec>();
}

std::shared_ptr<DcNetCodec> makeTextJsonCodec() {
	return std::make_shared<TextJsonCodec>();
}

/// 服务端镜像 codec：单张量进出，与出站 codec 共用同一 v1 报文格式。
class TensorJsonServerCodec final : public DcNetServerCodec {
public:
	TensorJsonServerCodec(std::string inputPort, std::string outputPort)
		: _inputPort(std::move(inputPort)), _outputPort(std::move(outputPort)) {}

	std::unordered_map<std::string, Tensor> decodeRequest(const Payload& request) override {
		const auto j = nlohmann::json::parse(request); // 解析失败 → 监听器 415
		return {{_inputPort, decodeTensor(j)}};        // 未知 dtype → 415
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
