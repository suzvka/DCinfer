// OpenAI 兼容远端引擎适配器：POST {basePath}/chat/completions
// 端口：prompt/system/params（Data）→ response（Data）
//
// 基于 DCNet 传输框架（HttpTransport + NetCodec 契约 + NetError 归一化），
// 自 NetCodec_Chat.cpp 迁移（2026-08，DCNet 收缩为张量传输框架后
// 协议级适配器归 DCEngines，与 OnnxRuntime 对称并列）。

#include "DCEngine/OpenAiEngine.h"

#include "DCNet/NetAdapter.h"
#include "DCNet/NetTransport_Http.h"
#include "Node.h"
#include "Tensor.hpp"

#include <nlohmann/json.hpp>

#include <cstring>
#include <cctype>
#include <memory>
#include <string>
#include <utility>

namespace DC::OpenAI {

namespace {

std::string textOf(const Tensor& t) {
	auto bytes = t.bytes();
	return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

Tensor makeText(const std::string& s) {
	Tensor::DataBlock block(s.size());
	if (!s.empty())
		std::memcpy(block.data(), s.data(), s.size());
	return Tensor(Tensor::TensorType::Data, 1, {static_cast<int64_t>(s.size())}, std::move(block));
}

/// 可选 Data 端口（手工构造：NodePort::optional<T> 会把默认值写入 Tensor，
/// 要求 T 平凡可拷贝，vector<char> 不满足；此处用空 Tensor 作为默认值）。
NodePort optionalDataPort(const std::string& name) {
	NodePort p;
	p.name = name;
	p.type = Tensor::TensorType::Data;
	p.typeSize = 1;
	p.required = false;
	p.defaultValue = Tensor(Tensor::TensorType::Data, 1); // 空文本默认
	return p;
}

/// OpenAI 兼容 chat codec（DCNet NetCodec 契约实现，双向翻译器）。
class ChatCodec : public DC::Net::DcNetCodec {
public:
	explicit ChatCodec(std::string model) : _model(std::move(model)) {}

	Node::Schema schema() const override {
		Node::Schema s;
		s.inputs = {
			NodePort::in<std::vector<char>>("prompt"),
			optionalDataPort("system"),
			optionalDataPort("params"),
		};
		s.outputs = {NodePort::out<std::vector<char>>("response")};
		return s;
	}

	std::string requestPath() const override { return "/chat/completions"; }

	DC::Net::Payload encodeRequest(const Node::RunContext& ctx) override {
		nlohmann::json j;
		j["model"] = _model;

		nlohmann::json messages = nlohmann::json::array();
		const auto& sysVal = ctx.peek("system");
		if (const auto* t = sysVal.as<Tensor>(); t) {
			const std::string sys = textOf(*t);
			if (!sys.empty())
				messages.push_back({{"role", "system"}, {"content", sys}});
		}
		const auto& promptVal = ctx.peek("prompt");
		const auto* pt = promptVal.as<Tensor>();
		if (!pt)
			throw std::runtime_error("chat codec: 'prompt' not a Tensor");
		messages.push_back({{"role", "user"}, {"content", textOf(*pt)}});
		j["messages"] = messages;
		j["stream"] = false;

		// params 端口（可选）：请求级采样参数 JSON，逐请求覆盖。
		// 非法 JSON / 非对象 → DcCodecInputError（标准 RunFn 映射为 InvalidInput），
		// 与"未提供 params"（空 Tensor，静默跳过）严格区分。
		const auto& paramsVal = ctx.peek("params");
		if (const auto* t = paramsVal.as<Tensor>(); t) {
			const std::string text = textOf(*t);
			if (!text.empty()) {
				nlohmann::json overrides;
				try {
					overrides = nlohmann::json::parse(text);
				} catch (const std::exception& e) {
					throw DC::Net::DcCodecInputError(std::string("params is not valid JSON: ") + e.what());
				}
				if (!overrides.is_object())
					throw DC::Net::DcCodecInputError("params must be a JSON object");
				for (auto it = overrides.begin(); it != overrides.end(); ++it)
					j[it.key()] = it.value();
			}
		}
		return j.dump();
	}

	void decodeResponse(DC::Net::Payload& payload, Node::RunContext& ctx) override {
		// 结构异常 → DcCodecRemoteError（标准 RunFn 映射为 RemoteMalformed）；
		// content 为合法空字符串（""）时正常成功返回，与字段缺失严格区分。
		nlohmann::json j;
		try {
			j = nlohmann::json::parse(payload);
		} catch (const std::exception& e) {
			throw DC::Net::DcCodecRemoteError(std::string("response is not valid JSON: ") + e.what());
		}
		if (!(j.contains("choices") && j["choices"].is_array() && !j["choices"].empty() &&
			  j["choices"][0].contains("message") && j["choices"][0]["message"].contains("content") &&
			  j["choices"][0]["message"]["content"].is_string()))
			throw DC::Net::DcCodecRemoteError(
				"response missing choices[0].message.content (string); protocol drift or non-chat endpoint?");
		const auto content = j["choices"][0]["message"]["content"].get<std::string>();
		ctx.output("response", Value(std::make_unique<Tensor>(makeText(content))));
	}

private:
	std::string _model;
};

} // namespace

void registerOpenAiEngine(EngineRegistry& reg, const OpenAiOptions& opts) {
	auto codec = std::make_shared<ChatCodec>(opts.model);
	DC::Net::DcNetAdapterDesc desc;
	desc.engineType = opts.engineType;
	desc.schema = codec->schema(); // 本地静态形状规则（不依赖远端）
	desc.codec = std::move(codec);
	desc.transportFactory = [] { return std::make_shared<DC::Net::HttpTransport>(); };

	// 鉴权：tokenProvider 优先（注册时求值一次）；裸 key 自动补 Bearer 前缀。
	// OpenAI 兼容协议的标准鉴权形如 "Authorization: Bearer <key>"；
	// 需自定义鉴权头时用 opts.headers 显式给出。
	std::string token = opts.authToken;
	if (opts.tokenProvider) {
		auto provided = opts.tokenProvider();
		if (!provided.empty())
			token = std::move(provided);
	}
	if (!token.empty()) {
		const std::string lowerPrefix = [&] {
			std::string p = token.substr(0, 7);
			for (auto& c : p)
				c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
			return p;
		}();
		desc.authToken = (lowerPrefix == "bearer ") ? token : "Bearer " + token;
	}
	desc.headers = opts.headers;
	desc.connectTimeout = opts.connectTimeout;
	desc.requestTimeout = opts.requestTimeout;
	desc.maxRetries = opts.maxRetries;
	DC::Net::registerDcNetAdapter(reg, std::move(desc));
}

} // namespace DC::OpenAI
