// OpenAI 兼容远端引擎适配器：POST {basePath}/chat/completions（DCNet HttpTransport + NetCodec）。

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
#include <cmath>

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

/// 可选 Data 端口（手工构造：NodePort::optional<T> 的默认值要求 T 平凡可拷贝，
/// vector<char> 不满足；改用空 Tensor）。
NodePort optionalDataPort(const std::string& name) {
	NodePort p;
	p.name = name;
	p.type = Tensor::TensorType::Data;
	p.typeSize = 1;
	p.required = false;
	p.defaultValue = Tensor(Tensor::TensorType::Data, 1);
	return p;
}

/// OpenAI 兼容 chat codec（NetCodec 契约实现）。
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
		if (const auto* t = ctx.input<Tensor>("system"); t) {
			const std::string sys = textOf(*t);
			if (!sys.empty())
				messages.push_back({{"role", "system"}, {"content", sys}});
		}
		const auto* pt = ctx.input<Tensor>("prompt");
		if (!pt)
			throw std::runtime_error("chat codec: 'prompt' not a Tensor");
		messages.push_back({{"role", "user"}, {"content", textOf(*pt)}});
		j["messages"] = messages;
		j["stream"] = false;

		// params：请求级采样参数，逐请求覆盖；非法 JSON/非对象 → DcCodecInputError
		//（映射 InvalidInput），与未提供 params（空 Tensor 静默跳过）严格区分。
		if (const auto* t = ctx.input<Tensor>("params"); t) {
			const std::string text = textOf(*t);
			if (!text.empty()) {
				nlohmann::json overrides;
				try {
					overrides = nlohmann::json::parse(text);
				} catch (const std::exception&) {
					throw DC::Net::DcCodecInputError("params is not valid JSON");
				}
				if (!overrides.is_object())
					throw DC::Net::DcCodecInputError("params must be a JSON object");
				for (auto it = overrides.begin(); it != overrides.end(); ++it) {
					const auto& key = it.key();
					const auto& value = it.value();
					if (key == "temperature" || key == "top_p" || key == "presence_penalty" || key == "frequency_penalty") {
						if (!value.is_number())
							throw DC::Net::DcCodecInputError("sampling parameter must be numeric");
						const double number = value.get<double>();
						const double lower = (key == "presence_penalty" || key == "frequency_penalty") ? -2.0 : 0.0;
						const double upper = key == "top_p" ? 1.0 : 2.0;
						if (!std::isfinite(number) || number < lower || number > upper)
							throw DC::Net::DcCodecInputError("sampling parameter outside supported range");
					} else if (key == "max_tokens") {
						// Compare without narrowing: huge unsigned values must not wrap to signed.
						if (!value.is_number_integer() || value <= 0 || value > 2147483647)
							throw DC::Net::DcCodecInputError("max_tokens must be an integer in [1,2147483647]");
					} else {
						throw DC::Net::DcCodecInputError("unsupported params field");
					}
					j[key] = value;
				}
			}
		}
		return j.dump();
	}

	void decodeResponse(DC::Net::Payload& payload, Node::RunContext& ctx) override {
		// 结构异常 → DcCodecRemoteError（映射 ExecutionFailed）；content 为空字符串
		// 是合法成功，与字段缺失严格区分。
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
	desc.schema = codec->schema();
	desc.codec = std::move(codec);
	desc.transportFactory = [] { return std::make_shared<DC::Net::HttpTransport>(); };

	// 鉴权：tokenProvider 优先；裸 key 自动补 Bearer 前缀（自定义头用 headers）。
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
	desc.allowInsecureCredentials = opts.allowInsecureCredentials;
	desc.connectTimeout = opts.connectTimeout;
	desc.requestTimeout = opts.requestTimeout;
	desc.maxRetries = opts.maxRetries;
	DC::Net::registerDcNetAdapter(reg, std::move(desc));
}

} // namespace DC::OpenAI
