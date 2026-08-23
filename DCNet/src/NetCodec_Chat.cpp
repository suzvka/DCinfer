// OpenAI 兼容 chat codec：POST {basePath}/chat/completions
// 端口：prompt/system/params（Data）→ response（Data）

#include "DCNet/DcNetHttp.h"
#include "DCNet/NetError.h"
#include "DCNet/NetTransport.h"
#include "Tensor.hpp"

#include <nlohmann/json.hpp>

#include <cstring>
#include <memory>
#include <string>

namespace DC::Net {

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

} // namespace

class ChatCodec : public DcNetCodec {
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

	Payload encodeRequest(const Node::RunContext& ctx) override {
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

		// params 端口（可选）：请求级采样参数 JSON，逐请求覆盖
		const auto& paramsVal = ctx.peek("params");
		if (const auto* t = paramsVal.as<Tensor>(); t) {
			try {
				const auto overrides = nlohmann::json::parse(textOf(*t));
				for (auto it = overrides.begin(); it != overrides.end(); ++it)
					j[it.key()] = it.value();
			} catch (const std::exception&) {
				// 非法 params：忽略，使用默认采样参数
			}
		}
		return j.dump();
	}

	void decodeResponse(Payload& payload, Node::RunContext& ctx) override {
		const auto j = nlohmann::json::parse(payload);
		std::string content;
		if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty() &&
			j["choices"][0].contains("message") && j["choices"][0]["message"].contains("content") &&
			j["choices"][0]["message"]["content"].is_string())
			content = j["choices"][0]["message"]["content"].get<std::string>();
		ctx.output("response", Value(std::make_unique<Tensor>(makeText(content))));
	}

private:
	std::string _model;
};

std::shared_ptr<DcNetCodec> makeChatCodec(std::string model) {
	return std::make_shared<ChatCodec>(std::move(model));
}

} // namespace DC::Net
