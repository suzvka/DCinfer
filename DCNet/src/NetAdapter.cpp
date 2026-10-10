#include "DCNet/NetAdapter.h"

#include "Node.h"

#include <chrono>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <utility>

namespace DC::Net {

namespace {

/// 标准 RunFn：encode → send → recv → decode；失败经 NetError 归一化出口上报。
///
/// send/recv 传输级失败按 maxRetries 退避重试（100ms × 已试次数），以 send+recv
/// 为原子单元重放（幂等性由服务语义保证）；encode / decode 异常不重试。
Node::RunFn makeDefaultRunFn(std::shared_ptr<DcNetCodec> codec) {
	return [codec = std::move(codec)](Node::RunContext& ctx) -> Node::Result {
		auto* transport = static_cast<DcNetTransport*>(ctx.engine());
		if (!transport)
			return ctx.failure(Node::Status::ExecutionFailed, "DCNet: no transport instance");
		if (!codec)
			return ctx.failure(Node::Status::InternalError, "DCNet: no codec configured");

		Payload request;
		try {
			request = codec->encodeRequest(ctx);
		} catch (const DcCodecInputError& e) {
			return ctx.failure(Node::Status::InvalidInput,
							   std::string("DCNet: invalid request input: ") + e.what());
		} catch (const std::exception& e) {
			return ctx.failure(Node::Status::InternalError,
							   std::string("DCNet: encode failed: ") + e.what());
		}

		const int maxRetries = transport->endpoint().maxRetries;
		NetError err;
		Payload response;
		for (int attempt = 0;; ++attempt) {
			err = transport->send(request);
			if (err.ok())
				err = transport->recv(response);
			if (err.ok() || attempt >= maxRetries)
				break;
			std::this_thread::sleep_for(std::chrono::milliseconds(100 * (attempt + 1)));
		}
		if (!err.ok())
			return ctx.failure(err.localStatus, err.localMessage, err.diagnostic);

		try {
			codec->decodeResponse(response, ctx);
		} catch (const DcCodecRemoteError& e) {
			const std::string msg = std::string("DCNet: malformed remote response: ") + e.what();
			return ctx.failure(Node::Status::ExecutionFailed, msg,
							   DC::Diagnostic{"dcnet", static_cast<int>(NetErrorCategory::RemoteMalformed), msg});
		} catch (const std::exception& e) {
			return ctx.failure(Node::Status::InternalError,
							   std::string("DCNet: decode failed: ") + e.what());
		}

		return ctx.success();
	};
}

} // namespace

void registerDcNetAdapter(EngineRegistry& reg, DcNetAdapterDesc desc) {
	EngineDescriptor ed;
	ed.engineType = desc.engineType;

	const std::string engineType = desc.engineType;
	const Node::Schema localSchema = desc.schema;
	const std::shared_ptr<DcNetCodec> codec = desc.codec;
	const Node::RunFn runFn = desc.runFn ? desc.runFn : makeDefaultRunFn(std::move(desc.codec));
	const std::function<std::shared_ptr<DcNetTransport>()> transportFactory = std::move(desc.transportFactory);

	// loadModel 钩子在 registry 锁外执行（single-flight），可反向调用 registry；
	// 同 key 并发只执行一次。覆盖项按值捕获：注册级配置固化。
	const std::string epAuthToken = desc.authToken;
	const std::vector<std::string> epHeaders = desc.headers;
	const auto epConnectTimeout = desc.connectTimeout;
	const auto epRequestTimeout = desc.requestTimeout;
	const int epMaxRetries = desc.maxRetries;
	const bool allowInsecureCredentials = desc.allowInsecureCredentials;
	ed.loadModel = [engineType, transportFactory, codec, epAuthToken, epHeaders,
					   epConnectTimeout, epRequestTimeout, epMaxRetries, allowInsecureCredentials](const EngineCore& /*core*/, const std::string& modelPath) -> EngineInstance {
		if (!transportFactory)
			throw NodeException(NodeException::ErrorType::InternalError, "DCNet",
								"engine '" + engineType + "' has no transportFactory");
		auto transport = transportFactory();
		if (!transport)
			throw NodeException(NodeException::ErrorType::InternalError, "DCNet",
								"engine '" + engineType + "' transportFactory returned null");
		NetEndpoint ep = NetEndpoint::parse(modelPath);
		ep.allowInsecureCredentials = allowInsecureCredentials;
		if (codec)
			ep.requestPath = codec->requestPath();
		if (!epAuthToken.empty())
			ep.authToken = epAuthToken;
		if (!epHeaders.empty())
			ep.headers = epHeaders;
		if (epConnectTimeout.count() > 0)
			ep.connectTimeout = epConnectTimeout;
		if (epRequestTimeout.count() > 0)
			ep.requestTimeout = epRequestTimeout;
		if (epMaxRetries > 0)
			ep.maxRetries = epMaxRetries;
		NetError err = transport->connect(ep);
		if (!err.ok())
			throw NodeException(NodeException::ErrorType::ExecutionFailed, "DCNet",
								"connect failed: " + err.localMessage);
		return EngineInstance(std::move(transport));
	};

	ed.getInputPorts = [localSchema](const EngineInstance&) { return localSchema.inputs; };
	ed.getOutputPorts = [localSchema](const EngineInstance&) { return localSchema.outputs; };

	ed.factory = [engineType, runFn](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		auto node = std::make_unique<Node>(engineType, p.nodeName, p.schema, runFn,
										   ResourceClass::System);
		if (p.engineInstance)
			node->bindEngine(p.engineInstance, p.engineInstance->descriptor());
		return node;
	};

	// 执行相位协议：HTTP 同步返回，逻辑内联 RunFn，执行相位全部留空。
	reg.registerEngine(ed);
}

} // namespace DC::Net
