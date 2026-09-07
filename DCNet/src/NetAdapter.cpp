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
/// 编排细节：
/// - encode 抛 DcCodecInputError → InvalidInput（调用方输入问题，不重试）
/// - decode 抛 DcCodecRemoteError → RemoteMalformed（远端结构异常，不重试）
/// - send/recv 传输级失败按 ep.maxRetries 退避重试（100ms × 已试次数）；
///   重试以 send+recv 整体为原子单元重放，幂等性由服务语义保证
Node::RunFn makeDefaultRunFn(std::shared_ptr<DcNetCodec> codec) {
	return [codec = std::move(codec)](Node::RunContext& ctx) -> Node::Result {
		auto* transport = static_cast<DcNetTransport*>(ctx.engine());
		if (!transport)
			return ctx.failure(Node::Status::ExecutionFailed, "DCNet: no transport instance");
		if (!codec)
			return ctx.failure(Node::Status::InternalError, "DCNet: no codec configured");

		// 1. 本地端口 → 对方请求报文（一次性；输入错误不重试）
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

		// 2+3. 发送 → 接收（传输级失败按 maxRetries 退避重试）
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
			return ctx.failure(err.localStatus, err.localMessage);

		// 4. 对方响应报文 → 本地端口（校验本地形状规则；结构异常不重试）
		try {
			codec->decodeResponse(response, ctx);
		} catch (const DcCodecRemoteError& e) {
			return ctx.failure(Node::Status::RemoteMalformed,
							   std::string("DCNet: malformed remote response: ") + e.what());
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

	// ── createEngine：modelPath 即远端端点 → 解析 + 覆盖 → 创建 transport → 连接 ──
	// 注意：getOrCreateEngine 在锁内调用本钩子，钩子内不得反向调用 registry。
	// 覆盖项捕获为值：注册级配置固化（authToken 不参与日志/错误信息）。
	const std::string epAuthToken = desc.authToken;
	const std::vector<std::string> epHeaders = desc.headers;
	const auto epConnectTimeout = desc.connectTimeout;
	const auto epRequestTimeout = desc.requestTimeout;
	const int epMaxRetries = desc.maxRetries;
	ed.createEngine = [engineType, transportFactory, codec, epAuthToken, epHeaders,
					   epConnectTimeout, epRequestTimeout, epMaxRetries](const std::string& modelPath) -> EngineInstance {
		if (!transportFactory)
			throw NodeException(NodeException::ErrorType::InternalError, "DCNet",
								"engine '" + engineType + "' has no transportFactory");
		auto transport = transportFactory();
		if (!transport)
			throw NodeException(NodeException::ErrorType::InternalError, "DCNet",
								"engine '" + engineType + "' transportFactory returned null");
		NetEndpoint ep = NetEndpoint::parse(modelPath);
		if (codec)
			ep.requestPath = codec->requestPath(); // 协议子路径（如 /infer）注入端点
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
								"connect " + ep.endpoint() + " failed: " + err.localMessage);
		return EngineInstance(std::move(transport));
	};

	// ── 端口推导：返回本地静态形状规则（不依赖远端）──
	ed.getInputPorts = [localSchema](const EngineInstance&) { return localSchema.inputs; };
	ed.getOutputPorts = [localSchema](const EngineInstance&) { return localSchema.outputs; };

	// ── factory：构造节点（System affinity）并绑定引擎实例 ──
	ed.factory = [engineType, runFn](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		auto* engineInstance = const_cast<EngineInstance*>(static_cast<const EngineInstance*>(p.engineConfig));
		auto node = std::make_unique<Node>(engineType, p.nodeName, p.schema, runFn,
										   ThreadPoolAffinity::System);
		if (engineInstance)
			node->bindEngine(engineInstance, engineInstance->descriptor());
		return node;
	};

	// ── 运行时钩子：v1 全 no-op（HTTP 同步返回；重连策略见 M3 / DESIGN.md §5.2）──
	ed.synchronize = nullptr;
	ed.preRun = nullptr;
	ed.postRun = nullptr;
	ed.releaseEngine = nullptr;
	ed.onError = nullptr;

	reg.registerEngine(ed);
}

} // namespace DC::Net
