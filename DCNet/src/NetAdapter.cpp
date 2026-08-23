#include "DCNet/NetAdapter.h"

#include "Node.h"

#include <exception>
#include <memory>
#include <string>
#include <utility>

namespace DC::Net {

namespace {

/// 标准 RunFn：encode → send → recv → decode；失败经 NetError 归一化出口上报。
Node::RunFn makeDefaultRunFn(std::shared_ptr<DcNetCodec> codec) {
	return [codec = std::move(codec)](Node::RunContext& ctx) -> Node::Result {
		auto* transport = static_cast<DcNetTransport*>(ctx.engine());
		if (!transport)
			return ctx.failure(Node::Status::ExecutionFailed, "DCNet: no transport instance");
		if (!codec)
			return ctx.failure(Node::Status::InternalError, "DCNet: no codec configured");

		// 1. 本地端口 → 对方请求报文
		Payload request;
		try {
			request = codec->encodeRequest(ctx);
		} catch (const std::exception& e) {
			return ctx.failure(Node::Status::InternalError,
							   std::string("DCNet: encode failed: ") + e.what());
		}

		// 2. 发送
		NetError err = transport->send(request);
		if (!err.ok())
			return ctx.failure(err.localStatus, err.localMessage);

		// 3. 接收（transport 已完成归一化；失败时 remoteDetail 保留原始报文）
		Payload response;
		err = transport->recv(response);
		if (!err.ok())
			return ctx.failure(err.localStatus, err.localMessage);

		// 4. 对方响应报文 → 本地端口（校验本地形状规则）
		try {
			codec->decodeResponse(response, ctx);
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

	// ── createEngine：modelPath 即远端端点 → 创建 transport → 连接 ──
	// 注意：getOrCreateEngine 在锁内调用本钩子，钩子内不得反向调用 registry。
	ed.createEngine = [engineType, transportFactory, codec](const std::string& modelPath) -> EngineInstance {
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
