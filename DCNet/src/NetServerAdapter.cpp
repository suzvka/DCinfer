// 变体 A 装配：把本地 DCinfer 节点暴露为可被出站 send→recv 驱动的监听服务
// （M-server / DESIGN.md §3.6）。执行走与本地完全相同的节点管线
// （setInput → tryExecute → collectOutputs），图级语义与本地执行无差别。

#include "DCNet/NetServerAdapter.h"

#include "DCNet/NetError.h"
#include "Node.h"
#include "NodeException.h"
#include "NetWire.h"

#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace DC::Net {

namespace {

/// @brief 输入边界本地形状规则校验（镜像出站 decodeResponse「校验本地形状规则
/// 后 ctx.output」的职责，DESIGN.md §3.2/§3.4）：端口名 / 类型 / 形状（-1 动态维）
/// 不符 → 返回 false 并填 reason。wire 上按提案 §5 schema 违例行应答
/// 400 → 对端 RemoteRejected → InvalidInput。
bool validateAgainstSchema(const Node::Schema& schema, const std::unordered_map<std::string, Tensor>& inputs,
						   std::string& reason) {
	for (const auto& [name, tensor] : inputs) {
		const Node::Port* port = schema.findInput(name);
		if (!port) {
			reason = "unknown input port '" + name + "'";
			return false;
		}
		if (port->type != Tensor::TensorType::Void && tensor.type() != port->type) {
			reason = "port '" + name + "' expects a different tensor type";
			return false;
		}
		if (!port->shape.empty()) {
			const auto& shape = tensor.shape();
			for (std::size_t i = 0; i < port->shape.size(); ++i) {
				if (port->shape[i] >= 0 && (i >= shape.size() || shape[i] != port->shape[i])) {
					reason = "port '" + name + "' shape violates local shape rule";
					return false;
				}
			}
		}
	}
	return true;
}

class ServerService final : public DcNetServerService {
public:
	ServerService(EngineRegistry& reg, DcNetServerAdapterDesc desc)
		: _reg(reg), _engineType(std::move(desc.engineType)), _localModelRef(std::move(desc.localModelRef)),
		  _codec(std::move(desc.codec)), _endpoint(desc.endpoint) {}

	void launch() {
		_listener = makeHttpListener();
		_listener->bind(_endpoint); // 配置期出口（[C1]）：失败抛 NodeException
		_listener->start([this](const std::string& path, const std::string& body) {
			return execute(path, body);
		});
	}

	void stop() override {
		if (_listener)
			_listener->stop();
	}

	bool alive() const override {
		return _listener && _listener->alive();
	}

	int port() const override {
		return _listener ? _listener->port() : -1;
	}

private:
	WireResponse execute(const std::string& /*path*/, const std::string& body) {
		try {
			// ① decodeRequest：报文 → 本地输入端口张量
			//    解析失败 = wire 级垃圾报文（提案 §5 拆行）→ 415，对端归一化
			//    RemoteMalformed → InternalError，与 schema 违例区分
			std::unordered_map<std::string, Tensor> inputs;
			try {
				inputs = _codec->decodeRequest(body);
			} catch (const std::exception& e) {
				return {415, detail::wireErrorBody("malformed_frame",
												   std::string("request payload not parseable: ") + e.what())};
			}

			// ② 每请求一个节点实例（[C2] 裁决：实例级隔离）；引擎实例由
			//    Registry 缓存复用，本地执行互斥串行（引擎单任务语义）
			const std::string taskId = "dcnet-server-" + std::to_string(_taskSeq.fetch_add(1));
			std::unique_ptr<Node> node;
			try {
				node = _reg.createNode(_engineType, taskId, _localModelRef);
			} catch (const std::exception& e) {
				return {500, detail::wireErrorBody("server_error", std::string("createNode failed: ") + e.what())};
			}
			if (!node)
				return {500, detail::wireErrorBody("server_error", "engine '" + _engineType + "' unavailable")};

			// ③ 输入边界 schema 校验（提案 §5 schema 违例行）→ 400 → InvalidInput
			std::string reason;
			if (!validateAgainstSchema(node->schema(), inputs, reason))
				return {400, detail::wireErrorBody("invalid_input", reason)};

			Node::Result result;
			std::unordered_map<std::string, Tensor> outputs;
			{
				std::lock_guard lk(_execMutex);
				try {
					node->setInput(taskId, std::move(inputs));
				} catch (const NodeException& e) {
					// 端口名不在 schema（PortNotFound）→ schema 违例 → 400
					return {400, detail::wireErrorBody("invalid_input", e.what())};
				}
				if (!node->isReady(taskId))
					return {400, detail::wireErrorBody("missing_input", "required input port(s) missing")};
				try {
					result = node->tryExecute(taskId);
				} catch (const NodeException& e) {
					// 深层形状/类型校验（ValidatorRegistry abort 漏网到 drain 层）
					// 同属输入违例 → 400；其余按本地执行失败语义 → 500
					switch (e.getErrorType()) {
					case NodeException::ErrorType::NotReady:
						return {400, detail::wireErrorBody("missing_input", e.what())};
					case NodeException::ErrorType::PortNotFound:
					case NodeException::ErrorType::TypeMismatch:
						return {400, detail::wireErrorBody("invalid_input", e.what())};
					default:
						return {500, detail::wireErrorBody("server_error", e.what())};
					}
				} catch (const std::exception& e) {
					return {500, detail::wireErrorBody("server_error", std::string("execute failed: ") + e.what())};
				}
				if (result.ok())
					outputs = node->collectOutputTensors(taskId);
				node->clearTask(taskId);
			}

			// ④ 本地执行失败 → wire 逆向映射（DESIGN.md §6.1；验收标准 1）
			if (!result.ok())
				return {wireHttpStatusFor(result.status),
						detail::wireErrorBody(wireCodeFor(result.status), result.message)};

			// ⑤ encodeResponse：输出张量 → 报文（编码失败按本地失败语义应答 5xx）
			try {
				return {200, _codec->encodeResponse(outputs)};
			} catch (const std::exception& e) {
				return {500, detail::wireErrorBody("server_error", std::string("encode response failed: ") + e.what())};
			}
		} catch (const std::exception& e) {
			return {500, detail::wireErrorBody("server_error", e.what())};
		} catch (...) {
			return {500, detail::wireErrorBody("server_error", "unknown server error")};
		}
	}

	EngineRegistry& _reg;
	std::string _engineType;
	std::string _localModelRef;
	std::shared_ptr<DcNetServerCodec> _codec;
	NetServerEndpoint _endpoint;
	std::unique_ptr<DcNetListener> _listener;
	std::mutex _execMutex; // 本地执行串行（引擎实例跨请求共享）
	std::atomic<std::uint64_t> _taskSeq{0};
};

} // namespace

std::shared_ptr<DcNetServerService> registerDcNetServerAdapter(EngineRegistry& reg, DcNetServerAdapterDesc desc) {
	// 配置期校验（[C1]）：失败抛 NodeException，不产生 wire
	if (!desc.codec)
		throw NodeException(NodeException::ErrorType::InternalError, "registerDcNetServerAdapter",
							"engine '" + desc.engineType + "' has no server codec");
	// 预创建引擎实例：engineType 未注册 / createEngine 失败在此暴露（配置期报错）
	if (!reg.getOrCreateEngine(desc.engineType, desc.localModelRef))
		throw NodeException(NodeException::ErrorType::ExecutionFailed, "registerDcNetServerAdapter",
							"engine '" + desc.engineType + ":" + desc.localModelRef +
								"' not registered or creation failed");
	// requestPath 由 codec 注入（镜像出站 createEngine 的端点装配）
	desc.endpoint.requestPath = desc.codec->requestPath();

	auto svc = std::make_shared<ServerService>(reg, std::move(desc));
	svc->launch();
	return svc;
}

} // namespace DC::Net
