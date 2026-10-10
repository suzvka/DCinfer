// 节点服务化装配：把本地节点暴露为可被出站 send 与 recv 驱动的监听服务。
// 执行走与本地相同的节点管线，NodeExecutor 承载 task 态，图级语义无差别。

#include "DCNet/NetServerAdapter.h"

#include "DCNet/NetError.h"
#include "Node.h"
#include "NodeExecutor.h"
#include "NodeException.h"
#include "NetWire.h"

#include <cstdio>
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

/// 输入边界形状规则校验：端口名、类型与形状，-1 为动态维；
/// 不符返回 false 并填 reason，wire 上应答 400。
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

	~ServerService() override {
		try { stop(); }
		catch (...) {
			std::fputs("DCNet fatal: service destruction from its handler is forbidden\n", stderr);
			std::terminate();
		}
	}

	void launch() {
		_listener = makeHttpListener();
		_listener->bind(_endpoint); // 配置期出口：失败抛 NodeException
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
			// decodeRequest：报文转输入端口张量；解析失败即 wire 级垃圾，回 415
			std::unordered_map<std::string, Tensor> inputs;
			try {
				inputs = _codec->decodeRequest(body);
			} catch (const std::exception& e) {
				return {415, detail::wireErrorBody("malformed_frame",
												   "request payload not parseable")};
			}

			// 每请求一个节点实例，实例级隔离；引擎实例由 Registry 缓存复用
			const std::string taskId = "dcnet-server-" + std::to_string(_taskSeq.fetch_add(1));
			std::unique_ptr<Node> node;
			try {
				node = _reg.createNode(_engineType, taskId, _localModelRef);
			} catch (const std::exception& e) {
				return {500, internalError("create_node_failed")};
			}
			if (!node)
				return {500, internalError("engine_unavailable")};

			// 输入边界 schema 校验，违例归 InvalidInput
			std::string reason;
			if (!validateAgainstSchema(node->schema(), inputs, reason))
				return {400, detail::wireErrorBody("invalid_input", reason)};

			Node::Result result;
			std::unordered_map<std::string, Tensor> outputs;
			{
				std::lock_guard lk(_execMutex);
				// task 态随执行器走，实例级隔离；节点仅作执行计划
				NodeExecutor exec(*node);
				try {
					exec.setInput(taskId, std::move(inputs));
				} catch (const NodeException& e) {
					// 端口名不在 schema 则 400
					return {400, detail::wireErrorBody("invalid_input", e.what())};
				}
				if (!exec.isReady(taskId))
					return {400, detail::wireErrorBody("missing_input", "required input port(s) missing")};
				try {
					result = exec.tryExecute(taskId);
				} catch (const NodeException& e) {
					// 深层校验如 ValidatorRegistry 漏网同属输入违例则 400；其余 500
					switch (e.getErrorType()) {
					case NodeException::ErrorType::NotReady:
						return {400, detail::wireErrorBody("missing_input", e.what())};
					case NodeException::ErrorType::PortNotFound:
					case NodeException::ErrorType::TypeMismatch:
						return {400, detail::wireErrorBody("invalid_input", e.what())};
					default:
						return {500, internalError("execute_failed")};
					}
				} catch (const std::exception& e) {
					return {500, internalError("execute_failed")};
				}
				if (result.ok())
					outputs = exec.collectOutputTensors(taskId);
				exec.clearTask(taskId);
			}

			// 本地执行失败经 wire 逆向映射
			if (!result.ok() && wireHttpStatusFor(result.status) >= 500) return {500, internalError("execute_failed")};
			if (!result.ok())
				return {wireHttpStatusFor(result.status),
						detail::wireErrorBody(wireCodeFor(result.status), result.message)};

			// encodeResponse：输出张量转报文，失败回 5xx
			try {
				return {200, _codec->encodeResponse(outputs)};
			} catch (const std::exception& e) {
				return {500, internalError("encode_response_failed")};
			}
		} catch (const std::exception& e) {
			return {500, internalError("execute_failed")};
		} catch (...) {
			return {500, internalError("execute_failed")};
		}
	}

	std::string internalError(const char* stage) {
		const auto id = "dcnet-service-" + std::to_string(_errorSeq.fetch_add(1));
		if (_endpoint.diagnosticSink) { try { _endpoint.diagnosticSink(id + " internal server error stage=" + stage); } catch (...) {} }
		return detail::wireErrorBody("server_error", "internal server error; correlation=" + id);
	}
	std::atomic<unsigned long long> _errorSeq{0};
	EngineRegistry& _reg;
	std::string _engineType;
	std::string _localModelRef;
	std::shared_ptr<DcNetServerCodec> _codec;
	NetServerEndpoint _endpoint;
	std::unique_ptr<DcNetListener> _listener;
	std::mutex _execMutex; // 本地执行串行，引擎实例跨请求共享
	std::atomic<std::uint64_t> _taskSeq{0};
};

} // namespace

std::shared_ptr<DcNetServerService> registerDcNetServerAdapter(EngineRegistry& reg, DcNetServerAdapterDesc desc) {
	// 配置期校验：失败抛 NodeException，不产生 wire
	if (!desc.codec)
		throw NodeException(NodeException::ErrorType::InternalError, "registerDcNetServerAdapter",
							"engine '" + desc.engineType + "' has no server codec");
	// 预创建引擎核心与实例：未注册 / 加载失败在配置期暴露
	if (!reg.getOrCreateEngine(desc.engineType, desc.localModelRef))
		throw NodeException(NodeException::ErrorType::ExecutionFailed, "registerDcNetServerAdapter",
							"engine '" + desc.engineType + ":" + desc.localModelRef +
								"' not registered or creation failed");
	// requestPath 由 codec 注入
	desc.endpoint.requestPath = desc.codec->requestPath();

	auto svc = std::make_shared<ServerService>(reg, std::move(desc));
	svc->launch();
	return svc;
}

} // namespace DC::Net
