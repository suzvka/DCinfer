#pragma once

#include "GraphException.h"
#include "InferGraph.h"
#include "Node.h"
#include "NodeException.h"
#include "TaskStatus.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace DC {

/// @brief 组合算子：把一整张推理图包装成普通 Node 的算子工厂。
///
/// 仅使用公开 API，无核心图特判；宿主用法与自定义算子一致：构造、makeNode、addNode。
/// 构造时以 shared_ptr 接管子图所有权并立即 freeze 推导 Schema，端口名即绑定 alias，
/// 其余属性如实拷贝目标端口；导出节点经 RunFn 捕获同一共享句柄，生命周期引用计数闭合。
///
/// 子任务命名：每次执行以 "父任务ID|实例号|节点名" 命名，段内分隔符转义保证拼接单射，
/// 同一子图被多节点或多父图复用时互不冲突。
///
/// 执行语义：RunFn 内同步喂入、提交并分段等待子图任务，周期性感知父任务取消以协作式解围。
/// 等待期间占住一个执行槽位；默认亲和 System 与子图业务节点默认的 Operator 类分离，
/// 但进程预算必须覆盖并发等待节点数，否则嵌套等待链叠加可能自锁。
///
/// 序列化：节点 type 为 "Builtin"，DCIr 往返仅保留结构，RunFn 由宿主重建。
class GraphOperator {
public:
	struct Options {
		uint32_t maxHops = InferGraph::kDefaultMaxHops;
		std::chrono::milliseconds pollInterval{100};
	};

	/// @brief 构造，默认参数形态。
	/// @note 不写默认实参 Options{}：嵌套类型的默认成员初始化器不得在默认实参中求值，GCC/Clang 拒绝。
	explicit GraphOperator(std::shared_ptr<InferGraph> graph)
		: GraphOperator(std::move(graph), Options{}) {}

	/// @brief 接管子图共享所有权，立即冻结并推导接口 Schema，fail-fast。
	explicit GraphOperator(std::shared_ptr<InferGraph> graph, Options opts)
		: _graph(std::move(graph)), _opts(opts) {
		if (!_graph)
			throw GraphException(GraphException::ErrorType::Other, "GraphOperator",
								 "graph must not be null");
		if (_opts.pollInterval <= std::chrono::milliseconds::zero())
			throw GraphException(GraphException::ErrorType::Other, "GraphOperator",
								 "pollInterval must be positive (0 would mean infinite wait)");
		_graph->freeze(); // 构造即冻结：接口定型并消灭懒冻结竞争窗口
		_schema = _deriveSchema();
	}

	/// @brief 生成组合节点，即普通 Node；可多次调用共享同一子图。
	std::unique_ptr<Node> makeNode(const std::string& nodeName,
								   ResourceClass affinity = ResourceClass::System) const {
		const uint64_t instanceId = _nextInstanceId.fetch_add(1, std::memory_order_relaxed) + 1;

		auto runFn = [graph = _graph, opts = _opts, nodeName, instanceId](Node::RunContext& ctx) -> Node::Result {
			const std::string childTid = _escapeIdSegment(ctx.taskId()) + "|"
									 + std::to_string(instanceId) + "|" + _escapeIdSegment(nodeName);

			// RAII 守卫：任何退出路径都回收子任务，已终态立即释放，在飞登记自动回收。
			struct ChildTaskGuard {
				InferGraph& graph;
				const std::string& childTid;

				~ChildTaskGuard() { graph.detachTask(childTid); }
			} childGuard{*graph, childTid};

			// 输入注入：未投递的输入跳过，默认值语义交还子图就绪判定。
			for (const auto& b : graph->inputBindings()) {
				Value v = _takeIfDelivered(ctx, b.alias);
				if (!v)
					continue;
				graph->feedInput(childTid, b.nodeName, b.portName, std::move(v));
			}

			// 提交，以全部输出绑定为声明
			graph->submitBound(childTid, opts.maxHops);

			// 分段等待并感知父取消，协作式解围。
			TaskResult res = graph->waitForResult(childTid, opts.pollInterval);
			while (res.status == TaskStatus::Running) {
				if (ctx.isCancellationRequested()) {
					graph->cancel(childTid);
					graph->waitForResult(childTid, kCancelGrace); // 限时收尾，结果不再使用
					return ctx.failure(Node::Status::ExecutionFailed,
									   "block '" + nodeName + "' (child task '" + childTid +
										   "'): parent task cancelled while awaiting subgraph completion");
				}
				res = graph->waitForResult(childTid, opts.pollInterval);
			}

			// 非成功：转发内层诊断，资源由守卫回收。
			if (res.status != TaskStatus::Succeeded) {
				std::string detail;
				if (!res.errors.empty())
					detail = ": " + res.errors[0].message;
				return ctx.failure(Node::Status::ExecutionFailed,
								   "block '" + nodeName + "' (child task '" + childTid +
									   "'): subgraph terminated with status " + _statusName(res.status) + detail);
			}

			// 输出收集：缺失即显式失败，避免真实根因被父节点无输出判败掩盖。
			std::vector<std::string> missing;
			for (const auto& b : graph->outputBindings()) {
				if (!graph->hasOutput(childTid, b.nodeName, b.portName)) {
					missing.push_back(b.alias);
					continue;
				}
				ctx.output(b.alias, graph->takeOutput(childTid, b.nodeName, b.portName));
			}
			if (!missing.empty()) {
				std::string list;
				for (const auto& alias : missing) {
					if (!list.empty())
						list += ", ";
					list += alias;
				}
				return ctx.failure(Node::Status::ExecutionFailed,
								   "block '" + nodeName + "' (child task '" + childTid +
									   "'): subgraph finished but bound outputs were not produced: " + list);
			}
			return ctx.success();
		};

		return std::make_unique<Node>("Builtin", nodeName, _schema, std::move(runFn), affinity);
	}

	/// @brief 组合节点 Schema，构造时推导。
	const Node::Schema& schema() const { return _schema; }

	/// @brief 子图引用。
	const InferGraph& graph() const { return *_graph; }

private:
	/// @brief 取消解围的限时收尾窗口。
	static constexpr std::chrono::seconds kCancelGrace{1};

	/// @brief 按绑定推导节点 Schema；端口名即 alias，其余属性拷贝目标端口。
	Node::Schema _deriveSchema() const {
		Node::Schema schema;
		for (const auto& b : _graph->inputBindings()) {
			const Node* n = _resolveTarget(b.nodeName, b.alias, "input");
			const auto* port = n->schema().findInput(b.portName);
			if (!port)
				throw GraphException(GraphException::ErrorType::PortNotFound, "GraphOperator",
									 "input binding '" + b.alias + "' references port '" + b.nodeName + "." +
										 b.portName + "' not found in schema");
			Node::Port p = *port;
			p.name = b.alias;
			schema.inputs.push_back(std::move(p));
		}
		for (const auto& b : _graph->outputBindings()) {
			const Node* n = _resolveTarget(b.nodeName, b.alias, "output");
			const auto* port = n->schema().findOutput(b.portName);
			if (!port)
				throw GraphException(GraphException::ErrorType::PortNotFound, "GraphOperator",
									 "output binding '" + b.alias + "' references port '" + b.nodeName + "." +
										 b.portName + "' not found in schema");
			Node::Port p = *port;
			p.name = b.alias;
			schema.outputs.push_back(std::move(p));
		}
		if (schema.inputs.empty() && schema.outputs.empty())
			throw GraphException(GraphException::ErrorType::Other, "GraphOperator",
								 "graph declares no interface; call bindInput/bindOutput before composing");
		return schema;
	}

	/// @brief 绑定目标解析；拒绝连接器，接口必须指向业务节点。
	const Node* _resolveTarget(const std::string& nodeName, const std::string& alias,
							   const char* direction) const {
		const InferGraph& g = *_graph; // const 视图走只读 node 重载，可写重载冻结后抛 Frozen
		const Node* n = g.node(nodeName);
		if (!n)
			throw GraphException(GraphException::ErrorType::NodeNotFound, "GraphOperator",
								 std::string(direction) + " binding '" + alias + "' references node '" + nodeName +
									 "' not found");
		if (n->isConnector())
			throw GraphException(GraphException::ErrorType::Other, "GraphOperator",
								 std::string(direction) + " binding '" + alias + "' targets connector node '" +
									 nodeName + "'; interface must reference business nodes");
		return n;
	}

	/// @brief 取出已投递的输入；无数据返回空 Value，peek 的 TypeMismatch 视为未投递。
	static Value _takeIfDelivered(Node::RunContext& ctx, const std::string& alias) {
		try {
			if (!ctx.peek(alias))
				return {};
			return ctx.pop(alias);
		} catch (const NodeException& e) {
			if (e.getErrorType() == NodeException::ErrorType::TypeMismatch)
				return {};
			throw;
		}
	}

	static const char* _statusName(TaskStatus st) {
		switch (st) {
		case TaskStatus::Unknown: return "Unknown";
		case TaskStatus::Running: return "Running";
		case TaskStatus::Succeeded: return "Succeeded";
		case TaskStatus::Failed: return "Failed";
		case TaskStatus::Cancelled: return "Cancelled";
		}
		return "Unknown";
	}

	/// @brief 段转义：把 \ 与 | 分别转义为 \\ 与 \|，保证 "段A|实例号|段B" 拼接单射。
	static std::string _escapeIdSegment(const std::string& s) {
		std::string out;
		out.reserve(s.size());
		for (char c : s) {
			if (c == '\\' || c == '|')
				out.push_back('\\');
			out.push_back(c);
		}
		return out;
	}

	std::shared_ptr<InferGraph> _graph;
	Node::Schema _schema;
	Options _opts;

	// 进程级实例号：每个 makeNode 产物拥有独立子任务命名空间。
	static inline std::atomic<uint64_t> _nextInstanceId{0};
};

} // namespace DC
