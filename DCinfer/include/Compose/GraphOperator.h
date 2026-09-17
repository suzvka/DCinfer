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

namespace DC {

/// @brief 组合算子：把一整张推理图（InferGraph）包装成普通 Node 的算子工厂。
///
/// ── 定位：组合工具，非核心图语义 ──
/// 仅使用公开 API（bindings / feedInput / submitBound / waitForResult /
/// cancel / detachTask / releaseTask / taskId / isCancellationRequested），
/// 不涉及任何 InferGraph 特判；宿主对"子图"的使用方式与自定义算子一致：
/// 构造 → makeNode → addNode。若要真正复用一个逻辑块，推荐将其建模为
/// 自定义算子（见 examples/03_custom_node）；本工具适用于"把现成图整体
/// 嵌入父图"的场景。
///
/// ── 生命周期：引用计数闭合 ──
/// 构造时以 shared_ptr 接管子图所有权；导出节点经 RunFn 捕获同一共享
/// 句柄——"子图先于节点析构"不可表示，无需生命周期契约或哨兵检测。
///
/// ── 接口与 Schema：构造即定型 ──
/// 构造时立即 freeze() 子图（幂等）并按绑定推导节点 Schema：
/// - 端口名 = 绑定 alias（bindInput/bindOutput 已保证 alias 唯一，
///   接口层命名碰撞不可表示）；
/// - 类型/形状/required/默认值如实拷贝目标端口。
/// 由此消灭"makeNode 后改绑定"的 schema 漂移窗口与懒冻结竞争。
///
/// ── 子任务空间：命名空间化 ──
/// 每次执行以 "父任务ID|实例号|节点名" 命名子任务——同一子图被多个组合
/// 节点、多个父图（含任务 ID 撞名）并发复用时互不冲突，无 DuplicateTask
/// 限制（逐节点实例号的进程级唯一性由原子计数器保证）。
///
/// ── 执行语义：等待型节点 ──
/// RunFn 内同步喂入 → 提交 → 分段等待子图任务，并周期性感知父任务取消
/// （协作式解围：宿主 cancel 父任务后，子图任务在 pollInterval 粒度内被
/// 取消，池线程有界释放，不会因内层信号停滞永久挂起）。
/// 注意：等待期间本节点占住一个执行线程——线程池容量需按
/// "每层池配置 × 并发组合节点数"规划。
///
/// ── 序列化 ──
/// 节点 type 为 "Builtin"，与注册算子待遇一致：DCIr 往返仅保留结构
/// （Schema 骨架），RunFn 由宿主在加载后重建。
class GraphOperator {
public:
	/// @brief 执行参数
	struct Options {
		uint32_t maxHops = InferGraph::kDefaultMaxHops; ///< 子图 TTL（防循环无限传播）
		std::chrono::milliseconds pollInterval{100};    ///< 父取消感知轮询间隔（必须 > 0）
	};

	/// @brief 构造：接管子图共享所有权，立即冻结并推导接口 Schema（fail-fast）。
	/// @throws GraphException(Other)         graph 为空 / 两侧绑定均空 / pollInterval <= 0
	/// @throws GraphException(NodeNotFound)  绑定引用的节点不存在
	/// @throws GraphException(PortNotFound)  绑定引用的端口不存在
	/// @throws GraphException(Other)         绑定目标为连接器（接口必须指向业务节点）
	explicit GraphOperator(std::shared_ptr<InferGraph> graph, Options opts = {})
		: _graph(std::move(graph)), _opts(opts) {
		if (!_graph)
			throw GraphException(GraphException::ErrorType::Other, "GraphOperator",
								 "graph must not be null");
		if (_opts.pollInterval <= std::chrono::milliseconds::zero())
			throw GraphException(GraphException::ErrorType::Other, "GraphOperator",
								 "pollInterval must be positive (0 would mean infinite wait)");
		_graph->freeze(); // 构造即冻结：接口定型 + 消灭懒冻结竞争窗口
		_schema = _deriveSchema();
	}

	/// @brief 生成组合节点（普通 Node；可多次调用——同一子图可被多个节点共享）。
	/// @param  nodeName  父图内节点名（唯一性由父图 addNode 校验）
	/// @param  affinity  执行线程池归属；等待型节点的占池语义见类注释
	std::unique_ptr<Node> makeNode(const std::string& nodeName,
								   ThreadPoolAffinity affinity = ThreadPoolAffinity::Operator) const {
		const uint64_t instanceId = _nextInstanceId.fetch_add(1, std::memory_order_relaxed) + 1;

		auto runFn = [graph = _graph, opts = _opts, nodeName, instanceId](Node::RunContext& ctx) -> Node::Result {
			// 子任务 ID：父任务空间 + 本节点实例命名空间——
			// 同一子图的多节点/多父图并发复用互不冲突（无 DuplicateTask 限制）。
			const std::string childTid =
				ctx.taskId() + "|" + std::to_string(instanceId) + "|" + nodeName;

			// ① 输入注入：父级已投递的数据按绑定代理到子图；未投递的输入
			//    跳过（把可选输入/默认值语义交还子图自身的就绪判定）。
			for (const auto& b : graph->inputBindings()) {
				Value v = _takeIfDelivered(ctx, b.alias);
				if (!v)
					continue;
				graph->feedInput(childTid, b.nodeName, b.portName, std::move(v));
			}

			// ② 提交：以子图全部输出绑定为声明
			graph->submitBound(childTid, opts.maxHops);

			// ③ 分段等待 + 父轮取消感知（协作式解围）：内层信号阻塞不再
			//    令父池线程无限期挂起。
			TaskResult res = graph->waitForResult(childTid, opts.pollInterval);
			while (res.status == TaskStatus::Running) {
				if (ctx.isCancellationRequested()) {
					graph->cancel(childTid);
					graph->waitForResult(childTid, kCancelGrace); // 限时收尾（结果不再使用）
					graph->detachTask(childTid); // 已终态立即回收；仍在飞 → 终态自动回收
					return ctx.failure(Node::Status::ExecutionFailed,
									   "block '" + nodeName + "' (child task '" + childTid +
										   "'): parent task cancelled while awaiting subgraph completion");
				}
				res = graph->waitForResult(childTid, opts.pollInterval);
			}

			// ④ 终止判定：非成功 → 转发内层诊断并回收子任务资源
			if (res.status != TaskStatus::Succeeded) {
				std::string detail;
				if (!res.errors.empty())
					detail = ": " + res.errors[0].message;
				graph->releaseTask(childTid);
				return ctx.failure(Node::Status::ExecutionFailed,
								   "block '" + nodeName + "' (child task '" + childTid +
									   "'): subgraph terminated with status " + _statusName(res.status) + detail);
			}

			// ⑤ 输出收集（仅已产出端口）→ 回收子任务资源
			for (const auto& b : graph->outputBindings()) {
				if (graph->hasOutput(childTid, b.nodeName, b.portName))
					ctx.output(b.alias, graph->takeOutput(childTid, b.nodeName, b.portName));
			}
			graph->releaseTask(childTid); // 终态回收：结果/诊断不滞留
			return ctx.success();
		};

		return std::make_unique<Node>("Builtin", nodeName, _schema, std::move(runFn), affinity);
	}

	/// @brief 组合节点 Schema（端口名 = 绑定 alias；构造时推导，随对象冻结）
	const Node::Schema& schema() const { return _schema; }

	/// @brief 子图指针（借用；所有权由共享句柄与本对象共同持有）
	const InferGraph& graph() const { return *_graph; }

private:
	/// @brief 取消解围的限时收尾窗口（子图已取消后等待其终止的兜底时长）
	static constexpr std::chrono::seconds kCancelGrace{1};

	/// @brief 按绑定推导节点 Schema。
	/// @note   端口名 = 绑定 alias；其余端口属性（类型/形状/required/默认值/
	///         锚定）如实拷贝目标端口——组合节点的就绪语义与子图入口一致。
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

	/// @brief 绑定目标节点解析与连接器拒绝（接口必须指向业务节点）
	const Node* _resolveTarget(const std::string& nodeName, const std::string& alias,
							   const char* direction) const {
		const InferGraph& g = *_graph; // const 视图：node() 走只读重载（构建期重载冻结后抛 Frozen）
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

	/// @brief 尝试从父上下文取出已投递的输入（无数据返回空 Value）。
	/// @note   RunContext::peek 对"已声明但未投递"的槽位抛
	///         NodeException(TypeMismatch)（空槽位无 Value 载荷）——组合语义
	///         把未投递输入交还子图自身的就绪判定，故在此按"跳过"处理；
	///         其余异常如实上抛。
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

	std::shared_ptr<InferGraph> _graph; ///< 子图（共享所有权：节点经 RunFn 捕获延长其生命周期）
	Node::Schema _schema;               ///< 组合节点 Schema（构造时推导，此后只读）
	Options _opts;                      ///< 执行参数（maxHops / pollInterval）

	/// 进程级实例号：使每个 makeNode 产物拥有独立子任务命名空间
	static inline std::atomic<uint64_t> _nextInstanceId{0};
};

} // namespace DC
