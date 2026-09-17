#include "GraphStore.h"
#include "Connector.h"
#include "GraphException.h"

#include <algorithm>

namespace DC {

// ════════════════════════════════════════════
// 查找
// ════════════════════════════════════════════

Node* GraphStore::node(const std::string& name) {
	auto it = _nodes.find(name);
	return it != _nodes.end() ? it->second.get() : nullptr;
}

const Node* GraphStore::node(const std::string& name) const {
	auto it = _nodes.find(name);
	return it != _nodes.end() ? it->second.get() : nullptr;
}

// ════════════════════════════════════════════
// 图构建（全部持锁 + 封印检查：冻结后抛 GraphException(Frozen)）
// ════════════════════════════════════════════

void GraphStore::seal() {
	std::lock_guard lk(_mutex);
	_sealed = true;
}

void GraphStore::_ensureMutable(const char* api) const {
	if (_sealed)
		throw GraphException(GraphException::ErrorType::Frozen, api,
							 "topology is sealed; graph is immutable after freeze");
}

Node& GraphStore::addNode(std::unique_ptr<Node> node) {
	std::lock_guard lk(_mutex);
	_ensureMutable("GraphStore::addNode");
	return _addNodeImpl(std::move(node));
}

Node& GraphStore::_addNodeImpl(std::unique_ptr<Node> node) {
	if (!node || node->name().empty())
		throw GraphException(GraphException::ErrorType::DuplicateNode, "GraphStore::addNode",
							 "node name is empty");

	const auto& name = node->name();
	if (_nodes.contains(name))
		throw GraphException(GraphException::ErrorType::DuplicateNode, "GraphStore::addNode",
							 "duplicate node name: '" + name + "'");

	auto* raw = node.get();
	_nodes.emplace(name, std::move(node));
	return *raw;
}

void GraphStore::connectRaw(const std::string& srcNode, const std::string& srcPort,
						 const std::string& dstNode, const std::string& dstPort) {
	std::lock_guard lk(_mutex);
	_ensureMutable("GraphStore::connectRaw");
	auto* src = node(srcNode);
	auto* dst = node(dstNode);
	if (!src)
		throw GraphException(GraphException::ErrorType::NodeNotFound, "GraphStore::connectRaw",
							 "source node '" + srcNode + "' not found");
	if (!dst)
		throw GraphException(GraphException::ErrorType::NodeNotFound, "GraphStore::connectRaw",
							 "destination node '" + dstNode + "' not found");

	// 约束：至少有一端是连接器（两个业务节点禁止直连，必须通过 Connector 中转）
	if (!src->isConnector() && !dst->isConnector())
		throw GraphException(GraphException::ErrorType::DirectConnect, "GraphStore::connectRaw",
							 "direct connect between non-connector nodes '" + srcNode + "' and '" + dstNode
								 + "' is forbidden. Use connect() instead.");

	// 验证端口存在
	if (!src->schema().findOutput(srcPort))
		throw GraphException(GraphException::ErrorType::PortNotFound, "GraphStore::connectRaw",
							 "output port '" + srcPort + "' not found on node '" + srcNode + "'");
	if (!dst->schema().findInput(dstPort))
		throw GraphException(GraphException::ErrorType::PortNotFound, "GraphStore::connectRaw",
							 "input port '" + dstPort + "' not found on node '" + dstNode + "'");

	_edges.push_back({srcNode, srcPort, dstNode, dstPort});
}

// ════════════════════════════════════════════
// connect：自动插入广播连接器（1→1，零拷贝 move 直通）
// ════════════════════════════════════════════
// 同一输出端口禁止二次 connect（CORE-01）：每次 connect 创建独立 1:1 wire，
// 二次连接会产生两条同源直连边（lowering 将 1 出边 wire 融合为直连边），
// 而传播期按边逐个消费式取数——首条边取走后 hasOutput=false，第二条边
// 静默跳过，下游永远收不到数据且任务永久挂起。1:N 分发必须显式创建
// Connector.Broadcast(N)（见 README「灵活的图拓扑」）。

Node& GraphStore::connect(const std::string& srcNode, const std::string& srcPort,
					   const std::string& dstNode, const std::string& dstPort) {
	std::lock_guard lk(_mutex);
	_ensureMutable("GraphStore::connect");
	auto* src = node(srcNode);
	auto* dst = node(dstNode);
	if (!src)
		throw GraphException(GraphException::ErrorType::NodeNotFound, "GraphStore::connect",
							 "source node '" + srcNode + "' not found");
	if (!dst)
		throw GraphException(GraphException::ErrorType::NodeNotFound, "GraphStore::connect",
							 "destination node '" + dstNode + "' not found");
	if (!src->schema().findOutput(srcPort))
		throw GraphException(GraphException::ErrorType::PortNotFound, "GraphStore::connect",
							 "output port '" + srcPort + "' not found on node '" + srcNode + "'");
	if (!dst->schema().findInput(dstPort))
		throw GraphException(GraphException::ErrorType::PortNotFound, "GraphStore::connect",
							 "input port '" + dstPort + "' not found on node '" + dstNode + "'");

	// 同目标端口已有入边 → 构图期 fail-fast：同口多驱动在传播期静默覆盖
	// （仅最后写入者生效，多上游数据仅存其一）。多上游汇聚必须使用不同
	// 输入口（等齐合并节点）。
	// TODO(串行化汇聚): 多上游共享同一输入口 + 串行投递（单值在途：下游消费
	// 后才推下一份）为规划特性——待调度器模式重构后实现（放行本检查 +
	// 投递层单值在途通道 + 消费触发）。
	for (const auto& e : _edges) {
		if (e.dstNode == dstNode && e.dstPort == dstPort) {
			throw GraphException(GraphException::ErrorType::DuplicateEdge, "GraphStore::connect",
								 "input port '" + dstNode + ":" + dstPort + "' already has an incoming edge; "
								   "N:1 fan-in requires distinct input ports (multi-input merge node); "
								   "serialized convergence connector is a planned feature");
		}
	}

	// 同源端口已有出边 → 二态：
	//  - 既有连接由广播导线承载（自动导线 / 显式 Broadcast 的 in 接线）→ 原地
	//    扩容（多分一份）并接上新下游：增加连接即扩扇出，无需手写 Broadcast(N)；
	//  - 其他拓扑（connectRaw 构造等）→ 构图期 fail-fast。
	for (const auto& e : _edges) {
		if (e.srcNode == srcNode && e.srcPort == srcPort) {
			const std::string wireName = e.dstNode; // 先复制：push_back 会使 e 失效
			auto* wire = node(wireName);
			if (!wire || !wire->isConnector() || wire->type() != "Connector.Broadcast"
				|| e.dstPort != "in") {
				throw GraphException(GraphException::ErrorType::DuplicateEdge, "GraphStore::connect",
									 "output port '" + srcNode + ":" + srcPort
										 + "' already has an outgoing edge; 1:N fan-out requires an explicit "
										   "Connector.Broadcast(N) (see README)");
			}
			// 扩容：追加输出口 + 新下游边（既有下游保持原口序）
			const size_t n = wire->schema().outputs.size();
			const std::string outPort = "out_" + std::to_string(n);
			wire->appendOutputPort({outPort, Node::TensorType::Void, 0, {}});
			_edges.push_back({wireName, outPort, dstNode, dstPort});
			return *wire;
		}
	}

	// 自动创建广播连接器（1 下游 → 零拷贝 move 直通，等效导线）
	auto wireName = "__wire_" + std::to_string(_nextWireId.fetch_add(1));
	auto wireNode = std::make_unique<Node>("Connector.Broadcast", wireName, Connector::broadcastSchema(1),
										   Connector::broadcastRunFn(), ThreadPoolAffinity::System);
	wireNode->setConnector(true);
	auto& wireRef = _addNodeImpl(std::move(wireNode));

	// 上游 → 广播
	_edges.push_back({srcNode, srcPort, wireName, "in"});
	// 广播(out_0) → 下游
	_edges.push_back({wireName, "out_0", dstNode, dstPort});

	return wireRef;
}

void GraphStore::bindInput(const std::string& nodeName, const std::string& portName,
						   const std::string& alias) {
	std::lock_guard lk(_mutex);
	_ensureMutable("GraphStore::bindInput");
	_inputZone.bind(nodeName, portName, alias);
}

// ════════════════════════════════════════════
// 查询
// ════════════════════════════════════════════

std::vector<std::string> GraphStore::nodeNames() const {
	std::vector<std::string> names;
	names.reserve(_nodes.size());
	for (const auto& [name, nodePtr] : _nodes) {
		names.push_back(name);
	}
	return names;
}

} // namespace DC
