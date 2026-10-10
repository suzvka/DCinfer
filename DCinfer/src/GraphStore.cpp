#include "GraphStore.h"
#include "Connector.h"
#include "GraphException.h"

#include <algorithm>

namespace DC {

Node* GraphStore::node(const std::string& name) {
	auto it = _nodes.find(name);
	return it != _nodes.end() ? it->second.get() : nullptr;
}

const Node* GraphStore::node(const std::string& name) const {
	auto it = _nodes.find(name);
	return it != _nodes.end() ? it->second.get() : nullptr;
}

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

	// 入口防御：常规构造已拒绝非法 schema，此处防御绕过构造路径的输入。
	if (!node->schema().valid())
		throw GraphException(GraphException::ErrorType::Other, "GraphStore::addNode",
							 "invalid schema on node '" + node->name() + "'");

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

	// 两个业务节点禁止直连，至少一端必须是连接器。
	if (!src->isConnector() && !dst->isConnector())
		throw GraphException(GraphException::ErrorType::DirectConnect, "GraphStore::connectRaw",
							 "direct connect between non-connector nodes '" + srcNode + "' and '" + dstNode
								 + "' is forbidden. Use connect() instead.");

	if (!src->schema().findOutput(srcPort))
		throw GraphException(GraphException::ErrorType::PortNotFound, "GraphStore::connectRaw",
							 "output port '" + srcPort + "' not found on node '" + srcNode + "'");
	if (!dst->schema().findInput(dstPort))
		throw GraphException(GraphException::ErrorType::PortNotFound, "GraphStore::connectRaw",
							 "input port '" + dstPort + "' not found on node '" + dstNode + "'");

	_edges.push_back({srcNode, srcPort, dstNode, dstPort});
}

// connect：自动插入 1:1 广播导线；同源口二次 connect 原地扩容该导线，等效 Broadcast(N)。

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

	// 同口多驱动会在传播期静默覆盖，构图期 fail-fast；多上游汇聚须用不同输入口。
	for (const auto& e : _edges) {
		if (e.dstNode == dstNode && e.dstPort == dstPort) {
			throw GraphException(GraphException::ErrorType::DuplicateEdge, "GraphStore::connect",
								 "input port '" + dstNode + ":" + dstPort + "' already has an incoming edge; "
								   "N:1 fan-in requires distinct input ports (multi-input merge node); "
								   "serialized convergence connector is a planned feature");
		}
	}

	// 同源已有出边：广播导线承载则原地扩容接新下游；其他拓扑构图期 fail-fast。
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
			// 追加输出口与新下游边；既有下游保持原口序
			const size_t n = wire->schema().outputs.size();
			const std::string outPort = "out_" + std::to_string(n);
			wire->appendOutputPort({outPort, Node::TensorType::Void, 0, {}});
			_edges.push_back({wireName, outPort, dstNode, dstPort});
			return *wire;
		}
	}

	// 自动创建广播连接器：1 下游时零拷贝直通
	auto wireName = "__wire_" + std::to_string(_nextWireId.fetch_add(1));
	auto wireNode = std::make_unique<Node>("Connector.Broadcast", wireName, Connector::broadcastSchema(1),
										   Connector::broadcastRunFn(), ResourceClass::System);
	wireNode->setConnector(true);
	auto& wireRef = _addNodeImpl(std::move(wireNode));

	_edges.push_back({srcNode, srcPort, wireName, "in"});
	_edges.push_back({wireName, "out_0", dstNode, dstPort});

	return wireRef;
}

void GraphStore::bindInput(const std::string& nodeName, const std::string& portName,
						   const std::string& alias) {
	std::lock_guard lk(_mutex);
	_ensureMutable("GraphStore::bindInput");
	_inputZone.bind(nodeName, portName, alias);
}

std::vector<std::string> GraphStore::nodeNames() const {
	std::lock_guard lk(_mutex); // 无锁遍历并发修改即 UB
	std::vector<std::string> names;
	names.reserve(_nodes.size());
	for (const auto& [name, nodePtr] : _nodes) {
		names.push_back(name);
	}
	return names;
}

} // namespace DC
