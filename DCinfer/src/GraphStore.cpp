#include "GraphStore.h"
#include "Connector.h"

#include <algorithm>

namespace DC {

// ════════════════════════════════════════════
// 查找
// ════════════════════════════════════════════

Node* GraphStore::findNode(const std::string& name) {
	auto it = _nodes.find(name);
	return it != _nodes.end() ? it->second.get() : nullptr;
}

const Node* GraphStore::findNode(const std::string& name) const {
	auto it = _nodes.find(name);
	return it != _nodes.end() ? it->second.get() : nullptr;
}

Node* GraphStore::node(const std::string& name) {
	return findNode(name);
}

const Node* GraphStore::node(const std::string& name) const {
	return findNode(name);
}

// ════════════════════════════════════════════
// 图构建
// ════════════════════════════════════════════

Node* GraphStore::addNode(std::unique_ptr<Node> node) {
	if (!node || node->name().empty())
		return nullptr;

	const auto& name = node->name();
	if (_nodes.contains(name))
		return nullptr;

	auto* raw = node.get();
	_nodes.emplace(name, std::move(node));
	return raw;
}

bool GraphStore::connect(const std::string& srcNode, const std::string& srcPort,
						 const std::string& dstNode, const std::string& dstPort) {
	auto* src = findNode(srcNode);
	auto* dst = findNode(dstNode);
	if (!src || !dst)
		return false;

	// 约束：至少有一端是连接器（两个业务节点禁止直连，必须通过 Connector 中转）
	if (!src->isConnector() && !dst->isConnector())
		return false;

	// 验证端口存在
	if (!src->schema().findOutput(srcPort))
		return false;
	if (!dst->schema().findInput(dstPort))
		return false;

	_edges.push_back({srcNode, srcPort, dstNode, dstPort});
	return true;
}

size_t GraphStore::connectAll(const std::string& srcNode, const std::string& dstNode) {
	auto* src = findNode(srcNode);
	auto* dst = findNode(dstNode);
	if (!src || !dst)
		return 0;

	size_t matched = 0;
	for (const auto& outPort : src->schema().outputs) {
		if (dst->schema().findInput(outPort.name)) {
			_edges.push_back({srcNode, outPort.name, dstNode, outPort.name});
			++matched;
		}
	}
	return matched;
}

// ════════════════════════════════════════════
// wire：自动插入广播连接器（1→1，零拷贝 move 直通）
// ════════════════════════════════════════════

Node* GraphStore::wire(const std::string& srcNode, const std::string& srcPort,
					   const std::string& dstNode, const std::string& dstPort) {
	auto* src = findNode(srcNode);
	auto* dst = findNode(dstNode);
	if (!src || !dst)
		return nullptr;
	if (!src->schema().findOutput(srcPort))
		return nullptr;
	if (!dst->schema().findInput(dstPort))
		return nullptr;

	// 自动创建广播连接器（1 下游 → 零拷贝 move 直通，等效导线）
	auto wireName = "__wire_" + std::to_string(_nextWireId.fetch_add(1));
	auto wireNode = std::make_unique<Node>("Connector.Broadcast", wireName, Connector::broadcastSchema(1),
										   Connector::broadcastRunFn(), nullptr, ThreadPoolAffinity::System);
	wireNode->setConnector(true);
	auto* wirePtr = addNode(std::move(wireNode));
	if (!wirePtr)
		return nullptr;

	// 上游 → 广播
	_edges.push_back({srcNode, srcPort, wireName, "in"});
	// 广播(out_0) → 下游
	_edges.push_back({wireName, "out_0", dstNode, dstPort});

	return wirePtr;
}

void GraphStore::bindInput(const std::string& nodeName, const std::string& portName) {
	_inputZone.bind(nodeName, portName);
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
