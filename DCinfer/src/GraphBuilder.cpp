#include "GraphBuilder.h"
#include "GraphLowering.h"

namespace DC {

// ════════════════════════════════════════════
// 图构建（全部经 _ensureMutable 守卫：冻结后拒绝）
// ════════════════════════════════════════════

Node& GraphBuilder::addNode(std::unique_ptr<Node> node) {
	_ensureMutable();
	return _store->addNode(std::move(node));
}

Node& GraphBuilder::connect(const std::string& srcNode, const std::string& srcPort,
							const std::string& dstNode, const std::string& dstPort) {
	_ensureMutable();
	return _store->connect(srcNode, srcPort, dstNode, dstPort);
}

void GraphBuilder::connectRaw(const std::string& srcNode, const std::string& srcPort,
							  const std::string& dstNode, const std::string& dstPort) {
	_ensureMutable();
	_store->connectRaw(srcNode, srcPort, dstNode, dstPort);
}

size_t GraphBuilder::connectAll(const std::string& srcNode, const std::string& dstNode) {
	_ensureMutable();
	return _store->connectAll(srcNode, dstNode);
}

void GraphBuilder::bindInput(const std::string& nodeName, const std::string& portName,
							 const std::string& alias) {
	_ensureMutable();
	_ensureAliasUnique(alias, _store->inputBindings(), "GraphBuilder::bindInput");
	_store->bindInput(nodeName, portName, alias);
}

void GraphBuilder::bindOutput(const std::string& nodeName, const std::string& portName,
							  const std::string& alias) {
	_ensureMutable();
	// 重复绑定同一 node:port 为无操作
	for (const auto& b : _outputBindings) {
		if (b.nodeName == nodeName && b.portName == portName)
			return;
	}
	_ensureAliasUnique(alias, _outputBindings, "GraphBuilder::bindOutput");
	_outputBindings.push_back({nodeName, portName, alias});
}

// ════════════════════════════════════════════
// 冻结
// ════════════════════════════════════════════

std::shared_ptr<const CompiledGraph> GraphBuilder::compile() {
	if (_snapshot)
		return _snapshot; // 幂等：重复 compile 返回同一快照

	// 图级签名快照：绑定列表构建期已冻结（此处的拷贝即最终形态）
	GraphSignature signature;
	signature.inputs = _store->inputBindings();
	signature.outputs = _outputBindings;

	// lowering pass：Broadcast(1) wire 从运行时视图擦除（源图不变）
	GraphRuntimeView view;
	GraphLoweringStats stats;
	buildRuntimeView(*_store, signature, view.nodes, view.edges, stats);

	// 拓扑所有权移交快照：冻结后构建面唯一入口消失，运行期只读。
	// 直接 new（非 make_shared）：构造函数为 private，仅 friend（本类）可达。
	_snapshot = std::shared_ptr<const CompiledGraph>(
		new CompiledGraph(std::move(_store), std::move(signature), std::move(view), stats));
	return _snapshot;
}

void GraphBuilder::_ensureMutable() const {
	if (_snapshot) {
		throw GraphException(GraphException::ErrorType::Frozen, "GraphBuilder",
							 "graph is compiled; topology is immutable after freeze");
	}
}

} // namespace DC
