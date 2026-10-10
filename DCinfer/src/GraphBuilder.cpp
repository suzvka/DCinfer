#include "GraphBuilder.h"
#include "GraphLowering.h"

#include <mutex>
#include <set>
#include <utility>

namespace DC {

// 构建 API 与 compile() 由 _mutex 串行化，构建操作要么纳入快照，要么冻结后抛 Frozen，无 TOCTOU 窗口。

Node& GraphBuilder::addNode(std::unique_ptr<Node> node) {
	std::lock_guard lk(_mutex);
	_ensureMutableLocked();
	return _store->addNode(std::move(node));
}

Node& GraphBuilder::connect(const std::string& srcNode, const std::string& srcPort,
							const std::string& dstNode, const std::string& dstPort) {
	std::lock_guard lk(_mutex);
	_ensureMutableLocked();
	return _store->connect(srcNode, srcPort, dstNode, dstPort);
}

void GraphBuilder::bindInput(const std::string& nodeName, const std::string& portName,
							 const std::string& alias) {
	std::lock_guard lk(_mutex);
	_ensureMutableLocked();
	_ensureAliasValid(alias, _store->inputBindings(), "GraphBuilder::bindInput");
	_store->bindInput(nodeName, portName, alias);
}

void GraphBuilder::bindOutput(const std::string& nodeName, const std::string& portName,
							  const std::string& alias) {
	std::lock_guard lk(_mutex);
	_ensureMutableLocked();
	// 重复绑定同一 node:port 为无操作
	for (const auto& b : _outputBindings) {
		if (b.nodeName == nodeName && b.portName == portName)
			return;
	}
	_ensureAliasValid(alias, _outputBindings, "GraphBuilder::bindOutput");
	_outputBindings.push_back({nodeName, portName, alias});
}

std::shared_ptr<const CompiledGraph> GraphBuilder::compile() {
	std::lock_guard lk(_mutex);
	if (_snapshot)
		return _snapshot; // 幂等：重复 compile 返回同一快照

	// 校验先于封印：失败可修正后重试 compile；先封印会把图锁死为不可修复状态。
	_validateOutputBindings();

	// 封印先行：拓扑与全部节点配置面先封闭，此后编译阶段无任何写者；封印不可回退。
	_store->seal();
	for (const auto& entry : _store->nodes())
		entry.second->_sealForExecution();

	// 只读阶段：图级签名快照，绑定列表拷贝即最终形态。
	GraphSignature signature;
	signature.inputs = _store->inputBindings();
	signature.outputs = _outputBindings;

	// lowering：从运行时视图擦除 Broadcast(1) wire，源图不变。
	GraphRuntimeView view;
	GraphLoweringStats stats;
	buildRuntimeView(*_store, signature, view.nodes, view.edges, stats);

	// 拓扑所有权移交快照；CompiledGraph 构造为 private，故显式 new。
	_snapshot = std::shared_ptr<const CompiledGraph>(
		new CompiledGraph(std::move(_store), std::move(signature), std::move(view), stats));
	return _snapshot;
}

void GraphBuilder::_ensureMutableLocked() const {
	if (_snapshot) {
		throw GraphException(GraphException::ErrorType::Frozen, "GraphBuilder",
							 "graph is compiled; topology is immutable after freeze");
	}
}

void GraphBuilder::_validateOutputBindings() const {
	// 输出绑定端口必须为终端端口：非终端绑定会把数据从下游数据流截走，导致下游饿死。
	// 声明侧完成条件语义不变，本校验只约束绑定。
	std::set<std::pair<std::string, std::string>> portsWithOutEdges;
	for (const auto& e : _store->edges())
		portsWithOutEdges.emplace(e.srcNode, e.srcPort);

	for (const auto& b : _outputBindings) {
		const Node* n = _store->node(b.nodeName);
		if (!n)
			continue; // 坐标存在性由 interface()/提交期校验
		if (portsWithOutEdges.contains({b.nodeName, b.portName})) {
			throw GraphException(GraphException::ErrorType::NonTerminalPort, "GraphBuilder::compile",
								 "output binding '" + b.alias + "' on port '" + b.nodeName + ":" + b.portName
									 + "' has outgoing edges; a bound (retrievable) output port must be terminal. "
									   "For a mid-pipeline value that must stay retrievable AND keep flowing "
									   "downstream, insert an explicit branch: connect the port through a "
									   "pass-through node and bind the branch leaf (repeat connect on the same "
									   "source port auto-expands the fan-out), or bind a downstream terminal "
									   "port instead");
		}
	}
}

} // namespace DC
