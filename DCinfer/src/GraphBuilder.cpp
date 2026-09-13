#include "GraphBuilder.h"
#include "GraphLowering.h"

#include <mutex>
#include <utility>

namespace DC {

// ════════════════════════════════════════════
// 图构建（全部持锁 + _ensureMutableLocked 守卫：冻结后拒绝）
//
// 构建 API 与 compile() 由 _mutex 串行化：构图与冻结并发时，每个构建
// 操作要么先于编译完成（纳入快照），要么在冻结后确定抛 Frozen，
// 不存在"检查通过后被冻结插入"的 TOCTOU 窗口。
// ════════════════════════════════════════════

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

// ════════════════════════════════════════════
// 冻结
// ════════════════════════════════════════════

std::shared_ptr<const CompiledGraph> GraphBuilder::compile() {
	std::lock_guard lk(_mutex);
	if (_snapshot)
		return _snapshot; // 幂等：重复 compile 返回同一快照

	// ① 封印先行：拓扑（GraphStore::seal）与全部节点配置面
	//   （Node::_sealForExecution）先封闭，再进入只读阶段——在飞构图
	//    操作的写经各自锁先于封印完成，此后源图不存在任何写者，
	//    签名构建 / lowering / 运行期对冻结前泄漏引用的修改一律被拒。
	//    注：封印不可回退——即使构建过程异常中止（仅极端资源耗尽），
	//    图保持封闭而非半冻结状态。
	_store->seal();
	for (const auto& entry : _store->nodes())
		entry.second->_sealForExecution();

	// ② 只读阶段：图级签名快照（绑定列表构建期已累积，此处的拷贝即最终形态）
	GraphSignature signature;
	signature.inputs = _store->inputBindings();
	signature.outputs = _outputBindings;

	// lowering pass：Broadcast(1) wire 从运行时视图擦除（源图不变）
	GraphRuntimeView view;
	GraphLoweringStats stats;
	buildRuntimeView(*_store, signature, view.nodes, view.edges, stats);

	// ③ 拓扑所有权移交快照：冻结后构建面唯一入口消失，运行期只读。
	// 直接 new（非 make_shared）：构造函数为 private，仅 friend（本类）可达。
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

} // namespace DC
