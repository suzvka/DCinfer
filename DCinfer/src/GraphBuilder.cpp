#include "GraphBuilder.h"
#include "GraphLowering.h"

#include <mutex>
#include <set>
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

	// ⓪ 校验先于封印：输出取数端口不变量失败可修正后重试 compile
	//   （封印不可回退——若先封印，用户错误将把图锁死为不可修复状态；
	//   提交期守卫「先于任何状态登记、失败可重试」的同款哲学）。
	_validateOutputBindings();

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

void GraphBuilder::_validateOutputBindings() const {
	// 输出取数端口不变量：绑定端口必须是终端端口（无出边）。
	// 传播期「输出区搬运」与「出边搬运」共享同一消费槽——非终端绑定会把
	// 数据从下游数据流中截走：下游饿死（下游声明无法满足 → 任务挂起，
	// 或侥幸跳过下游）。声明侧（submit）保持完成条件语义不变（循环/部分
	// 求值依赖它），本校验只约束绑定。
	//
	// 实现：一次扫源图建「有出边端口」索引，逐绑定 O(1) 查。
	// 构建期视图（持 _mutex，构建面独占）：调用时机先于封印。
	std::set<std::pair<std::string, std::string>> portsWithOutEdges;
	for (const auto& e : _store->edges())
		portsWithOutEdges.emplace(e.srcNode, e.srcPort);

	for (const auto& b : _outputBindings) {
		const Node* n = _store->node(b.nodeName);
		if (!n)
			continue; // 绑定的坐标存在性由 interface()/提交期校验（既有延迟语义不在此收紧）
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
