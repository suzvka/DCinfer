#pragma once

#include "GraphStore.h"
#include "OutputZone.h"  // OutputBinding
#include "CompiledGraph.h"
#include "GraphException.h"

#include <memory>
#include <string>
#include <vector>

namespace DC {

/// @brief 图构建器：Build → Freeze → Execute 生命周期中的唯一构建面。
///
/// 承接拓扑构建 API（addNode/connect/bindInput/bindOutput...）与构建期的
/// 输出绑定累积。compile() 产出不可变
/// CompiledGraph 快照并移交拓扑所有权：冻结后所有构建 API 抛
/// GraphException(Frozen)——"执行期可变拓扑"从数据竞争隐患变为确定性错误。
///
/// 与 InferGraph 的关系：InferGraph 是执行 Facade，内部持有一个
/// GraphBuilder；构建方法经冻结检查后转发，执行方法走冻结快照。
/// 拓扑演进 = 重新构建 GraphBuilder → compile 产生新快照。
class GraphBuilder {
public:
	using Edge = GraphStore::Edge;

	GraphBuilder() = default;
	~GraphBuilder() = default;

	GraphBuilder(const GraphBuilder&) = delete;
	GraphBuilder& operator=(const GraphBuilder&) = delete;

	// ── 图构建（冻结前可用）──

	/// @brief  添加节点（转移所有权），返回引用供后续接线引用
	/// @throws GraphException(DuplicateNode) 若节点名为空或重名
	Node& addNode(std::unique_ptr<Node> node);

	/// @brief  端口级接线（默认方式）：上游输出口 → 下游输入口，
	///         自动插入广播连接器（Broadcast Connector, N=1）
	/// @throws GraphException(NodeNotFound/PortNotFound) 若节点或端口不存在
	/// @return 指向自动创建的广播连接器的引用
	Node& connect(const std::string& srcNode, const std::string& srcPort,
				  const std::string& dstNode, const std::string& dstPort);

	/// @brief  标记输入：该节点的该端口为图级输入口
	/// @param  alias  公共别名（必填；须在全部输入绑定中唯一）
	/// @throws GraphException(InvalidBinding) 别名为空
	/// @throws GraphException(DuplicateBinding) 别名重复
	void bindInput(const std::string& nodeName, const std::string& portName,
				   const std::string& alias);

	/// @brief  标记输出：该节点的该端口产出进入输出区（与边目的地互斥）
	/// @param  alias  公共别名（必填；须在全部输出绑定中唯一）
	/// @throws GraphException(InvalidBinding) 别名为空
	/// @throws GraphException(DuplicateBinding) 别名重复
	/// @note   重复绑定同一 node:port 为无操作
	void bindOutput(const std::string& nodeName, const std::string& portName,
					const std::string& alias);

	// ── 构建期内省（冻结前可用；compile 后返回空/零）──

	GraphStore& store() { return *_store; }
	const GraphStore& store() const { return *_store; }
	Node* node(const std::string& name) { return _store->node(name); }
	const Node* node(const std::string& name) const { return _store->node(name); }
	size_t nodeCount() const { return _store->nodeCount(); }
	size_t edgeCount() const { return _store->edgeCount(); }
	std::vector<std::string> nodeNames() const { return _store->nodeNames(); }
	const std::vector<Edge>& edges() const { return _store->edges(); }
	const std::vector<InputBinding>& inputBindings() const { return _store->inputBindings(); }
	const std::vector<OutputBinding>& outputBindings() const { return _outputBindings; }

	// ── 冻结 ──

	/// @brief  编译构建面为不可变快照（幂等：重复调用返回同一快照）。
	///
	/// 构建 GraphSignature（输入/输出绑定快照）→ 拓扑所有权移交快照。
	/// 此后所有构建 API 抛 GraphException(Frozen)。
	std::shared_ptr<const CompiledGraph> compile();

private:
	/// @brief  构建守卫：冻结后调用构建 API 抛 GraphException(Frozen)
	void _ensureMutable() const;

	/// @brief  别名校验（别名是绑定的必填公共名）：拒绝空别名与重复别名
	template <typename Bindings>
	static void _ensureAliasValid(const std::string& alias, const Bindings& bindings,
								   const char* api) {
		if (alias.empty())
			throw GraphException(GraphException::ErrorType::InvalidBinding, api, "alias is required; bindInput/bindOutput take (alias, nodeName, portName)");
		for (const auto& b : bindings) {
			if (b.alias == alias)
				throw GraphException(GraphException::ErrorType::DuplicateBinding, api,
									 "alias '" + alias + "' is already bound; aliases must be unique");
		}
	}

	std::unique_ptr<GraphStore> _store = std::make_unique<GraphStore>();
	std::vector<OutputBinding> _outputBindings;     ///< 输出绑定（构建期累积；compile 时进签名）
	std::shared_ptr<const CompiledGraph> _snapshot; ///< compile 产物（幂等返回）
};

} // namespace DC
