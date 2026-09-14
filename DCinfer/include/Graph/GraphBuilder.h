#pragma once

#include "GraphStore.h"
#include "OutputZone.h"  // OutputBinding
#include "CompiledGraph.h"
#include "GraphException.h"

#include <memory>
#include <mutex>
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
/// 线程模型：全部公开方法以内部互斥锁串行化（构建 API 与 compile 互斥）——
/// 构图与冻结并发时，每个构建操作要么先于编译完成（纳入快照），要么在
/// 冻结后确定抛 Frozen，不存在"检查通过后被冻结插入"的 TOCTOU 窗口。
/// compile() 内部先封印（GraphStore::seal + 逐节点 _sealForExecution）
/// 再只读遍历，保证快照构建阶段与运行期对源图零写入。
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
	/// @param  alias  公共别名（必填；须在全部输入绑定中唯一；不参与运行时寻址）
	/// @throws GraphException(InvalidBinding) 别名为空
	/// @throws GraphException(DuplicateBinding) 别名重复
	void bindInput(const std::string& nodeName, const std::string& portName,
				   const std::string& alias);

	/// @brief  标记输出：该节点的该端口产出进入输出区（与边目的地互斥）
	/// @param  alias  公共别名（必填；须在全部输出绑定中唯一；不参与运行时寻址）
	/// @throws GraphException(InvalidBinding) 别名为空
	/// @throws GraphException(DuplicateBinding) 别名重复
	/// @note   重复绑定同一 node:port 为无操作
	void bindOutput(const std::string& nodeName, const std::string& portName,
					const std::string& alias);

	// ── 构建期内省（源图视角：冻结前读构建面；compile 后读快照，前后一致）──

	/// @brief  拓扑只读访问（compile 后返回快照持有的源图）
	const GraphStore& store() const {
		std::lock_guard lk(_mutex);
		return _snapshot ? _snapshot->store() : *_store;
	}

	/// @brief  获取节点指针（可写指针，构建期专用）
	/// @throws GraphException(Frozen) 若图已冻结（冻结后经快照只读访问）
	Node* node(const std::string& name) {
		std::lock_guard lk(_mutex);
		_ensureMutableLocked();
		return _store->node(name);
	}

	/// @brief  获取节点指针（只读）
	const Node* node(const std::string& name) const {
		std::lock_guard lk(_mutex);
		return _snapshot ? _snapshot->store().node(name) : _store->node(name);
	}

	/// @brief  节点数量
	size_t nodeCount() const {
		std::lock_guard lk(_mutex);
		return _snapshot ? _snapshot->store().nodeCount() : _store->nodeCount();
	}

	/// @brief  边数量
	size_t edgeCount() const {
		std::lock_guard lk(_mutex);
		return _snapshot ? _snapshot->store().edgeCount() : _store->edgeCount();
	}

	/// @brief  获取所有节点名的列表
	std::vector<std::string> nodeNames() const {
		std::lock_guard lk(_mutex);
		return _snapshot ? _snapshot->store().nodeNames() : _store->nodeNames();
	}

	/// @brief  获取所有边的只读引用
	const std::vector<Edge>& edges() const {
		std::lock_guard lk(_mutex);
		return _snapshot ? _snapshot->store().edges() : _store->edges();
	}

	/// @brief  获取所有输入绑定的只读引用
	const std::vector<InputBinding>& inputBindings() const {
		std::lock_guard lk(_mutex);
		return _snapshot ? _snapshot->store().inputBindings() : _store->inputBindings();
	}

	/// @brief  获取所有输出绑定的只读引用
	const std::vector<OutputBinding>& outputBindings() const {
		std::lock_guard lk(_mutex);
		return _snapshot ? _snapshot->signature().outputs : _outputBindings;
	}

	// ── 冻结 ──

	/// @brief  编译构建面为不可变快照（幂等：重复调用返回同一快照）。
	///
	/// 构建 GraphSignature（输入/输出绑定快照）→ 拓扑所有权移交快照。
	/// 此后所有构建 API 抛 GraphException(Frozen)。
	std::shared_ptr<const CompiledGraph> compile();

private:
	/// @brief  构建守卫：冻结后调用构建 API 抛 GraphException(Frozen)
	///         （调用方须持有 _mutex）
	void _ensureMutableLocked() const;

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

	// 构建面串行化：全部公开方法持锁；compile 与构建 API 互斥（详见类注释）
	mutable std::mutex _mutex;
};

} // namespace DC
