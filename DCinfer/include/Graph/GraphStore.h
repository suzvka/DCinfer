#pragma once

#include "Node.h"
#include "InputZone.h"

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DC {

/// @brief 图拓扑存储：持有所有顶点（Node）和边（Edge），管理端口级拓扑连接。
///
/// Build → Freeze → Execute 中的拓扑载体：
/// - 构建期：由 GraphBuilder 独占持有（增删改唯一入口，冻结检查在 builder）；
/// - 冻结后：所有权移交 CompiledGraph，结构与节点运行期状态均不可变——
///   节点 task 级状态已归 task 执行域（GraphRuntimeState::exec），
///   拓扑增删改无公开入口。
///
/// 封印（seal）：compile() 时由 GraphBuilder 调用，此后全部构图方法
/// （addNode/connect/connectRaw/bindInput）抛 GraphException(Frozen)——
/// 即使调用方在冻结前保存了 GraphStore& 引用，也无法绕过冻结守卫修改
/// 快照持有的拓扑。全部构图方法以内部互斥锁串行化，封印与在飞构图操作
/// 互斥：通过校验的写完成于封印之前（纳入快照），封印后的调用确定被拒。
/// 未封印的裸 GraphStore（lowering 单测等）行为不变。
///
/// Node 不知下游，Connector 即 Node。GraphStore 对一切顶点统一处理。
class GraphStore {
public:
	// ── 端口级边 ──
	struct Edge {
		std::string srcNode;
		std::string srcPort;
		std::string dstNode;
		std::string dstPort;
	};

	GraphStore() = default;

	// ── 图构建 ──

	/// @brief  添加节点（转移所有权），返回引用供后续接线引用
	/// @throws GraphException(DuplicateNode) 若节点名为空或重名
	/// @throws GraphException(Frozen) 若拓扑已被封印（图已冻结）
	Node& addNode(std::unique_ptr<Node> node);

	/// @brief  端口级接线（默认方式）：上游输出口 → 下游输入口，
	///         自动插入广播连接器（Broadcast Connector, N=1）
	///         适用于两个业务节点之间的 1→1 直连场景
	/// @throws GraphException(NodeNotFound) 若节点不存在
	/// @throws GraphException(PortNotFound) 若端口不存在
	/// @throws GraphException(Frozen) 若拓扑已被封印（图已冻结）
	/// @return 指向自动创建的广播连接器的引用
	Node& connect(const std::string& srcNode, const std::string& srcPort,
				  const std::string& dstNode, const std::string& dstPort);

	/// @brief  端口级接线原语（internal）：直接建边，不插入连接器。
	///         仅供框架内部与测试构造裸拓扑（如 lowering 的纯 wire 链/环）使用；
	///         常规构图一律走 connect()。
	///         约束：至少有一端是连接器（两个业务节点禁止直连）
	/// @throws GraphException(NodeNotFound) 若节点不存在
	/// @throws GraphException(PortNotFound) 若端口不存在
	/// @throws GraphException(DirectConnect) 若两个非连接器节点直连
	/// @throws GraphException(Frozen) 若拓扑已被封印（图已冻结）
	void connectRaw(const std::string& srcNode, const std::string& srcPort,
					const std::string& dstNode, const std::string& dstPort);

	/// @brief  标记输入：该节点的该端口为图级输入口
	/// @param  alias  公共别名（必填；须在全部输入绑定中唯一，唯一性由 GraphBuilder 负责）
	/// @throws GraphException(Frozen) 若拓扑已被封印（图已冻结）
	void bindInput(const std::string& nodeName, const std::string& portName,
				   const std::string& alias);

	// ── 查找 ──

	/// @brief  获取节点指针（非拥有），不存在返回 nullptr
	Node* node(const std::string& name);

	/// @brief  获取节点指针（只读）
	const Node* node(const std::string& name) const;

	// ── 查询 ──

	/// @brief  节点数量
	size_t nodeCount() const { return _nodes.size(); }

	/// @brief  边数量
	size_t edgeCount() const { return _edges.size(); }

	/// @brief  获取所有节点名的列表
	std::vector<std::string> nodeNames() const;

	/// @brief  获取所有边的只读引用
	const std::vector<Edge>& edges() const { return _edges; }

	/// @brief  获取所有输入绑定的只读引用
	const std::vector<InputBinding>& inputBindings() const { return _inputZone.bindings(); }

	/// @brief  获取所有节点的只读引用
	const std::unordered_map<std::string, std::unique_ptr<Node>>& nodes() const { return _nodes; }

private:
	friend class GraphBuilder;

	/// @brief 封印拓扑（compile 时由 GraphBuilder 调用；一次性，不可回退）
	void seal();

	/// @brief 冻结门校验（调用方须持有 _mutex）；封印后抛 GraphException(Frozen)
	void _ensureMutable(const char* api) const;

	/// @brief addNode 的无锁实现（调用方须持有 _mutex；connect 内部建 wire 复用）
	Node& _addNodeImpl(std::unique_ptr<Node> node);

	std::unordered_map<std::string, std::unique_ptr<Node>> _nodes;
	std::vector<Edge> _edges;

	// 导线连接器自动命名计数器
	std::atomic<size_t> _nextWireId{0};

	// 输入区：图级输入端口声明（纯结构，无 task 级状态）
	InputZone _inputZone;

	// 构图串行化与封印标志：全部构图方法持锁；compile() 先 seal() 再只读遍历，
	// 保证封印后的拓扑对快照构建与运行期完全只读（详见类注释）
	std::mutex _mutex;
	bool _sealed = false;
};

} // namespace DC
