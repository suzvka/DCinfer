#pragma once

#include "Node.h"
#include "InputZone.h"

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DC {

/// @brief 图拓扑存储：持有所有顶点（Node）和边（Edge），管理端口级拓扑连接。
///
/// 从 InferGraph 提取的独立组件，只负责图结构的增删查，
/// 不涉及执行、调度、数据传播。
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

	/// @brief  添加节点（转移所有权），返回裸指针供后续接线引用
	/// @return 指向已存入图内节点的非拥有指针；若节点名为空或重名则返回 nullptr
	Node* addNode(std::unique_ptr<Node> node);

	/// @brief  端口级接线：上游输出口 → 下游输入口
	///         约束：至少有一端是连接器（两个业务节点禁止直连）
	/// @return true 表示两端节点和端口均存在且接线成功
	bool connect(const std::string& srcNode, const std::string& srcPort,
				 const std::string& dstNode, const std::string& dstPort);

	/// @brief  快捷接线：自动匹配上游所有输出口到下游同名的输入口
	/// @return 成功匹配的端口对数
	size_t connectAll(const std::string& srcNode, const std::string& dstNode);

	/// @brief  接线：在两个节点间自动插入广播连接器（Broadcast Connector, N=1）
	///         适用于两个业务节点之间的 1→1 直连场景
	/// @return 指向自动创建的广播连接器的非拥有指针；若节点或端口不存在则返回 nullptr
	Node* wire(const std::string& srcNode, const std::string& srcPort,
			   const std::string& dstNode, const std::string& dstPort);

	/// @brief  标记输入：该节点的该端口为图级输入口
	void bindInput(const std::string& nodeName, const std::string& portName);

	// ── 查找 ──

	/// @brief  查找节点（非拥有），不存在返回 nullptr
	Node* findNode(const std::string& name);

	/// @brief  查找节点（只读）
	const Node* findNode(const std::string& name) const;

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

	/// @brief  获取所有节点的只读引用（供执行引擎遍历）
	const std::unordered_map<std::string, std::unique_ptr<Node>>& nodes() const { return _nodes; }

	/// @brief  获取所有节点的可写引用（供执行引擎遍历清理）
	std::unordered_map<std::string, std::unique_ptr<Node>>& nodes() { return _nodes; }

private:
	std::unordered_map<std::string, std::unique_ptr<Node>> _nodes;
	std::vector<Edge> _edges;

	// 导线连接器自动命名计数器
	std::atomic<size_t> _nextWireId{0};

	// 输入区：图级输入端口声明（纯结构，无 task 级状态）
	InputZone _inputZone;
};

} // namespace DC
