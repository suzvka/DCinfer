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

/// @brief 图拓扑存储：持有所有顶点（Node）与端口级边（Edge）。
///
/// 构建期由 GraphBuilder 独占持有；compile() 时经 seal() 封印，此后全部构图方法
/// 抛 GraphException(Frozen)——即使外部保存了 GraphStore& 引用也无法绕过。
/// 全部构图方法以内部互斥锁串行化；冻结后所有权移交 CompiledGraph。
class GraphStore {
public:
	struct Edge {
		std::string srcNode;
		std::string srcPort;
		std::string dstNode;
		std::string dstPort;
	};

	GraphStore() = default;

	// ── 图构建 ──

	/// @brief 添加节点（转移所有权）；空名或重名抛 DuplicateNode。
	Node& addNode(std::unique_ptr<Node> node);

	/// @brief 端口级接线：上游输出口 → 下游输入口，自动插入广播连接器（N=1）。
	///        同一输出口再次 connect 时原地扩容为 N 路广播扇出。
	/// @return 承载该连接的广播连接器引用（扩容返回同一对象）。
	Node& connect(const std::string& srcNode, const std::string& srcPort,
				  const std::string& dstNode, const std::string& dstPort);

	/// @brief 端口级接线原语（internal）：直接建边，不插入连接器。
	///        约束：至少一端是连接器，两个非连接器节点直连抛 DirectConnect。
	void connectRaw(const std::string& srcNode, const std::string& srcPort,
					const std::string& dstNode, const std::string& dstPort);

	/// @brief 标记图级输入口（alias 必填且唯一；不参与运行时寻址）。
	void bindInput(const std::string& nodeName, const std::string& portName,
				   const std::string& alias);

	/// @brief 获取节点指针（非拥有；不存在返回 nullptr）。
	Node* node(const std::string& name);

	/// @brief 获取节点指针（只读）。
	const Node* node(const std::string& name) const;

	size_t nodeCount() const { return _nodes.size(); }
	size_t edgeCount() const { return _edges.size(); }
	std::vector<std::string> nodeNames() const;

	/// @brief 所有边（内部引用版；仅限持锁或封印后调用，公开路径走值副本）。
	const std::vector<Edge>& edges() const { return _edges; }

	std::vector<InputBinding> inputBindings() const { return _inputZone.bindings(); }

	/// @brief 所有节点表（内部引用版；约束同 edges()）。
	const std::unordered_map<std::string, std::unique_ptr<Node>>& nodes() const { return _nodes; }

private:
	friend class GraphBuilder;

	/// @brief 封印拓扑（compile 时调用；一次性，不可回退）。
	void seal();

	/// @brief 冻结门校验；调用方须持有 _mutex。
	void _ensureMutable(const char* api) const;

	/// @brief addNode 的无锁实现；调用方须持有 _mutex。
	Node& _addNodeImpl(std::unique_ptr<Node> node);

	std::unordered_map<std::string, std::unique_ptr<Node>> _nodes;
	std::vector<Edge> _edges;

	std::atomic<size_t> _nextWireId{0};

	InputZone _inputZone;

	// 构图串行化与封印标志：全部构图方法持锁；mutable 因 const 查询也需取锁。
	mutable std::mutex _mutex;
	bool _sealed = false;
};

} // namespace DC
