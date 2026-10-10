#pragma once

#include "GraphStore.h"
#include "OutputZone.h"
#include "CompiledGraph.h"
#include "GraphException.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace DC {

/// @brief 图构建器：构建期的唯一可写面。
///
/// compile 产出不可变 CompiledGraph 快照并移交拓扑所有权；冻结后所有构建 API 抛
/// GraphException(Frozen)。公开方法以内部互斥锁串行化，构图与冻结互斥，无 TOCTOU 窗口。
/// 拓扑演进须重建 GraphBuilder 并重新 compile。
class GraphBuilder {
public:
	using Edge = GraphStore::Edge;

	GraphBuilder() = default;
	~GraphBuilder() = default;

	GraphBuilder(const GraphBuilder&) = delete;
	GraphBuilder& operator=(const GraphBuilder&) = delete;

	/// @brief 添加节点并转移所有权；空名或重名抛 DuplicateNode。
	Node& addNode(std::unique_ptr<Node> node);

	/// @brief 端口级接线：自动插入广播连接器；同源口再次 connect 扩容扇出。
	Node& connect(const std::string& srcNode, const std::string& srcPort,
				  const std::string& dstNode, const std::string& dstPort);

	/// @brief 标记图级输入口；alias 必填且唯一。
	void bindInput(const std::string& nodeName, const std::string& portName,
				   const std::string& alias);

	/// @brief 标记图级输出绑定；alias 必填且唯一，重复绑定同一 node:port 为无操作。
	/// @throws GraphException(NonTerminalPort) 端口有出边。
	void bindOutput(const std::string& nodeName, const std::string& portName,
					const std::string& alias);

	/// @brief 拓扑只读访问；compile 后读快照。
	const GraphStore& store() const {
		std::lock_guard lk(_mutex);
		return _snapshot ? _snapshot->store() : *_store;
	}

	/// @brief 获取节点可写指针；构建期专用，冻结后抛 Frozen。
	Node* node(const std::string& name) {
		std::lock_guard lk(_mutex);
		_ensureMutableLocked();
		return _store->node(name);
	}

	/// @brief 获取只读节点指针。
	const Node* node(const std::string& name) const {
		std::lock_guard lk(_mutex);
		return _snapshot ? _snapshot->store().node(name) : _store->node(name);
	}

	size_t nodeCount() const {
		std::lock_guard lk(_mutex);
		return _snapshot ? _snapshot->store().nodeCount() : _store->nodeCount();
	}

	size_t edgeCount() const {
		std::lock_guard lk(_mutex);
		return _snapshot ? _snapshot->store().edgeCount() : _store->edgeCount();
	}

	std::vector<std::string> nodeNames() const {
		std::lock_guard lk(_mutex);
		return _snapshot ? _snapshot->store().nodeNames() : _store->nodeNames();
	}

	/// @brief 获取所有边的值副本，隔离并发构图导致的引用悬空。
	std::vector<Edge> edges() const {
		std::lock_guard lk(_mutex);
		return _snapshot ? _snapshot->store().edges() : _store->edges();
	}

	std::vector<InputBinding> inputBindings() const {
		std::lock_guard lk(_mutex);
		return _snapshot ? _snapshot->store().inputBindings() : _store->inputBindings();
	}

	const std::vector<OutputBinding>& outputBindings() const {
		std::lock_guard lk(_mutex);
		return _snapshot ? _snapshot->signature().outputs : _outputBindings;
	}

	/// @brief 编译为不可变快照；幂等，此后构建 API 抛 Frozen。
	std::shared_ptr<const CompiledGraph> compile();

private:
	/// @brief 构建守卫；调用方须持有 _mutex。
	void _ensureMutableLocked() const;

	/// @brief 输出绑定不变量校验，compile 期封印前调用：绑定端口必须无出边。
	/// @throws GraphException(NonTerminalPort)
	void _validateOutputBindings() const;

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
	std::vector<OutputBinding> _outputBindings;
	std::shared_ptr<const CompiledGraph> _snapshot;

	mutable std::mutex _mutex;
};

} // namespace DC
