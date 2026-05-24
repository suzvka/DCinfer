#pragma once

#include "InferNode.h"
#include "Tensor/NativeTensor.h"

#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DC {

class Tensor;

// ── 引擎注册表（线程安全单例）──
class EngineRegistry {
public:
	static EngineRegistry& instance();

	bool registerEngine(const EngineDescriptor& desc);

	// ── 接口 1：从已注册引擎创建节点 ──
	std::unique_ptr<InferNode> createNode(
		const std::string& engineType,
		const std::string& nodeName,
		const void* engineConfig = nullptr) const;

	// ── 接口 2：直接创建一个节点（无需注册引擎，自动标记为 "Builtin"）──
	std::unique_ptr<InferNode> createNode(
		const std::string& nodeName,
		InferNode::Schema schema,
		std::function<InferNode::Result(InferNode&)> fn) const;

	const EngineDescriptor* find(const std::string& engineType) const;
	bool hasEngine(const std::string& engineType) const;
	std::vector<std::string> engineTypes() const;

private:
	EngineRegistry() = default;
	mutable std::shared_mutex _mutex;
	std::unordered_map<std::string, EngineDescriptor> _engines;
};

// ── 节点工厂辅助模板 ──
// 无配置版本
// 用法：makeNodeFactory("Builtin", schema, [](InferNode& n) { ... })
template<typename F>
NodeFactory makeNodeFactory(
	std::string engineType, InferNode::Schema schema, F&& fn)
{
	return [engineType = std::move(engineType),
	        schema    = std::move(schema),
	        fn        = std::forward<F>(fn)]
	       (std::string name, const void* /*engineConfig*/)
	       -> std::unique_ptr<InferNode>
	{
		return std::make_unique<InferNode>(
				engineType, std::move(name), schema, fn);
	};
}

// 带配置版本：engineConfig 在 createNode 时拷贝为值，lambda 接收 const C&
// 用法：makeNodeFactory<int>("Mock", schema, [](InferNode& n, int magic) { ... })
template<typename C, typename F>
NodeFactory makeNodeFactory(
	std::string engineType, InferNode::Schema schema, F&& fn)
{
	return [engineType = std::move(engineType),
	        schema    = std::move(schema),
	        fn        = std::forward<F>(fn)]
	       (std::string name, const void* engineConfig)
	       -> std::unique_ptr<InferNode>
	{
		C config = engineConfig ? *static_cast<const C*>(engineConfig) : C{};
		return std::make_unique<InferNode>(engineType, std::move(name), schema,
			[fn, config = std::move(config)](InferNode& node) -> InferNode::Result {
				return fn(node, config);
			});
	};
}

} // namespace DC
