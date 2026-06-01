#pragma once

#include <functional>
#include <memory>
#include <string>

#include "../Node.h"

namespace DC {

// ── 前向声明（定义见 Graph/EngineRegistry.h）──
struct EngineDescriptor;
class EngineInstance;

/// @brief 引擎适配器：封装引擎实例的非拥有引用，编排引擎生命周期钩子。
///
/// 彻底消除 EngineRegistry::instance() 的隐式依赖：
/// EngineDescriptor* 在构造时注入，Builtin 节点传入 nullptr。
///
/// 编排 preRun → onError → synchronize → postRun 的调用顺序。
class EngineAdapter {
public:
	/// @brief 构造引擎适配器（两个参数都可以为 nullptr）。
	/// @param instance    引擎实例指针（非拥有）。
	/// @param descriptor  引擎描述符指针（非拥有，构造时注入）。
	EngineAdapter(EngineInstance* instance, const EngineDescriptor* descriptor);

	// ── 钩子 ──

	/// @brief  preRun 钩子：推理前准备（I/O 绑定、warmup 等）。
	void preRun() const;

	/// @brief  synchronize 钩子：确保异步引擎计算已完成。
	void synchronize() const;

	/// @brief  postRun 钩子：同步后的后处理（D2H 传输等）。
	void postRun(class Node::RunContext& ctx) const;

	/// @brief  onError 钩子：执行失败时重置引擎状态。
	void onError() const;

	// ── 访问器 ──

	/// @brief  获取 TensorConverter 钩子指针（可能为 nullptr）。
	const struct TensorConverter* converter() const;

	/// @brief  获取 EngineDescriptor 指针。
	const EngineDescriptor* descriptor() const { return _desc; }

	/// @brief  获取引擎原生指针。
	void* engine() const;

	/// @brief  获取 EngineInstance 指针。
	const EngineInstance* instance() const { return _instance; }

private:
	EngineInstance* _instance;
	const EngineDescriptor* _desc;
};

} // namespace DC
