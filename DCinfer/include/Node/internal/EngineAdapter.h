#pragma once

#include <functional>
#include <memory>
#include <string>

#include "../Node.h"

namespace DC {

// ── 前向声明（定义见 Graph/EngineRegistry.h）──
struct EngineDescriptor;
class EngineInstance;

/// @brief 引擎适配器：持有引擎实例共享句柄，编排引擎生命周期钩子。
///
/// 彻底消除 EngineRegistry::instance() 的隐式依赖：
/// EngineDescriptor* 在构造时注入，Builtin 节点传入 nullptr。
/// 句柄由节点持有：节点存活 ⇒ 引擎实例存活，注册表释放仅移除缓存条目。
///
/// 编排执行相位协议：preRun → RunFn(框架外) → 成功: synchronize → postRun ｜ 失败: onError。
/// 相位顺序契约见 EngineDescriptor::ExecutionPhases（Graph/EngineRegistry.h）。
class EngineAdapter {
public:
	/// @brief 构造引擎适配器（两个参数都可以为空）。
	/// @param instance    引擎实例共享句柄（可空）。
	/// @param descriptor  引擎描述符指针（非拥有，构造时注入）。
	EngineAdapter(std::shared_ptr<EngineInstance> instance, const EngineDescriptor* descriptor);

	// ── 钩子 ──

	/// @brief  preRun 钩子：推理前引擎级准备（warmup、session 配置等；
	///         拿不到 task 输入——输入绑定在 RunFn 内经 TensorConverter 完成）。
	void preRun() const;

	/// @brief  synchronize 钩子：确保异步引擎计算已完成。
	void synchronize() const;

	/// @brief  postRun 钩子：同步后的后处理（D2H 传输等）。
	void postRun(class Node::RunContext& ctx) const;

	/// @brief  onError 钩子：任一执行相位失败时重置引擎状态（尽力而为）。
	void onError() const;

	// ── 访问器 ──

	/// @brief  获取 TensorConverter 钩子指针（可能为 nullptr）。
	const struct TensorConverter* converter() const;

	/// @brief  获取 EngineDescriptor 指针。
	const EngineDescriptor* descriptor() const { return _desc; }

	/// @brief  获取引擎原生指针。
	void* engine() const;

	/// @brief  获取 EngineInstance 指针（借用：适配器持有句柄保证其存活）。
	const EngineInstance* instance() const { return _instance.get(); }

private:
	std::shared_ptr<EngineInstance> _instance; ///< 拥有句柄：节点存活 ⇒ 引擎实例存活
	const EngineDescriptor* _desc;
};

} // namespace DC
