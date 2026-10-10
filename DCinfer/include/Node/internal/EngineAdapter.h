#pragma once

#include <functional>
#include <memory>
#include <string>

#include "../Node.h"

namespace DC {

// 前向声明（定义见 Graph/EngineRegistry.h）
struct EngineDescriptor;
class EngineInstance;

/// @brief 引擎适配器：持有引擎实例共享句柄，编排引擎生命周期钩子。
///
/// EngineDescriptor* 构造时注入（Builtin 节点为 nullptr）；句柄由节点持有，
/// 节点存活 ⇒ 实例存活。相位顺序：preRun → RunFn → 成功 synchronize→postRun / 失败 onError。
class EngineAdapter {
public:
	EngineAdapter(std::shared_ptr<EngineInstance> instance, const EngineDescriptor* descriptor);

	/// @brief preRun 钩子：推理前引擎级准备（拿不到 task 输入）。
	void preRun() const;

	/// @brief synchronize 钩子：等待异步计算完成。
	void synchronize() const;

	/// @brief postRun 钩子：同步后处理（D2H 传输等）。
	void postRun(class Node::RunContext& ctx) const;

	/// @brief onError 钩子：执行失败时重置引擎状态（尽力而为）。
	void onError() const;

	/// @brief 获取 TensorConverter 钩子指针（可能为 nullptr）。
	const struct TensorConverter* converter() const;

	const EngineDescriptor* descriptor() const { return _desc; }

	void* engine() const;

	/// @brief 获取 EngineInstance 指针（借用；适配器持有句柄保证存活）。
	const EngineInstance* instance() const { return _instance.get(); }

private:
	std::shared_ptr<EngineInstance> _instance;
	const EngineDescriptor* _desc;
};

} // namespace DC
