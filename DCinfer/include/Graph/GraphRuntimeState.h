#pragma once

#include "GraphStore.h"
#include "OutputZone.h"
#include "SignalStore.h"
#include "ErrorTracker.h"

#include <memory>

namespace DC {

/// @brief 图运行时状态：异步飞行任务的共享生命周期锚点。
///
/// InferGraph 与全部飞行中的任务 lambda / TaskGate / 超时看门狗共同持有
/// 本状态的 shared_ptr：图对象先行析构时，已提交任务所需的
/// 拓扑 / 输出区 / 信号 / 诊断组件仍然存活——异步任务的生命周期安全
/// 由所有权直接保证，不再依赖成员声明顺序与析构顺序约定。
struct GraphRuntimeState {
	GraphStore   store;   ///< 图拓扑存储（节点 + 边 + 输入绑定）
	OutputZone   output;  ///< 输出聚合（绑定 + 声明 + 结果 artifact）
	std::shared_ptr<SignalStore> signals = std::make_shared<SignalStore>(); ///< 信号仓库（节点经 bindSignal 共享持有）
	ErrorTracker errors;  ///< 异步执行诊断收集
};

} // namespace DC
