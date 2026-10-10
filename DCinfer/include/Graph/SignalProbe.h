#pragma once

#include "Node.h"

#include <string>
#include <unordered_set>
#include <vector>

namespace DC {

class GraphRuntimeView;

/// @brief 纯拓扑可达性检测：忽略信号阻断，判断 targetNodes 能否从 startNodes 沿运行时视图到达。
///
/// 供 ExecutionEngine::submit 提交期守卫使用，拓扑不可达立即抛错而非异步挂起。
/// 起点或目标为空时返回 true 放行。
bool canSatisfyTopologically(const GraphRuntimeView& view,
							 const std::vector<std::string>& startNodes,
							 const std::unordered_set<std::string>& targetNodes);

} // namespace DC
