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

/// @brief 致死锥（kill cone）：反向收集可经信号致目标节点于死地的全部上游节点（含自身）。
///
/// 死区语义 dead(n) = gated(n) OR ∃ 必需无默认值生产者 p: dead(p)：节点就绪是必需输入口
/// 的 AND，任一必需生产者被信号关闭则输入永凑不齐、节点永不运行。沿“必需且无默认值”
/// 输入边反向遍历即得致死锥；可选/默认值口不参与就绪 AND，不传播死区。
/// 拓扑冻结 ⇒ 锥静态可算：运行期只需对锥内节点现读 isBlocked(taskId)，任一阻塞即声明不活跃。
/// 目标不存在或无上游时返回仅含目标自身（可为空）的列表；环以 visited 兜底。
std::vector<const Node*> killCone(const GraphRuntimeView& view, const std::string& targetNode);

} // namespace DC
