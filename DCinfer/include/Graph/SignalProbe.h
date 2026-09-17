#pragma once

#include "Node.h"

#include <string>
#include <unordered_set>
#include <vector>

namespace DC {

class GraphRuntimeView; ///< 定义见 CompiledGraph.h（运行时视图：lowering 后参与调度的点/边）

/// @brief  纯拓扑声明通路检测：声明目标节点是否从给定起点集合沿运行时视图可达。
///
/// 语义：忽略信号阻断（不查 isBlocked），仅回答"构图/断链错误"——供
/// ExecutionEngine::submit 提交期守卫使用：目标纯拓扑不可达即必然无法满足
/// 声明，应立即抛错而非异步挂起。信号阻断属合法运行期状态，由宿主
/// wait+cancel 解围，不在此判定。
///
/// 起点集合由调用方按 task 上下文计算：已注入输入的节点 ∪ 图级输入绑定节点。
/// 起点或目标为空时返回 true（无可判别对象，放行）。
///
/// @param  view        运行时视图（与 _propagateFrom 实际传播面同构）
/// @param  startNodes  可能先行执行的节点（已注入输入/输入绑定）
/// @param  targetNodes 声明输出所在节点集合
/// @return true = 存在从任一起点到任一目标的拓扑通路；false = 全部不可达
bool canSatisfyTopologically(const GraphRuntimeView& view,
							 const std::vector<std::string>& startNodes,
							 const std::unordered_set<std::string>& targetNodes);

} // namespace DC
