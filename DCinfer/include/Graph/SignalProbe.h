#pragma once

#include "Node.h"

#include <string>
#include <unordered_set>
#include <vector>

namespace DC {

class GraphStore;
class GraphRuntimeView; ///< 定义见 CompiledGraph.h（运行时视图：lowering 后参与调度的点/边）
struct GraphSignature; ///< 与 GraphSignature.h 中的 struct 定义保持一致（MSVC class/struct 修饰名敏感）

/// @brief  声明通路检测：剔除被信号阻塞的节点后，
///         输入绑定算子是否仍存在可达输出绑定端口的通路。
///
/// 语义：与 ExecutionEngine::_propagateFrom 的"阻塞跳过边"同构，
/// 是子图内部传播的静态预演（dry-run）。供子图节点（exportNode 产物）
/// 注册 blockedOverride 时使用——内部全部输出绑定均不可达时，子图边界
/// 应答阻塞（执行必然无法满足任何声明）。
///
/// 目标集采用图级静态输出绑定（GraphSignature::outputs），而非 task 级声明：
/// 父级查询 isBlocked 时内部尚未 submit，声明不可知；而 exportNode 的
/// RunFn 每次执行都会声明全部输出绑定，故绑定集合与内部实际声明一致。
///
/// 复杂度 O(V+E)/次。
/// @param  store      图拓扑（冻结快照或构建期存储）
/// @param  signature  图级签名（输出绑定来源）
/// @param  taskId     查询的 task（taskId 空间贯穿父子边界）
/// @return true = 存在通路（声明可能满足）；false = 全部输出绑定均不可达
bool canSatisfyDeclarations(const GraphStore& store, const GraphSignature& signature,
							const Node::TaskId& taskId);

/// @brief  纯拓扑声明通路检测：声明目标节点是否从给定起点集合沿运行时视图可达。
///
/// 与 canSatisfyDeclarations 的区别：**忽略信号阻断**（不查 isBlocked），
/// 仅回答"构图/断链错误"——供 ExecutionEngine::submit 提交期守卫使用：
/// 目标纯拓扑不可达即必然无法满足声明，应立即抛错而非异步挂起。
/// 信号阻断属合法运行期状态，由宿主 wait+cancel 解围，不在此判定。
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
