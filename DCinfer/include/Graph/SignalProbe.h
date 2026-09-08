#pragma once

#include "Node.h"

#include <string>

namespace DC {

class GraphStore;
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
/// 复杂度 O(V+E)/次，第一版不做缓存（signal version 失效缓存列为后续优化）。
/// @param  store      图拓扑（冻结快照或构建期存储）
/// @param  signature  图级签名（输出绑定来源）
/// @param  taskId     查询的 task（taskId 空间贯穿父子边界）
/// @return true = 存在通路（声明可能满足）；false = 全部输出绑定均不可达
bool canSatisfyDeclarations(const GraphStore& store, const GraphSignature& signature,
							const Node::TaskId& taskId);

} // namespace DC
