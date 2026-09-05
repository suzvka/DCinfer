#pragma once

#include "Node.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace DC {

/// @brief 可切换候选描述：一个候选代表一种"同义实现"。
///
/// 每个候选提供自己的执行函数与可选的输入/输出整流钩子，
/// 用于适配 IO 略有差异但语义等价的同义实现。
struct SwitchCandidate {
    std::string label; // 人读标签（由调用方自定义，库内不参与逻辑判定）

    /// 候选执行函数：等价于普通 Node 的 RunFn。
    /// 从 ctx 读输入、执行计算、写输出。
    std::function<Node::Result(Node::RunContext&)> run;

    /// 输入整流（可选）：在 run 之前调用，用于将图级统一输入映射到
    /// 该候选实际需要的端口（转置、重命名、填充额外输入等）。
    std::function<void(Node::RunContext&)> preAdapt;

    /// 输出整流（可选）：在 run 之后调用（仅当 run 成功时），用于将
    /// 候选的实际输出映射回图级统一输出端口。
    std::function<void(Node::RunContext&)> postAdapt;
};

/// @brief 运行时切换句柄：外部持有，用于在不改变图拓扑的前提下切换 active 候选。
///
/// 线程安全：select() 可在任意线程调用；RunFn 内通过 load() 读取当前索引。
/// 语义：切换对"下一次数据派发"生效，不影响正在执行的 task。
struct SwitchHandle {
    std::shared_ptr<std::atomic<size_t>> index;

    /// @brief  切换到第 i 个候选（0-based）。
    /// @note   若 i >= 候选数则 undefined behavior；调用者保证合法性。
    void select(size_t i) const { index->store(i, std::memory_order_release); }

    /// @brief  查询当前 active 候选索引。
    size_t current() const { return index->load(std::memory_order_acquire); }
};

/// @brief 创建可切换节点（SwitchNode）。
///
/// SwitchNode 在图中表现为单个普通节点（一条入边 + 一条出边），
/// 内部持有 N 个候选实现，运行时通过 SwitchHandle 原子切换当前生效的候选。
/// 切换瞬间：正在执行的 task 使用旧候选完成，新到达的 task 使用新候选。
///
/// @param  name        节点名（图中唯一标识）
/// @param  schema      对外统一的端口契约（所有候选共享同一 IO 接口）
/// @param  candidates  候选列表（至少 1 个）
/// @param  affinity    线程池归属（默认 Compute）
/// @return (node, handle)：node 入图，handle 外部持有用于切换
///
/// @throws std::invalid_argument 若 candidates 为空
std::pair<std::unique_ptr<Node>, SwitchHandle>
createSwitchNode(std::string name,
                 Node::Schema schema,
                 std::vector<SwitchCandidate> candidates,
                 ThreadPoolAffinity affinity = ThreadPoolAffinity::Compute);

} // namespace DC
