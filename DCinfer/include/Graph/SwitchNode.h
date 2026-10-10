#pragma once

#include "Node.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace DC {

/// @brief 可切换候选描述：每个候选提供执行函数与可选的输入/输出整流钩子，
///        用于适配 IO 略有差异但语义等价的同义实现。
struct SwitchCandidate {
    std::string label; // 人读标签，不参与逻辑判定

    /// 候选执行函数（等价于普通 Node 的 RunFn）。
    std::function<Node::Result(Node::RunContext&)> run;

    /// 输入整流（可选）：run 前调用，把图级统一输入映射到该候选端口。
    std::function<void(Node::RunContext&)> preAdapt;

    /// 输出整流（可选）：run 成功后调用，把候选输出映射回图级统一输出端口。
    std::function<void(Node::RunContext&)> postAdapt;
};

/// @brief 运行时切换句柄：不改图拓扑即可切换 active 候选；select() 任意线程可调用。
///        切换对下一次数据派发生效，不影响正在执行的 task。
struct SwitchHandle {
    std::shared_ptr<std::atomic<size_t>> index;

    /// @brief 切换到第 i 个候选（0-based；调用者保证 i 合法）。
    void select(size_t i) const { index->store(i, std::memory_order_release); }

    /// @brief 查询当前 active 候选索引。
    size_t current() const { return index->load(std::memory_order_acquire); }
};

/// @brief 创建可切换节点：单个普通节点（一入一出），内部持 N 个候选，经 SwitchHandle 原子切换；
///        切换瞬间在飞 task 用旧候选完成，新 task 用新候选。
/// @return (node, handle)：node 入图，handle 外部持有用于切换。
/// @throws std::invalid_argument 若 candidates 为空
std::pair<std::unique_ptr<Node>, SwitchHandle>
createSwitchNode(std::string name,
                 Node::Schema schema,
                 std::vector<SwitchCandidate> candidates,
                 ResourceClass affinity = ResourceClass::Compute);

} // namespace DC
