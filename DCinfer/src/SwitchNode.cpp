#include "SwitchNode.h"

#include <stdexcept>
#include <utility>

namespace DC {

std::pair<std::unique_ptr<Node>, SwitchHandle>
createSwitchNode(std::string name,
                 Node::Schema schema,
                 std::vector<SwitchCandidate> candidates,
                 ThreadPoolAffinity affinity) {
    if (candidates.empty())
        throw std::invalid_argument("createSwitchNode: '" + name + "' has no candidates");

    // 共享原子索引：初始指向第 0 个候选
    auto sharedIndex = std::make_shared<std::atomic<size_t>>(0);
    SwitchHandle handle{sharedIndex};

    // 将候选列表移入 lambda 捕获（生命周期由 Node::RunFn 闭包持有）
    auto ownedCandidates = std::make_shared<std::vector<SwitchCandidate>>(std::move(candidates));

    // 构造 RunFn：读索引 → preAdapt → run → postAdapt
    Node::RunFn fn = [sharedIndex, ownedCandidates](Node::RunContext& ctx) -> Node::Result {
        size_t idx = sharedIndex->load(std::memory_order_acquire);
        if (idx >= ownedCandidates->size()) {
            return ctx.failure(Node::Status::InternalError,
                "SwitchNode '" + ctx.name() + "': active index " + std::to_string(idx) +
                " out of range (" + std::to_string(ownedCandidates->size()) + " candidates)");
        }

        const auto& candidate = (*ownedCandidates)[idx];

        // 输入整流
        if (candidate.preAdapt)
            candidate.preAdapt(ctx);

        // 执行候选计算
        auto result = candidate.run(ctx);

        // 输出整流（仅在成功时执行）
        if (result.ok() && candidate.postAdapt)
            candidate.postAdapt(ctx);

        return result;
    };

    auto node = std::make_unique<Node>("Switch", std::move(name), std::move(schema), std::move(fn), affinity);

    return {std::move(node), std::move(handle)};
}

} // namespace DC
