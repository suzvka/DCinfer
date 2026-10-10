#include "SwitchNode.h"

#include <stdexcept>
#include <utility>

namespace DC {

std::pair<std::unique_ptr<Node>, SwitchHandle>
createSwitchNode(std::string name,
                 Node::Schema schema,
                 std::vector<SwitchCandidate> candidates,
                 ResourceClass affinity) {
    if (candidates.empty())
        throw std::invalid_argument("createSwitchNode: '" + name + "' has no candidates");

    auto sharedIndex = std::make_shared<std::atomic<size_t>>(0);
    SwitchHandle handle{sharedIndex};

    auto ownedCandidates = std::make_shared<std::vector<SwitchCandidate>>(std::move(candidates));

    Node::RunFn fn = [sharedIndex, ownedCandidates](Node::RunContext& ctx) -> Node::Result {
        size_t idx = sharedIndex->load(std::memory_order_acquire);
        if (idx >= ownedCandidates->size()) {
            return ctx.failure(Node::Status::InternalError,
                "SwitchNode '" + ctx.name() + "': active index " + std::to_string(idx) +
                " out of range (" + std::to_string(ownedCandidates->size()) + " candidates)");
        }

        const auto& candidate = (*ownedCandidates)[idx];

        if (candidate.preAdapt)
            candidate.preAdapt(ctx);

        auto result = candidate.run(ctx);

        if (result.ok() && candidate.postAdapt)
            candidate.postAdapt(ctx);

        return result;
    };

    auto node = std::make_unique<Node>("Switch", std::move(name), std::move(schema), std::move(fn), affinity);

    return {std::move(node), std::move(handle)};
}

} // namespace DC
