#include "GraphRuntimeState.h"
#include "Graph/internal/TaskExecutionState.h"

namespace DC {

GraphRuntimeState::GraphRuntimeState()
	: exec(std::make_unique<TaskExecutionDomain>()) {}

GraphRuntimeState::~GraphRuntimeState() = default;

void GraphRuntimeState::attachGraph(std::shared_ptr<const CompiledGraph> snapshot) {
	graph = std::move(snapshot);
	// 闸表源自源图全节点集合（含被 lowering 擦除的 wire——feedInput 仍可能
	// 按名直喂未绑定的 wire，其 task 态与闸必须可达）
	exec->attachGraph(graph->store());
}

} // namespace DC
