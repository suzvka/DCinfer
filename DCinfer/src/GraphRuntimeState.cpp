#include "GraphRuntimeState.h"
#include "Graph/internal/TaskExecutionState.h"

namespace DC {

GraphRuntimeState::GraphRuntimeState()
	: exec(std::make_unique<TaskExecutionDomain>()) {}

GraphRuntimeState::~GraphRuntimeState() = default;

void GraphRuntimeState::attachGraph(std::shared_ptr<const CompiledGraph> snapshot) {
	// 预建执行闸：按源图全节点集合建表，含被 lowering 擦除的 wire；
	// feedInput 仍可能按名直喂未绑定节点。
	exec->attachGraph(snapshot->store());

	_graph = std::move(snapshot);

	// 最后 release 发布：以上写入对 acquire 读者全部可见，不存在半初始化中间态。
	_frozen.store(true, std::memory_order_release);
}

} // namespace DC
