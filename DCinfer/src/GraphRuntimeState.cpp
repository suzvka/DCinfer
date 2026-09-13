#include "GraphRuntimeState.h"
#include "Graph/internal/TaskExecutionState.h"

namespace DC {

GraphRuntimeState::GraphRuntimeState()
	: exec(std::make_unique<TaskExecutionDomain>()) {}

GraphRuntimeState::~GraphRuntimeState() = default;

void GraphRuntimeState::attachGraph(std::shared_ptr<const CompiledGraph> snapshot) {
	// ① 先完成全部派生状态：按源图节点集合预建节点执行闸。
	//    闸表源自源图全节点集合（含被 lowering 擦除的 wire——feedInput
	//    仍可能按名直喂未绑定的 wire，其 task 态与闸必须可达）
	exec->attachGraph(snapshot->store());

	// ② 填充快照所有者（普通写，此时尚未发布）
	_graph = std::move(snapshot);

	// ③ 最后一次性发布：release 保证 ①② 的写入对全部 acquire 读者可见。
	//    读取方只能观察到 nullptr（未发布）或完整初始化状态，不存在
	//    “快照已可见但闸表尚未就绪”的中间态。
	_frozen.store(true, std::memory_order_release);
}

} // namespace DC
