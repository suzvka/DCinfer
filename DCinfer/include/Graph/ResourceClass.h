#pragma once

namespace DC {

/// @brief 资源类：进程级资源调度器分配执行槽位的类别维度。
///
/// 节点经 Node::affinity() 声明所属资源类，进程级 ResourceScheduler 按类
/// 分配执行槽位（worker 预算）。隔离语义由调度器预算承载：
/// - 同类任务共享该类槽位上限（超出则排队等待，无抢占）；
/// - 异类任务互不干扰（Compute 类阻塞不占用 Operator/System 槽位）。
enum class ResourceClass {
	Compute,  ///< 计算类：模型推理、数值计算等重计算任务
	Operator, ///< 算子类：通用节点执行（默认归属）
	System,   ///< 系统类：连接器、数据搬运等基础设施任务
};

} // namespace DC
