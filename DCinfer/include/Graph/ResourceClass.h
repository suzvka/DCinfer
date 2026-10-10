#pragma once

namespace DC {

/// @brief 资源类：调度器分配执行槽位的类别维度（同类共享槽位上限，异类互不干扰）。
enum class ResourceClass {
	Compute,  ///< 计算类：模型推理、数值计算
	Operator, ///< 算子类：通用节点执行（默认）
	System,   ///< 系统类：连接器、数据搬运
};

} // namespace DC
