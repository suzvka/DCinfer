#pragma once

namespace DC {

/// @brief 资源类：调度器分配执行槽位的类别维度；同类共享槽位上限，异类互不干扰。
enum class ResourceClass {
	Compute,
	Operator,
	System,
};

} // namespace DC
