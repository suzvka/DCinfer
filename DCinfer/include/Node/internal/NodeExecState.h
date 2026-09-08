#pragma once

#include "TaskBuffer.h"
#include "SlotWorkspace.h"

#include <memory>

namespace DC {

struct NodeSchema; // 定义见 Node.h

/// @brief 单节点在某次 task 中的执行态：task IO 缓冲 + 工作槽位。
///
/// 原 Node 内嵌的 TaskBuffer/SlotWorkspace 迁入地——Node 由此退化为
/// 纯 plan 对象（Schema/RunFn/句柄），运行期可交出 const。
/// buffer 与 workspace 由同一 schema 构造，二者一致性由本类型保证。
struct NodeExecState {
	TaskBuffer buffer;
	std::unique_ptr<SlotWorkspace> workspace;

	explicit NodeExecState(const NodeSchema& schema)
		: workspace(std::make_unique<SlotWorkspace>(schema)) {}
};

} // namespace DC
