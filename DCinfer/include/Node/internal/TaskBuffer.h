#pragma once

#include "Value.h"
#include "NodeException.h"

#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DC {

class Node; // 前向声明
struct NodeSchema; // 前向声明（定义见 Node.h）

/// @brief 线程安全的 task 级 I/O 缓冲区管理器。
///
/// 封装原 Node 中的 _taskInputs / _taskOutputs / _bufferMutex，
/// 提供线程安全的 setInput / getOutput / isReady / 生命周期管理。
/// 所有需要 Schema 信息的操作通过参数传入，避免头文件循环依赖。
class TaskBuffer {
public:
	using TaskId = std::string;
	using TaskData = Value;

	TaskBuffer() = default;

	// ── 输入 ──

	/// @brief  单端口写入（Value），仅写入缓冲，不触发执行。
	/// @throws NodeException(PortNotFound) 若端口名不存在于 Schema 中。
	void setInput(const TaskId& taskId, const std::string& portName, Value data,
				  const NodeSchema& schema);

	/// @brief  批量写入，预校验所有端口名。
	void setInputBatch(const TaskId& taskId,
					   std::unordered_map<std::string, TaskData> inputs,
					   const NodeSchema& schema);

	// ── 就绪判断 ──

	/// @brief  查询指定任务是否所有必需输入已就绪（含默认值）。
	bool isReady(const TaskId& taskId, const NodeSchema& schema) const;

	// ── 输出 ──

	/// @brief  查询指定任务是否已产出指定输出端口的数据。
	bool hasOutput(const TaskId& taskId, const std::string& name) const;

	/// @brief  消费式取出输出数据（调用后缓冲区该槽位清空）。
	/// @throws NodeException(TaskNotFound) 若任务不存在。
	/// @throws NodeException(OutputNotProduced) 若输出端口为空。
	Value getOutput(const TaskId& taskId, const std::string& name);

	/// @brief  只读查看输出（不消费，数据保留在缓冲区）。
	const Value& peekOutput(const TaskId& taskId, const std::string& name) const;

	/// @brief  批量消费所有输出。
	std::unordered_map<std::string, TaskData> collectOutputs(const TaskId& taskId);

	// ── 生命周期 ──

	/// @brief  查询指定 task 是否存在（输入或输出缓冲区非空）。
	bool hasTask(const TaskId& taskId) const;

	/// @brief  清除指定 task 的所有 IO 缓冲区。
	void clearTask(const TaskId& taskId);

	/// @brief  当前活跃任务数量。
	size_t taskCount() const;

	// ── 批量传输（供 ExecutionPipeline 使用）──

	/// @brief  将 task 输入缓冲区数据 move 到工作槽位（含默认值回退）。
	///         调用前持有锁，操作完成后重置缓冲区槽位。
	void drainInputsTo(const TaskId& taskId, class SlotWorkspace& workspace,
					   const NodeSchema& schema);

	/// @brief  将工作槽位数据收集到 task 输出缓冲区。
	///         确保输出缓冲区条目存在，从工作槽位 take Value 填入。
	void fillOutputsFrom(const TaskId& taskId, class SlotWorkspace& workspace,
						 const NodeSchema& schema);

	/// @brief  验证指定 task 的所有必需输出端口已产生（在 fillOutputsFrom 之后调用）。
	bool validateOutputs(const TaskId& taskId, const NodeSchema& schema) const;

	/// @brief  仅擦除指定 task 的输入缓冲区（输出保留供调用方拉取）。
	void eraseInputs(const TaskId& taskId);

private:
	using TaskBufferEntry = std::unordered_map<std::string, std::optional<TaskData>>;
	using TaskBufferMap = std::unordered_map<TaskId, TaskBufferEntry>;

	/// @brief  惰性创建任务的输入缓冲区条目（内部使用，调用前须持锁）。
	void _ensureTaskExists(const TaskId& taskId, const NodeSchema& schema);

	mutable std::shared_mutex _mutex;
	TaskBufferMap _taskInputs;
	TaskBufferMap _taskOutputs;
};

} // namespace DC
