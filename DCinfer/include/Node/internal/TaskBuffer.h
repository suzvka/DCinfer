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

class Node;
struct NodeSchema; // 定义见 Node.h

/// @brief 线程安全的 task 级 I/O 缓冲区管理器（Schema 经参数传入，避免头文件循环依赖）。
class TaskBuffer {
public:
	using TaskId = std::string;
	using TaskData = Value;

	TaskBuffer() = default;

	/// @brief 单端口写入（仅写缓冲，不触发执行）；端口名未声明抛 PortNotFound。
	void setInput(const TaskId& taskId, const std::string& portName, Value data,
				  const NodeSchema& schema);

	/// @brief 单端口写入 + 就绪判定（同一临界区：多上游并发时仅"最后写入者"观察到就绪，无重复提交窗口）。
	/// @return 写入后该 task 的全部必需输入是否已就绪
	bool setInputAndCheckReady(const TaskId& taskId, const std::string& portName, Value data,
						   const NodeSchema& schema);

	/// @brief 批量写入（预校验所有端口名）。
	void setInputBatch(const TaskId& taskId,
					   std::unordered_map<std::string, TaskData> inputs,
					   const NodeSchema& schema);

	/// @brief 所有必需输入是否已就绪（含默认值）。
	bool isReady(const TaskId& taskId, const NodeSchema& schema) const;

	/// @brief 是否已产出指定输出端口的数据。
	bool hasOutput(const TaskId& taskId, const std::string& name) const;

	/// @brief 消费式取出输出（取出即清空）；任务不存在抛 TaskNotFound，端口为空抛 OutputNotProduced。
	Value takeOutput(const TaskId& taskId, const std::string& name);

	/// @brief 消费式取出输出（检查+取数同一临界区，无 check-then-act 竞态）。
	/// @return 任务不存在或端口为空返回 nullopt（不抛异常）
	std::optional<Value> tryTakeOutput(const TaskId& taskId, const std::string& name);

	std::unordered_map<std::string, TaskData> collectOutputs(const TaskId& taskId);

	bool hasTask(const TaskId& taskId) const;
	void clearTask(const TaskId& taskId);
	size_t taskCount() const;

	/// @brief 将输入缓冲 move 到工作槽位（含默认值回退）。
	void drainInputsTo(const TaskId& taskId, class SlotWorkspace& workspace,
					   const NodeSchema& schema);

	/// @brief 将工作槽位数据收集到输出缓冲区。
	void fillOutputsFrom(const TaskId& taskId, class SlotWorkspace& workspace,
						 const NodeSchema& schema);

	/// @brief 验证所有必需输出端口已产生（fillOutputsFrom 之后调用）。
	bool validateOutputs(const TaskId& taskId, const NodeSchema& schema) const;

	/// @brief 仅擦除输入缓冲区（输出保留）。
	void eraseInputs(const TaskId& taskId);

private:
	using TaskBufferEntry = std::unordered_map<std::string, std::optional<TaskData>>;
	using TaskBufferMap = std::unordered_map<TaskId, TaskBufferEntry>;

	/// @brief 惰性创建任务的输入缓冲条目；调用方须持锁。
	void _ensureTaskExists(const TaskId& taskId, const NodeSchema& schema);

	/// @brief 就绪判定核心；调用方须持锁。
	bool _isReadyLocked(const TaskId& taskId, const NodeSchema& schema) const;

	mutable std::shared_mutex _mutex;
	TaskBufferMap _taskInputs;
	TaskBufferMap _taskOutputs;
};

} // namespace DC
