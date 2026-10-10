#pragma once

#include "Node.h"
#include "InputZone.h"
#include "OutputZone.h"
#include "TaskStatus.h"

#include <chrono>
#include <string>
#include <vector>

namespace DC {

class InferGraph;

/// @brief 图公开接口：只按公开别名（bindInput / bindOutput 的 alias）喂数据 / 取结果。
///
/// 由 InferGraph::interface() 创建（构造即冻结图）；创建时解析绑定并校验坐标存在性，
/// 内部转发至 InferGraph 的坐标寻址 API。
class GraphInterface {
public:
	class Task;

	std::vector<std::string> inputAliases() const;
	std::vector<std::string> outputAliases() const;

	/// @brief 创建任务句柄（自动分配 taskId；句柄析构按状态回收：终态释放 / 在飞弃置并取消 / 未提交清输入）。
	Task createTask() &;

private:
	friend class InferGraph;

	GraphInterface(InferGraph& graph, std::vector<InputBinding> inputs,
				   std::vector<OutputBinding> outputs);

	const InputBinding& _resolveInput(const std::string& alias, const char* api) const;
	const OutputBinding& _resolveOutput(const std::string& alias, const char* api) const;

	InferGraph* _graph;
	std::vector<InputBinding> _inputs;
	std::vector<OutputBinding> _outputs;
};

/// @brief 统一任务句柄：按公开别名喂数据、执行（run / submit+wait）、取结果。
class GraphInterface::Task {
public:
	Task(Task&& other) noexcept;
	Task& operator=(Task&& other) noexcept;
	~Task();

	Task(const Task&) = delete;
	Task& operator=(const Task&) = delete;

	const std::string& taskId() const { return _taskId; }

	/// @brief 按公开输入别名喂数据（写入缓冲，不触发执行）。
	Task& feed(const std::string& alias, Value data);

	/// @brief 便捷：直接传入 DC::Tensor。
	Task& feed(const std::string& alias, Tensor data);

	/// @brief 异步启动：以全部输出绑定提交并返回（不等待）。
	void submit();

	/// @brief 同步运行：submit() + wait()。
	TaskResult run();

	/// @brief 同步运行（显式超时）：超时未终止返回 {status=Running}，不取消任务。
	TaskResult run(std::chrono::milliseconds timeout);

	/// @brief 同步等待 task 终止（无限等待）。
	TaskResult wait();

	/// @brief 同步等待 task 终止（显式超时；超时只放弃等待，不取消任务）。
	TaskResult wait(std::chrono::milliseconds timeout);

	/// @brief 查询 task 当前状态。
	TaskStatus status() const;

	/// @brief 请求取消活动中的 task（协作式；幂等，未知或已终止返回 false）。
	bool cancel();

	/// @brief 按公开输出别名取结果（消费式；取出即消耗）。
	Value take(const std::string& alias);

	/// @brief 便捷：消费式取出 DC::Tensor。
	Tensor takeTensor(const std::string& alias);

	/// @brief 检查指定输出别名是否已有结果。
	bool has(const std::string& alias) const;

	/// @brief 查询 task 在整条传播链上的错误记录。
	std::vector<TaskError> errors() const;

private:
	friend class GraphInterface;

	Task(GraphInterface& iface, std::string taskId);

	void _releaseOnDestroy() noexcept;

	GraphInterface* _iface = nullptr;
	std::string _taskId;
	/// 已成功 submit 后置位；析构路径据此区分：未提交（清输入）/ 在飞弃置（取消并回收）。
	bool _submitted = false;
};

} // namespace DC
