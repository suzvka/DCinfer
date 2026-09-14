#pragma once

#include "Node.h"
#include "InputZone.h"
#include "OutputZone.h"
#include "TaskStatus.h"

#include <string>
#include <vector>

namespace DC {

class InferGraph;

/// @brief 图公开接口：只按公开别名（bindInput / bindOutput 的 alias）喂数据 / 取结果。
///
/// 由 InferGraph::interface() 创建（构造即冻结图）：创建时一次性解析
/// 公开绑定（alias → (nodeName, portName)）并校验坐标存在性；此后
/// feed / take 仅按别名操作，内部转发至 InferGraph 的坐标寻址运行期 API——
/// 不引入第二套寻址语义。
class GraphInterface {
public:
	class Task;

	/// @brief 公开输入别名列表（按绑定顺序）
	std::vector<std::string> inputAliases() const;

	/// @brief 公开输出别名列表（按绑定顺序）
	std::vector<std::string> outputAliases() const;

	/// @brief 创建任务句柄（自动分配 taskId；句柄析构自动释放已终止任务）
	/// @note  仅限具名对象调用：临时接口对象上的任务句柄生命周期不安全
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

/// @brief 公开接口任务句柄：别名喂数据 → 同步运行 → 别名取结果。
///
/// 析构自动释放已终止任务的资源（状态表条目 / 结果 / 诊断）；
/// 不取消在飞任务。动作均转发 InferGraph 运行期 API（同步语义）。
class GraphInterface::Task {
public:
	Task(Task&& other) noexcept;
	Task& operator=(Task&& other) noexcept;
	~Task();

	Task(const Task&) = delete;
	Task& operator=(const Task&) = delete;

	const std::string& taskId() const { return _taskId; }

	/// @brief 按公开输入别名喂数据（写入缓冲，不触发执行）
	void feed(const std::string& alias, Value data);

	/// @brief 便捷接口：直接传入 DC::Tensor
	void feed(const std::string& alias, Tensor data);

	/// @brief 同步运行：以全部输出绑定提交（submitBound）并等待终止
	TaskResult run();

	/// @brief 按公开输出别名取结果（消费式；取出即消耗）
	Value take(const std::string& alias);

	/// @brief 便捷接口：消费式取出 DC::Tensor
	Tensor takeTensor(const std::string& alias);

private:
	friend class GraphInterface;

	Task(GraphInterface& iface, std::string taskId);

	void _releaseIfTerminated() noexcept;

	GraphInterface* _iface = nullptr;
	std::string _taskId;
};

} // namespace DC
