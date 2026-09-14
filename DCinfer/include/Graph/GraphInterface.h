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

/// @brief 统一任务句柄：按公开别名喂数据 →（同步）run /（异步）submit+wait → 取结果。
///
/// 宿主层对推理图的唯一操作入口（默认 API）：同步与异步同级——
/// run() 等价于 submit() 后 wait() 的同步组合，两者可互换表达；
/// 连接/等待/取消/诊断全部按公开别名操作，无 taskId / 坐标暴露。
/// 析构自动释放已终止任务的资源（状态表条目 / 结果 / 诊断）；
/// 不取消在飞任务。动作均转发 InferGraph 运行期 API。
class GraphInterface::Task {
public:
	Task(Task&& other) noexcept;
	Task& operator=(Task&& other) noexcept;
	~Task();

	Task(const Task&) = delete;
	Task& operator=(const Task&) = delete;

	/// @brief 自动分配的 taskId（供下沉互操作：需坐标级精细控制时可用）
	const std::string& taskId() const { return _taskId; }

	// ── 组装（按公开输入别名；链式）──

	/// @brief 按公开输入别名喂数据（写入缓冲，不触发执行）
	Task& feed(const std::string& alias, Value data);

	/// @brief 便捷接口：直接传入 DC::Tensor
	Task& feed(const std::string& alias, Tensor data);

	// ── 执行：同步与异步同级 ──

	/// @brief 异步启动：以全部输出绑定提交并返回（不等待）
	/// @throws GraphException(NoDeclaration/DuplicateTask/UnreachableDeclaration) 同 submitBound
	void submit();

	/// @brief 同步运行：submit() + wait()（无限等待直至终止）
	TaskResult run();

	/// @brief 同步运行（显式超时）：超时未终止返回 {status=Running}，
	///        不取消任务——仍可继续 wait()/cancel() 或由析构兜底
	TaskResult run(std::chrono::milliseconds timeout);

	// ── 等待 / 控制（异步路径配套）──

	/// @brief 同步等待 task 终止（无限等待）
	TaskResult wait();

	/// @brief 同步等待 task 终止（显式超时；超时只放弃等待，不取消任务）
	TaskResult wait(std::chrono::milliseconds timeout);

	/// @brief 查询 task 当前状态
	/// @return Unknown=未提交；Running=执行中；Succeeded/Failed/Cancelled=已终止
	TaskStatus status() const;

	/// @brief 请求取消活动中的 task（协作式取消；幂等，未知或已终止返回 false）
	bool cancel();

	// ── 结果 / 诊断（按公开输出别名）──

	/// @brief 按公开输出别名取结果（消费式；取出即消耗）
	Value take(const std::string& alias);

	/// @brief 便捷接口：消费式取出 DC::Tensor
	Tensor takeTensor(const std::string& alias);

	/// @brief 检查指定输出别名是否已有结果
	bool has(const std::string& alias) const;

	/// @brief 查询 task 在整条传播链上的错误记录
	std::vector<TaskError> errors() const;

private:
	friend class GraphInterface;

	Task(GraphInterface& iface, std::string taskId);

	void _releaseIfTerminated() noexcept;

	GraphInterface* _iface = nullptr;
	std::string _taskId;
};

} // namespace DC
