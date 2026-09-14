#pragma once

#include "InferGraph.h"
#include "GraphException.h"
#include "TaskStatus.h"
#include "OutputZone.h"
#include "Value.h"
#include "Tensor.hpp"

#include <chrono>
#include <string>
#include <utility>
#include <vector>

namespace DC {

// ── TaskScope：task 生命周期作用域句柄（RAII）+ 同步便捷入口 ──
//
// 定位：InferGraph 生命周期 API（submitBound / waitForResult / takeOutput /
// releaseTask）的作用域化薄封装——一个对象锚定一个 taskId 的完整生命周期：
//
//   同步场景（三行）：
//     TaskScope task{graph, "t1"};
//     task.feed("adder", "a", ta).feed("adder", "b", tb);
//     auto r = task.run();   // 提交 → 等待 → 取全部绑定输出 → 释放
//
//   异步场景：
//     task.feed(...).submit();   // 立即返回
//     ... 其他工作 ...
//     task.wait(timeout);        // 或经 setTaskCompleteCallback 驱动
//     task.takeTensor(...);
//   （离开作用域自动清理——见析构语义）
//
// 本类不新增任何引擎/图语义，全部方法转发 InferGraph 公开 API；现有细粒度
// API 保留给需要自定义调度策略（服务端跨线程取消 / 长驻等待 / 自定义取数
// 顺序）的场景。
//
// 层级定位：本类面向内核坐标层（(nodeName, portName) 寻址 + 显式输出声明）；
// 按公开别名操作的别名层对应物见 GraphInterface::Task（interface() →
// createTask()）。两者共享同一套运行时语义，仅放弃策略不同：
// GraphInterface::Task 仅同步 run、不暴露显式提交，析构回收已终止任务；
// TaskScope 暴露完整生命周期（submit/wait/cancel），析构按"销毁即放弃"
// 取消并释放。
//
// 所有权与生命周期契约：
// - move-only；图必须存活至本对象析构之后（对齐 exportNode 的契约表述）；
// - 作用域有效期内该 taskId 归本对象独占，勿在外部对同 ID 复用 / 取消 / 释放；
// - 移动后源对象析构为空操作。

class TaskScope {
public:
	using TaskId = InferGraph::TaskId;

	/// @brief  绑定图与 taskId（不创建任务；feed/submit/run 才启动生命周期）
	TaskScope(InferGraph& graph, TaskId taskId);

	/// @brief  析构：取消并释放（销毁即放弃）。
	///
	/// 若任务仍在运行：先 cancel()——取消为同步终止路径（发布终态、清理任务态、
	/// 唤醒等待者），不阻塞；在飞节点执行不被中断，其迟到结果被 gate 检查丢弃。
	/// 随后 releaseTask() 释放状态表条目、输出结果与诊断记录。
	/// 已终止或未知 ID：releaseTask() 幂等 no-op。
	/// 退出作用域不留残余状态 / 结果 / 诊断。
	~TaskScope();

	TaskScope(TaskScope&& other) noexcept;
	TaskScope& operator=(TaskScope&& other) noexcept;
	TaskScope(const TaskScope&) = delete;
	TaskScope& operator=(const TaskScope&) = delete;

	/// @brief  绑定的 taskId
	const TaskId& taskId() const noexcept { return _id; }

	// ── 组装（转发 feedInput；链式调用）──

	/// @brief  从图外注入数据到指定节点的输入端口（转发 InferGraph::feedInput）
	TaskScope& feed(const std::string& nodeName, const std::string& portName, Value data);

	/// @brief  便捷接口：直接传入 DC::Tensor
	TaskScope& feed(const std::string& nodeName, const std::string& portName, Tensor data);

	// ── 异步路径（转发）──

	/// @brief  以全部 bindOutput 绑定作为输出声明提交（转发 submitBound）
	/// @throws GraphException(DuplicateTask/NoDeclaration/UnreachableDeclaration) 同 submitBound
	void submit(uint32_t maxHops = InferGraph::kDefaultMaxHops);

	/// @brief  请求取消（幂等；未知或已终止返回 false）
	bool cancel();

	/// @brief  查询 task 当前状态（同 InferGraph::taskStatus）
	TaskStatus status() const;

	/// @brief  同步等待 task 终止（无限等待）
	TaskResult wait();

	/// @brief  同步等待 task 终止（显式超时；超时只放弃等待，不取消任务，
	///         返回 status 为 Running 的结构化结果）
	TaskResult wait(std::chrono::milliseconds timeout);

	/// @brief  消费式取出输出区结果（转发 InferGraph::takeOutput）
	Value take(const std::string& nodeName, const std::string& portName);

	/// @brief  消费式取出 DC::Tensor（转发 InferGraph::takeOutputTensor）
	Tensor takeTensor(const std::string& nodeName, const std::string& portName);

	// ── 同步便捷路径 ──

	/// @brief  run() 收集的单条输出：绑定元数据 + 数据本体
	struct Output {
		OutputBinding binding; ///< 绑定坐标与别名（与 bindOutput 一致）
		Value value;           ///< 消费式数据
	};

	/// @brief  run() 的结构化结果：终态 + 诊断 + 全部绑定输出
	struct Result {
		TaskStatus status = TaskStatus::Unknown; ///< 终态（宿主等待超时未终止则为 Running）
		std::vector<TaskError> errors;           ///< 诊断记录（可能为空）
		std::vector<Output> outputs;             ///< 覆盖全部 bindOutput 绑定（产出者可取）

		/// @brief  消费式取出指定坐标的输出（取出后从 outputs 移除；不存在抛
		///         GraphException(Other)）
		Value take(const std::string& nodeName, const std::string& portName);

		/// @brief  消费式取出 DC::Tensor（非 Tensor 数据抛 GraphException(Other)）
		Tensor takeTensor(const std::string& nodeName, const std::string& portName);
	};

	/// @brief  同步一发：submitBound → 无限等待 → 收集全部绑定输出 → releaseTask。
	///
	/// 返回后任务已释放（taskStatus 归 Unknown），作用域析构为空操作。
	/// 任务 Failed/Cancelled 时同样释放并返回结构化结果（输出为空 / 部分）。
	/// @throws GraphException(NoDeclaration) 图未 bindOutput 任何端口
	/// @throws GraphException(DuplicateTask/UnreachableDeclaration) 同 submitBound
	Result run();

	/// @brief  同步一发（显式超时）：超时未终止时返回 {status=Running, outputs 空}，
	///         不取不释放——作用域仍持有任务，可继续 wait()/cancel() 或由析构兜底。
	Result run(std::chrono::milliseconds timeout);

private:
	/// @brief  取消并释放：未终止则 cancel()，随后 releaseTask()（幂等）
	void _cleanup() noexcept;

	InferGraph* _graph = nullptr;
	TaskId _id;
};

} // namespace DC
