#pragma once

#include "TensorSlot.h"
#include "Value.h"
#include "NodeException.h"

#include <atomic>
#include <optional>
#include <string>
#include <unordered_map>

namespace DC {

struct NodeSchema; // 前向声明（定义见 Node.h）

/// @brief 工作槽位管理器：封装 Node 的 _inputSlots / _outputSlots / _executionGuard。
///
/// RunContext 的 peek/pop/output 操作委托至此。
/// 执行保护（互斥锁）和租约管理也在此集中。
class SlotWorkspace {
public:
	using SlotMap = std::unordered_map<std::string, TensorSlot>;

	/// @brief 从 Schema 构建工作槽位（含 shapeAnchor 的 DefaultProvider 安装）。
	explicit SlotWorkspace(const NodeSchema& schema);

	// ── RunContext 委托接口 ──

	/// @brief  只读查看输入槽中的 Value。
	const Value& peekInput(const std::string& name) const;

	/// @brief  消费式取出输入槽中的 Value（槽位清空）。
	Value popInput(const std::string& name);

	/// @brief  将 Value 写入输出槽。
	void writeOutput(const std::string& name, Value tensor);

	/// @brief  读取输出槽中的原始 Value（不消费），返回 nullptr 若不存在。
	const Value* peekOutputRaw(const std::string& name) const;

	// ── 执行保护 ──

	/// @brief  尝试获取执行租约（test_and_set），失败返回 false（重入拒绝）。
	bool tryAcquire();

	/// @brief  释放执行租约。
	void release();

	// ── 清空/跟踪 ──

	/// @brief  清空所有工作输出槽位（每轮执行前调用）。
	void clearOutputs();

	/// @brief  设置当前执行中的 task ID。
	void setCurrentTask(const std::string& taskId);

	/// @brief  清除当前 task ID。
	void clearCurrentTask();

	/// @brief  返回当前执行中的 task ID。
	std::optional<std::string> currentTask() const;

	// ── 槽位暴露（供 DefaultProvider / 序列化访问）──

	const SlotMap& inputSlots() const { return _inputSlots; }
	SlotMap& mutableInputSlots() { return _inputSlots; }
	const SlotMap& outputSlots() const { return _outputSlots; }
	SlotMap& mutableOutputSlots() { return _outputSlots; }

private:
	SlotMap _inputSlots;
	SlotMap _outputSlots;
	std::atomic_flag _executionGuard = ATOMIC_FLAG_INIT;
	std::optional<std::string> _currentTaskId;
};

} // namespace DC
