#pragma once

#include "TensorSlot.h"
#include "Value.h"
#include "NodeException.h"

#include <optional>
#include <string>
#include <unordered_map>

namespace DC {

struct NodeSchema; // 定义见 Node.h

/// @brief 工作槽位管理器：RunContext 的 peek/pop/output 操作委托至此。
class SlotWorkspace {
public:
	using SlotMap = std::unordered_map<std::string, TensorSlot>;

	/// @brief 从 Schema 构建工作槽位（含 shapeAnchor 的 DefaultProvider 安装）。
	explicit SlotWorkspace(const NodeSchema& schema);

	/// @brief 只读查看输入槽 Value。
	const Value& peekInput(const std::string& name) const;

	/// @brief 消费式取出输入槽 Value（槽位清空）。
	Value popInput(const std::string& name);

	void writeOutput(const std::string& name, Value tensor);

	/// @brief 读取输出槽原始 Value（不消费）；不存在返回 nullptr。
	const Value* peekOutputRaw(const std::string& name) const;

	/// @brief 清空所有工作输出槽位。
	void clearOutputs();

	const SlotMap& inputSlots() const { return _inputSlots; }
	SlotMap& mutableInputSlots() { return _inputSlots; }
	const SlotMap& outputSlots() const { return _outputSlots; }
	SlotMap& mutableOutputSlots() { return _outputSlots; }

private:
	SlotMap _inputSlots;
	SlotMap _outputSlots;
};

} // namespace DC
