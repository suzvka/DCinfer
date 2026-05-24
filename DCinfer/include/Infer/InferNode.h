#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "TensorSlot.h"
#include "SlotType.h"
#include "Tensor/NativeTensor.h"

namespace DC {

// ── 张量转换钩子：DC::Tensor ↔ 引擎原生张量 ──
// 每种引擎类型注册一份，所有同引擎节点共享
struct TensorConverter {
	// DC::Tensor → heap 原生张量（调用者在 Run 期间持有此 handle）
	std::function<NativeTensor(const Tensor&)> toNative;

	// 原生张量（借阅指针，不接管所有权）→ DC::Tensor（内部拷贝数据）
	std::function<Tensor(const void*)> toDC;

	// ── 自我识别钩子（供同引擎直传优化）──
	// 此引擎产出的原生张量类型标识（如 "Ort::Value", "nvinfer1::ITensor"）
	std::string nativeTypeTag;

	// 此引擎是否可直接消费指定类型的原生张量
	std::function<bool(const std::string& nativeTypeTag)> canAccept;
};

// ── 节点工厂：按名称 + 引擎特定配置创建节点 ──
using NodeFactory = std::function<std::unique_ptr<class InferNode>(
	std::string nodeName,
	const void* engineConfig
)>;

// ── 引擎描述符：注册一个引擎所需的全部信息 ──
struct EngineDescriptor {
	std::string     engineType;
	TensorConverter converter;
	NodeFactory     factory;
};

class InferNode {
public:
	using TensorType = Tensor::TensorType;
	using Shape      = Tensor::Shape;
	using SlotMap    = std::unordered_map<std::string, TensorSlotBase>;

	// ── 任务标识 ──
	using TaskId     = std::string;

	// ── 任务级数据：统一使用 NativeTensor 作为数据载体 ──
	// 实际类型通过 TensorConverter 钩子在边界处透明转换
	using TaskData      = NativeTensor;
	using TaskPortMap   = std::unordered_map<TaskId, std::unordered_map<std::string, TaskData>>;
	// 内部缓冲区（optional 表示"未设置"）
	using TaskBuffer   = std::unordered_map<std::string, std::optional<TaskData>>;
	using TaskBufferMap = std::unordered_map<TaskId, TaskBuffer>;

	// ── 端口定义 ──
	struct Port {
		std::string name;
		TensorType  type         = TensorType::Void;
		size_t      typeSize     = 0;
		Shape       shape;
		bool        required     = true;
		std::optional<Tensor> defaultValue;  // 有默认值时，就绪检查视为已填充
	};

	// ── Schema ──
	struct Schema {
		std::vector<Port> inputs;
		std::vector<Port> outputs;

		const Port* findInput(const std::string& name) const {
			return find(inputs, name);
		}
		const Port* findOutput(const std::string& name) const {
			return find(outputs, name);
		}
		bool valid() const {
			if (!hasUniqueNames(inputs) || !hasUniqueNames(outputs))
				return false;
			auto checkTypeSize = [](const std::vector<Port>& ports) {
				for (const auto& p : ports) {
					if (p.type != TensorType::Void && p.typeSize == 0)
						return false;
				}
				return true;
			};
			if (!checkTypeSize(inputs) || !checkTypeSize(outputs))
				return false;
			// 校验默认值类型一致性
			for (const auto& p : inputs) {
				if (p.defaultValue.has_value()) {
					const auto& dv = p.defaultValue.value();
					if (dv.type() != p.type || dv.typeSize() != p.typeSize)
						return false;
				}
			}
			return true;
		}
	private:
		static const Port* find(const std::vector<Port>& ports, const std::string& name) {
			for (const auto& port : ports) {
				if (port.name == name) return &port;
			}
			return nullptr;
		}
		static bool hasUniqueNames(const std::vector<Port>& ports) {
			std::unordered_set<std::string> names;
			names.reserve(ports.size());
			for (const auto& port : ports) {
				if (!names.insert(port.name).second) return false;
			}
			return true;
		}
	};

	// ── 状态 ──
	enum class Status {
		Ok,
		InvalidInput,
		SchemaMismatch,
		ExecutionFailed,
		InternalError
	};

	struct Result {
		Status      status  = Status::Ok;
		std::string message;
		bool ok() const { return status == Status::Ok; }
	};

	// ── RunFn 类型（依赖 Result）──
	using RunFn = std::function<Result(InferNode&)>;

	// ── 完成回调（纯通知，不传数据）──
	using CompletionFn = std::function<void(const TaskId& taskId, const Result& result)>;

	// ── 构造/析构 ──
	InferNode(std::string type, std::string name, Schema schema, RunFn fn);
	~InferNode();

	// 禁止拷贝/移动
	InferNode(const InferNode&)            = delete;
	InferNode& operator=(const InferNode&) = delete;
	InferNode(InferNode&&)                 = delete;
	InferNode& operator=(InferNode&&)      = delete;

	// ── 只读属性 ──
	const std::string& type()   const { return _type; }
	const std::string& name()   const { return _name; }
	const Schema&      schema() const { return _schema; }

	// ── 完成回调注册 ──
	void setCompletionCallback(CompletionFn fn);
	bool hasCompletionCallback() const;

	// ── 任务级输入 ──

	/// @brief  单端口写入（NativeTensor），写入后立即检查就绪
	/// 数据通过 TensorConverter 钩子在加载时转换为工作槽位所需类型
	bool setInput(const TaskId& taskId, const std::string& portName, NativeTensor data);

	/// @brief  批量写入，全部成功后触发一次检查
	bool setInputs(const TaskId& taskId, std::unordered_map<std::string, TaskData> inputs);

	// ── 任务级输出（始终从缓冲区拉取）──
	bool hasOutput(const TaskId& taskId, const std::string& name) const;

	template<typename T>
	T getOutput(const TaskId& taskId, const std::string& name);

	template<typename T>
	const T& peekOutput(const TaskId& taskId, const std::string& name) const;

	std::unordered_map<std::string, TaskData> collectOutputs(const TaskId& taskId);

	// ── 任务生命周期 ──
	void   clearTask(const TaskId& taskId);
	size_t taskCount() const;

	// ── RunFn 内部使用的计算 API ──
	const Tensor& input(const std::string& name) const;
	void          output(const std::string& name, const Tensor& tensor);
	void          output(const std::string& name, Tensor&& tensor);

	// ── 转换钩子访问器 ──
	const TensorConverter* converter() const;
	const EngineDescriptor* engineDescriptor() const;

	Result success(std::string message = {}) const;
	Result failure(Status status, std::string message) const;

	// ── 直接访问工作槽位（高级场景）──
	SlotMap&       inputSlots()       { return _inputSlots; }
	SlotMap&       outputSlots()      { return _outputSlots; }
	const SlotMap& inputSlots() const { return _inputSlots; }
	const SlotMap& outputSlots() const{ return _outputSlots; }

private:
	// ── 内部方法 ──
	void _ensureTaskExists(const TaskId& taskId);
	bool _isTaskReady(const TaskId& taskId) const;
	void _checkAndExecute(const TaskId& taskId, std::unique_lock<std::mutex>& lock);
	void _loadTaskToWorkingSlots(const TaskId& taskId);
	void _clearWorkingOutputs();
	void _collectAndSaveOutputs(const TaskId& taskId);

	// ── 成员 ──
	std::string       _type;
	std::string       _name;
	Schema            _schema;
	SlotMap           _inputSlots;     // 工作输入槽位
	SlotMap           _outputSlots;    // 工作输出槽位
	TaskBufferMap     _taskInputs;     // 任务级输入缓冲 (port → optional<TaskData>)
	TaskBufferMap     _taskOutputs;    // 任务级输出缓冲 (port → optional<TaskData>)
	RunFn             _fn;
	CompletionFn      _onComplete;
	mutable std::mutex _mutex;
};

// ────────────────────────────────────────────
// 模板实现（位于头文件）
// ────────────────────────────────────────────

template<typename T>
T InferNode::getOutput(const TaskId& taskId, const std::string& name) {
	static_assert(std::is_same_v<T, Tensor> || std::is_same_v<T, NativeTensor>,
	              "getOutput only supports Tensor or NativeTensor");
	std::lock_guard lock(_mutex);

	auto taskIt = _taskOutputs.find(taskId);
	if (taskIt == _taskOutputs.end()) {
		throw std::out_of_range("InferNode::getOutput: task '" + taskId + "' not found");
	}
	auto& optVal = taskIt->second.at(name);
	if (!optVal.has_value()) {
		throw std::out_of_range("InferNode::getOutput: output '" + name + "' is empty");
	}

	if constexpr (std::is_same_v<T, NativeTensor>) {
		T result = std::move(optVal.value());
		optVal.reset();  // 消费式取出
		return result;
	} else {
		// T == Tensor：通过 converter 从 NativeTensor 转换
		auto* nodeConverter = converter();
		if (!nodeConverter || !nodeConverter->toDC) {
			throw std::runtime_error("InferNode::getOutput: no TensorConverter registered for engine '" + _type + "'");
		}
		T result = nodeConverter->toDC(optVal.value().get());
		optVal.reset();  // 消费式取出
		return result;
	}
}

template<typename T>
const T& InferNode::peekOutput(const TaskId& taskId, const std::string& name) const {
	static_assert(std::is_same_v<T, NativeTensor>,
	              "peekOutput only supports NativeTensor; use getOutput<Tensor> for Tensor conversion");
	std::lock_guard lock(_mutex);

	auto taskIt = _taskOutputs.find(taskId);
	if (taskIt == _taskOutputs.end()) {
		throw std::out_of_range("InferNode::peekOutput: task '" + taskId + "' not found");
	}
	auto& optVal = taskIt->second.at(name);
	if (!optVal.has_value()) {
		throw std::out_of_range("InferNode::peekOutput: output '" + name + "' is empty");
	}
	return optVal.value();
}

} // namespace DC
