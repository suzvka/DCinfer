#pragma once

#include <cstddef>
#include <functional>
#include <memory>
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
#include "NativeTensor.h"

namespace DC {

// ── 张量转换钩子：DC::Tensor ↔ 引擎原生张量 ──
// 每种引擎类型注册一份，所有同引擎节点共享
struct TensorConverter {
	// DC::Tensor → heap 原生张量（调用者在 Run 期间持有此 handle）
	std::function<Value(const Tensor&)> toNative;

	// 原生张量（借阅指针，不接管所有权）→ DC::Tensor（内部拷贝数据）
	std::function<Tensor(const void*)> toDC;
};

// ── 节点工厂：按名称 + 引擎特定配置创建节点 ──
using NodeFactory = std::function<std::unique_ptr<class Node>(
	std::string nodeName,
	const void* engineConfig
)>;

// ── 引擎描述符 / 引擎实例（前向声明，定义见 Graph/EngineRegistry.h）──
struct EngineDescriptor;
class  EngineInstance;

class Node {
public:
	using TensorType = Tensor::TensorType;
	using Shape      = Tensor::Shape;

	// ── 任务标识 ──
	using TaskId     = std::string;

	// ── 任务级数据：统一使用 Value 作为数据载体 ──
	// 实际类型通过 TensorConverter 钩子在边界处透明转换
	using TaskData    = Value;

	// ── 端口定义 ──
	struct Port {
		std::string name;
		TensorType  type         = TensorType::Void;
		size_t      typeSize     = 0;
		Shape       shape;
		bool        required     = true;
		std::optional<Tensor> defaultValue;  // 有默认值时，就绪检查视为已填充

		// ── 工厂：必须输入端口 ──
		template<typename T>
		static Port in(std::string name, Shape shape = {}) {
			TensorMeta::ensureTypeMap();
			return {std::move(name), DC::Type::getType<TensorType, T>(), sizeof(T), std::move(shape), true};
		}

		// ── 工厂：可选输入端口（带默认值）──
		template<typename T>
		static Port optional(std::string name, T defaultValue, Shape shape = {}) {
			TensorMeta::ensureTypeMap();
			Tensor dv(DC::Type::getType<TensorType, T>(), sizeof(T));
			dv = defaultValue;
			return {std::move(name), DC::Type::getType<TensorType, T>(), sizeof(T), std::move(shape), false, std::move(dv)};
		}

		// ── 工厂：输出端口 ──
		template<typename T>
		static Port out(std::string name, Shape shape = {}) {
			TensorMeta::ensureTypeMap();
			return {std::move(name), DC::Type::getType<TensorType, T>(), sizeof(T), std::move(shape), true};
		}
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

	// ── RunContext（前向声明，定义见类外）──
	class RunContext;

	// ── RunFn 类型（依赖 Result 和 RunContext）──
	using RunFn = std::function<Result(RunContext&)>;

	// ── 完成回调（纯通知，不传数据）──
	using CompletionFn = std::function<void(const TaskId& taskId, const Result& result)>;

	// ── 构造/析构 ──
	Node(std::string type, std::string name, Schema schema, RunFn fn,
	          EngineInstance* engineInstance = nullptr);
	~Node();

	// 禁止拷贝/移动
	Node(const Node&)            = delete;
	Node& operator=(const Node&) = delete;
	Node(Node&&)                 = delete;
	Node& operator=(Node&&)      = delete;

	// ── 只读属性 ──
	const std::string& type()   const { return _type; }
	const std::string& name()   const { return _name; }
	const Schema&      schema() const { return _schema; }

	// ── 完成回调注册 ──
	void setCompletionCallback(CompletionFn fn);
	bool hasCompletionCallback() const;

	// ── 任务级输入 ──

	/// @brief  单端口写入（NativeTensor），写入后立即检查就绪
	bool setInput(const TaskId& taskId, const std::string& portName, Value data);

	/// @brief  便捷接口：直接传入 DC::Tensor，内部自动包装为 Value
	bool setInput(const TaskId& taskId, const std::string& portName, Tensor data);

	/// @brief  批量写入（NativeTensor），全部成功后触发一次检查
	bool setInput(const TaskId& taskId, std::unordered_map<std::string, TaskData> inputs);

	/// @brief  便捷接口：批量传入 DC::Tensor，内部自动包装
	bool setInput(const TaskId& taskId, std::unordered_map<std::string, Tensor> inputs);

	// ── 任务级输出（始终从缓冲区拉取）──
	bool hasOutput(const TaskId& taskId, const std::string& name) const;

	Value getOutput(const TaskId& taskId, const std::string& name);

	/// @brief  便捷接口：消费式取出 DC::Tensor，自动完成 Value → Tensor 解包
	/// @throws  std::out_of_range 若输出不存在或不是 Tensor 类型
	Tensor getOutputTensor(const TaskId& taskId, const std::string& name);

	const Value& peekOutput(const TaskId& taskId, const std::string& name) const;

	std::unordered_map<std::string, TaskData> collectOutputs(const TaskId& taskId);

	/// @brief  便捷接口：批量消费 DC::Tensor 输出
	std::unordered_map<std::string, Tensor> collectOutputTensors(const TaskId& taskId);

	// ── 阻塞式一次执行 ──
	/// @brief  一次性送入所有 Value 输入，同步执行，返回指定输出的 Value
	/// @throws  std::runtime_error 若输入不合法或执行失败
	Value execute(const std::string& outputName,
	                     std::unordered_map<std::string, Value> inputs);

	/// @brief  便捷接口：一次性送入 DC::Tensor，同步执行，返回指定输出的 Tensor
	/// @throws  std::runtime_error 若输入不合法、执行失败或输出不是 Tensor 类型
	Tensor executeTensor(const std::string& outputName,
	                     std::unordered_map<std::string, Tensor> inputs);

	// ── 任务生命周期 ──
	void   clearTask(const TaskId& taskId);
	size_t taskCount() const;

	// ── 直接访问工作槽位（只读，高级场景）──
	const std::unordered_map<std::string, TensorSlot>& inputSlots()  const { return _inputSlots; }
	const std::unordered_map<std::string, TensorSlot>& outputSlots() const { return _outputSlots; }

private:
	friend class RunContext;

	// ── 内部类型 ──
	using SlotMap       = std::unordered_map<std::string, TensorSlot>;
	using TaskPortMap   = std::unordered_map<TaskId, std::unordered_map<std::string, TaskData>>;
	using TaskBuffer    = std::unordered_map<std::string, std::optional<TaskData>>;
	using TaskBufferMap = std::unordered_map<TaskId, TaskBuffer>;

	// ── 内部方法 ──
	void _ensureTaskExists(const TaskId& taskId);
	bool _isTaskReady(const TaskId& taskId) const;
	void _checkAndExecute(const TaskId& taskId);
	void _loadTaskToWorkingSlots(const TaskId& taskId);
	void _clearWorkingOutputs();
	void _collectAndSaveOutputs(const TaskId& taskId);

	// ── RunContext 可调用的内部方法 ──
	const Value&     _inputImpl(const std::string& name) const;
	void                    _outputImpl(const std::string& name, Value tensor);
	const TensorConverter*  _converter() const;
	const EngineDescriptor* _engineDescriptor() const;
	void                    _synchronizeEngine() const;
	Result _makeSuccess(std::string message = {}) const;
	Result _makeFailure(Status status, std::string message) const;

	// ── 槽位可变访问（由 friend 类访问）──
	SlotMap& _mutableInputSlots()  { return _inputSlots; }
	SlotMap& _mutableOutputSlots() { return _outputSlots; }

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
	EngineInstance*   _engineInstance = nullptr;  // 非拥有引用，Registry 管理生命周期
};

// ── RunContext 定义（需在 Node 类体之后）──
class Node::RunContext {
public:
	const Value& input(const std::string& name) const {
		return _node._inputImpl(name);
	}
	void output(const std::string& name, Value tensor) {
		_node._outputImpl(name, std::move(tensor));
	}

	/// @brief  读取输出槽中的原始 Value（不消费），用于 postRun 钩子做 D2H 转换
	/// @return 指向 Value 的指针，若槽位不存在或无数据则返回 nullptr
	const Value* outputRaw(const std::string& name) const;

	Node::Result success(std::string message = {}) const {
		return _node._makeSuccess(std::move(message));
	}
	Node::Result failure(Node::Status status, std::string message) const {
		return _node._makeFailure(status, std::move(message));
	}
	const TensorConverter* converter() const {
		return _node._converter();
	}
	const EngineDescriptor* engineDescriptor() const {
		return _node._engineDescriptor();
	}
	const EngineInstance* engineInstance() const;
	void* engine() const;
	const Node::Schema& schema() const { return _node.schema(); }
	const std::string&       type()   const { return _node.type(); }
	const std::string&       name()   const { return _node.name(); }
private:
	friend class Node;
	explicit RunContext(Node& node) : _node(node) {}
	Node& _node;
};

} // namespace DC
