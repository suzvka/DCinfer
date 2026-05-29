#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "TensorSlot.h"
#include "SlotType.h"
#include "Value.h"
#include "NodeException.h"

#include <coroutine>

namespace DC {

/// @brief 张量转换钩子：DC::Tensor ↔ 引擎原生张量。
/// 每种引擎类型注册一份，所有同引擎节点共享。
struct TensorConverter {
	/// @brief DC::Tensor → 引擎原生张量（调用者在 Run 期间持有此 handle）。
	std::function<Value(const Tensor&)> toNative;

	/// @brief 原生张量（借阅指针，不接管所有权）→ DC::Tensor（内部拷贝数据）。
	std::function<Tensor(const void*)> toDC;
};

/// @brief 节点工厂：按引擎类型名称 + 引擎特定配置创建节点。
using NodeFactory = std::function<std::unique_ptr<class Node>(std::string nodeName, const void* engineConfig)>;

// ── 引擎描述符 / 引擎实例（前向声明，定义见 Graph/EngineRegistry.h）──
struct EngineDescriptor;
class EngineInstance;

/// @brief 线程池归属：标识节点应由三层线程池中的哪一个执行。
/// 仅在 Node 创建时赋值，之后只读。
enum class ThreadPoolAffinity {
	Compute, ///< 计算线程池 —— 执行推理过程（GPU 加速，如 ONNX/TensorRT）
	Operator, ///< 算子线程池 —— 执行 CPU 密集过程（如 Pre/Post 处理）
	System, ///< 系统线程池 —— 数据流动与连接器（Broadcast/Routing/Wire）
};

// ── 前向声明：co_await-able 节点完成通知 ──
struct NodeCompletion;

class Node {
public:
	using TensorType = Tensor::TensorType;
	using Shape = Tensor::Shape;

	// ── 任务标识 ──
	using TaskId = std::string;

	// ── 任务级数据：统一使用 Value 作为数据载体 ──
	// 实际类型通过 TensorConverter 钩子在边界处透明转换
	using TaskData = Value;

	// ── 端口定义 ──
	/// @brief 端口定义：描述节点的输入或输出端口元数据。
	struct Port {
		std::string name; ///< 端口名称（在 Schema 内唯一）。
		TensorType type = TensorType::Void; ///< 期望的张量逻辑类型。
		size_t typeSize = 0; ///< 单元素字节数。
		Shape shape; ///< 期望的形状（空=不校验）。
		bool required = true; ///< 执行前是否必须填充。
		std::optional<Tensor> defaultValue; ///< 有默认值时，就绪检查视为已填充。
		std::optional<std::string> shapeAnchor; ///< 形状锚定端口名：未填时形状跟随该端口运行时张量，内容填零。

		/// @brief 工厂：创建必须输入端口。
		/// @tparam T 期望的 C++ 类型。
		/// @param name 端口名。
		/// @param shape 期望形状。
		template <typename T>
		static Port in(std::string name, Shape shape = {}) {
			TensorMeta::ensureTypeMap();
			return {std::move(name), DC::Type::getType<TensorType, T>(), sizeof(T), std::move(shape), true};
		}

		/// @brief 工厂：创建可选输入端口（带默认值）。
		/// @tparam T 默认值的 C++ 类型。
		/// @param name 端口名。
		/// @param defaultValue 默认值。
		/// @param shape 期望形状。
		template <typename T>
		static Port optional(std::string name, T defaultValue, Shape shape = {}) {
			TensorMeta::ensureTypeMap();
			Tensor dv(DC::Type::getType<TensorType, T>(), sizeof(T));
			dv = defaultValue;
			return {std::move(name), DC::Type::getType<TensorType, T>(), sizeof(T), std::move(shape), false,
					std::move(dv)};
		}

		/// @brief 工厂：创建形状锚定输入端口（required=false，无静态默认值）。
		///        运行时若该端口未被显式填充，则自动生成与 anchorPort 同形全零张量。
		/// @tparam T 期望的 C++ 类型。
		/// @param name 端口名。
		/// @param anchorPort 锚定端口名（必须为本节点另一输入端口）。
		/// @param shape 期望形状（空=不校验，运行时跟随锚定端口）。
		template <typename T>
		static Port anchored(std::string name, std::string anchorPort, Shape shape = {}) {
			TensorMeta::ensureTypeMap();
			Port p;
			p.name = std::move(name);
			p.type = DC::Type::getType<TensorType, T>();
			p.typeSize = sizeof(T);
			p.shape = std::move(shape);
			p.required = false;
			p.shapeAnchor = std::move(anchorPort);
			return p;
		}

		/// @brief 工厂：创建输出端口。
		/// @tparam T 期望的 C++ 类型。
		/// @param name 端口名。
		/// @param shape 期望形状。
		template <typename T>
		static Port out(std::string name, Shape shape = {}) {
			TensorMeta::ensureTypeMap();
			return {std::move(name), DC::Type::getType<TensorType, T>(), sizeof(T), std::move(shape), true};
		}
	};

	// ── Schema ──
	/// @brief 节点 Schema：定义输入/输出端口的完整元数据。
	struct Schema {
		std::vector<Port> inputs; ///< 输入端口列表。
		std::vector<Port> outputs; ///< 输出端口列表。

		/// @brief 按名称查找输入端口。
		/// @return 指向 Port 的指针，若不存在则返回 nullptr。
		const Port* findInput(const std::string& name) const;

		/// @brief 按名称查找输出端口。
		/// @return 指向 Port 的指针，若不存在则返回 nullptr。
		const Port* findOutput(const std::string& name) const;

		/// @brief 校验 Schema 合法性：端口名唯一、非 Void 端口 typeSize>0、默认值类型一致。
		bool valid() const;

	private:
		static const Port* find(const std::vector<Port>& ports, const std::string& name);
		static bool hasUniqueNames(const std::vector<Port>& ports);
	};

	// ── 状态 ──
	/// @brief 节点执行结果状态枚举。
	enum class Status {
		Ok, ///< 执行成功。
		InvalidInput, ///< 输入数据不合法（值为空、类型错误等）。
		SchemaMismatch, ///< 输入与 Schema 声明不一致。
		ExecutionFailed, ///< RunFn 执行过程中抛出异常。
		InternalError ///< 节点内部状态错误。
	};

	/// @brief 节点执行结果。
	struct Result {
		Status status = Status::Ok; ///< 执行状态。
		std::string message; ///< 附加消息。
		/// @brief 是否执行成功。
		bool ok() const {
			return status == Status::Ok;
		}
	};

	// ── RunContext（前向声明，定义见类外）──
	class RunContext;

	// ── RunFn 类型（依赖 Result 和 RunContext）──
	using RunFn = std::function<Result(RunContext&)>;

	// ── 完成回调（纯通知，不传数据）──
	using CompletionFn = std::function<void(const TaskId& taskId, const Result& result)>;

	// ── 构造/析构 ──
	Node(std::string type, std::string name, Schema schema, RunFn fn, EngineInstance* engineInstance = nullptr,
		 ThreadPoolAffinity affinity = ThreadPoolAffinity::Operator);
	~Node();

	// 禁止拷贝/移动
	Node(const Node&) = delete;
	Node& operator=(const Node&) = delete;
	Node(Node&&) = delete;
	Node& operator=(Node&&) = delete;

	// ── 只读属性 ──
	const std::string& type() const {
		return _type;
	}
	const std::string& name() const {
		return _name;
	}
	const Schema& schema() const {
		return _schema;
	}

	// ── 线程池归属与分组 ──
	/// @brief  获取线程池归属（Compute / Operator / System）。
	ThreadPoolAffinity affinity() const {
		return _affinity;
	}

	/// @brief  设置节点分组标签（用于线程池分组限流）。
	void setTag(std::string tag) {
		_tag = std::move(tag);
	}
	/// @brief  获取节点分组标签。
	const std::string& tag() const {
		return _tag;
	}

	/// @brief  是否为连接器节点（导线/扇出等基础设施节点）。
	bool isConnector() const {
		return _isConnector;
	}
	/// @brief  设置连接器节点标记。
	void setConnector(bool v) {
		_isConnector = v;
	}

	// ── 完成回调注册 ──
	/// @brief  注册任务完成回调（纯通知，不传数据）。
	void setCompletionCallback(CompletionFn fn);
	/// @brief  是否已注册完成回调。
	bool hasCompletionCallback() const;

	// ── 任务级输入 ──

	/// @brief  单端口写入（Value），仅写入缓冲，不触发执行。
	/// @param taskId 任务标识符。
	/// @param portName 目标输入端口名。
	/// @param data 要写入的数据（Value 包装的原生张量）。
	/// @throws NodeException(PortNotFound) 若端口名不存在于 Schema 中。
	void setInput(const TaskId& taskId, const std::string& portName, Value data);

	/// @brief  便捷接口：直接传入 DC::Tensor，内部自动包装为 Value。
	/// @param taskId 任务标识符。
	/// @param portName 目标输入端口名。
	/// @param data 要写入的 DC::Tensor 数据。
	/// @throws NodeException(PortNotFound) 若端口名不存在于 Schema 中。
	void setInput(const TaskId& taskId, const std::string& portName, Tensor data);

	/// @brief  批量写入（Value），仅写入缓冲，不触发执行。
	/// @param taskId 任务标识符。
	/// @param inputs 端口名到数据的映射。
	/// @throws NodeException(PortNotFound) 若任一端口名不存在于 Schema 中。
	void setInput(const TaskId& taskId, std::unordered_map<std::string, TaskData> inputs);

	/// @brief  便捷接口：批量传入 DC::Tensor，内部自动包装。
	/// @param taskId 任务标识符。
	/// @param inputs 端口名到 Tensor 的映射。
	/// @throws NodeException(PortNotFound) 若任一端口名不存在于 Schema 中。
	void setInput(const TaskId& taskId, std::unordered_map<std::string, Tensor> inputs);

	// ── 任务级输出（始终从缓冲区拉取）──

	/// @brief 查询指定任务是否已产出指定输出端口的数据。
	/// @param taskId 任务标识符。
	/// @param name 输出端口名。
	/// @return true 若输出已就绪。
	bool hasOutput(const TaskId& taskId, const std::string& name) const;

	/// @brief 消费式取出输出数据（调用后输出缓冲区被清空）。
	/// @param taskId 任务标识符。
	/// @param name 输出端口名。
	/// @return 输出数据（Value）。
	/// @throws NodeException(TaskNotFound) 若任务不存在。
	/// @throws NodeException(OutputNotProduced) 若输出端口为空。
	Value getOutput(const TaskId& taskId, const std::string& name);

	/// @brief  便捷接口：消费式取出 DC::Tensor，自动完成 Value → Tensor 解包。
	/// @param taskId 任务标识符。
	/// @param name 输出端口名。
	/// @return 输出数据（DC::Tensor）。
	/// @throws NodeException(TaskNotFound) 若任务不存在。
	/// @throws NodeException(TypeMismatch) 若输出不是 DC::Tensor 类型。
	Tensor getOutputTensor(const TaskId& taskId, const std::string& name);

	/// @brief 只读查看输出（不消费，数据保留在缓冲区）。
	/// @param taskId 任务标识符。
	/// @param name 输出端口名。
	/// @return 输出数据的只读引用。
	/// @throws NodeException(TaskNotFound) 若任务不存在。
	/// @throws NodeException(OutputNotProduced) 若输出端口为空。
	const Value& peekOutput(const TaskId& taskId, const std::string& name) const;

	/// @brief 批量消费所有输出。
	/// @param taskId 任务标识符。
	/// @return 端口名到数据的映射。
	std::unordered_map<std::string, TaskData> collectOutputs(const TaskId& taskId);

	/// @brief 便捷接口：批量消费 DC::Tensor 输出。
	/// @param taskId 任务标识符。
	/// @return 端口名到 Tensor 的映射。
	std::unordered_map<std::string, Tensor> collectOutputTensors(const TaskId& taskId);

	// ── 协程支持 ──

	/// @brief  co_await-able 等待器：挂起当前协程，直到指定 task 执行完成
	///         完成后返回 Result，协程自动恢复
	NodeCompletion whenComplete(const TaskId& taskId);

	// ── 调度接口（供 Graph 调用）──

	/// @brief  查询指定任务是否所有必需输入已就绪（含默认值）
	bool isReady(const TaskId& taskId) const;

	/// @brief  尝试执行就绪任务：获取重入锁 → 加载输入 → RunFn → 收集输出。
	/// @param taskId 要执行的任务标识符。
	/// @throws NodeException(NotReady) 若任务输入尚未就绪。
	/// @throws NodeException(Reentrant) 若节点正忙（Exclusive 策略下重入被拒）。
	/// @throws NodeException(ExecutionFailed) 若 RunFn 执行失败。
	/// @note   线程安全；同一时刻最多一个 task 在执行。
	void tryExecute(const TaskId& taskId);

	/// @brief  返回当前正在执行的任务 ID（用于冒泡事件、任务-节点亲和性）
	/// @return 若节点空闲则返回 std::nullopt
	std::optional<TaskId> currentTaskId() const;

	// ── 任务生命周期 ──
	/// @brief  查询指定 task 是否存在于此节点（输入或输出缓冲区非空）。
	bool hasTask(const TaskId& taskId) const;
	/// @brief  清除指定任务的所有输入/输出缓冲区及等待者。
	void clearTask(const TaskId& taskId);
	/// @brief  终止指定 task：清除 IO 缓冲区并通知所有等待协程恢复
	void terminateTask(const TaskId& taskId);
	/// @brief  当前活跃任务数量。
	size_t taskCount() const;

	// ── 直接访问工作槽位（只读，高级场景）──
	const std::unordered_map<std::string, TensorSlot>& inputSlots() const {
		return _inputSlots;
	}
	const std::unordered_map<std::string, TensorSlot>& outputSlots() const {
		return _outputSlots;
	}

private:
	friend class RunContext;
	friend struct NodeCompletion;

	// ── 内部类型 ──
	using SlotMap = std::unordered_map<std::string, TensorSlot>;
	using TaskPortMap = std::unordered_map<TaskId, std::unordered_map<std::string, TaskData>>;
	using TaskBuffer = std::unordered_map<std::string, std::optional<TaskData>>;
	using TaskBufferMap = std::unordered_map<TaskId, TaskBuffer>;

	// ── 内部方法 ──
	/// @brief  惰性创建任务的输入缓冲区（若未存在）。
	void _ensureTaskExists(const TaskId& taskId);
	/// @brief  判断任务所有必需输入（含默认值）是否已就绪。
	bool _isTaskReady(const TaskId& taskId) const;
	/// @brief  执行完整流水线：加载输入 → 清空输出 → preRun → RunFn →
	///         synchronize → postRun → 收集输出 → 通知等待者 → 回调。
	void _checkAndExecute(const TaskId& taskId);
	/// @brief  将任务输入缓冲区数据移动到工作输入槽位（含默认值回退）。
	void _loadTaskToWorkingSlots(const TaskId& taskId);
	/// @brief  清空所有工作输出槽位。
	void _clearWorkingOutputs();
	/// @brief  将工作输出槽位数据移动到任务输出缓冲区。
	void _collectAndSaveOutputs(const TaskId& taskId);
	/// @brief  通知所有等待指定任务完成的协程。
	void _notifyWaiters(const TaskId& taskId, const Result& result);

	// ── RunContext 可调用的内部方法 ──
	/// @brief  从输入槽读取 Value（RunContext::input 委托）。
	const Value& _inputImpl(const std::string& name) const;
	/// @brief  从输入槽移出 Value（RunContext::takeInput 委托），消费后槽位清空。
	Value _takeInputImpl(const std::string& name);
	/// @brief  将 Value 写入输出槽（RunContext::output 委托）。
	void _outputImpl(const std::string& name, Value tensor);
	/// @brief  获取当前引擎的 TensorConverter 钩子指针。
	const TensorConverter* _converter() const;
	/// @brief  获取当前引擎的 EngineDescriptor。
	const EngineDescriptor* _engineDescriptor() const;
	/// @brief  同步引擎异步计算（调用引擎 synchronize 钩子）。
	void _synchronizeEngine() const;
	/// @brief  构造 Ok 状态 Result。
	Result _makeSuccess(std::string message = {}) const;
	/// @brief  构造失败状态 Result。
	Result _makeFailure(Status status, std::string message) const;

	// ── 槽位可变访问（由 friend 类访问）──
	SlotMap& _mutableInputSlots() {
		return _inputSlots;
	}
	SlotMap& _mutableOutputSlots() {
		return _outputSlots;
	}

	// ── 成员 ──
	std::string _type;
	std::string _name;
	Schema _schema;
	ThreadPoolAffinity _affinity = ThreadPoolAffinity::Operator;
	std::string _tag;
	bool _isConnector = false;
	SlotMap _inputSlots; // 工作输入槽位
	SlotMap _outputSlots; // 工作输出槽位
	TaskBufferMap _taskInputs; // 任务级输入缓冲 (port → optional<TaskData>)
	TaskBufferMap _taskOutputs; // 任务级输出缓冲 (port → optional<TaskData>)
	RunFn _fn;
	CompletionFn _onComplete;
	EngineInstance* _engineInstance = nullptr; // 非拥有引用，Registry 管理生命周期
	std::atomic_flag _executionGuard = ATOMIC_FLAG_INIT; // 保护 _inputSlots / _outputSlots 工作槽位，同一时刻仅允许一个 task 使用
	mutable std::shared_mutex _bufferMutex; // 保护 _taskInputs / _taskOutputs 的并发读写
	std::optional<TaskId> _currentTaskId; // 当前执行中的 task（由 tryExecute 设置/清除）

	// 协程等待者：key=taskId，value=等待该 task 完成的协程 handles
	std::unordered_map<TaskId, std::vector<std::coroutine_handle<>>> _waiters;
	mutable std::mutex _waitersMutex; // 保护 _waiters 的并发访问

	// 已完成标记：在 _waitersMutex 保护下充当原子桥接，
	// 解决 await_ready → await_suspend 之间的竞态窗口
	std::unordered_set<TaskId> _completedTasks;
};

// ── RunContext 定义（需在 Node 类体之后）──
class Node::RunContext {
public:
	const Value& input(const std::string& name) const {
		return _node._inputImpl(name);
	}
	/// @brief  消费式取出输入：从工作槽移出 Value（槽位清空），用于连接器零拷贝转发
	Value takeInput(const std::string& name) {
		return _node._takeInputImpl(name);
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
	const Node::Schema& schema() const {
		return _node.schema();
	}
	const std::string& type() const {
		return _node.type();
	}
	const std::string& name() const {
		return _node.name();
	}

private:
	friend class Node;
	explicit RunContext(Node& node) : _node(node) {}
	Node& _node;
};

// ── NodeCompletion：co_await-able 等待器 ──
// co_await node.whenComplete(taskId) 挂起当前协程，
// 直到 tryExecute(taskId) 完成后被 resume，返回执行结果
struct NodeCompletion {
	bool await_ready() const;
	void await_suspend(std::coroutine_handle<> h);
	Node::Result await_resume() const;

private:
	friend class Node;
	NodeCompletion(Node& node, const Node::TaskId& taskId) : _node(&node), _taskId(taskId) {}

	Node* _node;
	Node::TaskId _taskId;
	std::coroutine_handle<> _handle; // await_suspend 时设置
};

} // namespace DC
