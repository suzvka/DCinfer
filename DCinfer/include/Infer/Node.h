#pragma once

#include <atomic>
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
#include "Value.h"

#include <coroutine>

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

// ── 线程池归属：标识节点应由三层线程池中的哪一个执行 ──
// 仅在 Node 创建时赋值，之后只读
enum class ThreadPoolAffinity {
	Compute,   // 计算线程池 —— 执行推理过程（GPU 加速，如 ONNX/TensorRT）
	Operator,  // 算子线程池 —— 执行 CPU 密集过程（如 Pre/Post 处理）
	System,    // 系统线程池 —— 数据流动与连接器（Broadcast/Routing/Wire）
};

// ── 节点执行策略 ──
enum class ExecutionPolicy {
	Exclusive,    // 同时最多 1 个 task（默认行为）
	Concurrent,   // 多个 task 可并发（无状态节点）
	Serialized,   // 多 task 排队，FIFO 串行
	Batched,      // 攒 N 个 task 后批量执行
};

// ── 前向声明：co_await-able 节点完成通知 ──
struct NodeCompletion;

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
	          EngineInstance* engineInstance = nullptr,
	          ThreadPoolAffinity affinity = ThreadPoolAffinity::Operator);
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

	// ── 线程池归属与分组 ──
	ThreadPoolAffinity affinity() const { return _affinity; }

	/// @brief  设置节点分组标签（用于线程池分组限流）
	void setTag(std::string tag) { _tag = std::move(tag); }
	const std::string& tag() const { return _tag; }

	/// @brief  是否为连接器节点（导线/扇出等基础设施节点）
	bool isConnector() const { return _isConnector; }
	void setConnector(bool v) { _isConnector = v; }

	/// @brief  执行策略（当前默认 Exclusive）
	ExecutionPolicy policy() const { return _execPolicy; }
	void setExecutionPolicy(ExecutionPolicy p) { _execPolicy = p; }

	// ── 完成回调注册 ──
	void setCompletionCallback(CompletionFn fn);
	bool hasCompletionCallback() const;

	// ── 任务级输入 ──

	/// @brief  单端口写入（Value），仅写入缓冲，不触发执行
	/// @return 若端口名存在于 Schema 中则返回 true
	bool setInput(const TaskId& taskId, const std::string& portName, Value data);

	/// @brief  便捷接口：直接传入 DC::Tensor，内部自动包装为 Value
	bool setInput(const TaskId& taskId, const std::string& portName, Tensor data);

	/// @brief  批量写入（Value），仅写入缓冲，不触发执行
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

	// ── 协程支持 ──

	/// @brief  co_await-able 等待器：挂起当前协程，直到指定 task 执行完成
	///         完成后返回 Result，协程自动恢复
	NodeCompletion whenComplete(const TaskId& taskId);

	// ── 调度接口（供 Graph 调用）──

	/// @brief  查询指定任务是否所有必需输入已就绪（含默认值）
	bool isReady(const TaskId& taskId) const;

	/// @brief  尝试执行就绪任务：获取重入锁 → 加载输入 → RunFn → 收集输出
	/// @return true 表示执行成功完成，false 表示未就绪或节点正忙（重入被拒）
	/// @note   线程安全；同一时刻最多一个 task 在执行
	bool tryExecute(const TaskId& taskId);

	/// @brief  返回当前正在执行的任务 ID（用于冒泡事件、任务-节点亲和性）
	/// @return 若节点空闲则返回 std::nullopt
	std::optional<TaskId> currentTaskId() const;

	// ── 任务生命周期 ──
	void   clearTask(const TaskId& taskId);
	size_t taskCount() const;

	// ── 直接访问工作槽位（只读，高级场景）──
	const std::unordered_map<std::string, TensorSlot>& inputSlots()  const { return _inputSlots; }
	const std::unordered_map<std::string, TensorSlot>& outputSlots() const { return _outputSlots; }

private:
	friend class RunContext;
	friend struct NodeCompletion;

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
	void _notifyWaiters(const TaskId& taskId, const Result& result);

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
	ThreadPoolAffinity _affinity = ThreadPoolAffinity::Operator;
	std::string       _tag;
	ExecutionPolicy   _execPolicy = ExecutionPolicy::Exclusive;
	bool              _isConnector = false;
	SlotMap           _inputSlots;     // 工作输入槽位
	SlotMap           _outputSlots;    // 工作输出槽位
	TaskBufferMap     _taskInputs;     // 任务级输入缓冲 (port → optional<TaskData>)
	TaskBufferMap     _taskOutputs;    // 任务级输出缓冲 (port → optional<TaskData>)
	RunFn             _fn;
	CompletionFn      _onComplete;
	EngineInstance*   _engineInstance = nullptr;  // 非拥有引用，Registry 管理生命周期
	std::atomic_flag  _executing = ATOMIC_FLAG_INIT;  // 重入锁：同一时刻最多一个 task 在执行
	std::optional<TaskId> _currentTaskId;            // 当前执行中的 task（由 tryExecute 设置/清除）

	// 协程等待者：key=taskId，value=等待该 task 完成的协程 handles
	std::unordered_map<TaskId, std::vector<std::coroutine_handle<>>> _waiters;
	mutable std::mutex _waitersMutex;  // 保护 _waiters 的并发访问

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

// ── NodeCompletion：co_await-able 等待器 ──
// co_await node.whenComplete(taskId) 挂起当前协程，
// 直到 tryExecute(taskId) 完成后被 resume，返回执行结果
struct NodeCompletion {
	bool await_ready() const;
	void await_suspend(std::coroutine_handle<> h);
	Node::Result await_resume() const;

private:
	friend class Node;
	NodeCompletion(Node& node, const Node::TaskId& taskId)
		: _node(&node), _taskId(taskId) {}

	Node*             _node;
	Node::TaskId      _taskId;
	std::coroutine_handle<> _handle;  // await_suspend 时设置
};

} // namespace DC
