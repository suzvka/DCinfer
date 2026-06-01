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
struct TensorConverter {
	std::function<Value(const Tensor&)> toNative;
	std::function<Tensor(const void*)> toDC;
};

using NodeFactory = std::function<std::unique_ptr<class Node>(std::string nodeName, const void* engineConfig)>;

struct EngineDescriptor;
class EngineInstance;
class SignalStore;

// ── 内部组件前向声明 ──
class TaskBuffer;
class SlotWorkspace;
class CoroutineBridge;
class SignalGate;
class EngineAdapter;

enum class ThreadPoolAffinity {
	Compute,
	Operator,
	System,
};

struct NodeCompletion;

// ── 提取为顶层类型的 Node 嵌套类型（消除循环依赖）──

/// @brief 端口定义。
struct NodePort {
	std::string name;
	Tensor::TensorType type = Tensor::TensorType::Void;
	size_t typeSize = 0;
	Tensor::Shape shape;
	bool required = true;
	std::optional<Tensor> defaultValue;
	std::optional<std::string> shapeAnchor;

	template <typename T>
	static NodePort in(std::string name, Tensor::Shape shape = {}) {
		TensorMeta::ensureTypeMap();
		return {std::move(name), DC::Type::getType<Tensor::TensorType, T>(), sizeof(T), std::move(shape), true};
	}

	template <typename T>
	static NodePort optional(std::string name, T defaultValue, Tensor::Shape shape = {}) {
		TensorMeta::ensureTypeMap();
		Tensor dv(DC::Type::getType<Tensor::TensorType, T>(), sizeof(T));
		dv = defaultValue;
		return {std::move(name), DC::Type::getType<Tensor::TensorType, T>(), sizeof(T), std::move(shape), false,
				std::move(dv)};
	}

	template <typename T>
	static NodePort anchored(std::string name, std::string anchorPort, Tensor::Shape shape = {}) {
		TensorMeta::ensureTypeMap();
		NodePort p;
		p.name = std::move(name);
		p.type = DC::Type::getType<Tensor::TensorType, T>();
		p.typeSize = sizeof(T);
		p.shape = std::move(shape);
		p.required = false;
		p.shapeAnchor = std::move(anchorPort);
		return p;
	}

	template <typename T>
	static NodePort out(std::string name, Tensor::Shape shape = {}) {
		TensorMeta::ensureTypeMap();
		return {std::move(name), DC::Type::getType<Tensor::TensorType, T>(), sizeof(T), std::move(shape), true};
	}
};

/// @brief 节点 Schema。
struct NodeSchema {
	std::vector<NodePort> inputs;
	std::vector<NodePort> outputs;

	const NodePort* findInput(const std::string& name) const;
	const NodePort* findOutput(const std::string& name) const;
	bool valid() const;

private:
	static const NodePort* find(const std::vector<NodePort>& ports, const std::string& name);
	static bool hasUniqueNames(const std::vector<NodePort>& ports);
};

/// @brief 节点执行结果状态。
enum class NodeStatus {
	Ok,
	InvalidInput,
	SchemaMismatch,
	ExecutionFailed,
	InternalError
};

/// @brief 节点执行结果。
struct NodeResult {
	NodeStatus status = NodeStatus::Ok;
	std::string message;
	bool ok() const { return status == NodeStatus::Ok; }
};

// ── Node 类 ──

class Node {
public:
	using TensorType = Tensor::TensorType;
	using Shape = Tensor::Shape;
	using TaskId = std::string;
	using TaskData = Value;
	using Port = NodePort;
	using Schema = NodeSchema;
	using Status = NodeStatus;
	using Result = NodeResult;

	class RunContext;

	using RunFn = std::function<Result(RunContext&)>;
	using CompletionFn = std::function<void(const TaskId& taskId, const Result& result)>;

	Node(std::string type, std::string name, Schema schema, RunFn fn, EngineInstance* engineInstance = nullptr,
		 ThreadPoolAffinity affinity = ThreadPoolAffinity::Operator,
		 const EngineDescriptor* engineDesc = nullptr);
	~Node();

	Node(const Node&) = delete;
	Node& operator=(const Node&) = delete;
	Node(Node&&) = delete;
	Node& operator=(Node&&) = delete;

	// ── 只读属性 ──
	const std::string& type() const { return _meta.type; }
	const std::string& name() const { return _meta.name; }
	const Schema& schema() const { return _meta.schema; }

	// ── 线程池归属与分组 ──
	ThreadPoolAffinity affinity() const { return _meta.affinity; }
	void setTag(std::string tag) { _meta.tag = std::move(tag); }
	const std::string& tag() const { return _meta.tag; }
	bool isConnector() const { return _meta.isConnector; }
	void setConnector(bool v) { _meta.isConnector = v; }

	// ── 信号绑定 ──
	void bindSignal(std::shared_ptr<SignalStore> store, std::string name);
	bool isBlocked() const;
	bool isBlocked(const TaskId& taskId) const;

	// ── 模型路径 ──
	const std::string& modelPath() const { return _meta.modelPath; }
	void setModelPath(std::string path) { _meta.modelPath = std::move(path); }

	// ── 完成回调 ──
	void setCompletionCallback(CompletionFn fn);
	bool hasCompletionCallback() const;

	// ── 任务级输入 ──
	void setInput(const TaskId& taskId, const std::string& portName, Value data);
	void setInput(const TaskId& taskId, const std::string& portName, Tensor data);
	void setInput(const TaskId& taskId, std::unordered_map<std::string, TaskData> inputs);
	void setInput(const TaskId& taskId, std::unordered_map<std::string, Tensor> inputs);

	// ── 任务级输出 ──
	bool hasOutput(const TaskId& taskId, const std::string& name) const;
	Value getOutput(const TaskId& taskId, const std::string& name);
	Tensor getOutputTensor(const TaskId& taskId, const std::string& name);
	const Value& peekOutput(const TaskId& taskId, const std::string& name) const;
	std::unordered_map<std::string, TaskData> collectOutputs(const TaskId& taskId);
	std::unordered_map<std::string, Tensor> collectOutputTensors(const TaskId& taskId);

	// ── 协程支持 ──
	NodeCompletion whenComplete(const TaskId& taskId);

	// ── 调度接口 ──
	bool isReady(const TaskId& taskId) const;
	void tryExecute(const TaskId& taskId);
	std::optional<TaskId> currentTaskId() const;

	// ── 任务生命周期 ──
	bool hasTask(const TaskId& taskId) const;
	void clearTask(const TaskId& taskId);
	void terminateTask(const TaskId& taskId);
	size_t taskCount() const;

	// ── 槽位暴露 ──
	const std::unordered_map<std::string, TensorSlot>& inputSlots() const;
	const std::unordered_map<std::string, TensorSlot>& outputSlots() const;

private:
	friend class RunContext;
	friend struct NodeCompletion;

	struct NodeMeta {
		std::string type;
		std::string name;
		Schema schema;
		ThreadPoolAffinity affinity = ThreadPoolAffinity::Operator;
		std::string tag;
		bool isConnector = false;
		std::string modelPath;
		const EngineDescriptor* engineDescriptor = nullptr;
	};

	NodeMeta _meta;
	std::unique_ptr<TaskBuffer> _buffer;
	std::unique_ptr<SlotWorkspace> _workspace;
	std::unique_ptr<CoroutineBridge> _bridge;
	std::unique_ptr<SignalGate> _signal;
	std::unique_ptr<EngineAdapter> _engine;
	RunFn _fn;
	CompletionFn _onComplete;
};

// ── RunContext（方法定义在 Node.cpp，避免内联依赖组件完整类型）──
class Node::RunContext {
public:
	const Value& peek(const std::string& name) const;
	Value pop(const std::string& name);
	void output(const std::string& name, Value tensor);
	const Value* outputRaw(const std::string& name) const;

	Node::Result success(std::string message = {}) const;
	Node::Result failure(Node::Status status, std::string message) const;
	const TensorConverter* converter() const;
	const EngineDescriptor* engineDescriptor() const;
	const EngineInstance* engineInstance() const;
	void* engine() const;
	const Node::Schema& schema() const;
	const std::string& type() const;
	const std::string& name() const;

private:
	friend class Node;
	friend struct ExecutionPipeline;
	RunContext(SlotWorkspace& workspace, EngineAdapter& engine,
			   const Node::Schema& schema, const std::string& type, const std::string& name);

	SlotWorkspace& _workspace;
	EngineAdapter& _engine;
	const Node::Schema& _schema;
	std::string _type;
	std::string _name;
};

// ── NodeCompletion ──
struct NodeCompletion {
	bool await_ready() const;
	void await_suspend(std::coroutine_handle<> h);
	Node::Result await_resume() const;

private:
	friend class Node;
	friend class CoroutineBridge;
	NodeCompletion(TaskBuffer* buffer, CoroutineBridge* bridge,
				   const Node::TaskId& taskId, const Node::Schema* schema)
		: buffer(buffer), bridge(bridge), taskId(taskId), schema(schema) {}

	TaskBuffer* buffer = nullptr;
	CoroutineBridge* bridge = nullptr;
	Node::TaskId taskId;
	const Node::Schema* schema = nullptr;
	std::coroutine_handle<> handle;
};

} // namespace DC
