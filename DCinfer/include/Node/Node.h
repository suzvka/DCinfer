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
#include "Diagnostic.h"
#include "ResourceClass.h"

namespace DC {

/// @brief 张量转换钩子：DC::Tensor 与引擎原生张量互转。
struct TensorConverter {
	std::function<Value(const Tensor&)> toNative;
	std::function<Tensor(const void*)> toDC;
};

struct EngineDescriptor;
class EngineInstance;
class SignalStore;
class GraphBuilder;

class TaskBuffer;
class SlotWorkspace;
class SignalGate;
class EngineAdapter;

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

/// @brief 节点工厂参数。
///
/// engineConfig 仅承载用户自定义配置；engineInstance 在 modelPath 路径非空时经
/// Node::bindEngine 绑定，节点持有句柄；schema 依创建路径而定：modelPath 路径由框架推导，
/// createLazyNode 取调用方声明，engineConfig 路径为空。
struct NodeFactoryParams {
	std::string nodeName;
	const void* engineConfig = nullptr;
	std::shared_ptr<class EngineInstance> engineInstance;
	NodeSchema schema;
	std::string modelPath;
};

using NodeFactory = std::function<std::unique_ptr<class Node>(const NodeFactoryParams&)>;

/// @brief 节点执行结果状态；细分错误经 NodeResult::diagnostic 附带上报。
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
	std::optional<Diagnostic> diagnostic;
	bool ok() const { return status == NodeStatus::Ok; }
};

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

	Node(std::string type, std::string name, Schema schema, RunFn fn,
		 ResourceClass affinity = ResourceClass::Operator);
	~Node();

	Node(const Node&) = delete;
	Node& operator=(const Node&) = delete;
	Node(Node&&) = delete;
	Node& operator=(Node&&) = delete;

	/// @brief 绑定引擎实例：节点持有共享句柄，节点存活期间实例存活。
	void bindEngine(std::shared_ptr<EngineInstance> engineInstance, const EngineDescriptor* engineDesc = nullptr);

	const std::string& type() const { return _meta.type; }
	const std::string& name() const { return _meta.name; }
	const Schema& schema() const { return _meta.schema; }

	ResourceClass affinity() const { return _meta.affinity; }

	/// @brief 自由标签，纯序列化元数据，无调度语义。
	void setTag(std::string tag) {
		std::lock_guard lk(_mutationMutex);
		_ensureMutable("Node::setTag");
		_meta.tag = std::move(tag);
	}
	const std::string& tag() const { return _meta.tag; }
	bool isConnector() const { return _meta.isConnector; }

	void setConnector(bool v) {
		std::lock_guard lk(_mutationMutex);
		_ensureMutable("Node::setConnector");
		_meta.isConnector = v;
	}

	/// @brief 追加输出端口，构建期调用：新口按 out_{N} 命名，既有口序不变。
	void appendOutputPort(Port port) {
		std::lock_guard lk(_mutationMutex);
		_ensureMutable("Node::appendOutputPort");
		_meta.schema.outputs.push_back(std::move(port));
	}

	void bindSignal(std::shared_ptr<SignalStore> store, std::string name);
	bool isBlocked() const;
	bool isBlocked(const TaskId& taskId) const;

	/// @brief 注册就绪状态委托；未注册时回退 TaskBuffer 逻辑。
	void setReadyOverride(std::function<bool(const TaskId&)> fn) {
		std::lock_guard lk(_mutationMutex);
		_ensureMutable("Node::setReadyOverride");
		_readyOverride = std::move(fn);
	}

	const std::string& modelPath() const { return _meta.modelPath; }

	void setModelPath(std::string path) {
		std::lock_guard lk(_mutationMutex);
		_ensureMutable("Node::setModelPath");
		_meta.modelPath = std::move(path);
	}

	void setCompletionCallback(CompletionFn fn);

	/// @brief 节点执行函数。
	const RunFn& runFn() const { return _fn; }

	/// @brief 完成回调。
	const CompletionFn& completionCallback() const { return _onComplete; }

	/// @brief 引擎适配器，借用；由节点执行闸串行化。
	EngineAdapter& engine() const;

	/// @brief 就绪判定：readyOverride 优先，否则检查所有必选输入就绪或存在默认值/形状锚定。
	bool isReady(const TaskId& taskId, const TaskBuffer& buffer) const;

	/// @brief 是否注册了就绪覆盖委托。
	bool hasReadyOverride() const { return static_cast<bool>(_readyOverride); }

private:
	friend class RunContext;
	friend class GraphBuilder;

	struct NodeMeta {
		std::string type;
		std::string name;
		Schema schema;
		ResourceClass affinity = ResourceClass::Operator;
		std::string tag;
		bool isConnector = false;
		std::string modelPath;
		const EngineDescriptor* engineDescriptor = nullptr;
	};

	NodeMeta _meta;
	std::unique_ptr<SignalGate> _signal;
	std::unique_ptr<EngineAdapter> _engine;
	RunFn _fn;
	CompletionFn _onComplete;

	std::function<bool(const TaskId&)> _readyOverride;

	// 冻结门：compile 时置位 _frozen，此后一切配置入口抛 NodeException(Frozen)；
	// 互斥门保证封印与在飞 setter 互斥。未入图的独立节点永不封印。
	mutable std::mutex _mutationMutex;
	bool _frozen = false;

	/// @brief 冻结门校验；调用方须持有 _mutationMutex。
	void _ensureMutable(const char* api) const {
		if (_frozen)
			throw NodeException(NodeException::ErrorType::Frozen, api,
								"node '" + _meta.name +
									"' is frozen (graph compiled); node configuration is immutable after freeze");
	}

	/// @brief 封印节点配置面；compile 时调用，一次性。
	void _sealForExecution() {
		std::lock_guard lk(_mutationMutex);
		_frozen = true;
	}
};

// RunContext 方法定义在 Node.cpp，避免内联依赖组件完整类型。
class Node::RunContext {
public:
	const Value& peek(const std::string& name) const;
	Value pop(const std::string& name);
	void output(const std::string& name, Value tensor);
	const Value* outputRaw(const std::string& name) const;

	/// @brief 类型化输入访问器：peek、类型校验与空值检查的组合。
	/// @param error 失败原因输出，可为 nullptr
	/// @return 类型化只读指针，生命周期至本轮 RunFn 返回；失败返回 nullptr
	template <typename T>
	const T* input(const std::string& name, std::string* error = nullptr) const {
		return static_cast<const T*>(_inputChecked(name, ensureSlotType<T>(), error));
	}

	Node::Result success(std::string message = {}) const;
	Node::Result failure(Node::Status status, std::string message) const;
	Node::Result failure(Node::Status status, std::string message, Diagnostic diagnostic) const;
	const TensorConverter* converter() const;
	const EngineDescriptor* engineDescriptor() const;
	const EngineInstance* engineInstance() const;
	void* engine() const;
	const Node::Schema& schema() const;
	const std::string& type() const;
	const std::string& name() const;

	/// @brief 当前 task 标识；RunFn 内等待或轮询以此寻址。
	const TaskId& taskId() const { return _taskId; }

	/// @brief 取消感知，协作式：所属 task 是否已被请求取消或已终止；未注入谓词时恒 false。
	bool isCancellationRequested() const {
		return _cancelProbe ? _cancelProbe() : false;
	}

private:
	friend class Node;
	friend struct ExecutionPipeline;
	RunContext(SlotWorkspace& workspace, EngineAdapter& engine,
			   const Node::Schema& schema, const std::string& type, const std::string& name,
			   TaskId taskId, std::function<bool()> cancelProbe);

	/// @brief input<T> 的非模板实现，定义在 Node.cpp；失败返回 nullptr 并填充 error。
	const void* _inputChecked(const std::string& name, SlotDataType type, std::string* error) const;

	SlotWorkspace& _workspace;
	EngineAdapter& _engine;
	const Node::Schema& _schema;
	std::string _type;
	std::string _name;
	TaskId _taskId;
	std::function<bool()> _cancelProbe;
};

} // namespace DC
