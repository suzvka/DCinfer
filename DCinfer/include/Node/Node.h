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

namespace DC {

/// @brief 张量转换钩子：DC::Tensor ↔ 引擎原生张量。
struct TensorConverter {
	std::function<Value(const Tensor&)> toNative;
	std::function<Tensor(const void*)> toDC;
};

struct EngineDescriptor;
class EngineInstance;
class SignalStore;

// ── 内部组件前向声明 ──
class TaskBuffer;
class SlotWorkspace;
class SignalGate;
class EngineAdapter;

enum class ThreadPoolAffinity {
	Compute,
	Operator,
	System,
};

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

/// @brief 节点工厂参数：框架在 createNode 时收集并传入工厂。
///
/// - engineConfig：createNode(engineType, name, engineConfig) 透传的**用户配置**指针；
///   仅承载用户自定义配置（一字段一语义），引擎实例不经此字段传递。
/// - engineInstance：引擎实例共享句柄（modelPath 路径下非空）；
///   工厂经 Node::bindEngine 绑定后，节点持有句柄，引擎存活期覆盖节点存活期。
/// - schema：框架从 EngineInstance 推导（需引擎注册 getInputPorts/getOutputPorts）；
///   可为空，工厂可自行推导或使用内置 Schema 兜底。
/// - modelPath：createNode(engineType, name, modelPath) 路径下非空。
struct NodeFactoryParams {
	std::string nodeName;
	const void* engineConfig = nullptr;                 ///< 用户自定义配置指针（仅 createNode(engineType,name,engineConfig) 路径非空）
	std::shared_ptr<class EngineInstance> engineInstance; ///< 引擎实例共享句柄（modelPath 路径下非空）
	NodeSchema schema;                  ///< 框架推导的端口 Schema（可为空）
	std::string modelPath;              ///< 模型路径（modelPath 路径下非空）
};

using NodeFactory = std::function<std::unique_ptr<class Node>(const NodeFactoryParams&)>;

/// @brief 节点执行结果状态。
///
/// 保持最小通用词表：后端 / 协议 / 子系统的细分错误不进入本枚举，
/// 经 NodeResult::diagnostic（DC::Diagnostic，domain+code）附带上报。
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
	std::optional<Diagnostic> diagnostic; ///< 领域结构化诊断（可为空；细分分类由产生它的子系统定义）
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

	Node(std::string type, std::string name, Schema schema, RunFn fn,
		 ThreadPoolAffinity affinity = ThreadPoolAffinity::Operator);
	~Node();

	Node(const Node&) = delete;
	Node& operator=(const Node&) = delete;
	Node(Node&&) = delete;
	Node& operator=(Node&&) = delete;

	// ── 引擎绑定（引擎支持节点构造后绑定）──
	/// 节点持有引擎实例共享句柄：节点存活 ⇒ 引擎实例存活，
	/// releaseEngine/releaseAllEngines 移除缓存条目不影响已绑定节点。
	void bindEngine(std::shared_ptr<EngineInstance> engineInstance, const EngineDescriptor* engineDesc = nullptr);

	// ── 只读属性 ──
	const std::string& type() const { return _meta.type; }
	const std::string& name() const { return _meta.name; }
	const Schema& schema() const { return _meta.schema; }

	// ── 线程池归属与元数据 ──
	ThreadPoolAffinity affinity() const { return _meta.affinity; }
	/// @brief  自由标签（纯序列化元数据，随 DCIr JSON/.dcg 往返；无调度语义）
	void setTag(std::string tag) { _meta.tag = std::move(tag); }
	const std::string& tag() const { return _meta.tag; }
	bool isConnector() const { return _meta.isConnector; }
	void setConnector(bool v) { _meta.isConnector = v; }

	// ── 信号绑定 ──
	void bindSignal(std::shared_ptr<SignalStore> store, std::string name);
	bool isBlocked() const;
	bool isBlocked(const TaskId& taskId) const;

	// ── 状态委托（组合节点：子图节点等）──

	/// @brief  注册 task 级阻塞状态委托；注册后 isBlocked(taskId) 转发至此回调。
	///         未注册时回退 SignalGate 逻辑。典型用途：exportNode 产物的
	///         子图节点按内部"声明通路可达性"应答父级。
	void setBlockedOverride(std::function<bool(const TaskId&)> fn) { _blockedOverride = std::move(fn); }

	/// @brief 注册 task 级就绪状态委托；注册后 isReady(taskId, buffer) 转发至此回调。
	///         未注册时回退 TaskBuffer 逻辑。
	void setReadyOverride(std::function<bool(const TaskId&)> fn) { _readyOverride = std::move(fn); }

	// ── 模型路径 ──
	const std::string& modelPath() const { return _meta.modelPath; }
	void setModelPath(std::string path) { _meta.modelPath = std::move(path); }

	// ── 完成回调 ──
	void setCompletionCallback(CompletionFn fn);

	// ── 执行依赖访问器（task 态已归 task 域，pipeline 经此注入）──

	/// @brief  节点执行函数（ExecutionPipeline 注入用）
	const RunFn& runFn() const { return _fn; }

	/// @brief  完成回调（ExecutionPipeline 注入用）
	const CompletionFn& completionCallback() const { return _onComplete; }

	/// @brief  引擎适配器（借用；const Node 仍可驱动引擎——适配器钩子
	///         本身 const，引擎实例经 shared_ptr 共享，由节点执行闸串行化）
	EngineAdapter& engine() const;

	// ── 调度接口 ──
	/// @brief  就绪判定（task 态由外部 task 域持有，经 buffer 传入）：
	///         readyOverride 优先，否则检查所有必选输入已就绪（或存在默认值/形状锚定）
	bool isReady(const TaskId& taskId, const TaskBuffer& buffer) const;

private:
	friend class RunContext;

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
	std::unique_ptr<SignalGate> _signal;
	std::unique_ptr<EngineAdapter> _engine;
	RunFn _fn;
	CompletionFn _onComplete;

	// 状态委托回调（组合节点注册后覆盖默认 isBlocked/isReady 语义）
	std::function<bool(const TaskId&)> _blockedOverride;
	std::function<bool(const TaskId&)> _readyOverride;
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
	Node::Result failure(Node::Status status, std::string message, Diagnostic diagnostic) const;
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

} // namespace DC
