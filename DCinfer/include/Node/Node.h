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
class GraphBuilder;

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
	/// @throws NodeException(Frozen) 若节点所在图已冻结
	void bindEngine(std::shared_ptr<EngineInstance> engineInstance, const EngineDescriptor* engineDesc = nullptr);

	// ── 只读属性 ──
	const std::string& type() const { return _meta.type; }
	const std::string& name() const { return _meta.name; }
	const Schema& schema() const { return _meta.schema; }

	// ── 线程池归属与元数据 ──
	ThreadPoolAffinity affinity() const { return _meta.affinity; }
	/// @brief  自由标签（纯序列化元数据，随 JSON/.dcg 图序列化往返；无调度语义）
	/// @throws NodeException(Frozen) 若节点所在图已冻结
	void setTag(std::string tag) {
		std::lock_guard lk(_mutationMutex);
		_ensureMutable("Node::setTag");
		_meta.tag = std::move(tag);
	}
	const std::string& tag() const { return _meta.tag; }
	bool isConnector() const { return _meta.isConnector; }
	/// @throws NodeException(Frozen) 若节点所在图已冻结
	void setConnector(bool v) {
		std::lock_guard lk(_mutationMutex);
		_ensureMutable("Node::setConnector");
		_meta.isConnector = v;
	}

	// ── 信号绑定 ──
	/// @throws NodeException(Frozen) 若节点所在图已冻结
	void bindSignal(std::shared_ptr<SignalStore> store, std::string name);
	bool isBlocked() const;
	bool isBlocked(const TaskId& taskId) const;

	// ── 状态委托（组合节点：子图节点等）──

	/// @brief  注册 task 级阻塞状态委托；注册后 isBlocked(taskId) 转发至此回调。
	///         未注册时回退 SignalGate 逻辑。典型用途：exportNode 产物的
	///         子图节点按内部"声明通路可达性"应答父级。
	/// @throws NodeException(Frozen) 若节点所在图已冻结
	void setBlockedOverride(std::function<bool(const TaskId&)> fn) {
		std::lock_guard lk(_mutationMutex);
		_ensureMutable("Node::setBlockedOverride");
		_blockedOverride = std::move(fn);
	}

	/// @brief 注册 task 级就绪状态委托；注册后 isReady(taskId, buffer) 转发至此回调。
	///         未注册时回退 TaskBuffer 逻辑。
	/// @throws NodeException(Frozen) 若节点所在图已冻结
	void setReadyOverride(std::function<bool(const TaskId&)> fn) {
		std::lock_guard lk(_mutationMutex);
		_ensureMutable("Node::setReadyOverride");
		_readyOverride = std::move(fn);
	}

	// ── 模型路径 ──
	const std::string& modelPath() const { return _meta.modelPath; }
	/// @throws NodeException(Frozen) 若节点所在图已冻结
	void setModelPath(std::string path) {
		std::lock_guard lk(_mutationMutex);
		_ensureMutable("Node::setModelPath");
		_meta.modelPath = std::move(path);
	}

	// ── 完成回调 ──
	/// @throws NodeException(Frozen) 若节点所在图已冻结
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

	/// @brief  是否注册了就绪覆盖委托（调度侧据此选择原子就绪路径或委托路径；
	///         冻结后注册面关闭，运行期只读安全）
	bool hasReadyOverride() const { return static_cast<bool>(_readyOverride); }

private:
	friend class RunContext;
	friend class GraphBuilder;

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

	// ── 冻结门（Build → Freeze 边界）──
	//
	// compile()（冻结）时由 GraphBuilder::_sealForExecution 置位：此后一切
	// 配置入口抛 NodeException(Frozen)——“执行期节点配置不可变”由 API 边界
	// 强制，而非依赖调用方自律。用互斥门而非裸原子布尔：封印与在飞 setter
	// 互斥——通过校验的写完成于封印之前，封印后的调用确定被拒绝，同时关闭
	// 与执行流水线（isReady 读 _readyOverride、ExecutionPipeline 读
	// completionCallback）的竞争窗口。
	//
	// 未加入任何图的独立节点（NodeExecutor / NetServerAdapter 单节点路径）
	// 永不封印，配置语义不变。
	mutable std::mutex _mutationMutex;
	bool _frozen = false;

	/// @brief 冻结门校验（调用方须持有 _mutationMutex）；冻结后抛 NodeException(Frozen)
	void _ensureMutable(const char* api) const {
		if (_frozen)
			throw NodeException(NodeException::ErrorType::Frozen, api,
								"node '" + _meta.name +
									"' is frozen (graph compiled); node configuration is immutable after freeze");
	}

	/// @brief 封印节点配置面（compile 时由 GraphBuilder 调用；一次性，不可回退）
	void _sealForExecution() {
		std::lock_guard lk(_mutationMutex);
		_frozen = true;
	}
};

// ── RunContext（方法定义在 Node.cpp，避免内联依赖组件完整类型）──
class Node::RunContext {
public:
	const Value& peek(const std::string& name) const;
	Value pop(const std::string& name);
	void output(const std::string& name, Value tensor);
	const Value* outputRaw(const std::string& name) const;

	/// @brief  类型化输入访问器：peek + 类型标签校验 + 空值检查的组合，
	///         替代手动 "peek → as<T> → 判空" 三步样板。
	/// @param  name  输入端口名
	/// @param  error 失败原因输出（可为 nullptr）；覆盖端口不存在、数据未到达、
	///               类型标签不匹配（Value 持有其他原生类型）、空值
	/// @return 类型化只读指针（生命周期随槽位，至本轮 RunFn 返回）；失败返回 nullptr
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

	/// @brief  当前 task 标识（图路径为所属轮次 taskId；单节点路径为调用方给定值）。
	/// @note   子图嵌套场景该 ID 空间贯穿父子边界（exportNode 以父任务 ID 驱动子图），
	///         RunFn 内部等待/轮询以本 ID 寻址。
	const TaskId& taskId() const { return _taskId; }

	/// @brief  取消感知（协作式）：所属 task 是否已被请求取消/已终止。
	///         RunFn 中的长等待应周期性轮询本谓词并及时解围；未注入谓词的
	///         执行路径恒 false（单节点执行无取消语义）。
	bool isCancellationRequested() const {
		return _cancelProbe ? _cancelProbe() : false;
	}

private:
	friend class Node;
	friend struct ExecutionPipeline;
	RunContext(SlotWorkspace& workspace, EngineAdapter& engine,
			   const Node::Schema& schema, const std::string& type, const std::string& name,
			   TaskId taskId, std::function<bool()> cancelProbe);

	/// @brief 类型化输入检查的非模板实现（定义在 Node.cpp）：成功返回原生指针，
	///        失败返回 nullptr 并可选填充 error（模板 input<T> 薄包装）
	const void* _inputChecked(const std::string& name, SlotDataType type, std::string* error) const;

	SlotWorkspace& _workspace;
	EngineAdapter& _engine;
	const Node::Schema& _schema;
	std::string _type;
	std::string _name;
	TaskId _taskId;                        ///< 所属 task（子图嵌套时贯穿父子边界）
	std::function<bool()> _cancelProbe;    ///< 取消感知谓词（可空：单节点路径无取消语义）
};

} // namespace DC
