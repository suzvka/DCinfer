#include "Node.h"
#include "Node/internal/SignalGate.h"
#include "Node/internal/TaskBuffer.h"
#include "Node/internal/SlotWorkspace.h"
#include "Node/internal/EngineAdapter.h"
#include "Node/internal/ExecutionPipeline.h"
#include "EngineRegistry.h"
#include "SignalStore.h"

namespace DC {

// ── NodeSchema 方法实现 ──

const NodePort* NodeSchema::find(const std::vector<NodePort>& ports, const std::string& name) {
	for (const auto& port : ports) {
		if (port.name == name)
			return &port;
	}
	return nullptr;
}

bool NodeSchema::hasUniqueNames(const std::vector<NodePort>& ports) {
	std::unordered_set<std::string> names;
	names.reserve(ports.size());
	for (const auto& port : ports) {
		if (!names.insert(port.name).second)
			return false;
	}
	return true;
}

const NodePort* NodeSchema::findInput(const std::string& name) const {
	return find(inputs, name);
}

const NodePort* NodeSchema::findOutput(const std::string& name) const {
	return find(outputs, name);
}

bool NodeSchema::valid() const {
	if (!hasUniqueNames(inputs) || !hasUniqueNames(outputs))
		return false;
	auto checkTypeSize = [](const std::vector<NodePort>& ports) {
		for (const auto& p : ports) {
			if (p.type != Tensor::TensorType::Void && p.typeSize == 0)
				return false;
		}
		return true;
	};
	if (!checkTypeSize(inputs) || !checkTypeSize(outputs))
		return false;
	for (const auto& p : inputs) {
		if (p.defaultValue.has_value()) {
			const auto& dv = p.defaultValue.value();
			if (dv.type() != p.type || dv.typeSize() != p.typeSize)
				return false;
		}
	}
	for (const auto& p : inputs) {
		if (p.shapeAnchor.has_value()) {
			const auto& anchorName = p.shapeAnchor.value();
			if (anchorName == p.name)
				return false;
			if (!findInput(anchorName))
				return false;
		}
	}
	return true;
}

// ── 构造/析构 ──

Node::Node(std::string type, std::string name, Schema schema, RunFn fn,
		   ThreadPoolAffinity affinity)
	: _fn(std::move(fn)) {
	_meta.type = std::move(type);
	_meta.name = std::move(name);
	_meta.affinity = affinity;
	_meta.schema = std::move(schema);

	_signal = std::make_unique<SignalGate>();
	_engine = std::make_unique<EngineAdapter>(nullptr, nullptr);
}

void Node::bindEngine(std::shared_ptr<EngineInstance> engineInstance, const EngineDescriptor* engineDesc) {
	_meta.engineDescriptor = engineDesc;
	_engine = std::make_unique<EngineAdapter>(std::move(engineInstance), engineDesc);
}

Node::~Node() = default;

// ── 回调注册 ──

void Node::setCompletionCallback(CompletionFn fn) {
	_onComplete = std::move(fn);
}

// ── 信号绑定 ──

void Node::bindSignal(std::shared_ptr<SignalStore> store, std::string name) {
	_signal->bind(std::move(store), std::move(name));
}

bool Node::isBlocked() const {
	return _signal->isBlocked();
}

bool Node::isBlocked(const TaskId& taskId) const {
	if (_blockedOverride)
		return _blockedOverride(taskId);
	return _signal->isBlocked(taskId);
}

// ── 执行依赖访问器 / 调度接口 ──

EngineAdapter& Node::engine() const {
	return *_engine;
}

bool Node::isReady(const TaskId& taskId, const TaskBuffer& buffer) const {
	if (_readyOverride)
		return _readyOverride(taskId);
	return buffer.isReady(taskId, _meta.schema);
}

// ── RunContext 方法实现 ──

const Value& Node::RunContext::peek(const std::string& name) const {
	return _workspace.peekInput(name);
}

Value Node::RunContext::pop(const std::string& name) {
	return _workspace.popInput(name);
}

void Node::RunContext::output(const std::string& name, Value tensor) {
	_workspace.writeOutput(name, std::move(tensor));
}

const Value* Node::RunContext::outputRaw(const std::string& name) const {
	return _workspace.peekOutputRaw(name);
}

Node::Result Node::RunContext::success(std::string message) const {
	Node::Result r;
	r.status = Node::Status::Ok;
	r.message = std::move(message);
	return r;
}

Node::Result Node::RunContext::failure(Node::Status status, std::string message) const {
	Node::Result r;
	r.status = status;
	r.message = std::move(message);
	return r;
}

Node::Result Node::RunContext::failure(Node::Status status, std::string message, Diagnostic diagnostic) const {
	Node::Result r;
	r.status = status;
	r.message = std::move(message);
	r.diagnostic = std::move(diagnostic);
	return r;
}

const TensorConverter* Node::RunContext::converter() const {
	return _engine.converter();
}

const EngineDescriptor* Node::RunContext::engineDescriptor() const {
	return _engine.descriptor();
}

const EngineInstance* Node::RunContext::engineInstance() const {
	return _engine.instance();
}

void* Node::RunContext::engine() const {
	return _engine.engine();
}

const Node::Schema& Node::RunContext::schema() const {
	return _schema;
}

const std::string& Node::RunContext::type() const {
	return _type;
}

const std::string& Node::RunContext::name() const {
	return _name;
}

Node::RunContext::RunContext(SlotWorkspace& workspace, EngineAdapter& engine,
							 const Node::Schema& schema, const std::string& type, const std::string& name)
	: _workspace(workspace), _engine(engine), _schema(schema), _type(type), _name(name) {}

} // namespace DC
