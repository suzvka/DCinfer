#include "EngineRegistry.h"
#include "Node.h"

#include <stdexcept>
#include <memory>

namespace DC {

// ── EngineInstance 生命周期 ──

EngineInstance::~EngineInstance() {
	// 释放钩子：仅在最后一个共享句柄析构时调用一次（移动构造移交后
	// 源对象 _desc 置空，不会双发）。钩子在 _engine 成员仍存活的阶段
	// 执行，拿到原生指针做有序清理；随后 shared_ptr<void> 正常析构原生对象。
	if (_desc && _engine && _desc->releaseEngine) {
		_desc->releaseEngine(_engine.get());
	}
}

// ── Builtin 引擎的 TensorConverter（DC::Tensor ↔ NativeTensor）──
static Value builtinToNative(const Tensor& t) {
	return Value(std::make_unique<Tensor>(t));
}

static Tensor builtinToDC(const void* native) {
	return Tensor(*static_cast<const Tensor*>(native));
}

// ── 确保 Builtin 引擎已注册（std::call_once）──
static void ensureBuiltinEngine(EngineRegistry& reg) {
	EngineDescriptor desc;
	desc.engineType = "Builtin";
	desc.converter = {builtinToNative, builtinToDC};
	// Builtin 节点不通过工厂创建，由 createNode(name, schema, fn) 直接构造
	desc.factory = nullptr;
	reg.registerEngine(desc);
}

EngineRegistry& EngineRegistry::instance() {
	static EngineRegistry inst;
	static std::once_flag builtinFlag;
	std::call_once(builtinFlag, ensureBuiltinEngine, std::ref(inst));
	return inst;
}

bool EngineRegistry::registerEngine(const EngineDescriptor& desc) {
	std::lock_guard lk(_mutex);
	if (desc.engineType.empty()) {
		return false;
	}

	if (_engines.contains(desc.engineType)) {
		return false; // 不允许重复注册
	}

	_engines[desc.engineType] = desc;
	return true;
}

std::unique_ptr<Node> EngineRegistry::createNode(const std::string& engineType, const std::string& nodeName,
												 const void* engineConfig) const {
	// 锁内拷贝工厂，锁外调用（factory 是用户回调，可能重入注册表）
	NodeFactory factory;
	{
		std::lock_guard lk(_mutex);
		auto it = _engines.find(engineType);
		if (it == _engines.end() || !it->second.factory) {
			return nullptr;
		}
		factory = it->second.factory;
	}

	NodeFactoryParams params;
	params.nodeName = nodeName;
	params.engineConfig = engineConfig;
	return factory(params);
}

std::unique_ptr<Node> EngineRegistry::createNode(const std::string& nodeName, Node::Schema schema,
												 Node::RunFn fn) const {
	return std::make_unique<Node>("Builtin", nodeName, std::move(schema), std::move(fn),
								  ThreadPoolAffinity::Operator);
}

std::unique_ptr<Node> EngineRegistry::createNode(const std::string& engineType, const std::string& nodeName,
												 const std::string& modelPath) {
	// 单路径：一次加载并缓存引擎实例 → 从实例推导 Schema → factory 构造节点
	auto engineInstance = getOrCreateEngine(engineType, modelPath);
	if (!engineInstance)
		return nullptr;

	// 锁内拷贝钩子与工厂，锁外调用（防重入死锁）
	std::function<std::vector<Node::Port>(const EngineInstance&)> getInputs;
	std::function<std::vector<Node::Port>(const EngineInstance&)> getOutputs;
	NodeFactory factory;
	{
		std::lock_guard lk(_mutex);
		auto it = _engines.find(engineType);
		if (it == _engines.end() || !it->second.factory)
			return nullptr;
		factory = it->second.factory;
		getInputs = it->second.getInputPorts;
		getOutputs = it->second.getOutputPorts;
	}

	// 从实例推导 Schema（引擎未注册端口推导钩子时留空，由工厂兜底）
	Node::Schema schema;
	if (getInputs && getOutputs) {
		schema.inputs = getInputs(*engineInstance);
		schema.outputs = getOutputs(*engineInstance);
	}

	NodeFactoryParams params;
	params.nodeName = nodeName;
	params.engineInstance = engineInstance;    // 引擎实例一律经共享句柄（engineConfig 仅用户配置，不再承载实例指针）
	params.schema = std::move(schema);
	params.modelPath = modelPath;

	auto node = factory(params);
	if (node)
		node->setModelPath(modelPath);
	return node;
}

// ── 引擎实例管理 ──

std::string EngineRegistry::_makeEngineKey(const std::string& engineType, const std::string& modelPath) {
	return engineType + ":" + modelPath;
}

EngineHandle EngineRegistry::getOrCreateEngine(const std::string& engineType, const std::string& modelPath) {
	auto key = _makeEngineKey(engineType, modelPath);

	// 快路径：ready 缓存命中（无创建，锁开销极小）
	{
		std::lock_guard lk(_mutex);
		auto it = _engineInstances.find(key);
		if (it != _engineInstances.end() && it->second.ready)
			return it->second.ready;
	}

	// single-flight 登记：同 key 首个调用者成为领导者，其余成为跟随者
	std::promise<EngineHandle> promise;
	std::shared_future<EngineHandle> myFuture = promise.get_future().share();
	bool leader = false;
	std::function<EngineInstance(const std::string&)> createEngine;
	{
		std::lock_guard lk(_mutex);
		auto& slot = _engineInstances[key]; // 按需创建空槽位
		if (slot.ready) {
			return slot.ready; // 双重检查：登记竞态期间他人已完成创建
		}
		if (slot.loading.valid()) {
			myFuture = slot.loading; // 跟随者：等待首个创建者的同一结果
		} else {
			auto engIt = _engines.find(engineType);
			if (engIt == _engines.end() || !engIt->second.createEngine) {
				_engineInstances.erase(key); // 未注册：不留空槽位
				return nullptr;
			}
			slot.loading = myFuture; // 领导者登记 loading 条目
			createEngine = engIt->second.createEngine;
			leader = true;
		}
	}

	if (!leader) {
		// 跟随者：阻塞等待；成功拿句柄，失败透传领导者异常
		return myFuture.get();
	}

	// 领导者：锁外执行创建回调——ORT Session 加载等秒级操作不持有 _mutex，
	// backend 可安全重入 registry，其他 key 的创建/建图互不阻塞。
	EngineHandle handle;
	std::exception_ptr error;
	try {
		auto instance = createEngine(modelPath);
		if (instance) {
			handle = std::make_shared<EngineInstance>(std::move(instance));
			// 注入所属描述符（权威值，覆盖构造时传入值）；find 内部加锁，此处不持 _mutex。
			// _engines 注册后不擦除，节点地址稳定，指针可安全长存。
			if (const EngineDescriptor* desc = find(engineType))
				handle->setDescriptor(desc);
		}
	} catch (...) {
		error = std::current_exception();
	}

	// 发布：锁内写 ready / 清失败槽位；set_value 唤醒等待者放锁外，
	// 避免被唤醒者立即抢锁阻塞发布者自身
	{
		std::lock_guard lk(_mutex);
		if (handle) {
			auto& slot = _engineInstances[key];
			slot.ready = handle;
			slot.loading = {}; // 清除 loading 条目（valid() 变 false）
		} else {
			auto it = _engineInstances.find(key);
			if (it != _engineInstances.end() && !it->second.ready)
				_engineInstances.erase(it); // 失败：清槽位，后续调用重试创建
		}
	}
	if (error)
		promise.set_exception(std::move(error));
	else
		promise.set_value(handle);

	if (error)
		std::rethrow_exception(error); // 保持旧行为：创建异常透传给首个调用者
	return handle;
}

void EngineRegistry::releaseEngine(const std::string& engineType, const std::string& modelPath) {
	// 仅移除注册表缓存条目，不直接销毁实例：仍被节点持有的共享句柄
	// 保持实例存活，实际销毁（含 releaseEngine 钩子）发生在最后一个
	// 句柄释放时——与调用方无需任何释放顺序约定。
	// single-flight 创建中的条目（loading）无可释放对象，保留槽位，
	// 待领导者完成发布后由后续 release 生效。
	std::lock_guard lk(_mutex);
	auto it = _engineInstances.find(_makeEngineKey(engineType, modelPath));
	if (it == _engineInstances.end() || it->second.loading.valid())
		return;
	_engineInstances.erase(it);
}

void EngineRegistry::releaseAllEngines() {
	// 语义同 releaseEngine：逐条目移除缓存，实例销毁由句柄引用计数决定。
	// 旧实现在此处同步调用 releaseEngine 钩子，会悬空仍绑定实例的节点。
	// 创建中（loading）条目跳过，待创建完成后由后续 release 处理。
	std::lock_guard lk(_mutex);
	for (auto it = _engineInstances.begin(); it != _engineInstances.end();) {
		if (it->second.loading.valid()) {
			++it;
			continue;
		}
		it = _engineInstances.erase(it);
	}
}

const EngineDescriptor* EngineRegistry::find(const std::string& engineType) const {
	std::lock_guard lk(_mutex);
	auto it = _engines.find(engineType);
	if (it == _engines.end()) {
		return nullptr;
	}
	// 返回的指针指向 map 节点；_engines 注册后不擦除，地址稳定
	return &it->second;
}

bool EngineRegistry::hasEngine(const std::string& engineType) const {
	std::lock_guard lk(_mutex);
	return _engines.contains(engineType);
}

std::vector<std::string> EngineRegistry::engineTypes() const {
	std::lock_guard lk(_mutex);
	std::vector<std::string> types;
	types.reserve(_engines.size());
	for (const auto& [type, desc] : _engines) {
		types.push_back(type);
	}
	return types;
}

// ── 算子注册 ──

bool EngineRegistry::registerOperator(const std::string& operatorName, Node::Schema schema, Node::RunFn fn) {
	if (operatorName.empty())
		return false;
	{
		std::lock_guard lk(_mutex);
		if (_engines.contains(operatorName))
			return false;
	}

	EngineDescriptor desc;
	desc.engineType = operatorName;
	desc.converter = {builtinToNative, builtinToDC};

	// 工厂：捕获 schema 和 fn，创建算子节点
	desc.factory = [schema = std::move(schema),
					fn = std::move(fn)](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		return std::make_unique<Node>("Builtin", p.nodeName, schema, fn, ThreadPoolAffinity::Operator);
	};

	std::lock_guard lk(_mutex);
	_engines[operatorName] = std::move(desc);
	return true;
}

std::unique_ptr<Node> EngineRegistry::createOperator(const std::string& operatorName,
													 const std::string& nodeName) const {
	NodeFactory factory;
	{
		std::lock_guard lk(_mutex);
		auto it = _engines.find(operatorName);
		if (it == _engines.end() || !it->second.factory) {
			return nullptr;
		}
		factory = it->second.factory;
	}

	NodeFactoryParams params;
	params.nodeName = nodeName;
	return factory(params);
}

} // namespace DC
