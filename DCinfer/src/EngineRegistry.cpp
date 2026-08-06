#include "EngineRegistry.h"
#include "Node.h"

#include <stdexcept>
#include <memory>

namespace DC {

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
	auto* engineInstance = getOrCreateEngine(engineType, modelPath);
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
	params.engineConfig = engineInstance;
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

EngineInstance* EngineRegistry::getOrCreateEngine(const std::string& engineType, const std::string& modelPath) {
	auto key = _makeEngineKey(engineType, modelPath);

	// 快速路径：已缓存直接返回
	{
		std::lock_guard lk(_mutex);
		auto it = _engineInstances.find(key);
		if (it != _engineInstances.end()) {
			return &it->second;
		}
	}

	// 锁内拷贝 createEngine 钩子与所属描述符指针，锁外创建实例
	std::function<EngineInstance(const std::string&)> createFn;
	const EngineDescriptor* descPtr = nullptr;
	{
		std::lock_guard lk(_mutex);
		auto engIt = _engines.find(engineType);
		if (engIt == _engines.end() || !engIt->second.createEngine)
			return nullptr;
		createFn = engIt->second.createEngine;
		descPtr = &engIt->second;
	}

	auto instance = createFn(modelPath);
	if (!instance)
		return nullptr;

	// 双重检查：并发下其他线程可能已插入
	{
		std::lock_guard lk(_mutex);
		auto it = _engineInstances.find(key);
		if (it != _engineInstances.end()) {
			return &it->second;
		}

		auto [insertedIt, ok] = _engineInstances.emplace(std::move(key), std::move(instance));
		// 注入所属描述符（权威值，覆盖构造时传入值）。
		// _engines 注册后不擦除，节点地址稳定，指针可安全长存。
		insertedIt->second.setDescriptor(descPtr);
		return &insertedIt->second;
	}
}

void EngineRegistry::releaseEngine(const std::string& engineType, const std::string& modelPath) {
	std::lock_guard lk(_mutex);
	auto key = _makeEngineKey(engineType, modelPath);
	auto it = _engineInstances.find(key);
	if (it != _engineInstances.end()) {
		auto* desc = it->second.descriptor();
		if (desc && desc->releaseEngine) {
			desc->releaseEngine(it->second.get());
		}
	}
	_engineInstances.erase(key);
}

void EngineRegistry::releaseAllEngines() {
	std::lock_guard lk(_mutex);
	for (auto& [key, instance] : _engineInstances) {
		auto* desc = instance.descriptor();
		if (desc && desc->releaseEngine) {
			desc->releaseEngine(instance.get());
		}
	}
	_engineInstances.clear();
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
