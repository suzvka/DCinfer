#include "EngineRegistry.h"
#include "Node.h"

#include <stdexcept>
#include <memory>

namespace DC {

// ── Builtin 引擎的 TensorConverter（DC::Tensor ↔ NativeTensor）──
static Value builtinToNative(const Tensor& t) {
	auto* p = new Tensor(t);
	return Value(p, [](Tensor* ptr) { delete ptr; });
}

static Tensor builtinToDC(const void* native) {
	return Tensor(*static_cast<const Tensor*>(native));
}

// ── 确保 Builtin 引擎已注册（std::call_once）──
static void ensureBuiltinEngine(EngineRegistry& reg) {
	EngineDescriptor desc;
	desc.engineType = "Builtin";
	desc.converter  = { builtinToNative, builtinToDC };
	// Builtin 节点不通过工厂创建，由 createNode(name, schema, fn) 直接构造
	desc.factory    = nullptr;
	reg.registerEngine(desc);
}

EngineRegistry& EngineRegistry::instance() {
	static EngineRegistry inst;
	static std::once_flag builtinFlag;
	std::call_once(builtinFlag, ensureBuiltinEngine, std::ref(inst));
	return inst;
}

bool EngineRegistry::registerEngine(const EngineDescriptor& desc) {
	if (desc.engineType.empty()) {
		return false;
	}

	if (_engines.contains(desc.engineType)) {
		return false; // 不允许重复注册
	}

	_engines[desc.engineType] = desc;
	return true;
}

std::unique_ptr<Node> EngineRegistry::createNode(
	const std::string& engineType,
	const std::string& nodeName,
	const void* engineConfig) const
{
	auto it = _engines.find(engineType);
	if (it == _engines.end()) {
		return nullptr;
	}

	if (!it->second.factory) {
		return nullptr;
	}

	return it->second.factory(nodeName, engineConfig);
}

std::unique_ptr<Node> EngineRegistry::createNode(
	const std::string& nodeName,
	Node::Schema schema,
	Node::RunFn fn) const
{
	return std::make_unique<Node>("Builtin", nodeName,
		std::move(schema), std::move(fn));
}

std::unique_ptr<Node> EngineRegistry::createNode(
	const std::string& engineType,
	const std::string& nodeName,
	const std::string& modelPath)
{
	auto it = _engines.find(engineType);
	if (it == _engines.end())  return nullptr;
	if (!it->second.factory)   return nullptr;

	// 获取或创建引擎实例（生命周期由 Registry 管理）
	auto* engineInstance = getOrCreateEngine(engineType, modelPath);
	if (!engineInstance) return nullptr;

	// 从模型推导 Schema（loadModel 与 createEngine 可独立实现）
	Node::Schema schema;
	if (it->second.loadModel && it->second.getInputPorts && it->second.getOutputPorts) {
		auto handle = it->second.loadModel(modelPath);
		if (handle) {
			schema.inputs  = it->second.getInputPorts(handle);
			schema.outputs = it->second.getOutputPorts(handle);
		}
	}

	// 将 EngineInstance* 通过 engineConfig 传给工厂
	// 使用 makeNodeFactoryWithEngine 注册的工厂会正确提取
	return it->second.factory(nodeName, engineInstance);
}

// ── 引擎实例管理 ──

std::string EngineRegistry::_makeEngineKey(
	const std::string& engineType,
	const std::string& modelPath)
{
	return engineType + ":" + modelPath;
}

EngineInstance* EngineRegistry::getOrCreateEngine(
	const std::string& engineType,
	const std::string& modelPath)
{
	auto key = _makeEngineKey(engineType, modelPath);
	auto it = _engineInstances.find(key);
	if (it != _engineInstances.end()) {
		return &it->second;
	}

	auto engIt = _engines.find(engineType);
	if (engIt == _engines.end())        return nullptr;
	if (!engIt->second.createEngine)    return nullptr;

	auto instance = engIt->second.createEngine(modelPath);
	if (!instance) return nullptr;

	auto [insertedIt, ok] = _engineInstances.emplace(
		std::move(key), std::move(instance));
	return &insertedIt->second;
}

void EngineRegistry::releaseEngine(
	const std::string& engineType,
	const std::string& modelPath)
{
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
	for (auto& [key, instance] : _engineInstances) {
		auto* desc = instance.descriptor();
		if (desc && desc->releaseEngine) {
			desc->releaseEngine(instance.get());
		}
	}
	_engineInstances.clear();
}

const EngineDescriptor* EngineRegistry::find(const std::string& engineType) const {
	auto it = _engines.find(engineType);
	if (it == _engines.end()) {
		return nullptr;
	}

	return &it->second;
}

bool EngineRegistry::hasEngine(const std::string& engineType) const {
	return _engines.contains(engineType);
}

std::vector<std::string> EngineRegistry::engineTypes() const {
	std::vector<std::string> types;
	types.reserve(_engines.size());
	for (const auto& [type, desc] : _engines) {
		types.push_back(type);
	}
	return types;
}

// ── 算子注册 ──

bool EngineRegistry::registerOperator(
	const std::string& operatorName,
	Node::Schema schema,
	Node::RunFn fn)
{
	if (operatorName.empty()) return false;
	if (_engines.contains(operatorName)) return false;

	EngineDescriptor desc;
	desc.engineType = operatorName;
	desc.converter  = { builtinToNative, builtinToDC };

	// 工厂：捕获 schema 和 fn，创建 Builtin 节点
	desc.factory = [schema = std::move(schema), fn = std::move(fn)](
		std::string name, const void* /*engineConfig*/)
		-> std::unique_ptr<Node>
	{
		return std::make_unique<Node>("Builtin", std::move(name), schema, fn);
	};

	_engines[operatorName] = std::move(desc);
	return true;
}

std::unique_ptr<Node> EngineRegistry::createOperator(
	const std::string& operatorName,
	const std::string& nodeName) const
{
	auto it = _engines.find(operatorName);
	if (it == _engines.end() || !it->second.factory) {
		return nullptr;
	}
	return it->second.factory(nodeName, nullptr);
}

} // namespace DC
