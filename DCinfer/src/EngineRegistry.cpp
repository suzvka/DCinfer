#include "EngineRegistry.h"
#include "InferNode.h"

#include <mutex>
#include <stdexcept>

namespace DC {

// ── Builtin 引擎的 TensorConverter（DC::Tensor ↔ NativeTensor）──
static NativeTensor builtinToNative(const Tensor& t) {
	auto* p = new Tensor(t);
	return NativeTensor(p, [](Tensor* ptr) { delete ptr; });
}

static Tensor builtinToDC(const void* native) {
	return Tensor(*static_cast<const Tensor*>(native));
}

static bool builtinCanAccept(const std::string& tag) {
	return tag == "DC::Tensor";
}

// ── 确保 Builtin 引擎已注册（std::call_once）──
static void ensureBuiltinEngine(EngineRegistry& reg) {
	EngineDescriptor desc;
	desc.engineType = "Builtin";
	desc.converter  = { builtinToNative, builtinToDC, "DC::Tensor", builtinCanAccept };
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

	std::unique_lock lock(_mutex);

	if (_engines.contains(desc.engineType)) {
		return false; // 不允许重复注册
	}

	_engines[desc.engineType] = desc;
	return true;
}

std::unique_ptr<InferNode> EngineRegistry::createNode(
	const std::string& engineType,
	const std::string& nodeName,
	const void* engineConfig) const
{
	std::shared_lock lock(_mutex);

	auto it = _engines.find(engineType);
	if (it == _engines.end()) {
		return nullptr;
	}

	if (!it->second.factory) {
		return nullptr;
	}

	return it->second.factory(nodeName, engineConfig);
}

std::unique_ptr<InferNode> EngineRegistry::createNode(
	const std::string& nodeName,
	InferNode::Schema schema,
	std::function<InferNode::Result(InferNode&)> fn) const
{
	return std::make_unique<InferNode>("Builtin", nodeName,
		std::move(schema), std::move(fn));
}

const EngineDescriptor* EngineRegistry::find(const std::string& engineType) const {
	std::shared_lock lock(_mutex);

	auto it = _engines.find(engineType);
	if (it == _engines.end()) {
		return nullptr;
	}

	return &it->second;
}

bool EngineRegistry::hasEngine(const std::string& engineType) const {
	std::shared_lock lock(_mutex);
	return _engines.contains(engineType);
}

std::vector<std::string> EngineRegistry::engineTypes() const {
	std::shared_lock lock(_mutex);

	std::vector<std::string> types;
	types.reserve(_engines.size());
	for (const auto& [type, desc] : _engines) {
		types.push_back(type);
	}
	return types;
}

} // namespace DC
