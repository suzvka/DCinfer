#include "EnvRegistry.h"

namespace DC {

EnvRegistry& EnvRegistry::instance() {
	static EnvRegistry inst;
	return inst;
}

bool EnvRegistry::registerEnv(const std::string& envType,
							  std::function<std::shared_ptr<void>()> factory,
							  std::function<void(void*)> cleanup) {
	if (envType.empty())
		return false;
	if (!factory)
		return false;
	if (_factories.contains(envType))
		return false;

	_factories[envType] = {std::move(factory), std::move(cleanup)};
	return true;
}

void* EnvRegistry::getOrCreate(const std::string& envType) {
	// 已缓存则直接返回
	auto instIt = _instances.find(envType);
	if (instIt != _instances.end())
		return instIt->second.get();

	// 未缓存则从工厂创建
	auto factIt = _factories.find(envType);
	if (factIt == _factories.end())
		return nullptr;

	auto instance = factIt->second.factory();
	if (!instance)
		return nullptr;

	auto [it, _] = _instances.emplace(envType, std::move(instance));
	return it->second.get();
}

void EnvRegistry::release(const std::string& envType) {
	auto instIt = _instances.find(envType);
	if (instIt != _instances.end()) {
		auto factIt = _factories.find(envType);
		if (factIt != _factories.end() && factIt->second.cleanup) {
			factIt->second.cleanup(instIt->second.get());
		}
	}
	_instances.erase(envType);
}

void EnvRegistry::releaseAll() {
	for (auto& [envType, instance] : _instances) {
		auto factIt = _factories.find(envType);
		if (factIt != _factories.end() && factIt->second.cleanup) {
			factIt->second.cleanup(instance.get());
		}
	}
	_instances.clear();
}

bool EnvRegistry::hasEnv(const std::string& envType) const {
	return _factories.contains(envType);
}

} // namespace DC
