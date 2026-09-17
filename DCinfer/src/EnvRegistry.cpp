#include "EnvRegistry.h"

#include <vector>

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

	std::lock_guard lk(_mutex);
	if (_factories.contains(envType))
		return false;

	_factories[envType] = {std::move(factory), std::move(cleanup)};
	return true;
}

std::shared_ptr<void> EnvRegistry::getOrCreate(const std::string& envType) {
	// 锁内拷贝工厂，锁外调用（factory 是用户回调，可能重入注册表）
	std::function<std::shared_ptr<void>()> factory;
	{
		std::lock_guard lk(_mutex);
		// 已缓存则直接返回共享句柄
		auto instIt = _instances.find(envType);
		if (instIt != _instances.end())
			return instIt->second;

		auto factIt = _factories.find(envType);
		if (factIt == _factories.end())
			return nullptr;
		factory = factIt->second.factory;
	}

	// 未缓存则从工厂创建；并发下可能重复创建，emplace 先到者胜出，
	// 落选实例随局部句柄析构（不在锁内销毁）
	auto instance = factory();
	if (!instance)
		return nullptr;

	std::lock_guard lk(_mutex);
	auto [it, _] = _instances.emplace(envType, std::move(instance));
	return it->second;
}

void EnvRegistry::release(const std::string& envType) {
	// 待清理实例移入局部句柄、锁外调用 cleanup 与析构
	std::shared_ptr<void> doomed;
	std::function<void(void*)> cleanup;
	{
		std::lock_guard lk(_mutex);
		auto instIt = _instances.find(envType);
		if (instIt == _instances.end())
			return;
		auto factIt = _factories.find(envType);
		if (factIt != _factories.end())
			cleanup = factIt->second.cleanup;
		doomed = std::move(instIt->second);
		_instances.erase(instIt);
	}
	if (cleanup && doomed)
		cleanup(doomed.get());
}

void EnvRegistry::releaseAll() {
	// 待清理实例批量移入局部容器、锁外调用 cleanup 与析构
	std::vector<std::pair<std::shared_ptr<void>, std::function<void(void*)>>> doomed;
	{
		std::lock_guard lk(_mutex);
		doomed.reserve(_instances.size());
		for (auto& [envType, instance] : _instances) {
			auto factIt = _factories.find(envType);
			std::function<void(void*)> cleanup;
			if (factIt != _factories.end())
				cleanup = factIt->second.cleanup;
			doomed.emplace_back(std::move(instance), std::move(cleanup));
		}
		_instances.clear();
	}
	for (auto& [instance, cleanup] : doomed) {
		if (cleanup && instance)
			cleanup(instance.get());
	}
}

bool EnvRegistry::hasEnv(const std::string& envType) const {
	std::lock_guard lk(_mutex);
	return _factories.contains(envType);
}

} // namespace DC
