#include "ResourceScheduler.h"

#include <optional>
#include <stdexcept>

namespace DC {

// ════════════════════════════════════════════
// 进程级默认实例
// ════════════════════════════════════════════

namespace {
	/// 默认实例状态（进程级）：互斥 + 当前实例 + 预配置。
	/// shared_ptr 全局句柄在进程退出时释放（若届时已无使用方引用，
	/// 实例析构 → shutdown → join 全部 worker，顺序安全）。
	std::mutex g_instanceMutex;
	std::shared_ptr<ResourceScheduler> g_instance;
	std::optional<SchedulerConfig> g_pendingConfig;
} // namespace

std::shared_ptr<ResourceScheduler> ResourceScheduler::instance() {
	std::lock_guard lk(g_instanceMutex);
	if (!g_instance) {
		g_instance = std::make_shared<ResourceScheduler>(
			g_pendingConfig.value_or(SchedulerConfig{}));
		g_pendingConfig.reset();
	}
	return g_instance;
}

bool ResourceScheduler::configureInstance(const SchedulerConfig& config) {
	if (!config.valid())
		throw std::invalid_argument("ResourceScheduler::configureInstance: config workers must be > 0");
	std::lock_guard lk(g_instanceMutex);
	if (g_instance)
		return false; // 实例已创建：配置被忽略（改用自定义调度器注入）
	g_pendingConfig = config;
	return true;
}

void ResourceScheduler::resetInstance() {
	std::shared_ptr<ResourceScheduler> old;
	{
		std::lock_guard lk(g_instanceMutex);
		old = std::move(g_instance);
		g_pendingConfig.reset();
	}
	if (old)
		old->shutdown(); // 锁外关停：join 在飞任务，不阻塞其他 instance() 调用者
}

// ════════════════════════════════════════════
// 构造 / 析构
// ════════════════════════════════════════════

ResourceScheduler::ResourceScheduler(const SchedulerConfig& config)
	: _config(config) {
	if (!config.valid())
		throw std::invalid_argument("ResourceScheduler: config workers must be > 0");
}

ResourceScheduler::~ResourceScheduler() {
	shutdown();
}

// ════════════════════════════════════════════
// 提交 / 关停
// ════════════════════════════════════════════

ThreadPool* ResourceScheduler::_poolFor(ResourceClass cls) {
	// 快路径：已关停直接拒绝（锁外原子读）
	if (_stopped.load(std::memory_order_acquire))
		return nullptr;
	// 慢路径：惰性创建与关停遍历互斥（防"关停后新池创建"漏 join）
	std::lock_guard lk(_initMutex);
	if (_stopped.load(std::memory_order_relaxed))
		return nullptr;
	auto& pool = _pools[_indexOf(cls)];
	if (!pool)
		pool = std::make_unique<ThreadPool>(PoolConfig{_config.workersFor(cls)});
	return pool.get();
}

bool ResourceScheduler::submit(ResourceClass cls, std::function<void()> task) {
	ThreadPool* pool = _poolFor(cls);
	if (!pool)
		return false; // 已关停：拒绝（契约与 ThreadPool::submit 一致）
	// 锁外提交：池对象在调度器析构前保持存活（shutdown 只关停不销毁），
	// 与并发关停的竞态由池内锁串行化（关停后池自身返回 false）
	return pool->submit(std::move(task));
}

void ResourceScheduler::shutdown() {
	_stopped.store(true, std::memory_order_release);
	std::lock_guard lk(_initMutex);
	for (auto& pool : _pools) {
		if (pool)
			pool->shutdown();
	}
}

} // namespace DC
