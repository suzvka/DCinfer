#include "ResourceScheduler.h"

#include <array>
#include <cstdio>
#include <exception>
#include <optional>
#include <stdexcept>

namespace DC {

namespace {
	/// 全局句柄在进程退出时释放；实例析构自动 shutdown 并 join 全部 worker。
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
		return false; // 实例已创建；改用自定义调度器注入
	g_pendingConfig = config;
	return true;
}

void ResourceScheduler::resetInstance() {
	std::shared_ptr<ResourceScheduler> old;
	{
		std::lock_guard lk(g_instanceMutex);
		if (g_instance && g_instance->isWorkerThread())
			throw std::logic_error("ResourceScheduler::resetInstance: prohibited on own worker");
		old = std::move(g_instance);
		g_pendingConfig.reset();
	}
	if (old)
		old->shutdown(); // 锁外关停，不阻塞其他 instance() 调用者
}

ResourceScheduler::ResourceScheduler(const SchedulerConfig& config)
	: _config(config) {
	if (!config.valid())
		throw std::invalid_argument("ResourceScheduler: config workers must be > 0");
}

bool ResourceScheduler::isWorkerThread() const noexcept {
	const auto* current = ThreadPool::currentWorkerPool();
	if (!current)
		return false;
	std::lock_guard lk(_initMutex);
	for (const auto& pool : _pools)
		if (pool.get() == current)
			return true;
	return false;
}

ResourceScheduler::~ResourceScheduler() {
	if (isWorkerThread()) {
		// stderr 重定向后可能全缓冲，崩溃路径不经 flush 会丢诊断；显式 flush 兜底。
		std::fputs("ResourceScheduler: prohibited destruction on own worker; retain external ownership\n", stderr);
		std::fflush(stderr);
		std::terminate();
	}
	shutdown();
}

ThreadPool* ResourceScheduler::_poolFor(ResourceClass cls) {
	// 快路径：已关停直接拒绝
	if (_stopped.load(std::memory_order_acquire))
		return nullptr;
	// 非法枚举值按拒绝处理，禁止越界定长数组
	const size_t index = _indexOf(cls);
	if (index >= 3)
		return nullptr;
	// 慢路径：惰性创建与关停互斥，防关停后新建池漏 join
	std::lock_guard lk(_initMutex);
	if (_stopped.load(std::memory_order_relaxed))
		return nullptr;
	auto& pool = _pools[index];
	if (!pool)
		pool = std::make_unique<ThreadPool>(PoolConfig{_config.workersFor(cls)});
	return pool.get();
}

bool ResourceScheduler::submit(ResourceClass cls, std::function<void()> task) {
	ThreadPool* pool = _poolFor(cls);
	if (!pool)
		return false; // 已关停，与 ThreadPool::submit 契约一致
	// 锁外提交：池对象活到调度器析构；关停竞态由池内锁串行化。
	return pool->submit(std::move(task));
}

void ResourceScheduler::shutdown() {
	if (isWorkerThread())
		throw std::logic_error("ResourceScheduler::shutdown: prohibited on own worker");
	// 关停锁：消除并发 shutdown 交叠，池内已兜底双保险；重复调用幂等。
	std::lock_guard shutdownLk(_shutdownMutex);
	// 先置位拒绝新提交，再逐池关停；排队任务弃置，排水票据在弃置路径同样回收。
	_stopped.store(true, std::memory_order_release);
	// join 必须在锁外：持 _initMutex join 会与阻塞在 _poolFor 的 worker 循环等待。
	// 池对象活到调度器析构，快照指针锁外使用安全；置位后不会出现新池漏 join。
	std::array<ThreadPool*, 3> pools{};
	{
		std::lock_guard lk(_initMutex);
		for (size_t i = 0; i < 3; ++i)
			pools[i] = _pools[i].get();
	}
	for (auto* pool : pools) {
		if (pool)
			pool->shutdown();
	}
}

} // namespace DC
