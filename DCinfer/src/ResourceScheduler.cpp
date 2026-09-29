#include "ResourceScheduler.h"

#include <array>
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
	// 防御：非法枚举值（经 static_cast 强转传入）按拒绝处理，与
	// workersFor 的注释契约一致——禁止以下标形式越界定长数组（P2）
	const size_t index = _indexOf(cls);
	if (index >= 3)
		return nullptr;
	// 慢路径：惰性创建与关停遍历互斥（防“关停后新池创建”漏 join）
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
		return false; // 已关停：拒绝（契约与 ThreadPool::submit 一致）
	// 锁外提交：池对象在调度器析构前保持存活（shutdown 只关停不销毁），
	// 与并发关停的竞态由池内锁串行化（关停后池自身返回 false）
	return pool->submit(std::move(task));
}

void ResourceScheduler::shutdown() {
	// 先置位拒绝新提交，再逐池关停：join 在飞任务、清队弃置排队任务
	// （std::function 随队列析构——提交方的排水票据在弃置路径同样回收，
	// ExecutionEngine 析构的自排水等待因此必然终止，无逃逸依赖本函数
	// 的完成顺序）。
	_stopped.store(true, std::memory_order_release);
	// 锁内仅快照现有池指针，join 在锁外进行：若持 _initMutex join，
	// 在飞任务经 _poolFor 阻塞在同一把锁上（已通过第一道 _stopped 检查
	// 后关停插入），worker 永不退出而 shutdown 永等其退出——循环等待。
	// 锁外 join 后，被阻塞 worker 获得锁、经第二道检查按拒绝返回，
	// 任务收尾退出，join 正常完成。
	// 池对象生命期：shutdown 只关停不销毁，成员活到调度器析构，
	// 快照指针在锁外使用安全；置位后 _poolFor 双检查均拒绝，
	// 不会出现快照之外的新池漏 join。
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
