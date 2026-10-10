#pragma once

#include "ResourceClass.h"
#include "ThreadPool.h"

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>

namespace DC {

/// @brief 进程级资源预算：每资源类的执行槽位上限（worker 数）。
///
/// Compute/Operator 默认 1 槽位；System 默认 4（同承载 I/O、连接器与嵌套等待型编排节点）。
/// 服务器场景建议按 CPU 核数与任务画像显式配置。
struct SchedulerConfig {
	size_t computeWorkers = 1;
	size_t operatorWorkers = 1;
	size_t systemWorkers = 4;

	bool valid() const {
		return computeWorkers > 0 && operatorWorkers > 0 && systemWorkers > 0;
	}

	size_t workersFor(ResourceClass cls) const {
		switch (cls) {
		case ResourceClass::Compute:
			return computeWorkers;
		case ResourceClass::Operator:
			return operatorWorkers;
		case ResourceClass::System:
			return systemWorkers;
		}
		return 0; // 枚举全覆盖；未知资源类防御性返回 0（提交方按拒绝处理）
	}
};

/// @brief 进程级共享资源调度器：按资源类分配执行槽位的资源隔离机制。
///
/// 每资源类持有一个惰性创建的内部 ThreadPool，首次提交时创建；同类任务共享类预算，
/// 异类互不干扰；多图默认共享进程级实例（instance()），线程总量 = 各类预算之和。
/// 池队列为无界 FIFO（无内置背压）：宿主须在有界准入下控制队列规模。
///
/// 所有权约定：shutdown/resetInstance 必须在本调度器 worker 之外调用（违反抛
/// std::logic_error）；最后一个持有者须由外部释放。
/// 阻塞语义：worker 被阻塞任务（I/O、等待型节点）占住是池化执行固有语义，
/// 预算必须覆盖并发阻塞任务数，否则同类任务可能互相饥饿；调度器不做抢占与让渡。
class ResourceScheduler {
public:
	/// @brief 构造调度器（各资源类 worker 必须 > 0）。
	/// @throws std::invalid_argument config 无效
	explicit ResourceScheduler(const SchedulerConfig& config = {});
	/// 必须由外部非 worker 持有者销毁，违反打印诊断并 std::terminate。
	~ResourceScheduler();
	/// @brief 当前线程是否为本调度器的资源类 worker（含载荷析构）。
	bool isWorkerThread() const noexcept;

	ResourceScheduler(const ResourceScheduler&) = delete;
	ResourceScheduler& operator=(const ResourceScheduler&) = delete;

	/// @brief 提交任务到指定资源类（线程安全，首次提交惰性创建执行器）。
	/// @return true = 任务已入队；false = 已关闭或入队失败，任务未被执行。
	bool submit(ResourceClass cls, std::function<void()> task);

	/// @brief 关停调度器（幂等）：拒绝新提交，join 在飞任务，排队任务丢弃。
	/// @throws 在本调度器 worker 内调用抛 std::logic_error。
	void shutdown();

	/// @brief 是否已关停。
	bool isStopped() const {
		return _stopped.load(std::memory_order_acquire);
	}

	/// @brief 资源预算配置。
	const SchedulerConfig& config() const {
		return _config;
	}

	/// @brief 指定资源类的预算（worker 上限）。
	size_t workersFor(ResourceClass cls) const {
		return _config.workersFor(cls);
	}

	/// @brief 进程级默认调度器（惰性创建，线程安全；InferGraph 默认共享）。
	static std::shared_ptr<ResourceScheduler> instance();

	/// @brief 预配置进程级默认实例的预算（须在 instance() 首次创建之前调用）。
	/// @return true = 已接受（下次 instance() 创建时生效）；false = 实例已创建，配置被忽略。
	/// @throws std::invalid_argument config 无效
	static bool configureInstance(const SchedulerConfig& config);

	/// @brief 关停并清除进程级默认实例（此后 instance() 按新预配置重新创建）。
	/// @throws 在默认实例的 worker 内调用抛 std::logic_error。
	static void resetInstance();

private:
	/// @brief 取指定资源类的执行器（惰性创建）；已关停返回 nullptr。
	ThreadPool* _poolFor(ResourceClass cls);

	static size_t _indexOf(ResourceClass cls) {
		return static_cast<size_t>(cls);
	}

	SchedulerConfig _config;
	std::atomic<bool> _stopped{false};
	mutable std::mutex _initMutex;
	std::mutex _shutdownMutex;
	std::unique_ptr<ThreadPool> _pools[3]; ///< 下标 = ResourceClass 值
};

} // namespace DC
