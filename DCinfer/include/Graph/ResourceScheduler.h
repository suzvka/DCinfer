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
/// 默认 Compute/Operator 各 1 槽位（对齐单图历史默认）；System 默认 4——
/// System 类除 I/O/网络外还承载图连接器与等待型编排节点（GraphOperator
/// 默认亲和），嵌套等待链深度与子图内连接器并发均占用同类槽位，4 为
/// 开箱安全的保守值（覆盖 ≤3 层嵌套 + 并发连接器）。进程级共享语义下
/// 多图共存，服务器场景建议按 CPU 核数与任务画像显式配置（经
/// ResourceScheduler 构造或 configureInstance 预配置）。
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
		return 0; // 枚举全覆盖（防御：未知资源类返回 0，提交方按拒绝处理）
	}
};

/// @brief 进程级共享资源调度器：按资源类分配执行槽位的正式资源隔离机制。
///
/// ── 定位：资源隔离的权威组件 ──
/// 替代"每图私有线程池、线程数即配额"的隐式隔离（依赖线程副作用），以
/// 显式预算（SchedulerConfig）承载进程级资源隔离：
/// - 资源类隔离：同类任务共享该类预算（worker 上限），异类互不干扰；
/// - 跨图共享：多张 InferGraph 默认共享同一调度器（instance()），
///   服务器场景线程总量 = 各类预算之和，不随图数量线性膨胀；
/// - 用户可控：宿主可注入自定义调度器（InferGraph 构造），或经
///   configureInstance 预配置进程默认实例的预算。
///
/// ── 执行器：惰性启动 ──
/// 每资源类持有一个内部 ThreadPool，首次提交时创建（未使用的类不占线程）；
/// 池在调度器析构前保持存在——shutdown() 只关停（拒绝新提交、join 在飞），
/// 不销毁对象（并发提交方持有的池指针始终有效，与关停的竞态由池内锁串行）。
/// 内部池队列为无界 FIFO（无内置背压/丢弃）：提交速率长期超过执行速率时
/// 队列与任务载荷随之增长，长驻/服务部署由宿主以提交节流或扩大线程数
/// 控制队列规模（与 ThreadPool.h 契约一致）。
///
/// ── 阻塞语义（预算规划约束） ──
/// worker 被阻塞任务（网络 I/O、等待型节点）占住是池化执行的固有语义：
/// 预算必须覆盖"并发阻塞任务数"（如父子图嵌套等待链深度 × 并发组合节点
/// 数），否则同类任务可能互相饥饿（自锁）。调度器不做抢占与让渡。
class ResourceScheduler {
public:
	/// @brief 构造调度器。
	/// @param config 进程资源预算（各类 worker 必须 > 0）
	/// @throws std::invalid_argument config 无效
	explicit ResourceScheduler(const SchedulerConfig& config = {});
	~ResourceScheduler();

	ResourceScheduler(const ResourceScheduler&) = delete;
	ResourceScheduler& operator=(const ResourceScheduler&) = delete;

	/// @brief 提交任务到指定资源类（fire-and-forget）。
	/// @return true = 任务已入队；false = 调度器已关闭或入队失败（内存压力）
	///         ——任务未被接受，调用方按失败语义处理（不得假设任务会执行）
	/// @note   首次提交某资源类时惰性创建其内部执行器；线程安全
	bool submit(ResourceClass cls, std::function<void()> task);

	/// @brief 关停调度器（幂等）：此后拒绝新提交；已创建的池关停并等待
	///        在飞任务完成（排队未执行的任务按内部池语义丢弃——提交方的
	///        排水票据随 function 析构回收，消费方析构排水无需逃逸判定）。
	/// @note   建议先析构使用方（图/引擎），再关停调度器——反序亦安全
	///         （在飞任务由本函数 join 兜底执行完毕，排队任务弃置即回收，
	///         关停返回后无任何任务 lambda 可再回访使用方）。
	void shutdown();

	/// @brief 是否已关停（shutdown() 调用后为 true）
	bool isStopped() const {
		return _stopped.load(std::memory_order_acquire);
	}

	/// @brief 资源预算配置
	const SchedulerConfig& config() const {
		return _config;
	}

	/// @brief 指定资源类的预算（worker 上限）
	size_t workersFor(ResourceClass cls) const {
		return _config.workersFor(cls);
	}

	// ── 进程级默认实例 ──

	/// @brief 进程级默认调度器（首次调用惰性创建；线程安全）。
	///        InferGraph 默认构造共享本实例——多图默认共享进程预算。
	static std::shared_ptr<ResourceScheduler> instance();

	/// @brief 预配置进程级默认实例的预算（须在 instance() 首次创建之前调用）。
	/// @return true = 已接受（下次 instance() 创建时生效）；
	///         false = 实例已创建，配置被忽略（改用自定义调度器注入）
	/// @throws std::invalid_argument config 无效
	static bool configureInstance(const SchedulerConfig& config);

	/// @brief 关停并清除进程级默认实例（测试隔离 / 服务器确定性回收）。
	///        此后 instance() 按新预配置重新惰性创建；仍持有旧实例引用的
	///        使用方不受影响（旧实例已关停，拒绝新提交）。
	static void resetInstance();

private:
	/// @brief 取指定资源类的执行器（惰性创建）；已关停返回 nullptr
	ThreadPool* _poolFor(ResourceClass cls);

	static size_t _indexOf(ResourceClass cls) {
		return static_cast<size_t>(cls);
	}

	SchedulerConfig _config;
	std::atomic<bool> _stopped{false};
	std::mutex _initMutex;                 ///< 保护惰性创建与关停遍历
	std::unique_ptr<ThreadPool> _pools[3]; ///< 每资源类一个执行器（下标 = ResourceClass 值）
};

} // namespace DC
