#pragma once

#include <functional>
#include <future>
#include <utility>

namespace DC::Net {

/// @brief 核心 async→sync 桥（DESIGN.md §3.5 / ADR-6）。
///
/// 在 RunFn（System 池线程）内调用；submit 内可将任务投递到 transport 自持的
/// I/O 线程 / 事件循环，完成回调唤醒等待。适配器开发者不再手写 promise/condvar。
///
/// @code
///   NetError err = DcNet::syncAwait<NetError>([this](auto done) {
///       asyncClient->post(payload, [done](NetError e) { done(e); });
///   });
/// @endcode
template <typename R>
inline R syncAwait(const std::function<void(const std::function<void(R)>&)>& submit) {
	std::promise<R> p;
	submit([&p](R r) { p.set_value(std::move(r)); });
	return p.get_future().get();
}

/// @brief 无返回值便捷重载。
inline void syncAwaitVoid(const std::function<void(const std::function<void()>&)>& submit) {
	std::promise<void> p;
	submit([&p]() { p.set_value(); });
	p.get_future().get();
}

} // namespace DC::Net
