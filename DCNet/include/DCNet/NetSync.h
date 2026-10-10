#pragma once

#include <functional>
#include <future>
#include <utility>

namespace DC::Net {

/// 在 RunFn（System 池线程）内调用；submit 可将任务投递到 transport 自持的
/// I/O 线程，完成回调唤醒等待。
template <typename R>
inline R syncAwait(const std::function<void(const std::function<void(R)>&)>& submit) {
	std::promise<R> p;
	submit([&p](R r) { p.set_value(std::move(r)); });
	return p.get_future().get();
}

inline void syncAwaitVoid(const std::function<void(const std::function<void()>&)>& submit) {
	std::promise<void> p;
	submit([&p]() { p.set_value(); });
	p.get_future().get();
}

} // namespace DC::Net
