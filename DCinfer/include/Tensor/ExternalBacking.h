#pragma once
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace DC {

/// @brief 不透明外部存储契约：产出引擎声明载荷所在与访问方式，框架只存储/路由，不解释句柄内容。
///
/// 消费引擎在 toNative/RunFn 内单边判定能否直传：比较 coreToken 与本引擎核心身份，
/// 能接受则直接包装 handle；否则经 Tensor 的 host 读路径触发 exportToHost 兜底（= 无 backing 现状）。
/// 框架核心不依赖任何设备 API；句柄与同步令牌的语义完全由引擎定义。
///
/// 线程安全契约：backing 一经产出即不可变；exportToHost 必须可并发调用
/// （广播共享后多个消费者可能同时触发 host 导出）。
struct ExternalBacking {
	/// 产出引擎的 engineType，供诊断与路由展示。
	std::string producerEngine;

	/// 产出引擎核心身份令牌：消费方据此判定是否同核心可直传。
	/// 共享持有核心，防止核心销毁后地址复用造成身份误判；可空（无核心引擎）。
	std::shared_ptr<const void> coreToken;

	/// 不透明载荷句柄，由引擎解释；其 deleter 即释放回调。
	/// 引擎须保证释放不与在飞计算竞态（例如同流序保证或流序延迟释放）。
	/// 构造外部张量时不得为空。
	std::shared_ptr<void> handle;

	/// 不透明同步令牌（如设备事件），由消费引擎解释；空表示无需额外同步
	/// （例如同流序已保证可见性）。
	std::shared_ptr<const void> syncToken;

	/// 产出引擎提供的 host 导出兜底：返回与张量 meta（shape × typeSize）匹配的稠密字节。
	/// 空表示不支持导出，对此张量的 host 访问抛 TensorException。
	/// 字节数与 meta 不匹配时抛 std::invalid_argument（产出方契约违约显式暴露）。
	std::function<std::vector<std::byte>()> exportToHost;
};

} // namespace DC