#pragma once

#include "EngineRegistry.h"
#include "NetCodec.h"
#include "NetEndpoint.h"
#include "NetTransport.h"

#include <functional>
#include <memory>
#include <string>

namespace DC::Net {

/// @brief DCNet 适配器描述（对接契约的注册形态）。
///
/// 心智模型：**双向翻译器**（ADR-3）——transport 处理"收发"，codec 处理"映射"，
/// schema 声明本地形状规则（§3.4）；对方 SDK / 协议均为实现细节，
/// 对运行时零暴露。
struct DcNetAdapterDesc {
	/// 注册的 engineType（如 "DCNet.Tensor"）。
	std::string engineType;

	/// 本地形状规则（静态端口表，不依赖远端推导）。
	Node::Schema schema;

	/// 创建传输实例；createEngine 时调用并 connect（modelPath 即远端端点）。
	std::function<std::shared_ptr<DcNetTransport>()> transportFactory;

	/// 协议映射（无状态、可共享）。
	std::shared_ptr<DcNetCodec> codec;

	/// 可选：自定义 RunFn；为空时走标准编排（encode → send → recv → decode）。
	Node::RunFn runFn;
};

/// @brief 注册一个网络适配器到引擎注册表。
///
/// 注册后：
/// @code
///   auto node = EngineRegistry::instance().createNode(
///       "DCNet.Tensor", "llm", "http://192.168.1.10:8080/v1");   // modelPath = 端点
/// @endcode
///
/// 特性：
/// - createEngine(modelPath) 解析端点为 NetEndpoint → transportFactory 创建实例
///   → connect（失败抛 NodeException，配置期报错）
/// - getInputPorts/getOutputPorts 返回本地静态形状规则（不依赖远端）
/// - 节点归属 ThreadPoolAffinity::System（I/O 池，README 分工）
/// - RunFn 失败经 NetError 归一化出口上报（NodeResult + ErrorTracker 兼容）
///
/// @param reg  目标注册表，默认为全局单例 EngineRegistry::instance()
/// @param desc 适配器描述（schema / transportFactory / codec / 可选 runFn）
void registerDcNetAdapter(EngineRegistry& reg, DcNetAdapterDesc desc);

} // namespace DC::Net
