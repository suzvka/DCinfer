#pragma once

#include "SignalPlan.h"

#include <string>

namespace DC {

class SignalStore;

// ── 信号面保留名（编译期降级与运行期求值共用，单点定义）──

/// 信号逻辑节点 type 保留名
inline constexpr const char* kSignalTypeSource = "dc:SignalSource";
inline constexpr const char* kSignalTypeConst = "dc:SignalConst";
inline constexpr const char* kSignalTypeNot = "dc:SignalNot";
inline constexpr const char* kSignalTypeAnd = "dc:SignalAnd";
inline constexpr const char* kSignalTypeOr = "dc:SignalOr";

/// 标准信号端子名
inline constexpr const char* kSigGate = "sig.gate";
inline constexpr const char* kSigOut = "sig.out";

/// @brief 是否信号逻辑节点保留 type。
bool isSignalLogicType(const std::string& type);

/// @brief 信号 wire 名（SignalStore 键）：源节点用其对外名（宿主直达），其余用 `node.port`。
std::string signalWireName(const SignalNodePlan& producer, const std::string& port);

/// @brief 静态信号面求值：按依赖从信号源/常量/逻辑节点计算各 wire 的布尔值，写入 store（task 级）。
///
/// 覆盖不依赖运行期节点状态的静态控制面（宿主驱动的源、常量、其上的 AND/OR/NOT 组合）；
/// 计算节点状态端子（idle/running/blocked/error/done）的反馈属运行期，由执行侧另行发布，
/// 求值时按 store 现值读取（缺省 false）。环以 memo 兜底（缺省 false），不无限递归。
void evaluateStaticSignals(const SignalPlan& plan, SignalStore& store, const std::string& taskId);

} // namespace DC
