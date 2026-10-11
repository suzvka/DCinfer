#pragma once

#include <string>
#include <vector>

namespace DC {

/// @brief 信号面节点计划：一个节点的信号端子与逻辑参数（控制面，独立于张量数据流）。
///
/// 计算节点：isLogic=false，仅登记其信号端子（gate 输入 / 状态输出），本体仍是 DC::Node；
/// 逻辑节点：isLogic=true，纯信号构造（dc:Signal*），不物化为 DC::Node，只存在于本计划。
struct SignalNodePlan {
	std::string name;
	std::string type;				   ///< 逻辑节点保留名（dc:Signal*）；计算节点为空
	bool isLogic = false;			   ///< true = 纯信号节点，不物化为 DC::Node
	std::string signalName;			   ///< SignalSource 对外名（宿主 setSignal 入口）
	bool hasConstValue = false;		   ///< SignalConst 是否带常量值
	bool constValue = false;		   ///< SignalConst 常量布尔值
	std::vector<std::string> signalInputs;	///< 有序信号输入端子名
	std::vector<std::string> signalOutputs;	///< 有序信号输出端子名
};

/// @brief 信号边（端口到端口）：仅在信号面传播布尔，不入张量数据流。
struct SignalEdgePlan {
	std::string srcNode;
	std::string srcPort;
	std::string dstNode;
	std::string dstPort;
};

/// @brief 图级信号方案：编译期从图 JSON 剔离保存，供降级、序列化 roundtrip 与运行期求值。
struct SignalPlan {
	std::vector<SignalNodePlan> nodes;
	std::vector<SignalEdgePlan> edges;

	bool empty() const { return nodes.empty() && edges.empty(); }
};

} // namespace DC
