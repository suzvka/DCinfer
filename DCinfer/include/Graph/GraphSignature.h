#pragma once

#include "InputZone.h"   // InputBinding
#include "OutputZone.h"  // OutputBinding

#include <string>
#include <vector>

namespace DC {

/// @brief 图级静态签名：冻结时一次性构建的绑定快照（序列化/内省元数据）。
///
/// Build → Freeze → Execute 生命周期中的"静态签名"部分：
/// - inputs/outputs 是构建期经 bindInput/bindOutput 累积的绑定列表快照；
/// - 冻结后只读：执行期绑定读取（submitBound 声明推导 / 图导出 / 内省）
///   直接读本签名（无锁），与节点任务态（OutputZone 的声明/累加/artifact）分离；
/// - 运行时数据注入与取用不读本签名——寻址一律按 (nodeName, portName) 坐标；
/// - 绑定面（InputZone/OutputZone 的 bind 写入）仅存在于构建期 GraphBuilder。
struct GraphSignature {
	std::vector<InputBinding> inputs;   ///< 图级输入绑定快照（运行时寻址一律 (nodeName, portName)；alias 为元数据）
	std::vector<OutputBinding> outputs; ///< 图级输出绑定快照（运行时寻址一律 (nodeName, portName)；alias 为元数据）

	/// @brief  该 node:port 是否为图级输出绑定端口
	///         （传播路径上"产出进输出区 vs 流向下游边"的判定依据）
	bool isOutputBound(const std::string& nodeName, const std::string& portName) const {
		for (const auto& b : outputs)
			if (b.nodeName == nodeName && b.portName == portName)
				return true;
		return false;
	}
};

} // namespace DC
