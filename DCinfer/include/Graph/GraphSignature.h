#pragma once

#include "InputZone.h"
#include "OutputZone.h"

#include <string>
#include <vector>

namespace DC {

/// @brief 图级静态签名：冻结时一次性构建的绑定快照，供序列化与内省。
///
/// 冻结后只读、无锁读取；运行时寻址一律按 nodeName、portName 坐标，不读本签名。
struct GraphSignature {
	std::vector<InputBinding> inputs;
	std::vector<OutputBinding> outputs;

	/// @brief 该 node:port 是否为图级输出绑定端口。
	bool isOutputBound(const std::string& nodeName, const std::string& portName) const {
		for (const auto& b : outputs)
			if (b.nodeName == nodeName && b.portName == portName)
				return true;
		return false;
	}
};

} // namespace DC
