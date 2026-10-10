#pragma once

#include "EngineRegistry.h"
#include "NetCodec.h"
#include "NetEndpoint.h"
#include "NetTransport.h"

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace DC::Net {

/// 适配器描述：transport 负责收发，codec 负责映射，schema 声明本地形状规则。
struct DcNetAdapterDesc {
	std::string engineType;

	/// 静态端口表（不依赖远端）。
	Node::Schema schema;

	/// loadModel 时调用并 connect；modelPath 即远端端点。
	std::function<std::shared_ptr<DcNetTransport>()> transportFactory;

	/// 无状态，可共享。
	std::shared_ptr<DcNetCodec> codec;

	/// 为空时走标准编排（encode → send → recv → decode）。
	Node::RunFn runFn;

	// 端点级覆盖项：非空 / 非零值覆盖 modelPath 解析结果；鉴权 / 附加头 / 超时 /
	// 重试无法从 URL 表达，注册级统一注入。authToken 不写入日志与错误信息。
	bool allowInsecureCredentials = false;
	std::string authToken;
	std::vector<std::string> headers;
	std::chrono::milliseconds connectTimeout{0};
	std::chrono::milliseconds requestTimeout{0};
	int maxRetries = 0;
};

/// 注册网络适配器：loadModel 解析端点 → transportFactory 创建实例 → connect
/// （失败抛 NodeException，配置期报错）。节点归属 ResourceClass::System；
/// DCNet 无引擎级资源，不注册 createEngineCore。
void registerDcNetAdapter(EngineRegistry& reg, DcNetAdapterDesc desc);

} // namespace DC::Net
