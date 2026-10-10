#pragma once

#include "EngineRegistry.h"
#include "NetListener.h"
#include "NetServerCodec.h"
#include "NetServerEndpoint.h"

#include <memory>
#include <string>

namespace DC::Net {

/// 节点服务化实例句柄。
struct DcNetServerService {
	virtual ~DcNetServerService() = default;

	/// 优雅停止；析构等效调用，重复调用安全。
	virtual void stop() = 0;

	/// 服务端健康镜像。
	virtual bool alive() const = 0;

	/// 实际监听端口；未启动返回 -1。
	virtual int port() const = 0;
};

/// 节点服务化装配描述。
struct DcNetServerAdapterDesc {
	std::string engineType;                    ///< 本地节点引擎类型，须已注册
	std::string localModelRef;                 ///< 本地模型标识；不复用 modelPath，其约定为远端端点
	std::shared_ptr<DcNetServerCodec> codec;   ///< 服务端协议映射
	NetServerEndpoint endpoint;                ///< 监听端点，requestPath 由 codec 覆盖
};

/// 注册并启动节点服务化监听端：请求与本地执行同管线，
/// decodeRequest、createNode、tryExecute、encodeResponse，失败经 wireStatusFor 应答。
/// 引擎实例按 engineType 与 localModelRef 复用，互斥串行；配置期错误抛
/// NodeException，运行期失败一律 wire 应答。reg 须比返回句柄存活更久。
std::shared_ptr<DcNetServerService> registerDcNetServerAdapter(EngineRegistry& reg, DcNetServerAdapterDesc desc);

} // namespace DC::Net
