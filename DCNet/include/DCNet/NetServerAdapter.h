#pragma once

#include "EngineRegistry.h"
#include "NetListener.h"
#include "NetServerCodec.h"
#include "NetServerEndpoint.h"

#include <memory>
#include <string>

namespace DC::Net {

/// @brief 变体 A 服务实例句柄（registerDcNetServerAdapter 返回）。
struct DcNetServerService {
	virtual ~DcNetServerService() = default;

	/// @brief 优雅停止（graceful drain；析构等效调用，重复调用安全）。
	virtual void stop() = 0;

	/// @brief 服务端健康镜像。
	virtual bool alive() const = 0;

	/// @brief 实际监听端口（endpoint.port = 0 时 bind 后回读；未启动返回 -1）。
	virtual int port() const = 0;
};

/// @brief 变体 A（节点服务化 / Serve-a-Node）装配描述（DESIGN.md §3.6）。
struct DcNetServerAdapterDesc {
	std::string engineType;                    ///< 被驱动执行的本地节点引擎类型（须已注册）
	std::string localModelRef;                 ///< 本地模型标识（[C3]：语义为本地路径，不叫 modelPath）
	std::shared_ptr<DcNetServerCodec> codec;   ///< 服务端协议映射（载荷复用 NetCodec_Tensor）
	NetServerEndpoint endpoint;                ///< 监听端点（requestPath 由 codec 注入覆盖）
};

/// @brief 注册并启动一个「节点服务化」监听端（变体 A / M-server）。
///
/// 请求处理路径（与本地执行完全同管线，保证图级语义无差别——提案 FR-3）：
///   decodeRequest → EngineRegistry::createNode（一请求一节点实例）→
///   setInput → tryExecute → collectOutputs → encodeResponse → 200；
///   失败经 wireStatusFor 逆向映射为 wire 状态（DESIGN.md §6.1，提案 §5 表）。
/// 引擎实例按 engineType + localModelRef 复用（Registry 缓存），本地执行以
/// 互斥串行（引擎单任务语义）；监听器自持 I/O 线程（ADR-6(3)）。
///
/// 配置期错误（codec 缺失 / engineType 未注册 / createEngine 失败 / bind
/// 失败）抛 NodeException（[C1] 裁决）；运行期失败一律 wire 应答，不抛出、
/// 不静默丢弃（FR-5）。
///
/// @param reg  目标注册表；引用须比返回句柄存活更久
/// @param desc 装配描述
/// @return 服务实例句柄（stop / alive / port）
std::shared_ptr<DcNetServerService> registerDcNetServerAdapter(EngineRegistry& reg, DcNetServerAdapterDesc desc);

} // namespace DC::Net
