#pragma once

#include "EngineRegistry.h"
#include "NetCodec.h"
#include "Node.h"

#include <memory>
#include <string>

namespace DC::Net {

/// schema 为空时取 codec->schema()；同一 engineType 重复注册保留首次。
void registerDcNetHttp(EngineRegistry& reg, std::shared_ptr<DcNetCodec> codec,
					   Node::Schema schema = {}, std::string engineType = "DCNet.Tensor");

} // namespace DC::Net
