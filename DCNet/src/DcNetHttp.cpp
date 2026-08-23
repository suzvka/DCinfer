#include "DCNet/DcNetHttp.h"

#include "DCNet/NetAdapter.h"
#include "DCNet/NetTransport_Http.h"

#include <memory>

namespace DC::Net {

void registerDcNetHttp(EngineRegistry& reg, std::shared_ptr<DcNetCodec> codec, Node::Schema schema,
					   std::string engineType) {
	DcNetAdapterDesc desc;
	desc.engineType = std::move(engineType);
	if (codec && schema.inputs.empty() && schema.outputs.empty())
		schema = codec->schema(); // codec 自带本地形状规则
	desc.schema = std::move(schema);
	desc.codec = std::move(codec);
	desc.transportFactory = [] { return std::make_shared<HttpTransport>(); };
	registerDcNetAdapter(reg, std::move(desc));
}

} // namespace DC::Net
