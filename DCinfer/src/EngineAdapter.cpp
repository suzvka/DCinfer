#include "Node/internal/EngineAdapter.h"
#include "EngineRegistry.h"
#include "Node.h"

namespace DC {

EngineAdapter::EngineAdapter(EngineInstance* instance, const EngineDescriptor* descriptor)
	: _instance(instance), _desc(descriptor) {}

void EngineAdapter::preRun() const {
	if (!_instance)
		return;
	if (_desc && _desc->preRun) {
		_desc->preRun(_instance->get());
	}
}

void EngineAdapter::synchronize() const {
	if (!_instance)
		return;
	if (_desc && _desc->synchronize) {
		_desc->synchronize(_instance->get());
	}
}

void EngineAdapter::postRun(Node::RunContext& ctx) const {
	if (!_instance)
		return;
	if (_desc && _desc->postRun) {
		_desc->postRun(_instance->get(), ctx);
	}
}

void EngineAdapter::onError() const {
	if (!_instance)
		return;
	if (_desc && _desc->onError) {
		_desc->onError(_instance->get());
	}
}

const TensorConverter* EngineAdapter::converter() const {
	if (!_desc)
		return nullptr;
	return &_desc->converter;
}

void* EngineAdapter::engine() const {
	return _instance ? _instance->get() : nullptr;
}

} // namespace DC
