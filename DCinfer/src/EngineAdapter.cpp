#include "Node/internal/EngineAdapter.h"
#include "EngineRegistry.h"
#include "Node.h"

namespace DC {

EngineAdapter::EngineAdapter(std::shared_ptr<EngineInstance> instance, const EngineDescriptor* descriptor)
	: _instance(std::move(instance)), _desc(descriptor) {}

void EngineAdapter::preRun() const {
	if (!_instance)
		return;
	if (_desc && _desc->phases.preRun) {
		_desc->phases.preRun(_instance->get());
	}
}

void EngineAdapter::synchronize() const {
	if (!_instance)
		return;
	if (_desc && _desc->phases.synchronize) {
		_desc->phases.synchronize(_instance->get());
	}
}

void EngineAdapter::postRun(Node::RunContext& ctx) const {
	if (!_instance)
		return;
	if (_desc && _desc->phases.postRun) {
		_desc->phases.postRun(_instance->get(), ctx);
	}
}

void EngineAdapter::onError() const {
	if (!_instance)
		return;
	if (_desc && _desc->phases.onError) {
		_desc->phases.onError(_instance->get());
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
