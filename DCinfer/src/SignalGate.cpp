#include "Node/internal/SignalGate.h"
#include "SignalStore.h"

namespace DC {

void SignalGate::bind(std::shared_ptr<SignalStore> store, std::string name) {
	_store = std::move(store);
	_name = std::move(name);
}

bool SignalGate::isBlocked() const {
	if (!_store || _name.empty())
		return false;
	return !_store->get(_name, /*defaultVal=*/false);
}

bool SignalGate::isBlocked(const std::string& taskId) const {
	if (!_store || _name.empty())
		return false;
	// task 级优先，未命中回退全局
	return !_store->get(_name, taskId, /*defaultVal=*/false);
}

} // namespace DC
