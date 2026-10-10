#include "SlotType.h"
#include "Tensor.hpp"
#include "Value.h"

#include <mutex>

namespace DC {

ValidatorRegistry& ValidatorRegistry::instance() {
	static ValidatorRegistry inst;
	return inst;
}

void ValidatorRegistry::ensureDefaults() {
	static std::once_flag flag;
	std::call_once(flag, []() {
		auto dctensorId = ensureSlotType<DC::Tensor>();
		auto valueId = ensureSlotType<DC::Value>();

		ValidatorRegistry::instance().registerValidator(
			dctensorId, [](const void* data, SlotDataType, const TensorMeta& rule) -> SlotDataStatus {
				const auto* t = static_cast<const Tensor*>(data);
				if (!t || !t->valid()) {
					return SlotDataStatus{.invalid = true};
				}
				SlotDataStatus s;
				if (rule.type != TensorMeta::TensorType::Void && t->type() != rule.type) {
					s.needConvert = true;
				}
				if (!rule.checkShape(t->shape())) {
					s.needAlign = true;
				}
				return s;
			});

		// Value 放行，由具体引擎校验
		ValidatorRegistry::instance().registerValidator(
			valueId,
			[](const void*, SlotDataType, const TensorMeta&) -> SlotDataStatus { return SlotDataStatus{}; });

		// DCTensor 克隆器：深拷贝产出独立可变副本；拷贝构造自带非冻结语义
		ValueCloneRegistry::instance().registerClone(
			dctensorId, [](const void* data) -> std::shared_ptr<void> {
				return std::make_shared<Tensor>(*static_cast<const Tensor*>(data));
			});
	});
}

void ValidatorRegistry::registerValidator(SlotDataType type, SlotCheckFn fn) {
	std::lock_guard lk(_mutex);
	_validators[type] = std::move(fn);
}

const SlotCheckFn* ValidatorRegistry::find(SlotDataType type) const {
	std::lock_guard lk(_mutex);
	auto it = _validators.find(type);
	// 启动期注册完成后指针持续有效：map 节点地址稳定，运行期可并发读取
	return it != _validators.end() ? &it->second : nullptr;
}

SlotDataStatus ValidatorRegistry::validate(const void* data, SlotDataType type, const TensorMeta& rule) const {
	const auto* fn = find(type);
	if (!fn) {
		return SlotDataStatus{};
	}
	return (*fn)(data, type, rule);
}

ValueCloneRegistry& ValueCloneRegistry::instance() {
	static ValueCloneRegistry inst;
	return inst;
}

void ValueCloneRegistry::registerClone(SlotDataType type, ValueCloneFn fn) {
	std::lock_guard lk(_mutex);
	_clones[type] = std::move(fn);
}

const ValueCloneFn* ValueCloneRegistry::find(SlotDataType type) const {
	std::lock_guard lk(_mutex);
	auto it = _clones.find(type);
	return it != _clones.end() ? &it->second : nullptr;
}

} // namespace DC
