#include "SlotType.h"
#include "DCtype.h"
#include "EngineRegistry.h"

#include <mutex>

namespace DC {

ValidatorRegistry& ValidatorRegistry::instance() {
	static ValidatorRegistry inst;
	return inst;
}

void ValidatorRegistry::ensureDefaults() {
	static std::once_flag flag;
	std::call_once(flag, []() {
		// 类型映射
		DC::Type::registerType<DC::Tensor>(SlotDataType::DCTensor);
		DC::Type::registerType<DC::Value>(SlotDataType::Value);

		// DCTensor 校验器
		ValidatorRegistry::instance().registerValidator(
			SlotDataType::DCTensor, [](const void* data, SlotDataType, const TensorMeta& rule) -> SlotDataStatus {
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

		// Value 校验器：放行（由具体引擎校验）
		ValidatorRegistry::instance().registerValidator(
			SlotDataType::Value,
			[](const void*, SlotDataType, const TensorMeta&) -> SlotDataStatus { return SlotDataStatus{}; });
	});
}

void ValidatorRegistry::registerValidator(SlotDataType type, SlotCheckFn fn) {
	_validators[type] = std::move(fn);
}

const SlotCheckFn* ValidatorRegistry::find(SlotDataType type) const {
	auto it = _validators.find(type);
	return it != _validators.end() ? &it->second : nullptr;
}

SlotDataStatus ValidatorRegistry::validate(const void* data, SlotDataType type, const TensorMeta& rule) const {
	const auto* fn = find(type);
	if (!fn) {
		return SlotDataStatus{};
	}
	return (*fn)(data, type, rule);
}

} // namespace DC
