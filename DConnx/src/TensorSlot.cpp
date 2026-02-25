#include "TensorSlot.h"

namespace DC {
	using TensorType = TensorMeta::TensorType;

	TensorSlot::TensorSlot(
		const std::string& name,
		TensorMeta::TensorType type,
		size_t typeSize,
		const std::vector<int64_t>& shape
	) {
		setName(name);
		setShapes(shape);
		_rule.type = type;
		_rule.typeSize = typeSize;
	}

	TensorSlot& TensorSlot::setDefaultTensor(const Tensor& data) {
		if (!(*this == data)) {
			throw std::runtime_error("默认数据规则检查未通过");
		}

		_defaultData = std::make_unique<Tensor>(data);

		return *this;
	}

	TensorSlot& TensorSlot::setName(const std::string& name) {
		_rule.name = name;
		return *this;
	}

	TensorSlot& TensorSlot::setShapes(const std::vector<int64_t>& shape) {
		_rule.shape = shape;
		return *this;
	}

	Tensor& TensorSlot::getTensor() {
		if (!hasData()) {
			abort(ErrorType::Other, "TensorSlot has no data.");
		}

		if (!_data) {
			return *_defaultData;
		}

		return *_data;
	}
}