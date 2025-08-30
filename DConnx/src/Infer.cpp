#include "Infer.h"
#include "tool.h"

namespace DC {
	const Infer::TensorList& Infer::getInfo() const {
		return inputList;
	}

	Infer::ErrorCode Infer::check(const std::vector<Tensor>& inputs) {
		for (auto& inputValue : inputs) {
			auto itValue = inputList.find(inputValue.name());
			if (
				itValue == inputList.end() &&
				defaultList.find(inputValue.name()) == defaultList.end()
			) {
				errorMessage = "Missing tensor: " + inputValue.name();
				return ErrorCode::MISSING_TENSOR;
			}

			if (itValue->second != inputValue) {
				return ErrorCode::ERROR_TENSOR;
			}
		}

		return  ErrorCode::SUCCESS;
	}
}