#include "InferBase.h"
#include "tool.h"

namespace DC {
	const InferBase::TensorList& InferBase::getInput() const {
		return *inputList;
	}

	const InferBase::TensorList& InferBase::getOutput() const {
		return *outputList;
	}
}