#include "tensor.h"
#include "tensorort.h"

namespace DC {
	TypeManager<TensorMeta::TensorType> TensorMeta::_typeMap;
	TypeManager<ONNXTensorElementDataType> TensorOrt::_typeMap;
}