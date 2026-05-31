#include <stdexcept>
#include <numeric>

#include "DCtype.h"
#include "TensorMeta.h"

namespace DC {

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// TensorMeta implementation
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

TensorMeta::TensorMeta() {
	ensureTypeMap();
}

void TensorMeta::ensureTypeMap() {
	static std::once_flag flag;
	std::call_once(flag, []() { setTypeMap(); });
}

bool TensorMeta::checkShape(const std::vector<int64_t>& currentShape) const {
	if (shape.empty()) {
		return true;
	}
	if (shape.size() != currentShape.size()) {
		return false;
	}
	for (size_t i = 0; i < shape.size(); ++i) {
		if (shape[i] != -1 && shape[i] != currentShape[i]) {
			return false;
		}
	}
	return true;
}

void TensorMeta::setTypeMap() {
	Type::registerType<float>(TensorType::Float);
	Type::registerType<double>(TensorType::Float);

	Type::registerType<int64_t>(TensorType::Int);
	Type::registerType<int32_t>(TensorType::Int);
	Type::registerType<int16_t>(TensorType::Int);
	Type::registerType<int8_t>(TensorType::Int);
	Type::registerType<int>(TensorType::Int);

	Type::registerType<uint64_t>(TensorType::Uint);
	Type::registerType<uint32_t>(TensorType::Uint);
	Type::registerType<uint16_t>(TensorType::Uint);
	Type::registerType<uint8_t>(TensorType::Uint);
	Type::registerType<unsigned int>(TensorType::Uint);

	Type::registerType<bool>(TensorType::Bool);

	Type::registerType<char>(TensorType::Char);
	Type::registerType<unsigned char>(TensorType::Char);

	Type::registerType<std::vector<std::byte>>(TensorType::Data);
	Type::registerType<std::vector<char>>(TensorType::Data);
	Type::registerType<std::vector<unsigned char>>(TensorType::Data);
}

std::string TensorMeta::typeToString(TensorType type) {
	switch (type) {
	case TensorType::Float:
		return "Float";
	case TensorType::Int:
		return "Int";
	case TensorType::Uint:
		return "Uint";
	case TensorType::Bool:
		return "Bool";
	case TensorType::Char:
		return "Char";
	case TensorType::Data:
		return "Data";
	case TensorType::Void:
		return "Void";
	}
	return "Void";
}

TensorMeta::TensorType TensorMeta::stringToType(const std::string& str) {
	if (str == "Float")
		return TensorType::Float;
	if (str == "Int")
		return TensorType::Int;
	if (str == "Uint")
		return TensorType::Uint;
	if (str == "Bool")
		return TensorType::Bool;
	if (str == "Char")
		return TensorType::Char;
	if (str == "Data")
		return TensorType::Data;
	return TensorType::Void;
}

} // namespace DC