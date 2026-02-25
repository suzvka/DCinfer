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

    bool TensorMeta::check(const std::vector<int64_t>& currentShape) const {
        // Unset rule: skip check
        if (shape.empty()) {
            return true;
        }
        // Dimension count must match
        if (shape.size() != currentShape.size()) {
            return false;
        }
        for (size_t i = 0; i < shape.size(); ++i) {
            // -1 means dynamic dimension: skip comparison
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



}