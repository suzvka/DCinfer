#pragma once
#include <vector>
#include <onnxruntime_cxx_api.h>

namespace DC {
    static const std::unordered_map<std::string, ONNXTensorElementDataType> findType = {
        { typeid(float).name(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT    },
        { typeid(uint8_t).name(), ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8    },
        { typeid(int8_t).name(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8     },
        { typeid(uint16_t).name(), ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16   },
        { typeid(int16_t).name(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16    },
        { typeid(int32_t).name(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32    },
        { typeid(int64_t).name(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64    },
        { typeid(std::string).name(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING  },
        { typeid(bool).name(), ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL     },
        { typeid(double).name(), ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE   },
    };

    static const std::unordered_map<ONNXTensorElementDataType, std::string> findEnum = {
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT   , typeid(float).name()  },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8   , typeid(uint8_t).name()  },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8    , typeid(int8_t).name()  },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16  , typeid(uint16_t).name()  },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16   , typeid(int16_t).name()  },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32   , typeid(int32_t).name()  },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64   , typeid(int64_t).name()  },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING  , typeid(std::string).name() },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL    , typeid(bool).name()  },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE  , typeid(double).name()  },
    };

    static const std::unordered_map<ONNXTensorElementDataType, int> typeSize = {
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT   , sizeof(float)     },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8   , sizeof(uint8_t)   },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8    , sizeof(int8_t)    },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16  , sizeof(uint16_t)  },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16   , sizeof(int16_t)   },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32   , sizeof(int32_t)   },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64   , sizeof(int64_t)   },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING  , sizeof(std::string)},
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL    , sizeof(bool)      },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 , sizeof(float)     },
        { ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE  , sizeof(double)    },
    };

    //推导输入张量形状==================================================================
    // 辅助模板：判断是否为std::vector-------------------------------------
    template <typename T>
    struct is_std_vector : std::false_type {};
    template <typename T, typename Alloc>
    struct is_std_vector<std::vector<T, Alloc>> : std::true_type {};
    // 推导形状的核心递归模板----------------------------------------------
    template <typename T>
    std::vector<int64_t> inferShape(const T& value) {
        // 如果是标量，返回空形状
        return {};
    }
    template <typename T>
    std::vector<int64_t> inferShape(const std::vector<T>& vec) {
        // 当前维度是vector的大小
        std::vector<int64_t> shape = { static_cast<int64_t>(vec.size()) };
        // 如果内部还是vector，递归推导
        if constexpr (is_std_vector<T>::value) {
            std::vector<int64_t> subShape = inferShape(vec[0]);
            shape.insert(shape.end(), subShape.begin(), subShape.end());
        }
        return shape;
    }
    //==================================================================推导输入张量形状
}