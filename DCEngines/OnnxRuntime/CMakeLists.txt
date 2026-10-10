# DCEngine_OnnxRuntime - ONNX Runtime 引擎适配器
#
# 需要 vcpkg onnxruntime feature（vcpkg.json 已配置）：
#   cmake -B build -S . -DBUILD_ENGINE_ONNXRUNTIME=ON

find_package(onnxruntime CONFIG REQUIRED)

add_library(DCEngine_OnnxRuntime STATIC)
add_library(DCEngine::OnnxRuntime ALIAS DCEngine_OnnxRuntime)

target_compile_features(DCEngine_OnnxRuntime PUBLIC cxx_std_20)

# EP 编译宏注入：DCINFER_ORT_EP → 编译期默认 ExecutionProvider（依赖注入见 toolchain）
if (DCINFER_ORT_EP STREQUAL "CUDA")
    target_compile_definitions(DCEngine_OnnxRuntime PUBLIC DCINFER_ORT_ENABLE_CUDA=1)
elseif (DCINFER_ORT_EP STREQUAL "TENSORRT")
    target_compile_definitions(DCEngine_OnnxRuntime PUBLIC DCINFER_ORT_ENABLE_TENSORRT=1)
elseif (DCINFER_ORT_EP STREQUAL "OPENVINO")
    target_compile_definitions(DCEngine_OnnxRuntime PUBLIC DCINFER_ORT_ENABLE_OPENVINO=1)
elseif (NOT DCINFER_ORT_EP STREQUAL "CPU")
    # 兜底校验：wrapper toolchain 已拦截，此处双保险
    message(FATAL_ERROR
        "DCINFER_ORT_EP must be one of CPU/CUDA/TENSORRT/OPENVINO when "
        "BUILD_ENGINE_ONNXRUNTIME=ON (got '${DCINFER_ORT_EP}').")
endif()

target_link_libraries(DCEngine_OnnxRuntime PUBLIC
    DCinfer::DCinfer
    onnxruntime::onnxruntime
)

target_include_directories(DCEngine_OnnxRuntime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ORT 1.28 起导出 target 不再携带 include 路径（1.23 动态 target 自带，
# 1.28 静态 target 缺失）：按需 find_path 补齐，保证
# #include <onnxruntime_cxx_api.h> 在两种 linkage 下均可解析。
get_target_property(_ort_include_dirs onnxruntime::onnxruntime
                    INTERFACE_INCLUDE_DIRECTORIES)
if(NOT _ort_include_dirs OR _ort_include_dirs STREQUAL "_ort_include_dirs-NOTFOUND")
	# onnxruntime_DIR = <installed>/<triplet>/share/onnxruntime → 头文件在其上两级
	#（vcpkg 标准布局）；多候选兼容安装根层级差异。
	find_path(onnxruntime_INCLUDE_DIR
		NAMES onnxruntime_cxx_api.h
		HINTS "${onnxruntime_DIR}/../../include/onnxruntime"
				"${onnxruntime_DIR}/../../../include/onnxruntime"
		NO_DEFAULT_PATH)
	if(NOT onnxruntime_INCLUDE_DIR)
		message(FATAL_ERROR "ONNX Runtime C++ headers not found in the configured package")
	endif()
	target_include_directories(DCEngine_OnnxRuntime SYSTEM
		PUBLIC "${onnxruntime_INCLUDE_DIR}")
endif()

file(GLOB_RECURSE DCENGINE_ONNX_SRC CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
target_sources(DCEngine_OnnxRuntime PRIVATE ${DCENGINE_ONNX_SRC})

# 集成测试（DCINFER_BUILD_TESTS）
if (DCINFER_BUILD_TESTS)
    add_subdirectory(test)
endif()
