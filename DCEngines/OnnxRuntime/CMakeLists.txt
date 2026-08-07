# DCEngine_OnnxRuntime - ONNX Runtime 引擎适配器
#
# 需要 vcpkg onnxruntime feature：
#   1. 在 vcpkg.json 中启用 feature（已配置）
#   2. cmake -B build -S . -DBUILD_ENGINE_ONNXRUNTIME=ON
#
# 安装依赖（二选一）：
#   vcpkg install --feature onnxruntime          # 命令行
#   或在 CMake 配置时自动通过 manifest 安装

find_package(onnxruntime CONFIG REQUIRED)

add_library(DCEngine_OnnxRuntime STATIC)
add_library(DCEngine::OnnxRuntime ALIAS DCEngine_OnnxRuntime)

target_compile_features(DCEngine_OnnxRuntime PUBLIC cxx_std_20)

# ── EP 编译宏注入：由 DCINFER_ORT_EP 决定编译期默认 ExecutionProvider ──
# （依赖注入在 cmake/vcpkg-toolchain.cmake 中完成；此处仅映射编译宏）
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

file(GLOB_RECURSE DCENGINE_ONNX_SRC CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
target_sources(DCEngine_OnnxRuntime PRIVATE ${DCENGINE_ONNX_SRC})

# ── 集成测试（BUILD_TESTS 开启时编译）──
if (BUILD_TESTS)
    add_subdirectory(test)
endif()
