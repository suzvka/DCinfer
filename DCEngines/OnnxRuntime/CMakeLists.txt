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
