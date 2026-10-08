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

# ORT 1.28 起导出 target 不再携带 include 路径（上游移除了
# $<INSTALL_INTERFACE:...> 声明，头文件改经 PUBLIC_HEADER 安装到
# <prefix>/include/onnxruntime 平铺目录）——1.23 的动态导入 target 自带
# INTERFACE_INCLUDE_DIRECTORIES，1.28 的静态导入 target（x64-linux 默认
# static linkage）则完全缺失。按需补齐，保证 #include <onnxruntime_cxx_api.h>
# 在两种 linkage 下均可解析。
get_target_property(_ort_include_dirs onnxruntime::onnxruntime
                    INTERFACE_INCLUDE_DIRECTORIES)
if(NOT _ort_include_dirs OR _ort_include_dirs STREQUAL "_ort_include_dirs-NOTFOUND")
	# onnxruntime_DIR 指向 <installed>/<triplet>/share/onnxruntime，头文件在
	# <installed>/<triplet>/include/onnxruntime/（vcpkg 标准布局）——即
	# onnxruntime_DIR 的上两级。多候选兼容安装根层级差异。
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

# ── 集成测试（DCINFER_BUILD_TESTS 开启时编译）──
if (DCINFER_BUILD_TESTS)
    add_subdirectory(test)
endif()
