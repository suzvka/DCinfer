# cmake/vcpkg-toolchain.cmake - DCinfer vcpkg wrapper toolchain
# 将 DCINFER_ORT_EP 与模块开关映射为 vcpkg manifest feature，校验平台/链接
# 约束后委托 external/vcpkg 真实 toolchain。
# 用法：cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/vcpkg-toolchain.cmake \
#       -DBUILD_ENGINE_ONNXRUNTIME=ON -DDCINFER_ORT_EP=CUDA

set(VCPKG_MANIFEST_NO_DEFAULT_FEATURES ON)

# 项目定制 port overlay（不污染 external/vcpkg submodule）：onnx/onnxruntime
# CUDA 适配与 poco NetSSL_Win 定制见 cmake/overlay-ports/
set(VCPKG_OVERLAY_PORTS "${CMAKE_CURRENT_LIST_DIR}/overlay-ports")

# 定制 triplet overlay：x64-windows 追加 POCO_ENABLE_NETSSL_WIN
set(VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_LIST_DIR}/triplets")

# 构建并发上限（内存保护）：默认按逻辑核数并行，与 nvcc --threads 相乘会
# 耗尽内存；限制 4 个并行编译进程（可经 VCPKG_MAX_CONCURRENCY 覆盖）。
if (NOT DEFINED ENV{VCPKG_MAX_CONCURRENCY})
    set(ENV{VCPKG_MAX_CONCURRENCY} 4)
endif()

# ONNX Runtime ExecutionProvider 变体
set(DCINFER_ORT_EP "NONE" CACHE STRING
    "ONNX Runtime EP variant: NONE (no ORT deps), CPU, CUDA, TENSORRT, OPENVINO")
set_property(CACHE DCINFER_ORT_EP PROPERTY STRINGS NONE;CPU;CUDA;TENSORRT;OPENVINO)

# 预置模块开关默认值（与根 CMakeLists option 一致）：toolchain 在 project()
# 阶段加载、早于 option() 定义；-D/预设传入的值优先。
set(BUILD_ENGINE_ONNXRUNTIME OFF CACHE BOOL "Build ONNX Runtime engine adapter")
# 模块开关：新名优先（同步到旧名），否则预置旧名默认值
if (DEFINED DCINFER_BUILD_IR)
    set(BUILD_IR "${DCINFER_BUILD_IR}" CACHE BOOL "Build DCIr graph compiler" FORCE)
else()
    # 与根 CMakeLists 一致：IR 默认 OFF
    set(BUILD_IR OFF CACHE BOOL "Build DCIr graph compiler")
endif()
if (DEFINED DCINFER_BUILD_DCNET)
    set(BUILD_DCNET "${DCINFER_BUILD_DCNET}" CACHE BOOL "Build DCNet network adapters" FORCE)
else()
    set(BUILD_DCNET OFF CACHE BOOL "Build DCNet network adapters")
endif()

# EP / 模块 → manifest feature 注入：基础 dependencies 为空，模块依赖按
# feature 选择（ir → DCIr，net → DCNet）；core-only 零 vcpkg 安装。
if (BUILD_IR)
    list(APPEND VCPKG_MANIFEST_FEATURES "ir")
endif()
if (BUILD_DCNET)
    list(APPEND VCPKG_MANIFEST_FEATURES "net")
endif()

if (BUILD_ENGINE_ONNXRUNTIME)
    if (DCINFER_ORT_EP STREQUAL "NONE")
        message(FATAL_ERROR
            "DCINFER_ORT_EP is required when BUILD_ENGINE_ONNXRUNTIME=ON.\n"
            "Re-run cmake with e.g.: -DBUILD_ENGINE_ONNXRUNTIME=ON -DDCINFER_ORT_EP=CUDA\n"
            "Available EP variants: CPU, CUDA, TENSORRT, OPENVINO")
    endif()

    string(TOLOWER "${DCINFER_ORT_EP}" DCINFER_ORT_EP_LOWER)
    if (NOT DCINFER_ORT_EP_LOWER MATCHES "^(cpu|cuda|tensorrt|openvino)$")
        message(FATAL_ERROR
            "Unknown DCINFER_ORT_EP '${DCINFER_ORT_EP}'.\n"
            "Available EP variants: NONE, CPU, CUDA, TENSORRT, OPENVINO")
    endif()

    # vcpkg 按 VCPKG_MANIFEST_FEATURES 追加 --x-feature=<name>
    list(APPEND VCPKG_MANIFEST_FEATURES "ort-${DCINFER_ORT_EP_LOWER}")
endif()

# 委托真实 vcpkg toolchain（相对定位）
include("${CMAKE_CURRENT_LIST_DIR}/../external/vcpkg/scripts/buildsystems/vcpkg.cmake")

# 平台/链接方式校验（须在 vcpkg toolchain 加载后，VCPKG_TARGET_TRIPLET 才确定）
if (BUILD_ENGINE_ONNXRUNTIME)
    if (DCINFER_ORT_EP STREQUAL "CUDA" OR DCINFER_ORT_EP STREQUAL "TENSORRT")
        # onnxruntime[cuda] 端口仅支持 x64
        if (NOT VCPKG_TARGET_TRIPLET MATCHES "^x64-")
            message(FATAL_ERROR
                "DCINFER_ORT_EP=${DCINFER_ORT_EP} requires an x64 vcpkg triplet "
                "(got '${VCPKG_TARGET_TRIPLET}').")
        endif()
    endif()

    # cuda / tensorrt / openvino 端口 feature 均不支持静态链接
    if (NOT DCINFER_ORT_EP STREQUAL "CPU" AND VCPKG_TARGET_TRIPLET MATCHES "static")
        message(FATAL_ERROR
            "DCINFER_ORT_EP=${DCINFER_ORT_EP} is not supported with a static "
            "vcpkg triplet (onnxruntime port constraint).\n"
            "Use a dynamic triplet, e.g. x64-windows / x64-linux "
            "(got '${VCPKG_TARGET_TRIPLET}').")
    endif()
endif()
