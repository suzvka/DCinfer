# cmake/vcpkg-toolchain.cmake - DCinfer vcpkg wrapper toolchain
#
# 职责：
#   1. 将 DCINFER_ORT_EP 构建选项映射为 vcpkg manifest feature，自动安装对应
#      onnxruntime 变体——用户无需手动修改 vcpkg.json 或 CMake 配置
#   2. 显式选择才安装：禁用 manifest 默认 features（不再默认拉取 onnxruntime）
#   3. 平台/链接方式校验（CUDA / TensorRT / OpenVINO 端口仅支持动态库等约束）
#   4. 委托 external/vcpkg 真实 toolchain
#
# 用法：
#   cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=cmake/vcpkg-toolchain.cmake \
#         -DBUILD_ENGINE_ONNXRUNTIME=ON -DDCINFER_ORT_EP=CUDA

# ── 显式选择才安装：项目 manifest 的 features 不再默认全部安装 ──
set(VCPKG_MANIFEST_NO_DEFAULT_FEATURES ON)

# ── 项目定制 port（overlay，不污染 external/vcpkg submodule）──
# onnx / onnxruntime 的 CUDA 构建适配（MSVC 14.51 + CUDA 13.3 兼容性修复、
# ONNX_DISABLE_STATIC_REGISTRATION、ft_moe 排除等）存放在项目自己的
# cmake/overlay-ports/ 下，vcpkg 将优先使用 overlay 覆盖同名内置 port。
set(VCPKG_OVERLAY_PORTS "${CMAKE_CURRENT_LIST_DIR}/overlay-ports")

# ── 构建并发上限（内存保护）──
# vcpkg 默认按逻辑处理器数开 ninja 并行（16 线程机上为 -j17），与 nvcc 内部
# --threads 相乘后，CUTLASS/attention 重模板编译单元会耗尽 16GB 物理内存。
# 限制为 4 个并行编译进程（可用环境变量 VCPKG_MAX_CONCURRENCY 覆盖）。
if (NOT DEFINED ENV{VCPKG_MAX_CONCURRENCY})
    set(ENV{VCPKG_MAX_CONCURRENCY} 4)
endif()

# ── ONNX Runtime ExecutionProvider 变体（NONE / CPU / CUDA / TENSORRT / OPENVINO）──
set(DCINFER_ORT_EP "NONE" CACHE STRING
    "ONNX Runtime EP variant: NONE (no ORT deps), CPU, CUDA, TENSORRT, OPENVINO")
set_property(CACHE DCINFER_ORT_EP PROPERTY STRINGS NONE;CPU;CUDA;TENSORRT;OPENVINO)

# ── 预置引擎开关默认值（与根 CMakeLists option 保持一致；
#    toolchain 在 project() 阶段加载，早于 option() 定义）──
set(BUILD_ENGINE_ONNXRUNTIME OFF CACHE BOOL "Build ONNX Runtime engine adapter")

# ── EP → vcpkg manifest feature 注入 ──
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

    # vcpkg.cmake 按 VCPKG_MANIFEST_FEATURES 列表追加 --x-feature=<name>
    list(APPEND VCPKG_MANIFEST_FEATURES "ort-${DCINFER_ORT_EP_LOWER}")
endif()

# ── 委托真实 vcpkg toolchain（相对定位，随仓库移动）──
include("${CMAKE_CURRENT_LIST_DIR}/../external/vcpkg/scripts/buildsystems/vcpkg.cmake")

# ── 平台/链接方式校验（vcpkg toolchain 加载后 VCPKG_TARGET_TRIPLET 才确定）──
if (BUILD_ENGINE_ONNXRUNTIME)
    if (DCINFER_ORT_EP STREQUAL "CUDA" OR DCINFER_ORT_EP STREQUAL "TENSORRT")
        # onnxruntime[cuda] 端口约束：仅 x64 架构
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
