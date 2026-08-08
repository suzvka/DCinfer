# https://github.com/microsoft/onnxruntime/blob/v1.22.1/tools/python/util/vcpkg_helpers.py
message(WARNING "The port requires 'onnx' port build with CMake option ONNX_DISABLE_STATIC_REGISTRATION=ON")
if(VCPKG_TARGET_IS_OSX OR VCPKG_TARGET_IS_IOS)
    if("framework" IN_LIST FEATURES)
        # The Objective-C API requires onnxruntime_BUILD_SHARED_LIB
        vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)
    endif()
endif()

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO microsoft/onnxruntime
    REF "v${VERSION}"
    SHA512 31ee13a8b89f2bfd0c58086258c15b11218a675e577d287d94acc20723b0ca0110458bfaf268d3e9140fe86e9a0fe3f89abbc2d6d347f8ea65624fc0716f14c1
    PATCHES
        fix-cmake.patch # .framework install, re2 wasm override
        fix-cmake-cuda.patch # vcpkg deps: cuDNN / cudnn-frontend / nvidia-cutlass + MSVC /bigobj
)

find_program(PROTOC NAMES protoc PATHS "${CURRENT_HOST_INSTALLED_DIR}/tools/protobuf" REQUIRED NO_DEFAULT_PATH NO_CMAKE_PATH)
message(STATUS "Using protoc: ${PROTOC}")

find_program(FLATC NAMES flatc PATHS "${CURRENT_HOST_INSTALLED_DIR}/tools/flatbuffers" REQUIRED NO_DEFAULT_PATH NO_CMAKE_PATH)
message(STATUS "Using flatc: ${FLATC}")

vcpkg_find_acquire_program(PYTHON3)
get_filename_component(PYTHON_PATH "${PYTHON3}" PATH)
message(STATUS "Using python3: ${PYTHON3}")

vcpkg_execute_required_process(
    COMMAND "${PYTHON3}" onnxruntime/core/flatbuffers/schema/compile_schema.py --flatc "${FLATC}"
    LOGNAME compile_schema_core
    WORKING_DIRECTORY "${SOURCE_PATH}"
)
vcpkg_execute_required_process(
    COMMAND "${PYTHON3}" onnxruntime/lora/adapter_format/compile_schema.py --flatc "${FLATC}"
    LOGNAME compile_schema_lora
    WORKING_DIRECTORY "${SOURCE_PATH}"
)

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        python    onnxruntime_ENABLE_PYTHON
        training  onnxruntime_ENABLE_TRAINING
        training  onnxruntime_ENABLE_TRAINING_APIS
        cuda      onnxruntime_USE_CUDA
        cuda      onnxruntime_USE_CUDA_NHWC_OPS
        openvino  onnxruntime_USE_OPENVINO
        tensorrt  onnxruntime_USE_TENSORRT
        tensorrt  onnxruntime_USE_TENSORRT_BUILTIN_PARSER
        directml  onnxruntime_USE_DML
        directml  onnxruntime_USE_CUSTOM_DIRECTML
        winml     onnxruntime_USE_WINML
        coreml    onnxruntime_USE_COREML
        mimalloc  onnxruntime_USE_MIMALLOC
        valgrind  onnxruntime_USE_VALGRIND
        xnnpack   onnxruntime_USE_XNNPACK
        kleidiai  onnxruntime_USE_KLEIDIAI
        nnapi     onnxruntime_USE_NNAPI_BUILTIN
        azure     onnxruntime_USE_AZURE
        test      onnxruntime_BUILD_UNIT_TESTS
        test      onnxruntime_BUILD_BENCHMARKS
        test      onnxruntime_RUN_ONNX_TESTS
        framework onnxruntime_BUILD_APPLE_FRAMEWORK
        framework onnxruntime_BUILD_OBJC
        nccl      onnxruntime_USE_NCCL
    # 注：1.23.2 时代的 INVERTED_FEATURES（cuda → USE_MEMORY_EFFICIENT_ATTENTION=OFF）
    # 已移除——1.28 中 MEA 随 CUDA 默认开启，且 attention.cc 无条件引用
    # kCutlassSafeMaskFilterValue（定义在 USE_MEMORY_EFFICIENT_ATTENTION 守卫内），
    # 关闭 MEA 会导致 C2039 编译失败。
)

if("cuda" IN_LIST FEATURES)
    vcpkg_find_cuda(OUT_CUDA_TOOLKIT_ROOT cuda_toolkit_root)
    list(APPEND FEATURE_OPTIONS
        "-DCMAKE_CUDA_COMPILER=${NVCC}"
        "-DCUDAToolkit_ROOT=${cuda_toolkit_root}"
        # 注：1.23.2 时代的 -std=c++17 / --compiler-options=/std:c++17 方言 workaround 已移除——
        # ORT 1.25+ 源码构建强制 C++20（onnxruntime_language_standard_versions.cmake），
        # 且上游已内置 CUDA 13.3 cudafe++ regression workaround（ort_cuda133_patch_cccl_header）
        # 与 VS 2026 / CUDA 13.3 Windows 构建修复（#29042, #29266）。
        "-DCMAKE_CUDA_FLAGS=-Xcudafe --diag_suppress=2803 -Xcompiler=/Zc:preprocessor -Xcompiler=/wd4996 -D__NV_NO_VECTOR_DEPRECATION_DIAG"
        "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} /Zc:preprocessor /wd4996 -D__NV_NO_VECTOR_DEPRECATION_DIAG"
    )
endif()

if("tensorrt" IN_LIST FEATURES)
    if(DEFINED ENV{TENSORRT_HOME})
        set(TENSORRT_HOME "$ENV{TENSORRT_HOME}")
    endif()
    if(DEFINED TENSORRT_HOME)
        message(STATUS "Using TensorRT: ${TENSORRT_HOME}")
        list(APPEND FEATURE_OPTIONS "-Donnxruntime_TENSORRT_HOME:PATH=${TENSORRT_HOME}")
    else()
        message(WARNING "Define TENSORRT_HOME for onnxruntime_TENSORRT_HOME")
    endif()
endif()

string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "dynamic" BUILD_SHARED)

# 注：1.23.2 时代的三处源码 workaround 已随升级到 1.28.0 移除——
# 1) ft_moe/MoE 排除：1.28 的 MoE 实现已重写（moe.cc + qmoe_kernels.cu，ft_moe 已删除），
#    上游 CUDA 13 CI 正常编译，MoE/QMoE 算子恢复可用；
# 2) cuda_contrib_kernels.cc MoE 注册行清理：随 1) 一并移除；
# 3) concat_impl.cu 顶层 const 补写：上游已修复（显式实例化现为 const void** const）。

# see tools/ci_build/build.py
vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}/cmake"
    OPTIONS
        ${FEATURE_OPTIONS}
        "-DPython_EXECUTABLE:FILEPATH=${PYTHON3}"
        "-DProtobuf_PROTOC_EXECUTABLE:FILEPATH=${PROTOC}"
        "-DONNX_CUSTOM_PROTOC_EXECUTABLE:FILEPATH=${PROTOC}"
        -DBUILD_PKGCONFIG_FILES=ON
        -Donnxruntime_BUILD_SHARED_LIB=${BUILD_SHARED}
        -Donnxruntime_CROSS_COMPILING=${VCPKG_CROSSCOMPILING}
        -Donnxruntime_USE_EXTENSIONS=OFF
        -Donnxruntime_USE_NNAPI_BUILTIN=${VCPKG_TARGET_IS_ANDROID}
        -Donnxruntime_USE_VCPKG=ON
        -Donnxruntime_ENABLE_CPUINFO=ON
        -Donnxruntime_ENABLE_MICROSOFT_INTERNAL=OFF
        -Donnxruntime_ENABLE_BITCODE=OFF
        -Donnxruntime_ENABLE_PYTHON=OFF
        -Donnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS=OFF
        -Donnxruntime_ENABLE_MEMORY_PROFILE=OFF
        -Donnxruntime_ENABLE_LAZY_TENSOR=OFF
        -Donnxruntime_DISABLE_RTTI=OFF
        -Donnxruntime_DISABLE_ABSEIL=OFF
        # 内存保护：本机 16GB RAM。nvcc 默认 --threads 4 与 ninja 并行度相乘
        # 会耗尽内存（CUTLASS/attention 重模板 TU 单进程峰值数 GB），
        # 固定为 1；并行度由 VCPKG_MAX_CONCURRENCY=4（见 cmake/vcpkg-toolchain.cmake）统一控制
        -Donnxruntime_NVCC_THREADS=1
        # 目标 GPU：RTX 4070 (sm_89)；CUDA 13 已移除 sm_60 等旧架构，ORT 默认列表含 60 会配置失败
        "-DCMAKE_CUDA_ARCHITECTURES=89-real"
        # some other customizations ...
        --compile-no-warning-as-error
    OPTIONS_DEBUG
        -Donnxruntime_ENABLE_MEMLEAK_CHECKER=OFF
        -Donnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1
    MAYBE_UNUSED_VARIABLES
        Python_EXECUTABLE
        onnxruntime_TENSORRT_PLACEHOLDER_BUILDER
        onnxruntime_NVCC_THREADS
        CMAKE_CUDA_FLAGS
        onnxruntime_USE_CUSTOM_DIRECTML
)
if("cuda" IN_LIST FEATURES)
    vcpkg_cmake_build(TARGET onnxruntime_providers_cuda LOGFILE_BASE build-cuda)
endif()
if("tensorrt" IN_LIST FEATURES)
    vcpkg_cmake_build(TARGET onnxruntime_providers_tensorrt LOGFILE_BASE build-tensorrt)
endif()
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/onnxruntime)
vcpkg_fixup_pkgconfig() # pkg_check_modules(libonnxruntime)

# relocates the onnxruntime_providers_* binaries before vcpkg_copy_pdbs()
function(reolocate_ort_providers)
    if(VCPKG_TARGET_IS_WINDOWS AND (VCPKG_LIBRARY_LINKAGE STREQUAL "dynamic"))
        # the target is expected to be used without the .lib files
        file(GLOB PROVIDE_BINS_DBG  "${CURRENT_PACKAGES_DIR}/debug/lib/onnxruntime_providers_*.dll")
        file(COPY ${PROVIDE_BINS_DBG} DESTINATION "${CURRENT_PACKAGES_DIR}/debug/bin")
        file(GLOB PROVIDE_BINS_REL "${CURRENT_PACKAGES_DIR}/lib/onnxruntime_providers_*.dll")
        file(COPY ${PROVIDE_BINS_REL} DESTINATION "${CURRENT_PACKAGES_DIR}/bin")
        file(REMOVE ${PROVIDE_BINS_DBG} ${PROVIDE_BINS_REL})
    endif()
endfunction()

reolocate_ort_providers()
vcpkg_copy_pdbs()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
if(VCPKG_LIBRARY_LINKAGE STREQUAL "static")
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/bin" "${CURRENT_PACKAGES_DIR}/bin")
endif()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
