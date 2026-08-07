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
    SHA512 373c51575ada457b8aead5d195a5f3eba62fb747b6370a2a9889fff875c40ea30af8fd49104d58cc86f79247410e829086b0979f37ca8635c6dd34960e9cc424
    PATCHES
        fix-cmake.patch # .framework install, external library workarounds(abseil-cpp, eigen3)
        fix-cmake-cuda.patch
        fix-missing-cstdint.patch
        fix-cmake-mlas.patch
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
    INVERTED_FEATURES
        cuda      onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION
)

if("cuda" IN_LIST FEATURES)
    vcpkg_find_cuda(OUT_CUDA_TOOLKIT_ROOT cuda_toolkit_root)
    list(APPEND FEATURE_OPTIONS
        "-DCMAKE_CUDA_COMPILER=${NVCC}"
        "-DCUDAToolkit_ROOT=${cuda_toolkit_root}"
        # "-DCMAKE_CUDA_ARCHITECTURES=native"
        # too much warnings about attribute
        # NOTE: -std=c++17 必须与 host 侧 --compiler-options=/std:c++17 配对。
        # nvcc 前端（cudafe++）默认方言与 host 不一致时，解析 MSVC 14.51+
        # 的 xtr1common（_Is_any_of_v 折叠表达式）会崩溃
        # (internal error: form_constant, cudafe++ died with 0xC0000409)。
        "-DCMAKE_CUDA_FLAGS=-std=c++17 -Xcudafe --diag_suppress=2803 -Wno-deprecated-gpu-targets -Xcompiler=/Zc:preprocessor -Xcompiler=/wd4996 --compiler-options=/std:c++17 -D__NV_NO_VECTOR_DEPRECATION_DIAG"
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

# CUDA 13.x：FasterTransformer MoE (ft_moe) 内核与新版 CUDA 工具链/absl 组合编译失败（host pass 模板解析错误），
# 暂时排除 ft_moe 模块与 MoE 算子实现（不影响其他 CUDA 算子，运行时仅 MoE 算子不可用）
# 注意：vcpkg_replace_string 的 REPLACE 必须写成单个字符串参数（CMake 不拼接相邻字符串字面量），
# 否则只有第一行生效，生成残缺的 CMakeLists 语法（Parse error: Function missing ending ")"）。
vcpkg_replace_string("${SOURCE_PATH}/cmake/onnxruntime_providers_cuda.cmake"
    "    list(APPEND onnxruntime_providers_cuda_src \${onnxruntime_cuda_contrib_ops_cc_srcs} \${onnxruntime_cuda_contrib_ops_cu_srcs})"
    [=[
    list(REMOVE_ITEM onnxruntime_cuda_contrib_ops_cu_srcs
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/moe/ft_moe/moe_kernel.cu"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/moe/ft_moe/moe_gemm_kernels_bf16_bf16.cu"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/moe/ft_moe/moe_gemm_kernels_bf16_uint4.cu"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/moe/ft_moe/moe_gemm_kernels_bf16_uint8.cu"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/moe/ft_moe/moe_gemm_kernels_fp16_fp16.cu"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/moe/ft_moe/moe_gemm_kernels_fp16_uint4.cu"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/moe/ft_moe/moe_gemm_kernels_fp16_uint8.cu"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/moe/ft_moe/moe_gemm_kernels_fp32_fp32.cu"
    )
    list(REMOVE_ITEM onnxruntime_cuda_contrib_ops_cc_srcs
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/moe/ft_moe/cutlass_heuristic.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/moe/moe.cc"
        # moe_quantization.cc 引用 ft_moe 内核（CutlassMoeFCRunner 等），一并排除避免链接失败
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/quantization/moe_quantization.cc"
    )
    list(APPEND onnxruntime_providers_cuda_src ${onnxruntime_cuda_contrib_ops_cc_srcs} ${onnxruntime_cuda_contrib_ops_cu_srcs})
    ]=]
)

# cuda_contrib_kernels.cc 是自动生成的算子注册表，仍引用 MoE/QMoE 类（定义在已排除的 moe.cc 中），
# 移除其 class 前向声明与 BuildKernelCreateInfo 注册行，避免 LNK2001 链接失败
vcpkg_replace_string("${SOURCE_PATH}/onnxruntime/contrib_ops/cuda/cuda_contrib_kernels.cc"
    [=[
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MoE);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MoE);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, MoE);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, QMoE);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, QMoE);
]=]
    ""
)
vcpkg_replace_string("${SOURCE_PATH}/onnxruntime/contrib_ops/cuda/cuda_contrib_kernels.cc"
    [=[
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MoE)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MoE)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, MoE)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, QMoE)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, QMoE)>,
]=]
    ""
)

# MSVC 14.51（VS 2026）：模板显式实例化的按值指针形参会剥离顶层 const（符号 PEAPEBX），
# 而调用侧隐式实例化保留顶层 const（符号 QEAPEBX），导致 concat_impl.cu 与 concat.cc
# 链接符号不一致（LNK2019）。显式实例化补写 const 使两侧符号一致。
vcpkg_replace_string("${SOURCE_PATH}/onnxruntime/core/providers/cuda/tensor/concat_impl.cu"
    "                                                      void* output_data, const void** input_data,"
    "                                                      void* output_data, const void** const input_data,"
)

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
