cmake_minimum_required(VERSION 3.17)

project(DCinfer VERSION 0.4.0 LANGUAGES CXX)

# Guidance: on Windows, prefer configuring with an explicit platform (e.g. -A x64)
# or from an x64 developer command prompt. If CMake selects NMake Makefiles it may
# default to x86 toolchain; in that case re-run with -A x64 or use an x64 Native Tools prompt.

# ── 顶层判定（CMake < 3.21 无 PROJECT_IS_TOP_LEVEL，自实现）──
# 仅顶层工程执行全局设置；被第三方 add_subdirectory() 引入时零污染宿主。
get_directory_property(DCINFER_PARENT PARENT_DIRECTORY)
if (DCINFER_PARENT STREQUAL "")
    set(DCINFER_IS_TOP_LEVEL ON)
else()
    set(DCINFER_IS_TOP_LEVEL OFF)
endif()

# ── 选项：DCINFER_ 命名空间（防宿主工程选项名冲突）──
# 旧 BUILD_* 名保留为兼容别名：仅当用户显式传入（-D/预设）且新名未定义时映射。
macro(_dcinfer_option _ns _legacy _default _doc)
    if (DEFINED ${_legacy} AND NOT DEFINED ${_ns})
        set(${_ns} "${${_legacy}}" CACHE BOOL "${_doc}（旧名 ${_legacy} 兼容映射）" FORCE)
    elseif (NOT DEFINED ${_ns})
        set(${_ns} "${_default}" CACHE BOOL "${_doc}")
    endif()
endmacro()

# 消费为子工程时，测试/示例默认关闭（避免拖入宿主构建）；顶层保持原默认
if (DCINFER_IS_TOP_LEVEL)
    set(_dcinfer_tests_default ON)
    set(_dcinfer_examples_default ON)
else()
    set(_dcinfer_tests_default OFF)
    set(_dcinfer_examples_default OFF)
endif()

# IR 默认 OFF：新用户裸 `cmake -B build` 零依赖即可配置成功；需要 JSON/.dcg
# 序列化时显式 -DDCINFER_BUILD_IR=ON（或用 core-ir 预设，经 vcpkg feature 'ir' 拉依赖）。
_dcinfer_option(DCINFER_BUILD_IR        BUILD_IR        OFF "Build DCIr graph compiler")
_dcinfer_option(DCINFER_BUILD_TESTS     BUILD_TESTS     "${_dcinfer_tests_default}"    "Build DCinfer tests")
_dcinfer_option(DCINFER_BUILD_DCNET     BUILD_DCNET     OFF "Build DCNet network adapters")
_dcinfer_option(DCINFER_BUILD_ENGINES   BUILD_ENGINES   ON  "Build engine adapters")
_dcinfer_option(DCINFER_BUILD_EXAMPLES  BUILD_EXAMPLES  "${_dcinfer_examples_default}" "Build examples")

if (DCINFER_IS_TOP_LEVEL)
    if (MSVC AND CMAKE_GENERATOR MATCHES "NMake")
        message(WARNING "Selected generator appears to be NMake Makefiles. Ensure you configure for x64 when building (e.g. cmake -A x64) or use an x64 developer prompt.")
    endif()

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

    if (MSVC)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /manifest:no")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /manifest:no")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /manifest:no")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
    endif ()
endif()

add_subdirectory(DCinfer)

# ── DCIr：推理图编译器（JSON → InferGraph）──
if (DCINFER_BUILD_IR)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/DCIr/CMakeLists.txt)
        add_subdirectory(DCIr)
    endif()
endif()

# ── 测试 ──
if (DCINFER_BUILD_TESTS)
    enable_testing()
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/DCinfer/test/CMakeLists.txt)
        add_subdirectory(DCinfer/test)
    endif()
    if (DCINFER_BUILD_IR AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/DCIr/CMakeLists.txt AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/DCIr/test/CMakeLists.txt)
        add_subdirectory(DCIr/test)
    endif()
endif()

# ── DCNet：张量网络传输框架（传输/张量格式/错误归一化/对接契约，见 DCNet/DESIGN.md；
# 协议级适配器如 OpenAI 兼容位于 DCEngines/OpenAI，基于本框架契约）──
# 置于测试之后：enable_testing() 先行，DCNet/test 的 add_test 才能注册
# 默认 OFF：核心默认纯净，用户只想要核心时无需处理网络框架开关
if (DCINFER_BUILD_DCNET)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/DCNet/CMakeLists.txt)
        add_subdirectory(DCNet)
    endif()
endif()

# ── 引擎适配器（可选，默认全 OFF，按需启用各引擎）──
if (DCINFER_BUILD_ENGINES)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/DCEngines/CMakeLists.txt)
        add_subdirectory(DCEngines)
    endif()
endif()

# ── 示例（可选）──
if (DCINFER_BUILD_EXAMPLES)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt)
        add_subdirectory(examples)
    endif()
endif()

# ── 配置摘要：显式列出启用项与依赖关系，防"配置成功但目标缺失" ──
message(STATUS "════ DCinfer ${PROJECT_VERSION} configure summary ════")
message(STATUS "  Top-level build : ${DCINFER_IS_TOP_LEVEL}")
message(STATUS "  Core library    : DCinfer (C++20, zero dependencies)")
message(STATUS "  DCIr (JSON/.dcg): ${DCINFER_BUILD_IR}")
message(STATUS "  DCNet           : ${DCINFER_BUILD_DCNET}")
if (DCINFER_BUILD_ENGINES)
    message(STATUS "  Engine Builtin  : ${BUILD_ENGINE_BUILTIN}")
    message(STATUS "  Engine OnnxRT   : ${BUILD_ENGINE_ONNXRUNTIME}")
    message(STATUS "  Engine OpenAI   : ${BUILD_ENGINE_OPENAI} (requires DCNet)")
else()
    message(STATUS "  Engines         : OFF")
endif()
message(STATUS "  Tests           : ${DCINFER_BUILD_TESTS}")
message(STATUS "  Examples        : ${DCINFER_BUILD_EXAMPLES}")
