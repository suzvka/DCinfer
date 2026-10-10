include(GNUInstallDirs)

add_library(DCinfer STATIC)
add_library(DCinfer::DCinfer ALIAS DCinfer)

target_compile_features(DCinfer PUBLIC cxx_std_20)

# MSVC：源码 UTF-8（含中文注释），目标级启用
if (MSVC)
    target_compile_options(DCinfer PRIVATE /utf-8)
endif()

# Linux 需链接 pthread（Windows 的 Threads::Threads 为空操作）。
# 探测与 libatomic 同域隔离（RESET + 恢复）：宿主预置的 CMAKE_REQUIRED_*
# 会被 CMake 3.17 的 FindThreads 带入探测链接行，导致误判 not found。
include(CMakePushCheckState)
cmake_push_check_state(RESET)
find_package(Threads REQUIRED)
cmake_pop_check_state()
target_link_libraries(DCinfer PUBLIC Threads::Threads)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

    # GCC 的 <semaphore> 使用 64 位原子操作，部分平台须显式链接 libatomic
    # （MSVC/Clang 在标准库内）。先不带 atomic 探测，失败再尝试链接。
    function(_dcinfer_check_atomic)
        include(CheckCXXSourceCompiles)
        include(CMakePushCheckState)
        cmake_push_check_state(RESET)
        # 只探测 DCinfer 自身的链接需求；重配置时重评估，避免缓存反映其他上下文
        unset(DCINFER_ATOMIC_BUILTIN_OK CACHE)
        unset(DCINFER_NEEDS_LIBATOMIC CACHE)
        set(CMAKE_REQUIRED_LIBRARIES "")
        set(CMAKE_CXX_STANDARD 20)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
        set(CMAKE_CXX_EXTENSIONS OFF)
        set(_source "#include <semaphore>
                     #include <atomic>
                     #include <cstdint>
                     int main() {
                         std::counting_semaphore<> s(1);
                         s.acquire(); s.release();
                         std::atomic<std::int64_t> a{0};
                         a.fetch_add(1, std::memory_order_relaxed);
                         return 0;
                     }")
        check_cxx_source_compiles("${_source}" DCINFER_ATOMIC_BUILTIN_OK)
        if(NOT DCINFER_ATOMIC_BUILTIN_OK)
            list(APPEND CMAKE_REQUIRED_LIBRARIES atomic)
            check_cxx_source_compiles("${_source}" DCINFER_NEEDS_LIBATOMIC)
            if(NOT DCINFER_NEEDS_LIBATOMIC)
                message(FATAL_ERROR "C++20 semaphore/atomic probe failed both with and without libatomic")
            endif()
        endif()
        cmake_pop_check_state()
    endfunction()
    _dcinfer_check_atomic()
    if(NOT DCINFER_ATOMIC_BUILTIN_OK AND DCINFER_NEEDS_LIBATOMIC)
        target_link_libraries(DCinfer PUBLIC atomic)
    endif()
endif()

target_include_directories(DCinfer
    PUBLIC
        # BUILD_INTERFACE：源码内消费（扁平路径，裸文件名 include）；须用
        # 生成器表达式包装，裸绝对路径会被 install(EXPORT) 拒绝。
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Tensor>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Tools>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Node>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Graph>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Compose>
        # INSTALL_INTERFACE：安装树（公开头扁平合并到 include/，见下方 install 规则）
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

file(GLOB_RECURSE DCINFER_SRC_FILES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
)
target_sources(DCinfer PRIVATE ${DCINFER_SRC_FILES})

# 安装与 find_package 支持：消费方式 add_subdirectory() 或安装后
# find_package(DCinfer CONFIG REQUIRED)。
# 宿主直接 add_subdirectory 本目录时 PROJECT_VERSION 为宿主版本，须回落到
# 字面版本，否则写出的 *ConfigVersion.cmake 会带宿主版本号。
if (PROJECT_NAME STREQUAL "DCinfer" AND PROJECT_VERSION)
    set(DCINFER_PKG_VERSION "${PROJECT_VERSION}")
else()
    set(DCINFER_PKG_VERSION "0.7.6")
endif()

install(TARGETS DCinfer EXPORT DCinferTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 公开头扁平安装：源码以裸文件名 include（如 "InferGraph.h"），各子目录
# 公开头合并进 include/。Node/internal/ 为 Pimpl 私有头（公开头零引用），
# 不进入安装树。
install(DIRECTORY
    include/Graph/
    include/Node/
    include/Tensor/
    include/Tools/
    include/Compose/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN "internal" EXCLUDE
)

set(DCINFER_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/DCinfer")

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/DCinferConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/DCinferConfig.cmake"
    INSTALL_DESTINATION ${DCINFER_INSTALL_CMAKEDIR}
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/DCinferConfigVersion.cmake"
    VERSION ${DCINFER_PKG_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/DCinferConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/DCinferConfigVersion.cmake"
    DESTINATION ${DCINFER_INSTALL_CMAKEDIR}
)
install(EXPORT DCinferTargets
    FILE DCinferTargets.cmake
    NAMESPACE DCinfer::
    DESTINATION ${DCINFER_INSTALL_CMAKEDIR}
)
