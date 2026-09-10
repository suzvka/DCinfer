add_library(DCinfer STATIC)
add_library(DCinfer::DCinfer ALIAS DCinfer)

target_compile_features(DCinfer PUBLIC cxx_std_20)

# MSVC：源码为 UTF-8（含中文注释），目标级启用（add_subdirectory 消费时同样生效）
if (MSVC)
    target_compile_options(DCinfer PRIVATE /utf-8)
endif()

# Linux: <thread>, <mutex>, <semaphore>, <condition_variable> 均需链接 pthread
# Windows: Threads::Threads 是空操作，不影响
find_package(Threads REQUIRED)
target_link_libraries(DCinfer PUBLIC Threads::Threads)

# ── GCC 特化处理 ──
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

    # GCC 的 <semaphore> / std::counting_semaphore 内部使用 64 位原子操作，
    # 在某些平台/架构上必须显式链接 libatomic 才能解析符号。
    # MSVC / Clang 的对应实现在标准库内，无需额外链接。
    #
    # 先不带 atomic 测试，失败再尝试链接 libatomic。
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles(
        "#include <semaphore>
         #include <atomic>
         int main() {
             std::counting_semaphore<> s(1);
             s.acquire();
             s.release();
             std::atomic<std::int64_t> a{0};
             a.fetch_add(1, std::memory_order_relaxed);
             return 0;
         }"
        DCINFER_ATOMIC_BUILTIN_OK
    )
    if(NOT DCINFER_ATOMIC_BUILTIN_OK)
        set(CMAKE_REQUIRED_LIBRARIES "-latomic")
        check_cxx_source_compiles(
            "#include <semaphore>
             #include <atomic>
             int main() {
                 std::counting_semaphore<> s(1);
                 s.acquire();
                 s.release();
                 std::atomic<std::int64_t> a{0};
                 a.fetch_add(1, std::memory_order_relaxed);
                 return 0;
             }"
            DCINFER_NEEDS_LIBATOMIC
        )
        set(CMAKE_REQUIRED_LIBRARIES "")
        if(DCINFER_NEEDS_LIBATOMIC)
            target_link_libraries(DCinfer PUBLIC atomic)
        endif()
    endif()
endif()

target_include_directories(DCinfer
    PUBLIC
        # BUILD_INTERFACE：源码内消费（多子目录扁平路径，裸文件名 include）。
        # 必须用生成器表达式包装——裸源码绝对路径会被 install(EXPORT) 拒绝。
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Tensor>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Tools>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Node>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Graph>
        # INSTALL_INTERFACE：安装树（各子目录公开头扁平合并到 include/，见下方 install 规则）
        $<INSTALL_INTERFACE:include>
)

file(GLOB_RECURSE DCINFER_SRC_FILES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
)
target_sources(DCinfer PRIVATE ${DCINFER_SRC_FILES})
# Tests are handled in DCinfer/test when DCINFER_BUILD_TESTS is enabled from the root CMakeLists

# ── 安装与 find_package 支持 ──
# 消费方式二选一：add_subdirectory()（BUILD_INTERFACE 源码路径）或安装后
# find_package(DCinfer CONFIG REQUIRED)（INSTALL_INTERFACE 相对路径）。
include(GNUInstallDirs)

# 包版本：顶层 project(DCinfer VERSION) 为权威；宿主经 add_subdirectory 引入且
# 未声明版本号时兜底（需与顶层 project() 版本同步维护）。
set(DCINFER_PKG_VERSION "${PROJECT_VERSION}")
if(NOT DCINFER_PKG_VERSION)
    set(DCINFER_PKG_VERSION "0.4.0")
endif()

install(TARGETS DCinfer EXPORT DCinferTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 公开头扁平安装：源码与示例均以裸文件名 include（如 "InferGraph.h"、"Tensor.hpp"），
# 因此各子目录公开头直接合并进 include/，与 BUILD_INTERFACE 的搜索路径语义等价。
# Node/internal/ 为 Pimpl 私有组件头（公开头零引用），不进入安装树。
install(DIRECTORY
    include/Graph/
    include/Node/
    include/Tensor/
    include/Tools/
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
