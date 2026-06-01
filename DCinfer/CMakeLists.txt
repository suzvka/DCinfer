add_library(DCinfer STATIC)
add_library(DCinfer::DCinfer ALIAS DCinfer)

target_compile_features(DCinfer PUBLIC cxx_std_20)

# Linux: <thread>, <mutex>, <semaphore>, <condition_variable> 均需链接 pthread
# Windows: Threads::Threads 是空操作，不影响
find_package(Threads REQUIRED)
target_link_libraries(DCinfer PUBLIC Threads::Threads)

# ── GCC 特化处理 ──
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC 10~13 的 <coroutine> 需要显式启用编译期支持（GCC 14+ 内置）
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0")
        target_compile_options(DCinfer PUBLIC -fcoroutines)
    endif()

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
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Tensor
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Tools
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Node
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Graph
)

file(GLOB_RECURSE DCINFER_SRC_FILES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
)
target_sources(DCinfer PRIVATE ${DCINFER_SRC_FILES})
# ORT engine is intentionally not part of the core build.
# ORT-related sources/headers remain in the repository for local debugging only.

# Tests are handled in DCinfer/test when BUILD_TESTS is enabled from the root CMakeLists
