add_library(DCinfer STATIC)
add_library(DCinfer::DCinfer ALIAS DCinfer)

target_compile_features(DCinfer PUBLIC cxx_std_20)

# Linux: <thread>, <mutex>, <semaphore>, <condition_variable> 均需链接 pthread
# Windows: Threads::Threads 是空操作，不影响
find_package(Threads REQUIRED)
target_link_libraries(DCinfer PUBLIC Threads::Threads)

# GCC 10~13 的 <coroutine> 需要显式启用（GCC 14+ 内置）
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0")
    target_compile_options(DCinfer PUBLIC -fcoroutines)
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
