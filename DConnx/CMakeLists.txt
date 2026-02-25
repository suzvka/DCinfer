add_library(DConnx STATIC)
add_library(DConnx::DConnx ALIAS DConnx)

target_compile_features(DConnx PUBLIC cxx_std_20)

target_include_directories(DConnx
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Tensor
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Tools
)

file(GLOB_RECURSE DCONNX_SRC_FILES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
)
target_sources(DConnx PRIVATE ${DCONNX_SRC_FILES})

# ORT engine is intentionally not part of the core build.
# ORT-related sources/headers remain in the repository for local debugging only.

# Tests are handled in DConnx/test when BUILD_TESTS is enabled from the root CMakeLists
