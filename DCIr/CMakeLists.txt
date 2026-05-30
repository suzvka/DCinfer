# DCIr module CMakeLists - minimal target to enable tests and cross-platform builds
# This file defines the DCIr library and the DCIr::DCIr alias expected by tests.

# Do not set cmake_minimum_required or project() here; handled by top-level.

add_library(DCIr STATIC)
add_library(DCIr::DCIr ALIAS DCIr)

target_compile_features(DCIr PUBLIC cxx_std_20)

# DCIr 依赖 DCinfer 的头文件（InferGraph.h, GraphException.h 等）
target_link_libraries(DCIr PUBLIC DCinfer::DCinfer)

target_include_directories(DCIr
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
)

file(GLOB_RECURSE DCIr_SRC_FILES CONFIGURE_DEPENDS
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
)

target_sources(DCIr PRIVATE ${DCIr_SRC_FILES})
