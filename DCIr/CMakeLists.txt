# DCIr module CMakeLists - minimal target to enable tests and cross-platform builds
# This file defines the DCIr library and the DCIr::DCIr alias expected by tests.

# Do not set cmake_minimum_required or project() here; handled by top-level.

add_library(DCIr STATIC)
add_library(DCIr::DCIr ALIAS DCIr)

target_compile_features(DCIr PUBLIC cxx_std_20)

# MSVC：源码为 UTF-8（含中文注释），目标级启用（add_subdirectory 消费时同样生效）
if (MSVC)
	target_compile_options(DCIr PRIVATE /utf-8)
endif()

# 外部依赖
find_package(nlohmann_json CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(unofficial-minizip CONFIG REQUIRED)

# DCIr 依赖 DCinfer 的头文件（InferGraph.h, GraphException.h 等）
target_link_libraries(DCIr PUBLIC
	DCinfer::DCinfer
	nlohmann_json::nlohmann_json
	ZLIB::ZLIB
	unofficial::minizip::minizip
)

target_include_directories(DCIr
	PUBLIC
		# BUILD_INTERFACE：源码内消费（含 DCinfer 扁平头路径，经 DCinfer::DCinfer 传递）
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		# INSTALL_INTERFACE：安装树（include/Ir/ 结构保留，消费方 include "Ir/GraphCompiler.h"）
		$<INSTALL_INTERFACE:include>
)

file(GLOB_RECURSE DCIr_SRC_FILES CONFIGURE_DEPENDS
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
)

target_sources(DCIr PRIVATE ${DCIr_SRC_FILES})

# ── 安装与 find_package 支持 ──
# 依赖链：DCIrConfig → find_dependency(DCinfer) + json/zlib/minizip，与
# vcpkg feature 'ir' 分层一致；core-only 用户不安装本包、不拉这些依赖。
include(GNUInstallDirs)

# 包版本：与 DCinfer 模块同一约定（顶层 project(DCinfer VERSION) 权威 + 兜底）
set(DCINFER_PKG_VERSION "${PROJECT_VERSION}")
if(NOT DCINFER_PKG_VERSION)
	set(DCINFER_PKG_VERSION "0.4.0")
endif()

install(TARGETS DCIr EXPORT DCIrTargets
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 头文件保留 Ir/ 目录结构（源码以 "Ir/GraphCompiler.h" 前缀风格 include）
install(DIRECTORY include/Ir DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(DCIR_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/DCIr")

include(CMakePackageConfigHelpers)
configure_package_config_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/DCIrConfig.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/DCIrConfig.cmake"
	INSTALL_DESTINATION ${DCIR_INSTALL_CMAKEDIR}
)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/DCIrConfigVersion.cmake"
	VERSION ${DCINFER_PKG_VERSION}
	COMPATIBILITY SameMinorVersion
)

install(FILES
	"${CMAKE_CURRENT_BINARY_DIR}/DCIrConfig.cmake"
	"${CMAKE_CURRENT_BINARY_DIR}/DCIrConfigVersion.cmake"
	DESTINATION ${DCIR_INSTALL_CMAKEDIR}
)
install(EXPORT DCIrTargets
	FILE DCIrTargets.cmake
	NAMESPACE DCIr::
	DESTINATION ${DCIR_INSTALL_CMAKEDIR}
)
