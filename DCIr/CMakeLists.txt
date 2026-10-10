# DCIr module CMakeLists - minimal target to enable tests and cross-platform builds
# This file defines the DCIr library and the DCIr::DCIr alias expected by tests.

# Do not set cmake_minimum_required or project() here; handled by top-level.

include(GNUInstallDirs)

add_library(DCIr STATIC)
add_library(DCIr::DCIr ALIAS DCIr)

target_compile_features(DCIr PUBLIC cxx_std_20)

# MSVC：源码 UTF-8（含中文注释），需 /utf-8
if (MSVC)
	target_compile_options(DCIr PRIVATE /utf-8)
endif()

find_package(nlohmann_json CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(unofficial-minizip CONFIG REQUIRED)

target_link_libraries(DCIr PUBLIC
	DCinfer::DCinfer
	nlohmann_json::nlohmann_json
	ZLIB::ZLIB
	unofficial::minizip::minizip
)

if (WIN32)
	target_link_libraries(DCIr PRIVATE advapi32)
endif()

target_include_directories(DCIr
	PUBLIC
		# 源码内消费（扁平头路径经 DCinfer::DCinfer 传递）
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		# 安装树（消费方 include "Ir/GraphCompiler.h"）
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

file(GLOB_RECURSE DCIr_SRC_FILES CONFIGURE_DEPENDS
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
)

target_sources(DCIr PRIVATE ${DCIr_SRC_FILES})

# 安装与 find_package 支持：依赖链 DCIrConfig → find_dependency(DCinfer) +
# json/zlib/minizip（与 vcpkg feature 'ir' 分层一致）。
if (PROJECT_NAME STREQUAL "DCinfer" AND PROJECT_VERSION)
	set(DCINFER_PKG_VERSION "${PROJECT_VERSION}")
else()
	set(DCINFER_PKG_VERSION "0.7.6")
endif()

install(TARGETS DCIr EXPORT DCIrTargets
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 保留 Ir/ 目录结构
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
