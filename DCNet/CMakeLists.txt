# DCNet - 张量网络传输框架（设计见 DCNet/DESIGN.md）
# 依赖 DCinfer + nlohmann-json + POCO（vcpkg 管理）
# 构建：-DDCINFER_BUILD_DCNET=ON（默认 OFF，与 vcpkg feature 'net' 对应）

include(GNUInstallDirs)

add_library(DCNet STATIC)
add_library(DCNet::DCNet ALIAS DCNet)

target_compile_features(DCNet PUBLIC cxx_std_20)

# MSVC：源码 UTF-8（含中文注释），目标级启用
if (MSVC)
    target_compile_options(DCNet PRIVATE /utf-8)
endif()

find_package(nlohmann_json CONFIG REQUIRED)
# POCO TLS：Windows 用 NetSSLWin(SChannel)，POSIX 用 NetSSL(OpenSSL)；
# 两模块 HTTPSClientSession API 一致，业务代码零条件编译。
if (WIN32)
    find_package(Poco CONFIG REQUIRED COMPONENTS Foundation Net NetSSLWin)
else()
    find_package(Poco CONFIG REQUIRED COMPONENTS Foundation Net NetSSL)
endif()

target_link_libraries(DCNet PUBLIC
    DCinfer::DCinfer
    nlohmann_json::nlohmann_json
    Poco::Foundation
    Poco::Net
)

if (WIN32)
    target_link_libraries(DCNet PUBLIC Poco::NetSSLWin)
else()
    target_link_libraries(DCNet PUBLIC Poco::NetSSL)
endif()

target_include_directories(DCNet
    PUBLIC
        # BUILD_INTERFACE：源码内消费（DCinfer 扁平头路径经目标传递）
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        # INSTALL_INTERFACE：安装树（消费方 include "DCNet/NetAdapter.h"）
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

file(GLOB_RECURSE DCNET_SRC CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
target_sources(DCNet PRIVATE ${DCNET_SRC})

# 安装与 find_package 支持：依赖链 DCNetConfig → find_dependency(DCinfer)
# + json + Poco（与 vcpkg feature 'net' 分层一致）。MockServer.h 为
# header-only 测试基建，随包安装保持可用。
if (PROJECT_NAME STREQUAL "DCinfer" AND PROJECT_VERSION)
    set(DCINFER_PKG_VERSION "${PROJECT_VERSION}")
else()
    set(DCINFER_PKG_VERSION "0.7.6")
endif()

install(TARGETS DCNet EXPORT DCNetTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 保留 DCNet/ 目录结构（源码以前缀风格 include）
install(DIRECTORY include/DCNet DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(DCNET_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/DCNet")

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/DCNetConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/DCNetConfig.cmake"
    INSTALL_DESTINATION ${DCNET_INSTALL_CMAKEDIR}
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/DCNetConfigVersion.cmake"
    VERSION ${DCINFER_PKG_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/DCNetConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/DCNetConfigVersion.cmake"
    DESTINATION ${DCNET_INSTALL_CMAKEDIR}
)
install(EXPORT DCNetTargets
    FILE DCNetTargets.cmake
    NAMESPACE DCNet::
    DESTINATION ${DCNET_INSTALL_CMAKEDIR}
)

# 测试
if (DCINFER_BUILD_TESTS)
    add_subdirectory(test)
endif()
