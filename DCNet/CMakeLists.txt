# DCNet - 张量网络传输框架
#
# 依赖 DCinfer 核心库 + nlohmann-json + POCO（HTTP/TLS，跨平台，经 vcpkg 管理）。
# 形态与 DCEngines/OnnxRuntime 一致：EngineDescriptor 家族，运行时注入。
# 设计见 DCNet/DESIGN.md。
#
# 构建：cmake -B build -S . -DDCINFER_BUILD_DCNET=ON（默认 OFF，与 vcpkg feature 'net' 对应）

add_library(DCNet STATIC)
add_library(DCNet::DCNet ALIAS DCNet)

target_compile_features(DCNet PUBLIC cxx_std_20)

# MSVC：源码为 UTF-8（含中文注释），目标级启用（add_subdirectory 消费时同样生效）
if (MSVC)
    target_compile_options(DCNet PRIVATE /utf-8)
endif()

# 外部依赖
find_package(nlohmann_json CONFIG REQUIRED)
# POCO：HTTP 客户端 + TLS（见 vcpkg.json / cmake/overlay-ports/poco）。
# TLS 实现：Windows 用 NetSSLWin(SChannel，经 cmake/triplets/x64-windows.cmake
# 的 POCO_ENABLE_NETSSL_WIN)；POSIX 用 NetSSL(OpenSSL)。
# 两模块的 HTTPSClientSession 头文件与 API 一致，业务代码零条件编译。
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
        # BUILD_INTERFACE：源码内消费（DCinfer 扁平头路径经 DCinfer::DCinfer 传递）
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        # INSTALL_INTERFACE：安装树（include/DCNet/ 结构保留，消费方 include "DCNet/NetAdapter.h"）
        $<INSTALL_INTERFACE:include>
)

file(GLOB_RECURSE DCNET_SRC CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
target_sources(DCNet PRIVATE ${DCNET_SRC})

# ── 安装与 find_package 支持 ──
# 依赖链：DCNetConfig → find_dependency(DCinfer) + json + Poco，与
# vcpkg feature 'net' 分层一致。MockServer.h 为 header-only 测试基建
# （经 DCNet::DCNet 传递包含），随包安装以保持该用法可用。
include(GNUInstallDirs)

# 包版本：与 DCinfer 模块同一约定（顶层 project(DCinfer VERSION) 权威 + 兑底）
set(DCINFER_PKG_VERSION "${PROJECT_VERSION}")
if(NOT DCINFER_PKG_VERSION)
    set(DCINFER_PKG_VERSION "0.4.0")
endif()

install(TARGETS DCNet EXPORT DCNetTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 头文件保留 DCNet/ 目录结构（源码以 "DCNet/NetAdapter.h" 前缀风格 include）
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

# ── 测试（DCINFER_BUILD_TESTS 开启时编译）──
if (DCINFER_BUILD_TESTS)
    add_subdirectory(test)
endif()
