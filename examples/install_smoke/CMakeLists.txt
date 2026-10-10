# install_smoke - 安装版 DCinfer 的 find_package 冒烟验证（独立消费工程，不参与主构建）。
# 验证安装树闭环：find_package → 传递依赖解析 → 链接 → 运行。
#
# core-only（零外部依赖，无需 vcpkg）：
#   cmake -B build -S . -DCMAKE_PREFIX_PATH=<prefix>
#
# Builtin 引擎链（零外部依赖）：
#   cmake -B build -S . -DSMOKE_WITH_BUILTIN=ON -DCMAKE_PREFIX_PATH=<prefix>
#
# DCIr 链（依赖由本目录 vcpkg.json feature 'ir' 经主仓 toolchain wrapper 注入）：
#   cmake -B build -S . -DSMOKE_WITH_IR=ON -DDCINFER_BUILD_IR=ON \
#     -DCMAKE_PREFIX_PATH=<prefix> -DCMAKE_TOOLCHAIN_FILE=<主仓>/cmake/vcpkg-toolchain.cmake
#
# DCNet 链（同上，feature 'net'）：
#   cmake -B build -S . -DSMOKE_WITH_NET=ON -DDCINFER_BUILD_DCNET=ON \
#     -DCMAKE_PREFIX_PATH=<prefix> -DCMAKE_TOOLCHAIN_FILE=<主仓>/cmake/vcpkg-toolchain.cmake

cmake_minimum_required(VERSION 3.17)
project(DCinferInstallSmoke LANGUAGES CXX)

# Release smoke is intentionally a Core + DCIr consumer.  Keep the expected
# version explicit so a stale prefix or mismatched tag cannot pass silently.
set(SMOKE_EXPECT_VERSION "0.7.6" CACHE STRING "Expected DCinfer/ DCIr package version")
if (NOT SMOKE_EXPECT_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "SMOKE_EXPECT_VERSION must be a semantic version")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 版本对齐：导出文件用 SameMinorVersion 兼容策略，major.minor 不符时 find_package 失败；
# 每次 minor 升级需同步更新下面的版本号。
find_package(DCinfer ${SMOKE_EXPECT_VERSION} EXACT CONFIG REQUIRED)
if (NOT DCinfer_VERSION STREQUAL SMOKE_EXPECT_VERSION)
    message(FATAL_ERROR "DCinfer version mismatch: expected ${SMOKE_EXPECT_VERSION}, got ${DCinfer_VERSION}")
endif()
message(STATUS "DCinfer_VERSION = ${DCinfer_VERSION}")

add_executable(install_smoke main.cpp)
target_link_libraries(install_smoke PRIVATE DCinfer::DCinfer)

# MSVC：源码含中文注释，需 /utf-8
if (MSVC)
    target_compile_options(install_smoke PRIVATE /utf-8)
endif()

if (SMOKE_WITH_IR)
    find_package(DCIr ${SMOKE_EXPECT_VERSION} EXACT CONFIG REQUIRED)
    if (NOT DCIr_VERSION STREQUAL SMOKE_EXPECT_VERSION)
        message(FATAL_ERROR "DCIr version mismatch: expected ${SMOKE_EXPECT_VERSION}, got ${DCIr_VERSION}")
    endif()
    message(STATUS "DCIr_VERSION = ${DCIr_VERSION}")
    target_link_libraries(install_smoke PRIVATE DCIr::DCIr)
    target_compile_definitions(install_smoke PRIVATE SMOKE_WITH_IR=1)
endif()

# The SDK smoke must never accidentally validate an engine package.
if (SMOKE_SDK AND (SMOKE_WITH_BUILTIN OR SMOKE_WITH_NET))
    message(FATAL_ERROR "SMOKE_SDK excludes third-party engines and DCNet")
endif()

if (SMOKE_WITH_BUILTIN)
    find_package(DCEngine CONFIG REQUIRED)
    message(STATUS "DCEngine_VERSION = ${DCEngine_VERSION}")
    target_link_libraries(install_smoke PRIVATE DCEngine::Builtin)
    target_compile_definitions(install_smoke PRIVATE SMOKE_WITH_BUILTIN=1)
endif()

if (SMOKE_WITH_NET)
    find_package(DCNet CONFIG REQUIRED)
    message(STATUS "DCNet_VERSION = ${DCNet_VERSION}")
    target_link_libraries(install_smoke PRIVATE DCNet::DCNet)
    target_compile_definitions(install_smoke PRIVATE SMOKE_WITH_NET=1)
endif()
