# install_smoke - 安装版 DCinfer 的 find_package 冒烟验证（独立消费工程）
#
# 验证安装树闭环：find_package → 传递依赖解析 → 链接 → 运行。
# 独立工程，不参与主构建（examples/CMakeLists.txt 不引入本目录）。
#
# 用法（先按 README“作为库消费”完成安装，<prefix> 为安装前缀）：
#   core-only 冒烟（零外部依赖，无需 vcpkg）：
#     cmake -B build -S . -DCMAKE_PREFIX_PATH=<prefix>
#     cmake --build build && ./build/install_smoke
#
# 需要一并验证 DCIr 安装链时（DCIr 传递依赖由本目录 vcpkg.json 的 feature 'ir'
# 提供，经主仓 toolchain wrapper 由 -DDCINFER_BUILD_IR=ON 注入）：
#   cmake -B build -S . -DSMOKE_WITH_IR=ON -DCMAKE_PREFIX_PATH=<prefix> \
#     -DCMAKE_TOOLCHAIN_FILE=<主仓>/cmake/vcpkg-toolchain.cmake -DDCINFER_BUILD_IR=ON

cmake_minimum_required(VERSION 3.17)
project(DCinferInstallSmoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(DCinfer 0.2 CONFIG REQUIRED)
message(STATUS "DCinfer_VERSION = ${DCinfer_VERSION}")

add_executable(install_smoke main.cpp)
target_link_libraries(install_smoke PRIVATE DCinfer::DCinfer)

# MSVC：源码含中文注释，目标级启用 /utf-8（与主工程各模块一致）
if (MSVC)
    target_compile_options(install_smoke PRIVATE /utf-8)
endif()

if (SMOKE_WITH_IR)
    find_package(DCIr CONFIG REQUIRED)
    message(STATUS "DCIr_VERSION = ${DCIr_VERSION}")
    target_link_libraries(install_smoke PRIVATE DCIr::DCIr)
    target_compile_definitions(install_smoke PRIVATE SMOKE_WITH_IR=1)
endif()
