# DCNet - 网络适配器算子集合
#
# 依赖 DCinfer 核心库 + nlohmann-json + POCO（HTTP/TLS，跨平台，经 vcpkg 管理）。
# 形态与 DCEngines/OnnxRuntime 一致：EngineDescriptor 家族，运行时注入。
# 设计见 DCNet/DESIGN.md。
#
# 构建：cmake -B build -S . -DBUILD_DCNET=ON（默认 ON）

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
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

file(GLOB_RECURSE DCNET_SRC CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
target_sources(DCNet PRIVATE ${DCNET_SRC})

# ── 测试（DCINFER_BUILD_TESTS 开启时编译）──
if (DCINFER_BUILD_TESTS)
    add_subdirectory(test)
endif()
