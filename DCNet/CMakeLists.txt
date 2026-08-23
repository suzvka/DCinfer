# DCNet - 网络适配器算子集合
#
# 仅依赖 DCinfer 核心库 + nlohmann-json（已在核心依赖树中，零新增必选依赖）。
# 形态与 DCEngines/OnnxRuntime 一致：EngineDescriptor 家族，运行时注入。
# 设计见 DCNet/DESIGN.md。
#
# 构建：cmake -B build -S . -DBUILD_DCNET=ON（默认 ON）

add_library(DCNet STATIC)
add_library(DCNet::DCNet ALIAS DCNet)

target_compile_features(DCNet PUBLIC cxx_std_20)

# 外部依赖
find_package(nlohmann_json CONFIG REQUIRED)

target_link_libraries(DCNet PUBLIC
    DCinfer::DCinfer
    nlohmann_json::nlohmann_json
)

# HTTP transport 后端：Windows 用 WinHTTP（零新增依赖）；非 Windows 见 DESIGN.md §9 展望
if (WIN32)
    target_link_libraries(DCNet PUBLIC winhttp)
endif()

target_include_directories(DCNet
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

file(GLOB_RECURSE DCNET_SRC CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
target_sources(DCNet PRIVATE ${DCNET_SRC})

# ── 测试（BUILD_TESTS 开启时编译）──
if (BUILD_TESTS)
    add_subdirectory(test)
endif()
