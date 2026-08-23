# DCEngine_OpenAI - OpenAI 兼容远端服务引擎适配器
#
# 基于 DCNet 传输框架（HttpTransport + NetCodec 契约 + NetError 归一化），
# 对接任何 OpenAI 兼容服务（vLLM / TGI / 云 API 等）的 /v1/chat/completions。
# 形态与 OnnxRuntime 一致：EngineDescriptor 家族，运行时注入。
#
# 构建：cmake -B build -S . -DBUILD_ENGINE_OPENAI=ON（默认 ON）

add_library(DCEngine_OpenAI STATIC)
add_library(DCEngine::OpenAI ALIAS DCEngine_OpenAI)

target_compile_features(DCEngine_OpenAI PUBLIC cxx_std_20)

# 外部依赖（nlohmann-json 已在核心依赖树；此处 find_package 为独立声明，
# 供本目录及下游直接使用 target，不依赖 DCNet 子目录作用域）
find_package(nlohmann_json CONFIG REQUIRED)

target_link_libraries(DCEngine_OpenAI PUBLIC
    DCinfer::DCinfer
    DCNet::DCNet
    nlohmann_json::nlohmann_json
)

target_include_directories(DCEngine_OpenAI
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

file(GLOB_RECURSE DCENGINE_OPENAI_SRC CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
target_sources(DCEngine_OpenAI PRIVATE ${DCENGINE_OPENAI_SRC})

# ── 集成测试（BUILD_TESTS 开启时编译）──
if (BUILD_TESTS)
    add_subdirectory(test)
endif()
