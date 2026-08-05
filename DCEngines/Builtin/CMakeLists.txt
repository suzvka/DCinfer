# DCEngine_Builtin - 零依赖 CPU 算子引擎
#
# 仅依赖 DCinfer 核心库，无需任何外部推理框架。
# 提供 Add / Mul / Identity 等基础算子，用于教学、测试和快速验证图拓扑。

add_library(DCEngine_Builtin STATIC)
add_library(DCEngine::Builtin ALIAS DCEngine_Builtin)

target_compile_features(DCEngine_Builtin PUBLIC cxx_std_20)

target_link_libraries(DCEngine_Builtin PUBLIC DCinfer::DCinfer)

target_include_directories(DCEngine_Builtin
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

file(GLOB_RECURSE DCENGINE_BUILTIN_SRC CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
target_sources(DCEngine_Builtin PRIVATE ${DCENGINE_BUILTIN_SRC})
