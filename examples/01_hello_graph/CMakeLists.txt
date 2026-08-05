# 01_hello_graph - 最简推理图示例
#
# 展示：注册算子 → 构建图 → 注入数据 → 提交 → 获取结果
# 依赖：仅需 DCEngine_Builtin（零外部依赖）

add_executable(hello_graph main.cpp)

target_link_libraries(hello_graph
    PRIVATE
        DCEngine::Builtin
)
