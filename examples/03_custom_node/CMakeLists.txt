# 03_custom_node - 编写自定义节点的完整教程
#
# 展示：NodePort 工厂声明 Schema、ctx.input<Tensor> 类型化读取、算子注册
# 依赖：仅需 DCinfer 核心（零外部依赖）

add_executable(custom_node main.cpp)

target_link_libraries(custom_node
    PRIVATE
        DCinfer::DCinfer
)
