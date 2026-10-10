# 03_custom_node：编写自定义节点的完整教程

add_executable(custom_node main.cpp)

target_link_libraries(custom_node
    PRIVATE
        DCinfer::DCinfer
)
