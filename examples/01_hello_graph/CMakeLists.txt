# 01_hello_graph：最简推理图示例

add_executable(hello_graph main.cpp)

target_link_libraries(hello_graph
    PRIVATE
        DCEngine::Builtin
)
