# 02_lowering_benchmark：Broadcast(1) lowering 内省与延迟基线

add_executable(lowering_benchmark main.cpp)

target_link_libraries(lowering_benchmark
    PRIVATE
        DCinfer::DCinfer
)
