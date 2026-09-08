# 02_lowering_benchmark - Broadcast(1) lowering 内省与延迟基线
#
# 展示：Build → Freeze → Execute 边界的量化收益
# 依赖：仅需 DCinfer 核心（零外部依赖）

add_executable(lowering_benchmark main.cpp)

target_link_libraries(lowering_benchmark
    PRIVATE
        DCinfer::DCinfer
)
