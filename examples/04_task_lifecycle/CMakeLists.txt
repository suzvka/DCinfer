# 04_task_lifecycle - 统一任务句柄：同步与异步同级
#
# 展示：同一套 GraphInterface::Task 的两种节奏（同步 run / 异步 submit+wait）
# 依赖：仅需 DCEngine_Builtin（零外部依赖）

add_executable(task_lifecycle main.cpp)

target_link_libraries(task_lifecycle
    PRIVATE
        DCEngine::Builtin
)