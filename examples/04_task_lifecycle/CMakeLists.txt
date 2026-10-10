# 04_task_lifecycle：统一任务句柄，同步与异步同级

add_executable(task_lifecycle main.cpp)

target_link_libraries(task_lifecycle
    PRIVATE
        DCEngine::Builtin
)