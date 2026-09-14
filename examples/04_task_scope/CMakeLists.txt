# 04_task_scope - 内核坐标层作用域句柄示例
#
# 展示：同步一发（feed 链式 + run）与异步作用域（submit/wait + 析构清理）
# 依赖：仅需 DCEngine_Builtin（零外部依赖）

add_executable(task_scope main.cpp)

target_link_libraries(task_scope
    PRIVATE
        DCEngine::Builtin
)
