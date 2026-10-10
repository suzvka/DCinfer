# 示例程序集合：各示例为独立可执行文件，展示 DCinfer 核心使用模式

if (NOT BUILD_ENGINE_BUILTIN)
	message(STATUS "BUILD_ENGINE_BUILTIN is OFF — skipping examples (Builtin engine required)")
	return()
endif()

add_subdirectory(01_hello_graph)
add_subdirectory(02_lowering_benchmark)
add_subdirectory(03_custom_node)
add_subdirectory(04_task_lifecycle)
