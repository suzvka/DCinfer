# 示例程序集合
#
# 每个示例是独立的可执行文件，展示 DCinfer 的核心使用模式。
# 大部分示例仅需 Builtin 引擎（零依赖），无需安装 ONNX Runtime。

# Builtin 引擎是示例的最低依赖
if (NOT BUILD_ENGINE_BUILTIN)
	message(STATUS "BUILD_ENGINE_BUILTIN is OFF — skipping examples (Builtin engine required)")
	return()
endif()

add_subdirectory(01_hello_graph)
