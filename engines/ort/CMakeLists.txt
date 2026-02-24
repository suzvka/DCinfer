# Placeholder for optional ONNX Runtime engine module.
# Intentionally not enabled by default.
#
# Future shape:
# add_library(DConnxEngineOrt ...)
# target_link_libraries(DConnxEngineOrt PUBLIC DConnx::DConnx onnxruntime::onnxruntime)
# target_compile_definitions(DConnxEngineOrt PUBLIC DCONNX_ENABLE_ORT=1)
