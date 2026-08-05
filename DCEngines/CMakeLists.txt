# DCEngines - 引擎适配器集合
#
# 每个引擎适配器是独立的静态库，按需编译。
# 核心库 DCinfer 不反向依赖任何引擎适配器——
# 引擎通过 EngineRegistry::registerEngine() 在运行时注入。
#
# 启用方式：
#   cmake -B build -S . -DBUILD_ENGINE_BUILTIN=ON
#   cmake -B build -S . -DBUILD_ENGINE_ONNXRUNTIME=ON

option(BUILD_ENGINE_BUILTIN "Build Builtin CPU engine (zero dependencies)" ON)
if (BUILD_ENGINE_BUILTIN)
    add_subdirectory(Builtin)
endif()

option(BUILD_ENGINE_ONNXRUNTIME "Build ONNX Runtime engine adapter (requires vcpkg onnxruntime feature)" OFF)
if (BUILD_ENGINE_ONNXRUNTIME)
    add_subdirectory(OnnxRuntime)
endif()
