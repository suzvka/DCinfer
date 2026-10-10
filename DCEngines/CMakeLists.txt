# DCEngines - 引擎适配器集合
#
# 每个适配器是独立静态库，按需编译（默认全 OFF）。核心库不反向依赖引擎——
# 引擎通过 EngineRegistry::registerEngine() 运行时注入。本地引擎（Builtin/
# OnnxRuntime）与远端适配器（OpenAI，基于 DCNet）对称并列（EngineDescriptor 家族）。
#
# 启用：-DBUILD_ENGINE_BUILTIN=ON / -DBUILD_ENGINE_ONNXRUNTIME=ON / -DBUILD_ENGINE_OPENAI=ON

option(BUILD_ENGINE_BUILTIN "Build Builtin CPU engine (zero dependencies)" OFF)
if (BUILD_ENGINE_BUILTIN)
    add_subdirectory(Builtin)
endif()

option(BUILD_ENGINE_ONNXRUNTIME "Build ONNX Runtime engine adapter (requires vcpkg onnxruntime feature)" OFF)
if (BUILD_ENGINE_ONNXRUNTIME)
    add_subdirectory(OnnxRuntime)
endif()

option(BUILD_ENGINE_OPENAI "Build OpenAI-compatible remote engine adapter (requires DCNet)" OFF)
if (BUILD_ENGINE_OPENAI)
    if (TARGET DCNet::DCNet)
        add_subdirectory(OpenAI)
    else()
        # 显式失败：静默跳过会让"配置成功"被误认为功能已构建
        message(FATAL_ERROR
            "BUILD_ENGINE_OPENAI=ON requires DCINFER_BUILD_DCNET=ON (OpenAI adapter uses DCNet HTTP transport).\n"
            "Re-configure with: -DDCINFER_BUILD_DCNET=ON")
    endif()
endif()
