# DCEngines - 引擎适配器集合
#
# 每个引擎适配器是独立的静态库，按需编译（默认全 OFF：核心默认纯净，
# 用户只想要 DCinfer 核心时无需处理任何引擎开关）。
# 核心库 DCinfer 不反向依赖任何引擎适配器——
# 引擎通过 EngineRegistry::registerEngine() 在运行时注入。
# 本地引擎（Builtin/OnnxRuntime）与远端服务适配器（OpenAI，基于 DCNet
# 张量传输框架）对称并列，均属 EngineDescriptor 家族。
#
# 启用方式：
#   cmake -B build -S . -DBUILD_ENGINE_BUILTIN=ON
#   cmake -B build -S . -DBUILD_ENGINE_ONNXRUNTIME=ON
#   cmake -B build -S . -DBUILD_ENGINE_OPENAI=ON

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
        # 显式失败：静默跳过会让"配置成功"被误认为功能已构建（issue P1-7）
        message(FATAL_ERROR
            "BUILD_ENGINE_OPENAI=ON requires BUILD_DCNET=ON (OpenAI adapter uses DCNet HTTP transport).\n"
            "Re-configure with: -DBUILD_DCNET=ON")
    endif()
endif()
