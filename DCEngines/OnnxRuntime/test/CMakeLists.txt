# DCEngine_OnnxRuntime 集成测试
#
# 测试在运行时用手工编码的 protobuf 字节流生成小型 ONNX 模型，
# 无需外部模型文件，也不依赖 onnx/protobuf 库——
# 避免 onnxruntime.dll 内嵌的 ONNX 描述符与 onnx 静态库重复注册导致崩溃。
#
# ⚠ 已知问题（vcpkg onnxruntime:x64-windows@1.23.2 Debug 配置）：
# Debug 版 onnxruntime.dll 内嵌的 ONNX schema 静态注册表不完整
# （断言 "4 schema were exposed ... 741 were expected"），任何模型都无法加载，
# 且 vcpkg 会在每次构建末期重新部署覆盖手工替换的 DLL。
# 因此 Windows 上测试通过 RunOrtTest.bat 包装执行：运行前先把 Release 版
# ORT DLL（仅 C ABI 交互，兼容）复制到输出目录。前置条件：先构建过 Release：
#   cmake --build build --config Release

cmake_minimum_required(VERSION 3.12)

include(CTest)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(test_src IN LISTS TEST_SOURCES)
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    # TestHarness.h 位于核心库测试目录
    target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/DCinfer/test)
    target_link_libraries(${test_name} PRIVATE DCEngine::OnnxRuntime)

    if (WIN32)
        # 包装器：运行前部署 Release 版 ORT DLL（Debug workaround）
        add_test(NAME ${test_name}
                 COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/RunOrtTest.bat
                         ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release
                         $<TARGET_FILE_DIR:${test_name}>
                         $<TARGET_FILE:${test_name}>)
    else()
        add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    endif()
endforeach()
