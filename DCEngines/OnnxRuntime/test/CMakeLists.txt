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
# 因此仅 Windows 多配置 Debug 测试通过 RunOrtTest.bat 包装执行；
# Release 与单配置预设直接执行目标。Debug 的前置条件是先构建 Release：
#   cmake --build build --config Release

cmake_minimum_required(VERSION 3.12)

include(CTest)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(test_src IN LISTS TEST_SOURCES)
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    # TestHarness.h 位于核心库测试目录
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../DCinfer/test)
    target_link_libraries(${test_name} PRIVATE DCEngine::OnnxRuntime)

    if (WIN32 AND CMAKE_CONFIGURATION_TYPES)
        # Only multi-config Debug needs the historical Release DLL workaround.
        # Release and single-config presets execute their actual target directly.
        add_test(NAME ${test_name}
                 COMMAND $<IF:$<CONFIG:Debug>,${CMAKE_CURRENT_SOURCE_DIR}/RunOrtTest.bat,$<TARGET_FILE:${test_name}>>
                         $<$<CONFIG:Debug>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release>
                         $<$<CONFIG:Debug>:$<TARGET_FILE_DIR:${test_name}>>
                         $<$<CONFIG:Debug>:$<TARGET_FILE:${test_name}>>
                 COMMAND_EXPAND_LISTS)
    else()
        add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    endif()
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 120)
endforeach()
