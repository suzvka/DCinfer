# DCEngine_OnnxRuntime 集成测试
#
# 测试在运行时用手工编码的 protobuf 字节流生成 ONNX 模型，无需外部模型文件，
# 也不依赖 onnx/protobuf 库（避免描述符重复注册崩溃，见 OnnxEngineTest.cpp）。
#
# ⚠ 已知问题（vcpkg onnxruntime:x64-windows@1.23.2 Debug）：Debug DLL 内嵌的
# ONNX schema 注册表不完整（加载任何模型即断言），且构建末期会被重新部署覆盖。
# 因此 Windows 多配置 Debug 经 RunOrtTest.bat 包装执行（前置：先构建 Release）；
# 其余配置直接执行目标。

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
