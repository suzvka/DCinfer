# DCEngine_Builtin 集成测试
#
# 单文件独立测试（每文件一个 main），链接 DCEngine::Builtin（传递依赖 DCinfer）。
# 覆盖内置张量算子 Shape/Gather/Unsqueeze/Slice/Sub/Pad 的 ONNX-13 参考值对拍。

cmake_minimum_required(VERSION 3.12)

include(CTest)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(test_src IN LISTS TEST_SOURCES)
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE DCEngine::Builtin)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 120)
endforeach()
