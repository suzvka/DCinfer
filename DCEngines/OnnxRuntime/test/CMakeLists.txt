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

    # CUDA EP：provider DLL 由 onnxruntime.dll 运行时 LoadLibrary 加载（不在导入表
    # 内，applocal 不会复制），需手动部署到可执行文件目录；运行期再依赖 cuDNN /
    # CUDA runtime，故把两者的 bin 目录前置进测试 PATH（配置环境存在时才注入）。
    if (DCINFER_ORT_EP STREQUAL "CUDA")
        add_custom_command(TARGET ${test_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE_DIR:onnxruntime::onnxruntime>/onnxruntime_providers_shared.dll"
                "$<TARGET_FILE_DIR:onnxruntime::onnxruntime>/onnxruntime_providers_cuda.dll"
                "$<TARGET_FILE_DIR:${test_name}>"
            VERBATIM)

        if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.22)
            set(_ep_env "")
            if (DEFINED ENV{CUDNN_ROOT_DIR})
                string(APPEND _ep_env "PATH=path_list_prepend:$ENV{CUDNN_ROOT_DIR}/bin;")
                file(GLOB _cudnn_bin_dirs LIST_DIRECTORIES true "$ENV{CUDNN_ROOT_DIR}/bin/*")
                foreach(_dir IN LISTS _cudnn_bin_dirs)
                    if (IS_DIRECTORY "${_dir}")
                        string(APPEND _ep_env "PATH=path_list_prepend:${_dir};")
                    endif()
                endforeach()
            endif()
            if (DEFINED ENV{CUDA_PATH})
                # CUDA 13 起 Windows DLL 位于 bin\x64，两个目录都前置。
                string(APPEND _ep_env "PATH=path_list_prepend:$ENV{CUDA_PATH}/bin/x64;")
                string(APPEND _ep_env "PATH=path_list_prepend:$ENV{CUDA_PATH}/bin;")
            endif()
            if (_ep_env)
                set_tests_properties(${test_name} PROPERTIES
                    ENVIRONMENT_MODIFICATION "${_ep_env}")
            endif()
        endif()
    endif()
endforeach()
