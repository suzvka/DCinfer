# Standalone single-file tests (one main per file)

cmake_minimum_required(VERSION 3.12)

if (NOT TARGET DCNet::DCNet)
    message(FATAL_ERROR "DCNet::DCNet target not found. Ensure parent adds DCNet first.")
endif()

include(CTest)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(test_src IN LISTS TEST_SOURCES)
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE DCNet::DCNet)
    if (WIN32 AND test_name STREQUAL "HttpTlsTest")
        target_link_libraries(${test_name} PRIVATE crypt32 advapi32)
    endif()
    # 测试可访问 src/ 内部头（NetBase64.h 等）
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    # MockHttpServer（POCO）依赖经 DCNet::DCNet 传递
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 120)
endforeach()
