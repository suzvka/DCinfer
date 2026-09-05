# DCNet tests - standalone single-file tests (one main per file)

cmake_minimum_required(VERSION 3.12)

# This CMakeLists is included from DCNet/CMakeLists.txt; ensure target exists
if (NOT TARGET DCNet::DCNet)
    message(FATAL_ERROR "DCNet::DCNet target not found. Ensure parent adds DCNet first.")
endif()

include(CTest)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(test_src IN LISTS TEST_SOURCES)
    get_filename_component(test_name ${test_src} NAME_WE)
    # Each test source is expected to be a standalone test with its own main.
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE DCNet::DCNet)
    # 测试可访问内部头（NetBase64.h 等）
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    # MockHttpServer（POCO）依赖经 DCNet::DCNet 传递
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
endforeach()
