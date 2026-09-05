# DCEngine_OpenAI 集成测试（MockHttpServer 假远端，真实 HTTP 传输）

cmake_minimum_required(VERSION 3.12)

# This CMakeLists is included from DCEngines/OpenAI/CMakeLists.txt; ensure target exists
if (NOT TARGET DCEngine::OpenAI)
    message(FATAL_ERROR "DCEngine::OpenAI target not found. Ensure parent adds OpenAI first.")
endif()

include(CTest)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(test_src IN LISTS TEST_SOURCES)
    get_filename_component(test_name ${test_src} NAME_WE)
    # Each test source is expected to be a standalone test with its own main.
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE DCEngine::OpenAI)
    # MockHttpServer（POCO）依赖经 DCNet::DCNet → DCEngine::OpenAI 传递
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
endforeach()
