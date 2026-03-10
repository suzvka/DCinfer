# Auto-discover single-file tests in this directory and register them
# Expects tests to be standalone .cpp files (one main per file)

cmake_minimum_required(VERSION 3.12)

# This CMakeLists is included from the root; ensure DConnx target exists
if (NOT TARGET DConnx::DConnx)
    message(FATAL_ERROR "DConnx::DConnx target not found. Ensure parent adds DConnx first.")
endif()

# Enable CTest support
include(CTest)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(test_src IN LISTS TEST_SOURCES)
    get_filename_component(test_name ${test_src} NAME_WE)
    # Each test source is expected to be a standalone test with its own main.
    # Do not force-include a common test harness to avoid multiple main definitions.
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE DConnx::DConnx)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
