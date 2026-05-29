cmake_minimum_required(VERSION 3.17)

project(DCinfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if (MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /manifest:no")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /manifest:no")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /manifest:no")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
endif ()

add_subdirectory(DCinfer)

# ── DCIr：推理图编译器（JSON → InferGraph）──
option(BUILD_IR "Build DCIr graph compiler" ON)
if (BUILD_IR)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/DCIr/CMakeLists.txt)
        add_subdirectory(DCIr)
    endif()
endif()

# ── 测试 ──
option(BUILD_TESTS "Build DCinfer tests" ON)
if (BUILD_TESTS)
    enable_testing()
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/DCinfer/test/CMakeLists.txt)
        add_subdirectory(DCinfer/test)
    endif()
    if (BUILD_IR AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/DCIr/test/CMakeLists.txt)
        add_subdirectory(DCIr/test)
    endif()
endif()
