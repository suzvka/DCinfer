# DCIr 测试
cmake_minimum_required(VERSION 3.12)

if (NOT TARGET DCIr::DCIr)
	message(STATUS "DCIr::DCIr target not found; skipping DCIr tests. Ensure parent adds DCIr first to enable these tests.")
	return()
endif()

include(CTest)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(test_src IN LISTS TEST_SOURCES)
	get_filename_component(test_name ${test_src} NAME_WE)
	add_executable(${test_name} ${test_src})
	target_include_directories(${test_name} PRIVATE
			${CMAKE_SOURCE_DIR}/DCinfer/test
		)
		target_link_libraries(${test_name} PRIVATE DCIr::DCIr)
	add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
endforeach()
