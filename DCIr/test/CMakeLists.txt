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
	# 相对定位（外层 add_subdirectory 消费同样成立）；../src 供测试引用模块内私有头
	target_include_directories(${test_name} PRIVATE
			${CMAKE_CURRENT_SOURCE_DIR}/../../DCinfer/test
			${CMAKE_CURRENT_SOURCE_DIR}/../src
		)
		target_link_libraries(${test_name} PRIVATE DCIr::DCIr)
	add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
	set_tests_properties(${test_name} PROPERTIES TIMEOUT 120)
endforeach()
