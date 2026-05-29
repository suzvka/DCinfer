# DCIr — 推理图编译器（JSON → InferGraph）
#
# 依赖：
#   - DCinfer::DCinfer（核心库）
#   - nlohmann_json（通过 FetchContent 获取，header-only）

include(FetchContent)

FetchContent_Declare(
	nlohmann_json
	GIT_REPOSITORY https://github.com/nlohmann/json.git
	GIT_TAG v3.11.3
)
# header-only 库，无需编译
FetchContent_MakeAvailable(nlohmann_json)

add_library(DCIr STATIC)
add_library(DCIr::DCIr ALIAS DCIr)

target_compile_features(DCIr PUBLIC cxx_std_20)

target_include_directories(DCIr
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(DCIr
	PUBLIC
		DCinfer::DCinfer
	PRIVATE
		nlohmann_json::nlohmann_json
)

file(GLOB_RECURSE DCIR_SRC_FILES CONFIGURE_DEPENDS
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
target_sources(DCIr PRIVATE ${DCIR_SRC_FILES})
