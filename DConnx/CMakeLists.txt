add_library(DConnx)
add_library(DConnx::DConnx ALIAS DConnx)

target_compile_features(DConnx PUBLIC cxx_std_20)

target_include_directories(DConnx
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/API
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_sources(DConnx
    PRIVATE
        src/Infer.cpp
        src/InferBase.cpp
        src/Info.cpp
        src/TensorMods.cpp
        src/TensorSlot.cpp
)

# ORT engine is intentionally not part of the core build.
# ORT-related sources/headers remain in the repository for local debugging only.
