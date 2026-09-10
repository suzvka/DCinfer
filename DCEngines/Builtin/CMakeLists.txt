# DCEngine_Builtin - 零依赖 CPU 算子引擎
#
# 仅依赖 DCinfer 核心库，无需任何外部推理框架。
# 提供 Add / Mul / Identity 等基础算子，用于教学、测试和快速验证图拓扑。

add_library(DCEngine_Builtin STATIC)
add_library(DCEngine::Builtin ALIAS DCEngine_Builtin)

target_compile_features(DCEngine_Builtin PUBLIC cxx_std_20)

target_link_libraries(DCEngine_Builtin PUBLIC DCinfer::DCinfer)

target_include_directories(DCEngine_Builtin
    PUBLIC
        # BUILD_INTERFACE：源码内消费（DCinfer 扁平头路径经 DCinfer::DCinfer 传递）
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        # INSTALL_INTERFACE：安装树（include/DCEngine/ 结构保留，消费方 include "DCEngine/BuiltinOps.h"）
        $<INSTALL_INTERFACE:include>
)

file(GLOB_RECURSE DCENGINE_BUILTIN_SRC CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
target_sources(DCEngine_Builtin PRIVATE ${DCENGINE_BUILTIN_SRC})

# ── 安装与 find_package 支持 ──
# 包名 DCEngine：目标导出为 DCEngine::Builtin，与源码内 alias 一致
# （target 名 DCEngine_Builtin 需经 EXPORT_NAME 改名）。OnnxRuntime / OpenAI
# 适配器依赖 vcpkg 重型依赖，暂不纳入安装导出，仍以 add_subdirectory 消费。
include(GNUInstallDirs)

# 包版本：与 DCinfer 模块同一约定（顶层 project(DCinfer VERSION) 权威 + 兑底）
set(DCINFER_PKG_VERSION "${PROJECT_VERSION}")
if(NOT DCINFER_PKG_VERSION)
    set(DCINFER_PKG_VERSION "0.4.0")
endif()

set_target_properties(DCEngine_Builtin PROPERTIES EXPORT_NAME Builtin)

install(TARGETS DCEngine_Builtin EXPORT DCEngineTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 头文件保留 DCEngine/ 目录结构（源码以 "DCEngine/BuiltinOps.h" 前缀风格 include）
install(DIRECTORY include/DCEngine DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(DCENGINE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/DCEngine")

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/DCEngineConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/DCEngineConfig.cmake"
    INSTALL_DESTINATION ${DCENGINE_INSTALL_CMAKEDIR}
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/DCEngineConfigVersion.cmake"
    VERSION ${DCINFER_PKG_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/DCEngineConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/DCEngineConfigVersion.cmake"
    DESTINATION ${DCENGINE_INSTALL_CMAKEDIR}
)
install(EXPORT DCEngineTargets
    FILE DCEngineTargets.cmake
    NAMESPACE DCEngine::
    DESTINATION ${DCENGINE_INSTALL_CMAKEDIR}
)
