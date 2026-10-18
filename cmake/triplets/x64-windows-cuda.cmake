# cmake/triplets/x64-windows-cuda.cmake - CUDA EP 专用 overlay triplet
# CUDA 13.4 官方支持 VS2026（host_config 接受 _MSC_VER 1920-1959），无需固定
# 工具集（12.8 时代曾固定 14.44：12.8 拒绝 VS2026 的 14.51，而 14.29 太老，
# 会让 nvcc 静默丢弃 ORT 所需的 -std=c++20）。
# 另：vcpkg 端口环境默认仅自动透传 CUDA_PATH，CUDNN_ROOT_DIR 必须显式声明，
# 否则 FindCUDNN 会命中系统残留的旧 cuDNN 而非本树使用的 9.27。
# VCPKG_BUILD_TYPE release：本树只出 Release（环境变量方式实测不生效，须写进 triplet），
# ORT debug 构建会让编译时间翻倍且无收益。
# 注意：triplet 文件内容参与 vcpkg ABI hash，与官方 x64-windows 不可混用
# （切换 = 已装包全量重编一次）。

set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)

set(VCPKG_ENV_PASSTHROUGH "CUDNN_ROOT_DIR")
set(VCPKG_BUILD_TYPE release)
