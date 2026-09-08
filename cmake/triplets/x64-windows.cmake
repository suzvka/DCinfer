# cmake/triplets/x64-windows.cmake - DCinfer overlay triplet
#
# 基于官方 x64-windows，追加 POCO_ENABLE_NETSSL_WIN：
# Windows 上 POCO 使用 NetSSL_Win（SChannel，系统 TLS），避免 OpenSSL
# 及其 perl/nasm 构建链；POSIX 平台沿用官方 triplet + NetSSL(OpenSSL)。
# 经 cmake/vcpkg-toolchain.cmake 的 VCPKG_OVERLAY_TRIPLETS 生效。

set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)

set(POCO_ENABLE_NETSSL_WIN ON)
