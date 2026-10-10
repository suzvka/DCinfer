# cmake/triplets/x64-windows.cmake - DCinfer overlay triplet
# 基于官方 x64-windows 追加 POCO_ENABLE_NETSSL_WIN：Windows 用 NetSSL_Win
# (SChannel)，避免 OpenSSL 的 perl/nasm 构建链。

set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)

set(POCO_ENABLE_NETSSL_WIN ON)
