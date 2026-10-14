# cmake/triplets/x64-windows-dcnet.cmake - DCNet 专用 overlay triplet
# 在官方 x64-windows 基础上追加 POCO_ENABLE_NETSSL_WIN：poco 用 NetSSL_Win
# (SChannel)，避免 OpenSSL 的 perl/nasm 构建链。配套 poco 定制 port 见
# cmake/overlay-ports-dcnet/（wrapper toolchain 仅在 DCINFER_BUILD_DCNET=ON
# 且 Windows 时注入，见 cmake/vcpkg-toolchain.cmake）。
# 注意：triplet 文件内容参与 vcpkg ABI hash，与官方 x64-windows 不可混用
# （切换 = 已装包全量重编一次，见 README「Windows TLS 路线」矩阵）。

set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)

set(POCO_ENABLE_NETSSL_WIN ON)
