# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **DCNet 传输层 POCO 化（跨平台）**：`NetTransport_Http` 从 WinHTTP 迁至 POCO
  （vcpkg `poco[netssl]`，HTTP/HTTPS 单一实现覆盖 Windows/Linux/macOS）；
  `connect()` 新增 TCP 就绪探测（拒连/DNS 失败提前到 createEngine 配置期报告）；
  独立 connect/send/receive 超时
- **Windows TLS 走 SChannel**：新增 overlay triplet（`cmake/triplets/x64-windows.cmake`
  设置 `POCO_ENABLE_NETSSL_WIN`）与 overlay port（`cmake/overlay-ports/poco`），
  Windows 上 POCO 用 NetSSL_Win（系统 TLS），避免 OpenSSL 及其 perl/nasm 构建链；
  POSIX 仍用 NetSSL(OpenSSL)；两者 `HTTPSClientSession` API 一致，代码零条件编译
- **MockServer 迁至 POCO `ServerSocket`**（原 WinSock）：DCNet 与 DCEngines
  测试设施跨平台可用，测试目标不再链接 `ws2_32`
- 外部消费方注意：vcpkg manifest 需新增声明 `poco[netssl]`；Windows 下随项目
  wrapper toolchain 自动启用 SChannel 实现（见 `DCNet/DESIGN.md` §9）

## [0.2.0] - 2026-08-24

### Changed

- **DCNet 重构为张量网络传输框架**（原"网络适配器算子集合"，见 `DCNet/DESIGN.md`）
  - 职责收缩：传输（`NetTransport_Http`，WinHTTP）+ 张量 JSON 数据格式
    （数值 base64 / **Data 文本 UTF-8 直传**，新增 `makeTextJsonCodec`）
    + 错误归一化（`NetError`）+ 对接契约（`NetTransport` / `NetCodec` /
    `NetEndpoint` / `NetSync`）；协议级适配器不再内置
  - 默认 engineType 改名：`"DCNet.Http"` → `"DCNet.Tensor"`（破坏性变更）
  - `MockServer` 升为公开测试基础设施（`DCNet/include/DCNet/MockServer.h`，
    DCNet 与 DCEngines 适配器测试共用）

- **构建默认值翻转：核心默认纯净**
  - `BUILD_ENGINE_BUILTIN` / `BUILD_ENGINE_OPENAI` / `BUILD_DCNET` 默认改为 OFF——
    全新配置只构建核心库 + DCIr + 核心测试，引擎适配器与网络框架按需启用
  - 新增 `core-only` CMake 预设（`cmake --preset core-only`）；README 新增
    "仅使用核心（Core-only）"章节（含 git sparse-checkout 只取核心目录指引）
  - 修复 `CMakePresets.json` 编码损坏（中文 description 乱码，重写为规范要求
    的 UTF-8）

### Added

- **`DCEngines/OpenAI`：OpenAI 兼容远端服务引擎适配器（新模块）**
  - `DC::OpenAI::registerOpenAiEngine` / `OpenAiOptions`（engineType `"OpenAI"`），
    与 OnnxRuntime 对称并列（本地模型后端 vs 远端 LLM 服务）
  - chat codec 自 DCNet 迁移：`/v1/chat/completions`，端口
    prompt/system/params（Data，可选）→ response（Data）
  - 基于 DCNet 传输框架：HttpTransport + `NetCodec` 契约 + `NetError` 归一化
  - `OpenAiEngineTest`：chat 端到端（model/messages/stream/参数覆盖断言）
    + 远端 500 归一化（MockHttpServer，真实 HTTP）
  - 构建：`BUILD_ENGINE_OPENAI`（默认 OFF，依赖 `BUILD_DCNET`）
  - 文档：`DCEngines/OpenAI/README.md`

### Removed

- `DCNet` 的 `makeChatCodec` / `"DCNet.HttpChat"` engineType（迁移至
  `DCEngines/OpenAI` 后退役）
- `DCNet/src/NetCodec_Chat.cpp`（迁移后删除）

## [0.1.0] - 2026-08-19

### Added

- **Core Runtime**
  - Node + Port + Connector graph topology supporting multi-branch, fan-in, cyclic, and dynamic routing patterns
  - Data-driven execution model with automatic node triggering on input readiness
  - Three-tier thread pool isolation (Compute / Operator / System)
  - Schema-safe tensor system with runtime type & shape validation via `ValidatorRegistry`
  - Type-erased `TensorSlot` with zero-copy pass-by-reference for same-memory-space nodes
  - NumPy-style chained view indexing (slice, reshape, transpose) without data copy
  - Plugin-based `EngineRegistry` for runtime engine discovery and registration
  - `SignalStore` / `SignalProbe` for cross-node conditional execution control
  - `GraphStore` for named subgraph management
  - `ErrorTracker` for per-task error collection and diagnostics

- **Tensor System (`DC::Tensor`)**
  - Header-only `Tensor.hpp` with `Create<T>()`, item access, and view operations
  - `TensorData` backing store with span-based API and move semantics
  - `TensorMeta` for logical type, element size, and rule-shape constraints
  - Custom exception hierarchy (`TensorException`) with typed error codes

- **Graph Compiler (`DCIr`)**
  - JSON → `InferGraph` compilation via `GraphCompiler`
  - `.dcg` archive format with embedded model payloads (`DcgArchive`)
  - Schema derivation from engine instances with JSON override support
  - Round-trip serialization for `InputZone` / `OutputZone` declarations

- **Engine Adapters (`DCEngines`)**
  - `BuiltinOps`: CPU Add, Identity operators for testing and lightweight pipelines
  - `OnnxRuntime`: Full ONNX Runtime adapter with CPU / CUDA / TensorRT / OpenVINO EP selection
    - FP16 graceful degradation (mapped to Void with warning)
    - Dynamic shape (-1 dim) preservation and execution
    - `DC::Tensor ↔ Ort::Value` bidirectional converter
    - Per-session customization via `OnnxOptions::sessionCustomizer`

- **Build System**
  - CMake 3.17+ with vcpkg manifest mode
  - 8 CMake presets: gcc / clang / msvc × debug / release + ort-cpu / ort-cuda
  - GCC libatomic auto-detection for `<semaphore>` support
  - Zero external dependencies for core library (only C++20 standard library)
  - Optional ONNX Runtime feature flags in `vcpkg.json`

- **Testing**
  - 11 standalone test suites covering all core modules
  - `TestHarness` fixture for multi-threaded graph scenario testing
  - CTest integration with auto-discovery of test sources

- **Documentation**
  - Bilingual README (English + Chinese) with architecture overview
  - 4 Mermaid use-case diagrams (hybrid local+cloud, PC+cloud, LAN heterogeneous, multi-model)
  - Doxygen-style API documentation across all public headers
  - Quick Start guide with build, test, and ONNX Runtime setup instructions

### Known Limitations

- **Platform verification**: Only Windows / MSVC has been locally built and tested. Linux (GCC / Clang) builds are configured via presets but not yet verified in CI.
- **ONNX Runtime adapter**: Experimental; FP16 and some ONNX ML operators map to Void with warnings. Not all ONNX data types are supported.
- **No install target**: The library is designed for `add_subdirectory()` consumption only. `cmake --install` and `find_package(DCinfer)` are not yet supported.
- **Single example**: Only `01_hello_graph` is provided. More complex scenarios (multi-branch, cyclic, cloud offload) are documented but not exemplified.
- **No Python bindings**: C++ only; no language bindings or scripting interface.

[0.2.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.2.0
[0.1.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.1.0
