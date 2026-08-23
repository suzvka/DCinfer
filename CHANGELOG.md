# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **DCNet（网络适配器算子集合，新增顶层模块，`BUILD_DCNET` 默认 ON）**
  - 对接契约：`DcNetTransport`（传输抽象）/ `DcNetCodec`（协议映射）/
    `NetEndpoint`（端点配置）/ `NetError`（归一化中间结构）——契约内置、协议外置
  - 错误归一化：网络错误 / HTTP 状态 / 远端错误报文 → 本地标准报错
    （`NodeStatus` + 消息前缀 + `retryable`），纯函数映射表可单测
  - 本地形状规则：网络算子端口 Schema 本地声明（复用 `NodePort`，含 -1 动态维与锚定）
  - 内置适配器 `DCNet.Http`：WinHTTP transport（零新增依赖）+ 张量 JSON codec
    （`{"dtype","shape","data(base64)"}`）+ OpenAI 兼容 chat codec
  - `DcNetSync` 核心 async→sync 桥（ADR-6：默认不派生线程，异步 SDK 由 transport 内部承载）
  - 注册入口：`registerDcNetAdapter`（通用）/ `registerDcNetHttp`（便捷）
  - 测试：`NetErrorTest`（映射表）/ `NetAdapterTest`（契约 + FakeTransport）/
    `HttpTransportTest`（MockServer 真实 HTTP：传输往返、404/500/拒连归一化、张量与 chat 端到端）
  - 设计文档：`DCNet/DESIGN.md`（ADR-1~6 决策记录 + 完整契约规格）

- **文档**
  - README 新增 "网络适配器算子（DCNet）" 章节（定位、契约哲学、最小用法）
  - DCinfer-test 外部消费验证：`net_smoke`（外部开发者自定义 transport/codec）与
    `net_mnist`（网络化 MNIST 端到端，预测 7）

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

[0.1.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.1.0
