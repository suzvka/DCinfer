# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
