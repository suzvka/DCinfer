# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **任务生命周期 API（issue P0-2/P0-3/P1-6）**：新增 `TaskStatus`（Unknown/Running/
  Succeeded/Failed/TimedOut/Cancelled）与 `TaskResult`；`InferGraph` 新增
  `taskStatus()` / `waitForResult()` / `cancel()`（幂等） / `releaseTask()`。
  **输出在任务终止后保留**（至下次同 ID submit 或 `releaseTask()`），
  `submit → wait → getOutput` 无需注册回调即可取结果；
  终止时自动将声明端口残余数据从节点缓冲护送至 OutputZone（含超时/取消路径的部分结果）
- **taskId 生命周期定义**：活动任务重复提交抛 `GraphException(DuplicateTask)`；
  已终止 ID 可安全复用（自动清理上一轮声明/结果/诊断）；
  修复旧 `_terminatedTasks` 只增不减导致 wait 立即返回、传播被拦截、回调不触发、内存无限增长
- **wait 默认超时语义对齐**：`waitForResult` 超时未终止时返回 `Running`（区分"仍在运行"
  与各类终止态）；原 `bool wait()` 保持兼容；传播/节点执行采用 gate 级终止检查，
  同 ID 复用后旧 lambda 不得污染新任务
- **OpenAI 适配器鉴权与可观测性（issue P0-4/P1-8）**：`OpenAiOptions` 新增
  `authToken` / `tokenProvider` / `headers` / `connectTimeout` / `requestTimeout` /
  `maxRetries`（裸 key 自动补 `Bearer ` 前缀；token 不写入日志与错误信息）；
  非法 params JSON → `InvalidInput`（不再吞掉）；响应非 JSON / 缺
  `choices[0].message.content` → `RemoteMalformed`（不再以空字符串成功）；
  `maxRetries` 实现传输级退避重试（原字段无任何行为）
- **DCNet 错误分类细化**：`NodeStatus` 新增 `RemoteMalformed`（原归 InternalError）；
  新增 `DcCodecInputError` / `DcCodecRemoteError` codec 异常契约；
  `DcNetTransport` 新增 `endpoint()` 访问器；`DcNetAdapterDesc` 新增端点级覆盖项
  （鉴权/附加头/超时/重试）
- **图 API 便捷绑定（issue P2-11）**：`feedBoundInput`（按 bindInput 端口名注入，
  歧义显式报错）与 `submitBound`（以 bindOutput 绑定作为输出声明，无需重复声明）；
  hello_graph 示例同步改用并补 `connect` vs `wire` 语义注释
- **构建体验（issue P0-1/P0-5/P1-7/P1-9/P1-10）**：
  - README 新增「10 分钟上手」章节（配置→构建→运行 hello_graph 及预期输出）；
    CI 逐字执行该命令并断言输出
  - 构建选项迁移至 `DCINFER_BUILD_*` 命名空间（旧 `BUILD_*` 名兼容映射）；
    根 CMakeLists 全局设置加顶层保护（`add_subdirectory()` 引入零污染宿主，
    子工程模式下测试/示例默认关）；MSVC `/utf-8` 下沉至目标级；
    配置结束输出 configure summary
  - `BUILD_ENGINE_OPENAI` 缺 DCNet 时 FATAL_ERROR（原 WARNING 后静默跳过）
  - vcpkg 依赖按 feature 分层（`ir` / `net` / `ort-*`，基础依赖清空）；
    `core-only` 预设零 vcpkg 依赖；新增 `core-ir` 预设；
    CI 新增 core-only / README 上手路径（双平台）/ DCNet+OpenAI mock 作业，
    时间敏感测试以 `ctest --repeat until-fail:3` 加严

### Fixed

- 文档漂移（issue P2-12）：`OpenAiEngine.h` 头注释 WinHTTP → POCO；
  OpenAI README 构建命令补充 `BUILD_DCNET=ON` 并修正"默认 ON"错误说明

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

### Fixed

- **ThreadPool 分组限流空转**：队列非空但分组信号量不可用时，工作线程原会忙等
  自旋（wait 谓词恒真，占核 100%）；改为限时休眠（2ms 兜底轮询覆盖跨池释放，
  同池释放由完成后 `notify_all` 即时唤醒）
- **ExecutionEngine 看门狗自 join 崩溃**：超时路径在看门狗自身线程内 erase 并
  join 自身 `jthread`，抛 `resource_deadlock_would_occur` 并因自 noexcept 析构
  逃逸触发 `std::terminate`；现移交退役列表由引擎析构统一 join
- **`_watchdogs` 数据竞争**：`submit` 注册与 `_terminate` 回收（看门狗线程、
  池 worker、提交方线程）并发访问无同步；新增独立互斥锁，join 一律移出锁外
- 新增看门狗超时回归测试（`InferGraphTest`；旧实现下该测试将使进程崩溃）

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
