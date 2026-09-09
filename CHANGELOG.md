# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **超时看门狗 → 引擎级共享 TimerService**：`ExecutionEngine` 不再为每条带超时的
  submit 创建看门狗线程（原 100ms 轮询 `jthread` + per-task 注册/回收），改为单定时器
  线程 + deadline 最小堆：每任务仅登记一个 deadline 条目，终止路径 O(1) 作废、零 join。
  超时触发从 ~100ms 轮询粒度变为精确 deadline 唤醒；引擎析构时定时器线程先于
  线程池停止（原看门狗晚于池析构，存在池关闭期间触发超时的窗口）；
  每 task 成本从"一线程 + 周期轮询"降为"一个堆条目"。
  同 ID 复用后旧条目到点时经活动门控身份校验失配退出，不会误杀新任务
  （该校验取代原 per-task join 带来的提交唯一性保证）。
  公开 API（`submit` / `wait` / `cancel` / `waitForResult` 等）签名与语义不变
- **exportNode 子图驱动简化**：RunFn 内不再经引擎级 `setTaskCompleteCallback` +
  手动条件变量捕获输出，改为 `submit → wait → 逐绑定 takeOutput`——
  `_terminate` 抢救声明输出（步骤⑥）先于唤醒等待者（步骤⑦），wait 返回后
  声明输出必已入 OutputZone，回调捕获机制随之删除；错误判定由全局
  `hasErrors()/clearErrors()` 收敛为 task 级 `taskErrors(tid)`，消除跨 task
  诊断污染；顺带删除未使用的 fedCount。行为等价（超时路径的部分输出反而更完整）
- **执行引擎传播链去重**：`_submitNodeRun` / `_propagateFrom` 复用已持有的
  task 执行态句柄，消除每节点/每边传播中重复的 `findTaskState` + `find`
  加锁查找（每节点执行省 1 次、每边传播省 2 次锁获取，行为不变）
- **`InferGraph::_ensureSubmittable` 注释澄清**：补充"必须先于
  clearTask/declare 执行"的因果说明（该守卫保护在飞任务状态不被重复提交
  破坏，引擎内校验发生在 facade 状态变更之后，二者非冗余）与已知 TOCTOU
  窗口说明；方法本体无变化

### Removed

- `_retiredWatchdogs` 看门狗自 join 补丁与 `_watchdogs` 线程表（原超时路径在看门狗
  自身线程内 erase + join 自身 `jthread` 会抛 `resource_deadlock_would_occur`，
  需移交退役列表延迟回收；共享定时器无线程可回收，补丁随之移除）。
  纯内部实现，无 API 变化
- **`Connector.Routing`（轮询连接器）全链路移除**：其轮询计数器经注册表按值
  拷贝共享，进程内所有 Routing 实例共享同一全局计数器（跨实例隐式耦合 +
  并发下分发顺序非确定），故将轮询语义整体迁移至应用层——以自定义节点实现
  （实例局部选择逻辑，引擎 partial-output 传播语义天然支持）。
  连带变更：注册名 `"Connector.Routing"` 消失（`Connector::routingSchema/`
  `routingRunFn` 一并删除）；DCIr 序列化不再产出 `mode:"routing"`，反序列化
  遇 `routing` 模式显式抛 `GraphException(Other)`（不再静默降级为普通连线）；
  连接器内置行为收敛为 Broadcast 单一语义（isConnector 扩展点框架保留）。
  迁移：将 Routing 节点替换为 N 输出的自定义节点，RunFn 每次仅产出一个输出口

### Fixed

- **lowering 悬空边（组合反例）**：`buildRuntimeView` 边融合原只向后回跳一层，
  wire→wire 链（`connectRaw` 允许连接器与连接器相连，合法构图）会产生指向已擦除
  节点的悬空边（如 `a→w1→w2→b` 被改写为 `a→w2`），下游不可达、数据静默滞留、
  任务只能靠看门狗终止。现改为沿唯一出边链追踪到首个保留节点（visited 集合防
  纯 wire 环，环上融合边丢弃——数据在源语义中同样无法抵达业务节点），并收尾
  新增不变量校验：运行边端点必须存在于运行节点集合，违反即抛 `GraphException`。
  新增链式 / 链终止于保留连接器 / 纯 wire 环三个组合测试
- **wait 结果就绪窗口**：`wait()` 谓词原只绑定终态发布（`_terminate` 首步），
  早于声明输出抢救（步骤⑥）；窗口内返回的调用方 `takeOutput` 可能抛
  `OutputNotProduced` / `TaskNotFound`（成功路径的声明输出本就完全依赖步骤⑥
  搬运——打卡满足即 return，绑定搬运被跳过）。`_taskStates` 值改为
  `{status, resultsReady}`，`resultsReady` 在步骤⑥完成后、notify 前置位，
  wait 谓词改为 `terminated && resultsReady`（`status()` 语义不变，仍在终态
  写入即返回）。同步文档化完成回调约束：回调先于结果就绪发布触发，
  回调内不得对同一 taskId 调用 `wait()`。新增裸 InferGraph 循环压测与
  多声明可读性测试

## [0.3.0] - 2026-09-08

### Added

- **图级公共 I/O 别名**：`bindInput(alias, nodeName, portName)` /
  `bindOutput(alias, nodeName, portName)` 三参重载为图级绑定赋予公共别名
  （别名须唯一，重复抛 `GraphException(DuplicateBinding)`；输入/输出别名独立命名空间）；
  `feedBoundInput` 优先按别名解析，跨节点同名端口可用唯一别名消歧；
  `takeOutput` / `takeOutputTensor` / `hasOutput` 新增 2 参重载，
  按公共别名或唯一绑定端口名定位，调用方无需感知内部节点/端口名
- **README 新增「环境要求」矩阵**：CMake/C++ 标准/各编译器下限与 CI 验证平台、
  可选依赖与模块的对应关系
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
  hello_graph 示例同步改用便捷绑定 API
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

- **Build → Freeze → Execute 生命周期（freeze/lowering 边界）**：
  - 新增 `GraphBuilder`（构建期唯一可变面）与 `compile()`：产出不可变
    `CompiledGraph` 快照（冻结拓扑 + `GraphSignature` 图级签名 + lowering 后
    运行时视图）；`InferGraph` 保留为执行 Facade，惰性冻结（首次
    `submit`/`feedInput` 自动编译），新增显式 `freeze()`（幂等）
  - 新增 `GraphSignature`：图级输入/输出绑定快照；执行期别名/绑定解析
    无锁（原 `OutputZone` 绑定面移除，仅承载声明/累加/artifact 纯任务态）
  - 新增 lowering pass（`GraphLowering`）：`Broadcast(1)` 导线连接器从
    运行时视图擦除（入边改写为直连；源图不变——DCIr 序列化/exportNode/
    内省查询仍反映源图）；防护：绑定图级输入/输出或多出边的 wire 不擦除；
    `CompiledGraph` 新增 `runtimeNodeCount()/runtimeEdgeCount()/loweringStats()`；
    新增 `examples/02_lowering_benchmark`（100 节点链：199→100 调度顶点）
- **领域结构化诊断**：新增 `DC::Diagnostic`（domain/code/message，见
  `Node/Diagnostic.h`）；`NodeResult`/`TaskError` 携带可选诊断；
  `ErrorTracker::recordError` 新增带诊断重载；`RunContext::failure` 新增
  三参重载

### Fixed

- 文档漂移（issue P2-12）：`OpenAiEngine.h` 头注释 WinHTTP → POCO；
  OpenAI README 构建命令补充 `BUILD_DCNET=ON` 并修正"默认 ON"错误说明

### Changed

- **破坏性重命名：`getOutput` → `takeOutput`、`getOutputTensor` → `takeOutputTensor`**
  （InferGraph 与 Node/TaskBuffer 同步更名）：输出取用一直是消费式语义
  （取出即从 OutputZone/节点缓冲清除，不可重复读取），旧名隐匿了该行为；
  非破坏式预览仍可用 Node::peekOutput
- **破坏性重命名：`wire()` → `connect()`、`connect()` → `connectRaw()`**
  （GraphStore 与 InferGraph 门面同步更名）：自动插入广播连接器的节点连线
  回收最直观的 `connect()` 命名，对齐“连接两个节点”的用户心智（可用性评审
  第 2 点）；低层建边原语更名为 `connectRaw()`（语义不变，仍要求两端至少
  一端为连接器，直连报错提示同步改为 “Use connect() instead.”）；
  `connectAll()` 语义不变（低层批量直连，不插入连接器）
- **`wait()` / `waitForResult()` 默认改为无限等待直至终止**：原默认 5s 隐式超时
  与方法名语义相悖；显式超时改经重载传入，`timeout <= 0` 视为无限等待
  （与 `submit` 的执行超时 0=不限时约定一致）；未知 taskId（从未提交/已释放）
  在无限等待模式下立即返回，防误拼写挂死；超时只放弃等待、不取消任务
  （取消须显式 `cancel()`）
- **README quickstart 与 CI 逐字对齐**：`readme-hello-graph` CI 作业不再注入
  vcpkg toolchain、不再初始化 submodule——逐字验证 README「10 分钟上手」的
  零依赖命令（新增"未安装 vcpkg 依赖"断言）；README 同步将 submodule 初始化
  与 toolchain 参数移至扩展模块段落，消除文档与 CI 的信任偏差
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

- **执行期拓扑不可变（破坏性）**：首次 `submit`/`feedInput` 触发惰性冻结后，
  `addNode/connect/connectRaw/connectAll/bindInput/bindOutput/declareSubgraph`
  抛 `GraphException(Frozen)`——"任务执行期间拓扑能否改变"的答案恒为否；
  拓扑演进路径：重建 `GraphBuilder` 重新 compile 产生新快照，旧图任务排空后
  替换（绑定须在首次运行期调用前完成，InferGraphTest 别名用例已同步调整）
- **破坏性：`NodeStatus` 移除 `RemoteMalformed`**：核心枚举保持最小通用词表，
  远端结构异常改报 `ExecutionFailed` + `Diagnostic{domain="dcnet",
  code=RemoteMalformed}`（分类法保留在 DCNet `NetErrorCategory`，核心只透传）；
  `NetError` 新增 `diagnostic` 字段，`finalize` 统一填充；DCNet/OpenAI
  测试断言同步更新
- **破坏性：`registerGroupLimit(affinity, tag, limit)` → `registerGroupLimit(tag, limit)`**：
  组信号量由三个线程池共享、注册一次全局生效，公开 API 不再暴露模型并不
  区分的 affinity 维度（`DCNet/DESIGN.md` 引用同步更新）
- **`NodeFactoryParams::engineConfig` 语义单一化（一字段一语义）**：仅承载
  `createNode(engineType, name, engineConfig)` 透传的用户配置指针；
  modelPath 路径不再兼容性塞入引擎实例裸指针（在树工厂均只读
  `engineInstance`，零消费者），引擎实例一律经共享句柄传递
- **TTL 语义（lowering 配套）**：`maxHops` 只统计运行时顶点，被擦除的
  1:1 导线不再消耗 hop——成环图 TTL 触发时机后移（方向安全：更不易误杀
  深图）；直连后传播握手在上游节点的完成线程执行（原 System 池；N=1 导线
  本就是零拷贝 move 直通，无数据搬运；Broadcast(N>1)/Routing 不受影响）

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

[0.3.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.3.0
[0.2.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.2.0
[0.1.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.1.0
