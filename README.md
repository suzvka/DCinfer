# DCinfer

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*面向端云协同 AI 的数据驱动推理运行时。*

DCinfer 是一个 C++20 推理管线编排器，目标是让 AI 应用能够在 **本地、云端、PC、局域网设备**间，以**不同推理引擎** 自由放置模型节点，同时把网络、调度、数据传输、引擎生命周期和执行细节隔离在运行时。

---

## 核心功能

### 灵活的图拓扑

传统 DAG 框架将推理管线约束为线性或树形结构——这在多模型、多分支场景下很快会成为瓶颈。DCinfer 采用 **节点 + 端口 + Connector** 的抽象：节点通过类型化端口声明输入输出 Schema，Connector 负责节点间的数据路由。

得益于此模型，你可以轻松构建：
- **多分支管线**：单输出同时驱动多个下游节点并行推理
- **汇聚模式**：多个上游输出合并注入同一节点
- **成环拓扑**：支持反馈回路，用于迭代优化、强化学习或流式场景
- **动态路由**：Connector 支持 `Broadcast`（1→N 广播）和 `Routing`（1→N 条件分发），根据运行时条件决定数据流向

### 原生并发执行

DCinfer 采用**数据驱动执行模型**：节点在所有输入就绪后自动触发，下游节点由上游数据到达事件自动唤醒，无需手动编排执行顺序。独立分支间可乱序并发执行，天然利用多核与异构硬件。

内置三种亲和预设，隔离不同类型的负载：
- **Compute Pool**：专用于模型推理（GPU / NPU 密集型），避免推理任务被 CPU 计算抢占
- **Operator Pool**：承担 CPU 预处理、后处理、特征工程等算子
- **System Pool**：处理 I/O、网络传输和系统维护任务

三层隔离确保重计算不会拖慢系统响应，I/O 延迟也不会阻塞推理吞吐。

### Schema 安全的张量系统

类型和形状错误是 ML 管线中最常见的运行时故障。DCinfer 的张量系统允许在图构造阶段就定义端口 Schema——数据类型和形状在图上声明，执行前即可校验。

核心能力：

- **运行时类型校验**：`TensorSlot` 通过 `ValidatorRegistry` 在数据写入时自动校验类型与形状，将错误拦截在执行前
- **类型擦除传输**：`TensorSlot` 提供泛型通道，在端点保持类型安全的同时实现中间层零耦合
- **类 NumPy 链式视图索引**：切片、重塑、转置均通过视图实现，不拷贝底层数据
- **零拷贝路径**：同内存空间的节点间张量以引用传递，消除不必要的内存搬运

### 插件式引擎注册

DCinfer 将"节点做什么"与"用什么引擎执行"解耦。`EngineRegistry` 提供统一接口，在运行时注册和发现后端引擎——同一张推理图可以逐步适配不同执行后端。本库提供通用注册接口而非具体引擎实现，用户通过 `EngineDescriptor` 注册自有引擎（ONNX Runtime、TensorRT 或其他自定义后端），接入专有或研究模型。

节点通过引擎名称引用后端——替换引擎无需修改管线结构。

### 零依赖核心

DCinfer 核心库为静态库，**零外部依赖**——仅需 C++20 和标准库。DCIr 序列化模块依赖 nlohmann-json、minizip、zlib（通过 vcpkg 管理），核心库本身不引入任何外部依赖。

## 开始使用

### 10 分钟上手（hello_graph 示例）

```bash
cd DCinfer

# 初始化 submodule（首次克隆后必须执行）
git submodule update --init --recursive

# 配置：核心库 + DCIr + Builtin 引擎 + 示例（一条命令，无需额外依赖）
cmake -B build -S . -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=cmake/vcpkg-toolchain.cmake \
  -DBUILD_ENGINE_BUILTIN=ON

# 构建
cmake --build build

# 运行示例（Ninja 单配置输出在 build/bin/；MSVC 多配置为 build/bin/Release/）
./build/bin/hello_graph
```

预期输出：

```text
3.0 + 4.0 = 7
```

示例代码（[examples/01_hello_graph](examples/01_hello_graph/main.cpp)）展示标准任务生命周期：

```cpp
graph.bindInput("adder", "a");            // 标记图级输入端口
graph.bindOutput("pass", "y");            // 标记图级输出端口
graph.feedBoundInput("task1", "a", ...);  // 按绑定名注入（无需重复节点名）
graph.submitBound("task1");               // 以 bindOutput 绑定作为输出声明
if (graph.waitForResult("task1").status == DC::TaskStatus::Succeeded) {
    auto result = graph.getOutputTensor("task1", "pass", "y");  // wait 后取结果
}
```

输出在任务终止后仍有效（无需注册回调即可读取）；复用已终止的 taskId 合法；
活动任务重复提交会抛出明确错误；支持 `cancel()` / `taskStatus()` / `releaseTask()`。

### 默认构建内容

不带任何 `-D` 开关的默认配置只构建核心库（`DCinfer`）+ DCIr + 核心测试；
引擎适配器（Builtin / OnnxRuntime / OpenAI）与网络框架 DCNet **全部默认 OFF**，按需启用。
启用/停用情况在配置结束时以 **configure summary** 汇总输出。

### 仅使用核心（Core-only）

DCinfer 核心库零外部依赖，不反向依赖任何引擎适配器。**不启用任何模块时，
不会安装任何 vcpkg 依赖**（模块依赖已按 feature 分层）：

```bash
# 方式一：预设（推荐）——只构建核心库，零 vcpkg 依赖（无 DCIr/引擎/DCNet/测试/示例）
cmake --preset core-only
cmake --build build/core-only --config Release

# 方式二：需要 JSON / .dcg 序列化 —— core-ir 预设（仅安装 nlohmann-json/minizip/zlib）
cmake --preset core-ir
cmake --build build/core-ir --config Release

# 方式三：手动开关（等价于 core-only）
cmake -B build -S . \
  -DDCINFER_BUILD_IR=OFF -DDCINFER_BUILD_ENGINES=OFF -DDCINFER_BUILD_DCNET=OFF \
  -DDCINFER_BUILD_TESTS=OFF -DDCINFER_BUILD_EXAMPLES=OFF
```

模块依赖分层（`vcpkg.json` feature）：`ir` → DCIr（nlohmann-json/minizip/zlib）、
`net` → DCNet（nlohmann-json + poco[netssl]）、`ort-*` → ONNX Runtime 适配器
（由 `DCINFER_ORT_EP` 自动注入）。构建选项已迁移至 `DCINFER_BUILD_*` 命名空间
（旧 `BUILD_*` 名仍被识别，避免宿主工程经 `add_subdirectory()` 引入时冲突）。

仅需源码的下载层面，可用 git sparse-checkout 只取核心目录：

```bash
git sparse-checkout init --cone
git sparse-checkout set DCinfer DCIr cmake vcpkg.json CMakeLists.txt
```

核心之上注册自有引擎：实现 `EngineDescriptor` 钩子 → `EngineRegistry::registerEngine()`
（详见 `DCinfer/include/Graph/EngineRegistry.h`），图级语义与内置引擎完全一致。

### 运行测试

```bash
ctest --test-dir build -C Release
```

---

## License

This project is licensed under the [MIT License](LICENSE).
