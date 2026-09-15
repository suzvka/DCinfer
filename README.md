# DCinfer

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*可嵌入的端云协同数据驱动推理运行时*

DCinfer 是一个 C++20 推理管线编排器，目标是让 AI 应用能够在 **本地、云端、PC、局域网设备**间，以**不同推理引擎** 自由放置模型节点，同时把网络、调度、数据传输、引擎生命周期和执行细节隔离在运行时。

---

## 核心功能

### 灵活的图拓扑

传统 DAG 框架将推理管线约束为线性或树形结构——这在多模型、多分支场景下很快会成为瓶颈。DCinfer 采用电路图语义的节点化、端口化、连接器化设计。得益于此模型，你可以轻松构建：
- **多分支管线**：单输出同时驱动多个下游节点并行推理
- **汇聚模式**：多个上游输出合并注入同一节点
- **成环拓扑**：支持反馈回路，用于迭代优化、强化学习或流式场景

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

## 环境要求

| 组件 | 要求 |
|---|---|
| CMake | >= 3.17（裸构建最低 3.17；CMakePresets version 3 需 ≥ 3.21） |
| C++ 标准 | C++20 |
| GCC | >= 11（Linux；CI 在 ubuntu-24.04 上验证） |
| Clang | >= 14（CI 在 ubuntu-24.04 上验证） |
| MSVC | VS 2026（Windows；CI 在 windows-latest 上验证） |
| Ninja | 可选（README 示例使用 `-G Ninja`） |
| vcpkg | 仅 DCIr / DCNet / OnnxRuntime / OpenAI 模块需要（仓库以 submodule 提供） |

平台支持：Linux、Windows

## 开始使用

### 10 分钟上手（hello_graph 示例）

```bash
cd DCinfer

# 配置：核心库 + Builtin 引擎 + 示例
# （核心与 Builtin 均零外部依赖：无需 vcpkg，也无需初始化 submodule）
cmake -B build -S . -G Ninja -DBUILD_ENGINE_BUILTIN=ON

# 构建
cmake --build build

# 运行示例（Ninja 单配置输出在 build/bin/；MSVC 多配置为 build/bin/Release/）
./build/bin/hello_graph
```

预期输出：

```text
3.0 + 4.0 = 7
```

后续需要 JSON/.dcg 序列化（DCIr）、DCNet 或 OnnxRuntime/OpenAI 适配器时，
再初始化 vcpkg submodule 并追加 toolchain 参数：

```bash
git submodule update --init --recursive
cmake -B build -S . -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/vcpkg-toolchain.cmake -DBUILD_ENGINE_BUILTIN=ON
```

注意：DCIr 随主库交付；DCNet / OnnxRuntime / OpenAI 适配器为实验性组件、本次不交付（见上方"发布状态"）。

示例代码（[examples/01_hello_graph](examples/01_hello_graph/main.cpp)）展示标准任务生命周期：

```cpp
graph.bindInput("a", "adder", "a");        // 图公开输入别名
graph.bindOutput("result", "pass", "y");   // 图公开输出别名

auto api = graph.interface();              // 取接口即定型：冻结图并解析别名 → 坐标
auto task = api.createTask();              // 任务句柄：析构自动回收（终态即释放；在飞弃置后自动回收）
task.feed("a", tensorA);                   // 按公开别名喂数据
task.feed("b", tensorB);
auto result = task.run();                  // 同步运行（内部 submitBound + 等待终止）
if (result.status == DC::TaskStatus::Succeeded) {
    auto output = task.takeTensor("result"); // 按公开别名取结果（消费式）
}
```

编写自定义节点/算子（Schema 声明 → RunFn → 注册 → 执行）从
[examples/03_custom_node](examples/03_custom_node/main.cpp) 开始——完整可运行教程；
其他常见问题见[常见问题](#常见问题)。

### 寻址模型与 API 分层

`graph.interface()` 冻结图并一次性解析公开绑定后，
`GraphInterface::Task` 按公开别名操作，同步与异步同级——`run()` 等价于
`submit()` 后 `wait()` 的同步组合（可用别名见 `api.inputAliases()` /
`api.outputAliases()`；未知别名抛错并列出全部可用别名）：

```cpp
auto api = graph.interface();       // 取接口即定型

// 同步一发
auto task = api.createTask();
task.feed("a", x).feed("b", y);    // feed 链式：公开别名 → 内部坐标（一次性解析）
auto result = task.run();           // submit + 无限等待
if (result.status == DC::TaskStatus::Succeeded)
    auto output = task.take("answer"); // 消费式取出

// 异步（同一句柄，同级表达）
task.feed("a", x).feed("b", y).submit(); // 异步启动（不等待）
// …… 此处可做其他工作 ……
auto r = task.wait(5s);             // 显式超时；超时返回 Running，不取消任务
if (r.status == DC::TaskStatus::Succeeded)
    auto output = task.take("answer");
```

任务句柄同时提供 `status()` / `cancel()` / `has(alias)` / `errors()`。
析构语义按任务状态分三路：**已终止** → 立即释放全部资源（无需手工 `releaseTask`）；
**已提交仍在飞** → 弃置托管（不取消任务，完成后自动回收）；
**从未提交** → 立即释放已喂入的输入。
两种节奏的完整演示见 [examples/04_task_lifecycle](examples/04_task_lifecycle/main.cpp)。

**坐标层仅供扩展作者**：内核唯一按 `(nodeName, portName)` 复合坐标寻址
（`feedInput` / `takeOutput` / `takeOutputTensor` / `hasOutput` / `submit` 声明）。
宿主默认不需要接触 taskId 与坐标；只有框架/运行时扩展作者（动态多输出声明、
taskId 复用、引擎嵌入、精细资源控制）才直接驱动 `InferGraph` 坐标运行期 API：

```cpp
graph.feedInput(tid, "llm", "input", data);
graph.submit(tid, "llm", "output");
auto r = graph.takeOutputTensor(tid, "llm", "output");
graph.releaseTask(tid); 
```

输出在任务终止后仍保留；`take` / `takeOutput` / `takeOutputTensor` 为消费式取出。
`waitForResult(taskId)` 默认无限等待直至终止，执行超时由节点实现方自行负责；
复用已终止的taskId 合法；活动任务重复提交会抛出明确错误；
submit 时声明目标在拓扑上不可达会立即抛构图/断链错误。

### 构建选项

```bash
# 方式一：预设（推荐）——只构建核心库，零 vcpkg 依赖（无 DCIr/引擎/DCNet/测试/示例）
cmake --preset core-only
cmake --build build/core-only --config Release

# 方式二：需要 JSON / .dcg 序列化 —— core-ir 预设（仅安装 nlohmann-json/minizip/zlib）
cmake --preset core-ir
cmake --build build/core-ir --config Release

# 方式三：手动开关——IR/引擎/DCNet 均已默认 OFF，最简即一条裸命令（默认含核心测试）
cmake -B build -S .
```

模块依赖分层（`vcpkg.json` feature）：`ir` → DCIr（nlohmann-json/minizip/zlib）、`net` → DCNet（nlohmann-json + poco[netssl]）、`ort-*` → ONNX Runtime 适配器

仅需源码的下载层面，可用 git sparse-checkout 只取核心目录：

```bash
git sparse-checkout init --cone
git sparse-checkout set DCinfer DCIr cmake vcpkg.json CMakeLists.txt
```

核心之上注册自有引擎：实现 `EngineDescriptor` 钩子 → `EngineRegistry::registerEngine()`。
执行钩子遵循相位协议——同步引擎可全部留空、逻辑内联 RunFn。

### 作为库消费（安装与 find_package）

除源码内 `add_subdirectory()` 外，DCinfer 支持 `cmake --install` 后以`find_package` 消费：

```bash
# 构建 + 安装（core-only；其他预设同理，预设名见 CMakePresets.json）
cmake --preset core-only
cmake --build build/core-only --config Release
cmake --install build/core-only --prefix <安装前缀>
```

安装树按包分层（与 vcpkg feature 分层一致），传递依赖由各包 Config 自动解析：

| 包 | 消费方式 | 导出目标 | 传递依赖 |
|---|---|---|---|
| DCinfer | `find_package(DCinfer CONFIG REQUIRED)` | `DCinfer::DCinfer` | 标准库 |
| DCIr | `find_package(DCIr CONFIG REQUIRED)` | `DCIr::DCIr` | DCinfer, zlib |
| DCNet | `find_package(DCNet CONFIG REQUIRED)` | `DCNet::DCNet` | DCinfer, Poco |

宿主工程 CMakeLists 示例：

```cmake
find_package(DCinfer 0.4 CONFIG REQUIRED)
find_package(DCEngine CONFIG REQUIRED)   # 需要 Builtin 引擎时

target_link_libraries(my_app PRIVATE DCinfer::DCinfer DCEngine::Builtin)
```

安装闭环可运行 [examples/install_smoke](examples/install_smoke/CMakeLists.txt)

### 运行测试

```bash
ctest --test-dir build -C Release
```

---

## 常见问题

**循环提交大量短任务，内存持续增长？**

高层默认写法无需关注：任务句柄析构自动回收资源——已终止任务立即释放，
在飞任务被弃置（不取消），完成后自动回收（循环示范见
[examples/02_lowering_benchmark](examples/02_lowering_benchmark/main.cpp)）。
仅当直接使用坐标层 `InferGraph` 运行期 API 且复用同一 taskId 时，才需在消费结果后
调用 `releaseTask(taskId)`（仅终态可释放，活动任务会被拒绝且资源保持不动；
在飞任务交给 `detachTask(taskId)` 弃置）。

**我应该用 `interface()` 还是直接操作 `InferGraph`？**

宿主一律用 `interface()` 高层任务句柄——同步 `run()` 与异步 `submit()+wait()` 同级，
见「寻址模型与 API 分层」。只有框架/运行时扩展作者（动态多输出声明、taskId 复用、
引擎嵌入）才直接驱动 `InferGraph` 坐标 API。

**如何编写自定义节点/算子？**

完整流程见 [examples/03_custom_node](examples/03_custom_node/main.cpp)：
NodePort 工厂声明 Schema → `registerOperator` 注册 → 建图执行。两个关键点：

端口 Schema 用工厂声明（类型与 typeSize 单点书写，改型不漏改）——与聚合初始化等价：

```cpp
s.inputs = {Node::Port::in<float>("x")};                          // 工厂（推荐）
// 等价：s.inputs = {{"x", Tensor::TensorType::Float, sizeof(float), {}}};
```

RunFn 内用类型化访问器读取输入（内建空值/类型校验，失败返回 nullptr），
免手写 `peek → as<T> → 判空` 样板：

```cpp
const auto* x = ctx.input<Tensor>("x");
if (!x)
    return ctx.failure(Node::Status::InvalidInput, "x must be a Tensor");
```

---

## License

This project is licensed under the [MIT License](LICENSE).
