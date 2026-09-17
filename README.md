# DCinfer

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*可嵌入的端云协同数据驱动推理运行时*

DCinfer 是一个 C++20 推理管线编排器，目标是让 AI 应用能够在**本地、云端、PC、局域网设备**间，以**不同推理引擎**自由放置模型节点，同时把网络、调度、数据传输、引擎生命周期和执行细节隔离在运行时。

## 核心功能

### 灵活的图拓扑

传统 DAG 框架将推理管线约束为线性或树形结构——这在多模型、多分支场景下很快会成为瓶颈。DCinfer 采用电路图语义的节点化、端口化、连接器化设计。得益于此模型，你可以轻松构建：

- **多分支管线**：单输出同时驱动多个下游节点并行推理（对同一输出口再次
  `connect()` 即自动扩容为广播扇出，见下方示例）。
- **汇聚模式**：多个上游输出经不同输入口汇入同一节点（同一输入口二次驱动会在构图期被拒绝）。
- **成环拓扑**：支持反馈回路，用于迭代优化、强化学习或流式场景。

```cpp
// 1:N 分发：src.y 同时驱动 b、c 两个下游
graph.connect("src", "y", "b", "x");  // 分支 1（自动插入广播连接器）
graph.connect("src", "y", "c", "x");  // 分支 2（同一连接器原地扩容，返回同一引用）
```

### 原生并发执行

DCinfer 采用**数据驱动执行模型**：节点在所有输入就绪后自动触发，下游节点由上游数据到达事件自动唤醒，无需手动编排执行顺序。独立分支间可乱序并发执行，天然利用多核与异构硬件。

内置三种亲和预设，隔离不同类型的负载：

- **Compute Pool**：专用于模型推理（GPU / NPU 密集型），避免推理任务被 CPU 计算抢占。
- **Operator Pool**：承担 CPU 预处理、后处理、特征工程等算子。
- **System Pool**：处理 I/O、网络传输和系统维护任务。

三层隔离确保重计算不会拖慢系统响应，I/O 延迟也不会阻塞推理吞吐。

### Schema 安全的张量系统

类型和形状错误是 ML 管线中最常见的运行时故障。DCinfer 的张量系统允许在图构造阶段就定义端口 Schema——数据类型和形状在图上声明，执行前即可校验。

核心能力：

- **运行时类型校验**：`TensorSlot` 通过 `ValidatorRegistry` 在数据写入时自动校验类型与形状，将错误拦截在执行前。
- **类型擦除传输**：`TensorSlot` 提供泛型通道，在端点保持类型安全的同时实现中间层零耦合。
- **类 NumPy 链式视图索引**：切片、重塑、转置均通过视图实现，不拷贝底层数据。

### 插件式引擎注册

- 将“节点做什么”与“用什么引擎执行”解耦。
- `EngineRegistry` 提供统一接口，在运行时注册和发现后端引擎——同一张推理图可以逐步适配不同执行后端。
- 我们提供通用注册接口，用户可通过 `EngineDescriptor` 注册自有引擎（ONNX Runtime、TensorRT 或其他自定义后端）
- 节点通过引擎名称引用后端，换引擎无需修改管线结构。

### 零依赖核心

DCinfer 核心库为静态库，**零外部依赖**——仅需 C++20 和标准库。DCIr 序列化模块依赖 nlohmann-json、minizip、zlib（通过 vcpkg 管理），核心库本身不引入任何外部依赖。

## 环境要求

| 组件 | 要求 |
|---|---|
| CMake | ≥ 3.17（裸构建最低 3.17；CMakePresets version 3 需 ≥ 3.21） |
| C++ 标准 | C++20 |
| GCC | ≥ 11（Linux；CI 在 ubuntu-24.04 上验证） |
| Clang | ≥ 14（CI 在 ubuntu-24.04 上验证） |
| MSVC | VS 2026（Windows；CI 在 windows-latest 上验证） |
| Ninja | 可选（README 示例使用 `-G Ninja`） |
| vcpkg | 仅 DCIr / DCNet / OnnxRuntime / OpenAI 模块需要（仓库以 submodule 提供） |

平台支持：Linux、Windows。

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

> 标准任务生命周期 [examples/01_hello_graph](examples/01_hello_graph/main.cpp)

```cpp
graph.bindInput("a", "adder", "a");        // 声明输入端点
graph.bindOutput("result", "pass", "y");   // 声明输出端点

auto api = graph.interface();              // 取接口即定型：冻结图并固化绑定
auto task = api.createTask();              // 任务句柄：析构后自动回收
task.feed("a", tensorA);                   // 将数据推入指定端点
task.feed("b", tensorB);
auto result = task.run();                  // 同步运行
if (result.status == DC::TaskStatus::Succeeded) {
    auto output = task.takeTensor("result"); // 从输出端点中取结果
}
```

> 编写自定义节点/算子 [examples/03_custom_node](examples/03_custom_node/main.cpp)

### 寻址模型与 API 分层

- 构图时设置端点命名，推理时即可用端点名存取数据。
- 同步 `run()` 等价于 `submit()` 后 `wait()`。
- 使用 `api.inputAliases()` / `api.outputAliases()` 获取可用端点名，访问无效端点时报错。

```cpp
auto api = graph.interface();       // 取接口即定型

// 同步一发
auto task = api.createTask();
task.feed("a", x).feed("b", y);    // feed 链式：按端点名喂入
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

### 任务生命周期

执行操作：

- `status()`：获取任务状态。
- `cancel()`：请求取消（协作式；不中断在飞节点，传播链随即停止），幂等。
- `has(name)`：检查指定输出端点是否已有结果。
- `errors()`：获取任务错误信息。

析构时：

- 已终止：立即释放全部资源。
- 已提交（在飞）：请求取消（协作式），随后回收。
- 未提交：立即释放已有输入。

> 任务生命周期管理 [examples/04_task_lifecycle](examples/04_task_lifecycle/main.cpp)。

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

### 引入

- `add_subdirectory()`：源码级、同一构建系统集成。
- `cmake --install` + `find_package`：跨项目二进制包集成。

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
find_package(DCinfer 0.5 CONFIG REQUIRED)
find_package(DCEngine CONFIG REQUIRED)   # 需要 Builtin 引擎时

target_link_libraries(my_app PRIVATE DCinfer::DCinfer DCEngine::Builtin)
```

> 安装闭环 [examples/install_smoke](examples/install_smoke/CMakeLists.txt)。

### 运行测试

```bash
ctest --test-dir build -C Release
```

## 常见问题

### 循环提交大量短任务，内存持续增长？

高层默认写法无需关注：任务句柄析构自动回收资源——已终止任务立即释放，
在飞任务被弃置即请求取消（协作式），随后回收（循环示范见
[examples/02_lowering_benchmark](examples/02_lowering_benchmark/main.cpp)）。

### 如何编写自定义节点/算子？

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

### 单输出如何同时驱动多个下游节点（1:N 分发）？

对同一输出端口再次 `connect()` 即可：引擎自动插入的广播连接器会原地扩容
（多分一份），所有下游均获得数据副本——不存在“静默丢数据”的兼容行为。
也可显式创建 `Connector.Broadcast(N)` 手动建模分发点。

### 如何组合/复用一张推理图？

使用组合算子 `GraphOperator`（[include/Compose/GraphOperator.h](DCinfer/include/Compose/GraphOperator.h)）
把整张图包装成普通 Node——组合发生在算子层，核心图语义保持扁平：

```cpp
auto sub = std::make_shared<InferGraph>();      // 构建子图并声明接口
sub->addNode(...);
sub->bindInput("x", "entry", "x");
sub->bindOutput("y", "exit", "y");

GraphOperator op(sub);                          // 构造即冻结，接管共享所有权
parent.addNode(op.makeNode("Block"));           // 生成普通 Node 嵌入父图
```

- 端口名 = 绑定 alias；类型/形状/required 从目标端口拷贝——构造后子图即冻结，
  再改拓扑抛 `GraphException(Frozen)`；
- 生命周期由 `shared_ptr` 闭合：节点存活期间子图必然存活，无悬垂契约；
- 同一子图可被多个组合节点/多个父图并发复用（子任务 ID 按节点实例命名空间
  隔离，无 DuplicateTask 限制）；
- 等待型节点语义：子图执行期间占住一个执行线程（线程池按“每层池配置 ×
  并发组合节点数”规划）；内层信号停滞时宿主 `cancel()` 父任务可在
  `Options.pollInterval`（默认 100ms）粒度解围，不会永久挂起；
- 节点 type 为 `"Builtin"`：DCIr 序列化往返仅保留结构（Schema 骨架），
  与注册算子同等待遇。

## License

This project is licensed under the [MIT License](LICENSE).
