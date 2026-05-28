# DCinfer

Not DAG 的 C++20 推理管线编排器。提供节点编排、多引擎热插拔、协程驱动的异步数据流传播能力，适用于云端混合推理、多模型推理管道、预处理/后处理编排、自定义算子集成等场景。

---

## 核心特性

- **Not DAG 拓扑编排** —— 节点 + 端口级边连接，支持复杂推理管道建模，支持成环
- **多引擎支持** —— 注册 ONNX / TensorRT / 自定义引擎，热插拔
- **原生面向并发** —— 可配置线程池，数据驱动，内部 算子/引擎 乱序执行，最大化硬件利用率
- **三层线程池隔离** —— Compute（GPU）、Operator（CPU 密集）、System（数据搬运）物理隔离
- **类型安全** —— 编译期 Schema 定义 + 运行时 类型/形状 校验
- **电路图语义** —— 用 节点、导线、执行器 概念自然建模高并发推理模式
- **无外部依赖** —— 核心库为 static lib，零第三方依赖

---

## 环境要求

| 项目 | 版本 |
|------|------|
| C++ 标准 | C++20 |
| CMake | ≥ 3.17 |
| 依赖管理 | vcpkg |

---

## 快速开始

### 构建

```bash
cd DCinfer
cmake -B build -S . ^
  -DCMAKE_TOOLCHAIN_FILE=external/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

### 运行测试

```bash
cmake --build build --config Release
ctest --test-dir build -C Release
```

---

## 架构概览

```
┌────────────────────────────────────────────────────────┐
│                     InferGraph                         │
│  ┌───────┐   connect    ┌───────────┐   connect   ┌──┐ │
│  │ Node  │────────────▶│ Connector │───────────▶│..│ │
│  │ (add) │              │ (Wire/BC) │             │  │ │
│  └───────┘              └───────────┘             └──┘ │
│       ▲                      ▲                         │
│       │  execute             │  execute                │
│       ▼                      ▼                         │
│  ┌───────────┐    ┌───────────────────────┐            │
│  │ CoroScheduler │  │   Three-Tier ThreadPool  │       │
│  │  co_await     │  │  Compute/Operator/System │       │
│  └───────────┘    └───────────────────────┘            │
└────────────────────────────────────────────────────────┘
                        │
                        ▼
             ┌───────────────────┐
             │  EngineRegistry    │
             │  ONNX / TensorRT   │
             │  Custom / Builtin  │
             └───────────────────┘
```

- **Node** —— 基本计算单元 = Schema（输入/输出端口） + RunFn（执行体） + 线程池归属
- **Connector** —— 数据路由节点（Wire/Broadcast/Routing），遵循"边即节点"语义
- **InferGraph** —— 推理图管理器，持有节点和边拓扑
- **EngineRegistry** —— 推理引擎插件注册表，支持多种节点创建方式
- **CoroScheduler** —— 协程调度器，分配线程执行任务

---

## 核心概念

### 1. Node（节点）

节点是图的基本计算单元。每个节点由 Schema 和 RunFn 组成：

```cpp
// 定义一个加法算子
Node::Schema addSchema() {
    Node::Schema s;
    s.inputs  = { Node::Port::in<float>("a"), Node::Port::in<float>("b") };
    s.outputs = { Node::Port::out<float>("s") };
    return s;
}

Node::RunFn addRunFn() {
    return [](Node::RunContext& ctx) -> Node::Result {
        const auto& aVal = ctx.input("a");
        const auto& bVal = ctx.input("b");
        const auto* a = aVal.as<Tensor>();
        const auto* b = bVal.as<Tensor>();
        if (!a || !b) return ctx.failure(Node::Status::InvalidInput, "not a Tensor");

        auto result = std::make_unique<Tensor>(TensorType::Float, sizeof(float));
        *result = a->item<float>() + b->item<float>();
        ctx.output("s", Value(std::move(result)));
        return ctx.success();
    };
}

// 创建节点
auto node = std::make_unique<Node>(
    "Builtin", "adder", addSchema(), addRunFn(),
    nullptr, ThreadPoolAffinity::Operator);
```

**线程池归属**（`ThreadPoolAffinity`）：

| 归属 | 说明 | 典型用途 |
|------|------|----------|
| `Compute` | 计算线程池 | ONNX / TensorRT 推理（GPU 加速） |
| `Operator` | 算子线程池 | 预处理、后处理、自定义算子（CPU 密集） |
| `System` | 系统线程池 | Connector、数据搬运 |

### 2. InferGraph（推理图）

图是整个推理管道的容器。核心 API：

```cpp
InferGraph graph;

// 添加节点
graph.addNode(std::make_unique<Node>("Builtin", "add", addSchema(), addRunFn()));
graph.addNode(std::make_unique<Node>("Builtin", "proc", procSchema(), procRunFn()));

// 接线：wire() 自动在两业务节点间插入 Wire Connector
graph.wire("add", "s", "proc", "x");

// 从图外注入数据
graph.feedInput("task1", "add", "a", makeTensor(3.0f));
graph.feedInput("task1", "add", "b", makeTensor(4.0f));

// 同步执行（适合简单管道）
graph.run();

// 获取结果
auto result = graph.getOutputTensor("task1", "proc", "out");

// 异步提交（协程驱动，适合持续流式场景）
graph.declareOutput("task2", "proc", "out", 10);  // 需要产出 10 次
graph.submit("task2", std::chrono::milliseconds(5000));
```

**关键约束**：两个业务节点不允许直连，必须通过 Connector 中转。`connect()` 会校验此规则。

### 3. Connector（连接器）

| 类型 | 语义 | 注册名 |
|------|------|--------|
| **Wire** | 1→1 直通，`graph.wire()` 自动创建 | `Connector.Wire` |
| **Broadcast** | 1→N 广播，拷贝 N - 1 份到所有下游 | `Connector.Broadcast` |
| **Routing** | 1→N 轮询分发，每次仅写一个下游 | `Connector.Routing` |

```cpp
// 广播：1 upstream → 3 downstream
auto bcSchema = Connector::broadcastSchema(3);
auto bcNode = std::make_unique<Node>(
    "Connector.Broadcast", "bc", bcSchema,
    Connector::broadcastRunFn(), nullptr, ThreadPoolAffinity::System);
bcNode->setConnector(true);
graph.addNode(std::move(bcNode));

graph.connect("add", "s", "bc", "in");
graph.connect("bc", "out_0", "proc_a", "x");
graph.connect("bc", "out_1", "proc_b", "x");
graph.connect("bc", "out_2", "proc_c", "x");
```

### 4. EngineRegistry（引擎注册表）

支持三种节点创建方式：

```cpp
auto& reg = EngineRegistry::instance();

// 方式 1：Builtin 算子（纯 DC::Tensor）
auto node1 = reg.createNode("myOp", mySchema, myRunFn);

// 方式 2：注册引擎 + 配置
auto node2 = reg.createNode("ONNX", "onnx_node", &onnxConfig);

// 方式 3：注册引擎 + 模型路径（自动推导 Schema）
auto node3 = reg.createNode("ONNX", "onnx_node", "resnet50.onnx");

// 注册算子
auto& reg = EngineRegistry::instance();
reg.registerOperator("MyOperator", mySchema, myRunFn);
auto opNode = reg.createOperator("MyOperator", "op1");
```

### 5. 协程数据流

数据流由协程自动驱动：

```
 feedInput → Node 就绪 → 提交线程池 → co_await 完成
    → 消费输出 → 写入下游 → 下游就绪 → spawn 传播协程
    → 累加产出计数 → 满足 OutputDeclaration → 终止
```

- 使用 `TaskGate`（`shared_ptr`）实现协程链生命周期管理
- 超时看门狗独立线程保护
- 数据耗尽时自动检测声明是否满足

---

## 完整示例

```cpp
#include "InferGraph.h"
#include "Connector.h"

using namespace DC;

// 1. 定义加法算子
Node::Schema addSchema() { /* ... 同上 ... */ }
Node::RunFn addRunFn()   { /* ... 同上 ... */ }

// 2. 定义恒等算子
Node::Schema identitySchema() {
    Node::Schema s;
    s.inputs  = { Node::Port::in<float>("x") };
    s.outputs = { Node::Port::out<float>("y") };
    return s;
}

Node::RunFn identityRunFn() {
    return [](Node::RunContext& ctx) -> Node::Result {
        const auto* t = ctx.input("x").as<Tensor>();
        ctx.output("y", Value(std::make_unique<Tensor>(*t)));
        return ctx.success();
    };
}

int main() {
    InferGraph graph;

    // 3. 构建图: add → broadcast → [id_a, id_b]
    graph.addNode(std::make_unique<Node>("Builtin", "add", addSchema(), addRunFn()));

    auto bcNode = std::make_unique<Node>(
        "Connector.Broadcast", "bc",
        Connector::broadcastSchema(2), Connector::broadcastRunFn(),
        nullptr, ThreadPoolAffinity::System);
    bcNode->setConnector(true);
    graph.addNode(std::move(bcNode));

    graph.addNode(std::make_unique<Node>("Builtin", "id_a", identitySchema(), identityRunFn()));
    graph.addNode(std::make_unique<Node>("Builtin", "id_b", identitySchema(), identityRunFn()));

    graph.connect("add", "s", "bc", "in");
    graph.connect("bc", "out_0", "id_a", "x");
    graph.connect("bc", "out_1", "id_b", "x");

    // 4. 注入输入 & 执行
    auto makeTensor = [](float v) {
        auto t = std::make_unique<Tensor>(Tensor::TensorType::Float, sizeof(float));
        *t = v;
        return Value(std::move(t));
    };

    graph.feedInput("task1", "add", "a", makeTensor(10.0f));
    graph.feedInput("task1", "add", "b", makeTensor(20.0f));
    graph.run();

    // 5. 获取结果：两个下游均应有 30.0
    auto rA = graph.getOutputTensor("task1", "id_a", "y");
    auto rB = graph.getOutputTensor("task1", "id_b", "y");
    // rA.item<float>() == 30.0f, rB.item<float>() == 30.0f
}
```

更多示例参见 `DCinfer/test/InferGraphTest.cpp`。

---

## 目录结构

```
DCinfer/
├── CMakeLists.txt                    # 根 CMake 构建配置
├── CMakePresets.json                 # CMake 预设
├── vcpkg-configuration.json          # vcpkg manifest
├── README.md
├── DCinfer/
│   ├── CMakeLists.txt                # 核心库构建配置
│   ├── include/
│   │   ├── Graph/                    # 推理图、引擎注册、线程池、协程调度
│   │   │   ├── InferGraph.h          # 推理图
│   │   │   ├── EngineRegistry.h      # 引擎注册表 + 引擎描述符
│   │   │   ├── Connector.h           # 内置连接器（Wire/Broadcast/Routing）
│   │   │   ├── CoroScheduler.h       # C++20 协程调度器
│   │   │   ├── ThreadPool.h          # 三层线程池
│   │   │   └── GraphException.h      # 图级异常
│   │   ├── Node/
│   │   │   ├── Node.h                # 节点定义（Schema、RunFn、协程支持）
│   │   │   ├── TensorSlot.h          # 类型擦除的数据槽位
│   │   │   ├── Value.h               # 数据载体（DC::Tensor ↔ 引擎原生张量桥接）
│   │   │   ├── SlotType.h            # 槽位数据类型枚举
│   │   │   ├── NodeException.h       # 节点级异常
│   │   │   └── NodeMods.h            # 节点模块聚合头文件
│   │   ├── Tensor/
│   │   │   ├── Tensor.hpp            # DC 原生张量（模板实现）
│   │   │   ├── TensorData.h          # 张量数据存储（稀疏/稠密）
│   │   │   ├── TensorMeta.h          # 张量元数据
│   │   │   ├── TensorException.h     # 张量级异常
│   │   │   └── TensorMods.h          # 张量模块聚合头文件
│   │   └── Tools/
│   │       ├── DCtype.h              # 类型映射系统
│   │       ├── Exception.h           # 基础异常类
│   │       └── tool.h                # 通用工具宏
│   ├── src/                          # 源文件实现
│   │   ├── InferGraph.cpp            # 推理图核心逻辑
│   │   ├── Node.cpp                  # 节点实现
│   │   ├── Connector.cpp             # 连接器实现
│   │   ├── EngineRegistry.cpp        # 引擎注册表实现
│   │   ├── CoroScheduler.cpp         # 协程调度器实现
│   │   ├── ThreadPool.cpp            # 线程池实现
│   │   ├── Tensor.cpp                # 张量实现
│   │   ├── TensorData.cpp            # 张量数据实现
│   │   ├── TensorMeta.cpp            # 张量元数据
│   │   ├── TensorSlot.cpp            # 张量槽位实现
│   │   └── SlotType.cpp              # 槽位类型实现
│   └── test/                         # 单元测试
│       ├── CMakeLists.txt
│       ├── InferGraphTest.cpp        # 推理图集成测试
│       ├── NodeTest.cpp
│       ├── TensorTest.cpp
│       ├── TensorDataTest.cpp
│       ├── TensorSlotTest.cpp
│       ├── ConnectorTest.cpp
│       └── EngineRegistryTest.cpp
├── external/vcpkg/                   # 内置 vcpkg 依赖管理
└── build/                            # 构建输出目录
```

---

## 扩展指南

### 注册自定义推理引擎

```cpp
auto& reg = EngineRegistry::instance();

EngineDescriptor desc;
desc.engineType  = "MyEngine";
desc.converter   = TensorConverter{ /* toNative, toDC */ };
desc.factory     = makeNodeFactory("MyEngine", mySchema, myRunFn);
desc.loadModel   = [](const std::string& path) -> ModelHandle { /* ... */ };
desc.createEngine = [](const std::string& path) -> EngineInstance { /* ... */ };
desc.getInputPorts  = [](ModelHandle model) -> std::vector<Node::Port> { /* 推导输入端口 */ };
desc.getOutputPorts = [](ModelHandle model) -> std::vector<Node::Port> { /* 推导输出端口 */ };
desc.synchronize  = [](void* engine) { /* cudaStreamSynchronize */ };
desc.preRun       = [](void* engine) { /* 推理前准备 */ };
desc.postRun      = [](void* engine, Node::RunContext& ctx) { /* D2H 传输 */ };
desc.releaseEngine = [](void* engine) { /* 资源释放 */ };
desc.onError       = [](void* engine) { /* 异常恢复/状态重置 */ };

reg.registerEngine(desc);
```

### 注册自定义算子

```cpp
auto& reg = EngineRegistry::instance();
reg.registerOperator("Softmax", softmaxSchema, softmaxRunFn);
auto node = reg.createOperator("Softmax", "softmax1");
```

### 自定义连接器

实现 Schema + RunFn，标记 `setConnector(true)` + `ThreadPoolAffinity::System`：

```cpp
auto myConn = std::make_unique<Node>(
    "Connector.Custom", "myC", mySchema, myRunFn,
    nullptr, ThreadPoolAffinity::System);
myConn->setConnector(true);
graph.addNode(std::move(myConn));
```

---

## 开发状态

核心模块已完成：

- [x] Node — 节点定义、Schema、RunFn、协程支持
- [x] InferGraph — Not DAG 构建、同步/异步执行、数据传播
- [x] Tensor — DC 原生张量（多类型、形状变换、View 代理）
- [x] TensorSlot — 类型擦除存储 + 运行时校验
- [x] Connector — Wire / Broadcast / Routing
- [x] EngineRegistry — 引擎注册、算子注册、模型加载
- [x] ThreadPool — 三层线程池
- [x] CoroScheduler — 协程调度
- [x] 单元测试 — 覆盖核心模块

待扩展：

- [ ] ONNX Runtime 引擎完整集成（当前仅本地调试）
- [ ] TensorRT 引擎适配
- [ ] 性能基准测试
- [ ] 跨平台支持（Linux / macOS）

---

## License

[TBD]