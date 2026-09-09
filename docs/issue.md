# DCinfer 建议演进路径

## 1. 背景

DCinfer 的定位决定了它与强调强领域约束的 Runtime 有一个根本区别：

> DCinfer 面向的是开放的数据流与推理运行场景，而不是一个类型集合和扩展协议相对封闭的领域 Runtime。

因此，对 DCinfer 而言，通过进一步加强静态类型约束来降低内部复杂度，并不是一个理想方向。

在通用运行时中，节点的数据可能来自不同推理框架、不同设备、不同扩展模块，甚至可能进一步发展出：

* 自定义 Value；
* GPU / NPU resource handle；
* 远程数据引用；
* 流式 packet；
* 用户自定义 Engine；
* 动态 Routing；
* 跨设备或跨进程的数据传输。

如果试图在核心层提前枚举这些组合，类型系统本身很容易成为新的复杂度来源，并削弱扩展能力。

因此，建议 DCinfer 保持当前相对开放的数据模型：

> **数据类型可以动态，但运行时生命周期和状态所有权必须严格。**

后续架构演进的重点，不应是进一步限制“什么数据可以进入 Runtime”，而应该是降低核心模块内部同时存在的状态、职责和生命周期数量。

---

# 2. 核心判断

目前 DCinfer 的主要复杂度并不来自数据类型开放，而来自：

> **不同生命周期的状态在核心对象之间存在较强耦合。**

一个完整的数据流 Runtime 通常同时存在至少三种完全不同的生命周期：

1. **Graph 生命周期**

   * 描述计算拓扑；
   * 描述节点及其连接；
   * 描述执行需求；
   * 生命周期通常较长；
   * 编译完成后原则上应保持稳定。

2. **Runtime 生命周期**

   * 管理线程、Engine、设备、Timer 等共享资源；
   * 可以服务多个 Graph；
   * 可以跨多个 Task 重用。

3. **Task 生命周期**

   * 表示一次具体执行；
   * 包含输入、输出、中间状态、取消、超时、Signal、诊断等；
   * 生命周期短；
   * 多个 Task 可能并发运行在同一个 Graph 上。

如果这三类状态集中在 Node、Graph、ExecutionEngine 等少量对象中，就会产生大量横向协调逻辑：

* Executor 需要了解 Node 的内部状态；
* Graph 虽然已经冻结，却仍然需要暴露可变对象；
* Task 状态分散在多个 map / gate / watchdog 中；
* task termination 必须跨多个模块执行清理；
* ownership 同时承担资源管理和执行状态机语义；
* 对一个局部功能的修改容易影响整体生命周期。

因此建议把 DCinfer 下一阶段的架构目标概括为：

> **减少 Runtime 中的“可变状态权威”，而不是减少 Runtime 可以表达的数据类型。**

---

# 3. 目标架构

建议逐步形成以下三个明确的架构层次：

```text
Compiled Graph
    │
    │ describes
    ▼
What should be executed
    │
    │
    ├─────────────────────┐
    ▼                     ▼
Runtime                Task
shared resources       one execution
```

分别对应：

## 3.1 Graph：描述“做什么”

Graph 应主要负责：

* 节点；
* Port；
* Edge；
* Routing / Connector 等拓扑语义；
* 节点执行描述；
* affinity / execution requirement；
* Graph 对外输入输出契约。

Graph 编译完成以后，应尽可能成为真正意义上的不可变对象。

Graph 不应该持续承担某个具体 Task 的：

* 输入缓存；
* 输出缓存；
* workspace；
* cancellation state；
* timeout state；
* task signal；
* task completion；
* execution progress。

换句话说：

> **Graph 是 Plan，而不是 Execution。**

---

## 3.2 Runtime：描述“使用什么资源执行”

Runtime 应代表长期存在、可以被多次执行共享的运行环境。

它可以拥有：

* worker pools；
* Engine / backend services；
* device resources；
* timer services；
* runtime configuration；
* backend policy；
* diagnostics infrastructure；
* 其他跨 Task 的共享服务。

这类对象的生命周期通常明显长于一次 Graph execution。

将这些资源收敛到显式 Runtime 后，还可以自然解决未来的一些扩展问题，例如：

* 同一进程创建多个独立 Runtime；
* 不同 Runtime 使用不同 backend；
* 不同 Runtime 使用不同线程配置；
* 测试环境隔离；
* multi-tenant；
* backend policy 隔离；
* Engine registry 生命周期控制。

因此长期来看，更推荐：

```text
Runtime
 ├── Executor
 ├── Engines
 ├── Worker Resources
 └── System Services
```

而不是由 process-global singleton 承担所有长期运行态。

---

## 3.3 Task：描述“这一次执行发生了什么”

每次执行都应该存在一个明确的 task-level context。

它统一拥有本次执行相关的状态，例如：

* execution status；
* cancellation；
* deadline；
* 输入；
* 输出；
* node-local execution state；
* signals；
* diagnostics；
* execution progress；
* result collection。

这样，每个 Task 都成为自身状态的唯一主要 authority。

理想情况下：

> 创建 Task 时，这些状态一起产生；Task 结束后，这些状态一起释放。

而不是让一个 Task 的不同部分分别存在于：

* ExecutionEngine；
* Node；
* GraphStore；
* watchdog；
* signal manager；
* output manager；
* active task map；

等多个地方。

这会成为降低 DCinfer 核心复杂度最重要的一步。

---

# 4. 第一阶段：收敛 Task 生命周期

建议第一阶段优先解决：

> **一个 Task 的状态应该由谁拥有？**

这比修改 Node API、Graph API 或 Engine Registry 更重要。

当前 Runtime 中如果存在大量以 TaskId 为索引、分别维护 execution state 的内部结构，那么本质上代表：

```text
Task = 多个子系统中的若干记录
```

建议逐步演化成：

```text
Task = 一个明确的运行时对象
```

ExecutionEngine 可以继续管理 active Task，但不应该成为 Task 内部所有状态的直接所有者。

## 为什么优先做这一层？

因为 Task 是并发系统中变化最快的生命周期。

Graph 和 Runtime 通常比较稳定，而 Task：

* 会创建；
* 会完成；
* 会失败；
* 会取消；
* 会 timeout；
* 会等待；
* 会被 signal 唤醒；
* 会在多个 worker 间传播。

如果 Task 状态分散，几乎所有并发问题都会变成跨模块问题。

一旦 Task 生命周期收敛，大量原本属于 ExecutionEngine 的复杂度可以自然向 Task execution domain 内聚。

这里的目标不是简单地“把几个 map 塞进一个 class”，而是建立一个更重要的不变量：

> **任何属于某次 execution 的 mutable state，都应该能够明确追溯到这次 Task。**

---

# 5. 第二阶段：让 Graph 真正不可变

Task 生命周期收敛后，建议进一步推动 Graph 和 execution state 分离。

目前如果一个已经 freeze / compile 的 Graph 内部仍然包含 task-specific mutable state，那么“Graph 已冻结”实际上只冻结了 topology，而没有真正冻结 Graph object model。

这种设计会产生一个长期问题：

> Graph 的 constness 无法代表真实语义。

结果往往是：

```text
const CompiledGraph
        │
        ▼
mutable Node / Store
        │
        ▼
task-local mutation
```

这会削弱很多原本应该由架构天然保证的性质：

* 多 Task 并发执行；
* Graph safely shared；
* execution isolation；
* const correctness；
* caching；
* debugging；
* future serialization；
* remote execution；
* graph-level optimization。

因此建议明确区分：

```text
Node Definition / Node Plan
```

与：

```text
Node Execution State
```

前者属于 Graph。

后者属于 Task。

概念上可以理解为：

```text
CompiledGraph

Node A ───── Node B ───── Node C
  │            │            │
  ▼            ▼            ▼

Task #1
Frame A      Frame B      Frame C

Task #2
Frame A      Frame B      Frame C
```

这里所谓 Frame 只是生命周期概念，并不要求开发者采用某一种具体实现。

核心要求只有一个：

> **Node 本身描述节点，而节点在某次 Task 中产生的状态属于 Task。**

---

# 6. 第三阶段：把 ExecutionEngine 降级为“调度者”

当 Task state 与 Graph state 分离以后，可以重新审视 ExecutionEngine 的职责。

理想情况下，ExecutionEngine 不应该知道太多业务状态。

它最主要应该负责：

```text
ready event
    ↓
dispatch
    ↓
execute
    ↓
propagate
    ↓
new ready event
```

即：

> **Executor 管理 execution events，而不是管理所有 execution state。**

如果 Executor 同时负责：

* task state；
* node state；
* output cleanup；
* signal cleanup；
* watchdog lifecycle；
* task completion；
* diagnostic state；
* result recovery；
* graph mutation；
* backend registry；

那么它必然会成为整个 Runtime 的复杂度汇聚点。

更合理的职责关系应该是：

```text
Executor
   │
   ├── schedules
   ▼
Task

Task
   │
   ├── owns execution state
   │
   └── refers to
   ▼
CompiledGraph

Executor
   │
   └── uses
       Runtime resources
```

这样 ExecutionEngine 的 correctness 就主要围绕调度状态机，而不是围绕整个系统所有模块的生命周期展开。

---

# 7. 第四阶段：让执行进度成为显式概念

DCinfer 支持的运行模型比传统一次性 DAG execution 更开放。

例如未来或当前可能存在：

* loop；
* dynamic routing；
* signal；
* blocking；
* delayed execution；
* asynchronous resource；
* external event。

因此应避免把：

> “当前没有 worker 正在执行”

直接解释成：

> “Task 已完成”。

更合理的抽象是区分两个维度。

## Execution progress

```text
Active
   ↕
Quiescent
```

表示当前执行传播是否还有 activity。

## Task lifecycle

```text
Running
   │
   ├── Succeeded
   ├── Failed
   ├── Cancelled
   └── TimedOut
```

二者属于不同概念。

一个 Task 可以：

```text
Running + Quiescent
```

例如正在等待 Signal。

随后又变成：

```text
Running + Active
```

最终才进入：

```text
Succeeded
```

这种划分非常适合 DCinfer，因为它没有把 Runtime 限制在纯 DAG execution 上。

---

# 8. 不建议让 ownership 本身承担调度状态机语义

现代 C++ 中使用 RAII、shared ownership 管理 lifetime 是合理的。

但建议尽量保持一个边界：

> **ownership 负责对象是否存活，execution state machine 负责系统当前处于什么执行状态。**

如果某个重要执行事件依赖：

```text
最后一个 shared_ptr 被释放
```

才触发，那么控制流会变得隐式。

这种设计通常可以工作，而且有时非常巧妙，但长期维护成本较高：

* 调用路径中看不到实际状态变化；
* refcount 意外变化可能改变 execution semantics；
* debugging 困难；
* cancellation 与 destruction 容易相互影响；
* correctness 依赖对象被谁捕获；
* lifetime 和 scheduler semantics 被绑定。

因此对于诸如：

* outstanding work；
* quiescence；
* completion；
* termination；

等 Runtime 核心事件，更推荐存在明确的状态表达。

这并不意味着不能继续使用 `shared_ptr`。

相反：

> `shared_ptr` 可以解决“Task 是否仍然存活”，但不应该同时负责回答“Task 是否已经完成”。

这两个问题最好保持正交。

---

# 9. 第五阶段：统一 Runtime 级系统服务

随着 Task 状态收敛，部分当前分布在 ExecutionEngine 内部的系统能力可以进一步抽象为 Runtime service。

其中一个典型例子是 deadline / timeout。

从架构上看：

```text
timeout
```

并不是 Task 自己需要拥有一条 thread，而是：

```text
Runtime 提供时间服务

Task 注册 deadline
```

即：

```text
Runtime
   │
   └── Timer Service
            │
            ├── Task A deadline
            ├── Task B deadline
            └── Task C deadline
```

这种选型的价值并不主要是减少线程数量，而是：

> **让 timeout 生命周期从 execution lifecycle 中解耦。**

类似地，未来其他跨 Task 系统能力也可以遵循同样原则：

* tracing；
* metrics；
* device monitoring；
* scheduling resources；
* backend resources。

这样 Runtime 就逐渐形成真正意义上的 system boundary。

---

# 10. 第六阶段：重新划分 Edge 与 Connector 的语义

另一个值得逐渐简化的地方，是 Graph 中“连接关系”与“连接行为”之间的区别。

建议保持两个概念：

## Edge

表示：

> 一个节点的输出依赖会传播到另一个节点的输入。

Edge 是 Graph topology 本身。

## Connector / Routing Node

表示：

> 数据传输本身存在需要执行的行为。

例如：

* dynamic routing；
* remote transfer；
* device transfer；
* serialization；
* format conversion；
* batching；
* network transport。

因此建议长期采用这样的架构原则：

```text
普通依赖
A ─────────→ B

普通 fan-out
A ─────────→ B
 └─────────→ C

存在连接行为
A → Router → B
           → C
```

这样可以保留 Connector 作为高级扩展能力，同时避免最普通的数据依赖也需要经历额外的 runtime abstraction。

其核心判断标准可以是：

> **如果“连接”不需要执行，那么它应该首先是 topology；只有连接本身需要执行时，它才应该成为 executable entity。**

---

# 11. 第七阶段：建立 Runtime Instance 边界

Engine Registry 等全局服务不建议作为第一阶段重构目标。

原因是：

> singleton 本身虽然会影响长期架构，但它目前并不是 ExecutionEngine 复杂度最大的来源。

过早替换 singleton 很可能只改变 dependency injection 方式，却没有降低 execution model 的复杂度。

建议等：

1. Task lifecycle；
2. Graph immutability；
3. Runtime execution state；
4. Executor responsibilities；

这些边界基本稳定后，再逐步建立：

```text
Runtime Instance
```

最终让：

```text
Runtime A
Runtime B
```

可以在同一进程中拥有相互隔离的运行资源。

Engine Registry 届时自然会变成 Runtime-owned service，而不是为了“去 singleton”而去 singleton。

这个顺序可以避免进行一次收益有限的大规模 API 改造。

---

# 12. InferGraph 应保留为易用 façade

架构收敛不应该以牺牲 DCinfer 的易用性为代价。

因此即使内部逐步拆分为：

```text
Builder
CompiledGraph
Runtime
Executor
Task
```

也没有必要要求普通用户直接操作所有这些对象。

`InferGraph` 仍然可以作为高层 façade：

```text
User
 │
 ▼
InferGraph
 │
 ├── build
 ├── freeze
 ├── execute
 └── inspect
```

而内部再映射到更清晰的架构层。

这意味着：

> **内部架构应该严格，外部 API 可以继续宽松。**

这是通用 Runtime 很重要的一条设计原则。

内部 decomposition 是为了让 maintainers 更容易推理系统，不意味着应该把这些复杂度暴露给使用者。

---

# 13. 不建议优先推进的方向

## 13.1 不建议优先加强 Value 类型约束

DCinfer 的开放 Value 模型与其通用 Runtime 定位并不冲突。

runtime validation 本身不是低水平设计。

只要：

* ownership 清晰；
* failure semantics 清晰；
* validation boundary 清晰；

动态数据类型完全可以与高质量 Runtime 架构共存。

---

## 13.2 不建议为了架构感而大量增加接口层

降低复杂度不等于增加：

```text
IExecutor
ITask
IGraph
INode
IRuntime
IService
...
```

抽象层的数量不是设计成熟度的衡量标准。

本次演进真正需要减少的是：

> state ownership ambiguity

而不是增加 abstract class 数量。

---

## 13.3 不建议第一步重写 EngineRegistry

Engine Registry 的问题主要属于长期：

* Runtime isolation；
* multi-instance；
* service ownership。

而当前最值得解决的是 execution lifecycle 本身。

建议等 Runtime boundary 明确后再自然迁移。

---

## 13.4 不建议优先做 PImpl 化

公共头文件收敛当然有价值，但这更接近 API engineering，而不是当前核心复杂度的根源。

如果内部 execution model 本身仍然高度耦合，那么：

```text
把复杂代码移动到 .cpp
```

只是在隐藏复杂度，而不是减少复杂度。

应该先降低模型复杂度，再进行 API surface cleanup。

---

## 13.5 不建议把“无 scheduler thread”视为架构目标

线程少并不天然意味着 Runtime 更简单。

如果一个小型系统服务可以消除大量：

* lifecycle special case；
* self-join handling；
* destruction ordering；
* per-task bookkeeping；

那么增加一个明确职责的系统线程，可能反而显著降低总体复杂度。

真正应该优化的是：

> **系统需要多少种状态和特殊路径才能保证 correctness。**

而不是单纯统计 thread 数量。

---

# 14. 建议的整体演进顺序

建议按照 dependency direction，而不是按照类的重要程度进行演进。

## Phase 1 — Task State Consolidation

目标：

> 让一次 execution 成为明确的状态所有权边界。

重点解决：

* task-scoped state 的归属；
* task lifecycle；
* cancellation；
* diagnostics；
* execution progress。

---

## Phase 2 — Graph / Execution Separation

目标：

> Graph 真正成为 immutable execution plan。

重点解决：

* Node definition 与 Node runtime state 分离；
* task-local workspace 分离；
* frozen graph deep const；
* 多 Task 共享 Graph 的语义明确化。

---

## Phase 3 — Executor Simplification

目标：

> ExecutionEngine 只负责 execution event scheduling。

重点解决：

* ready；
* dispatch；
* propagate；
* execution progress；
* termination coordination。

同时逐渐把 task-specific cleanup 移出 Executor。

---

## Phase 4 — Runtime Services

目标：

> 把跨 Task 的长期资源形成明确 Runtime boundary。

包括：

* Timer；
* worker resources；
* Engines；
* backend resources；
* diagnostics infrastructure。

---

## Phase 5 — Graph Semantic Simplification

目标：

> 区分 topology 与 executable connection behavior。

重点重新审视：

* Edge；
* Connector；
* Routing；
* Transfer。

减少普通 Graph 表达所需要的中间抽象。

---

## Phase 6 — Runtime Instance & Public API Cleanup

最后再处理：

* global registry；
* runtime isolation；
* façade 收口；
* public/private API；
* PImpl；
* dependency exposure。

此时这些修改将建立在已经稳定的 internal architecture 上，而不是反过来。

---

# 15. 每个阶段都应保持兼容性优先

建议整个演进尽量遵循：

> **先改变内部 ownership，再改变外部 API。**

DCinfer 当前已经形成了一定的使用方式，因此没有必要为了架构纯粹性一次性重构整个 public API。

比较稳妥的策略是：

```text
Existing API
     │
     ▼
Compatibility façade
     │
     ▼
New internal model
```

当新模型稳定以后，再判断哪些 public API 确实需要改变。

这可以避免一次大规模 rewrite，同时也方便利用现有测试验证行为等价性。

---

# 16. 判断演进是否成功的标准

本次重构的成果不应该主要通过“减少多少代码”判断。

更重要的评价指标是以下问题能否得到简单答案。

### 1. 一次 Task 的所有状态在哪里？

理想答案：

> Task execution domain。

而不是：

> 分散在几个 manager、Node 和 ExecutionEngine 里面。

---

### 2. 一个已经编译的 Graph 是否还会因为执行某个 Task 而变化？

理想答案：

> 不会。

---

### 3. ExecutionEngine 的主要职责是什么？

理想答案：

> 调度 execution event。

而不是列出十几个生命周期相关职责。

---

### 4. 一个 Node 的数据分成哪两类？

理想答案：

> Graph-level definition 与 Task-level execution state。

---

### 5. timeout 属于谁？

理想答案：

> Task 有 deadline，Runtime 提供 timer capability。

---

### 6. `shared_ptr` 的作用是什么？

理想答案：

> 管理 lifetime。

而不是：

> 同时编码关键 execution state transition。

---

### 7. 普通连接为什么存在？

理想答案：

> 因为存在数据依赖。

只有连接本身存在运行行为时，才需要额外 executable abstraction。

---

### 8. 两个 Graph 是否可以安全并发执行？

理想答案应该可以直接从 architecture 得出，而不需要分析大量内部锁。

---

### 9. 同一个 Graph 是否可以安全执行多个 Task？

同样应该主要由 state ownership 保证，而不是依赖特殊 synchronization。

---

# 17. 最终目标

DCinfer 不需要演化成一个类型更加严格的 Runtime。

相反，它可以继续保持：

```text
Open Value Model
Open Engine Model
Open Graph Model
Dynamic Routing
Runtime Validation
```

同时在内部形成非常严格的：

```text
Ownership Model
Lifetime Model
Execution Model
State Transition Model
Runtime Boundary
```

这是两种完全不同的“严格”。

对于通用 Runtime，更值得追求的是后者。

最终希望 DCinfer 的核心架构能够清晰表达为：

```text
CompiledGraph
    │
    │ immutable plan
    ▼

Task
    │
    │ execution state
    ▼

Executor
    │
    │ scheduling
    ▼

Runtime
    │
    │ shared resources
    ▼

Backend / Engine / Device
```

其中：

> **Graph 描述做什么。**

> **Runtime 提供用什么做。**

> **Task 描述这一次发生了什么。**

> **Executor 负责推动它发生。**

如果能够建立这几个稳定边界，那么 DCinfer 即使未来继续增加：

* 更多数据类型；
* 更多 Engine；
* 更多 Routing 模式；
* GPU / NPU backend；
* asynchronous execution；
* remote execution；
* streaming；
* distributed graph；

这些能力也不必继续线性增加核心执行器的复杂度。

因此，建议将下一阶段架构演进的核心目标定义为：

> **保持数据模型开放，收紧状态所有权；保持外部表达能力，降低内部生命周期耦合。**

相比进一步扩展类型系统或增加新的 framework abstraction，这更可能成为 DCinfer 从“功能强大的通用执行框架”走向“长期可演进 Runtime 基础设施”的关键一步。
