# DCNet 设计文档

> 张量网络传输框架：为远端推理服务（云端 API、局域网 GPU worker、私有服务）
> 提供统一的**传输 + 张量数据格式 + 错误归一化 + 对接契约**，让远端节点以与
> 本地引擎一致的方式接入 DCinfer 图运行时（EngineDescriptor 家族）。
> 协议级适配器（如 OpenAI 兼容）位于 `DCEngines`，基于本框架契约实现。
>
> 本文档为权威约定，实现与文档不一致处以实现为准并回改文档。

---

## 1. 背景与定位

### 1.1 定位

DCNet 是**张量网络传输框架**，与 `DCEngines` 的本地引擎适配器互补：

| 维度 | 本地引擎适配器（DCEngines） | DCNet 框架（+ 其上协议适配器） |
|---|---|---|
| Schema 来源 | 模型文件推导（如 `getPortsFromSession`） | **本地声明**：`getInputPorts/getOutputPorts` 返回静态端口表 |
| 实例键 | `engineType + modelPath`（模型文件路径） | 同一机制，modelPath 语义重载为**远端端点**（URL / host:port） |
| 错误来源 | 本地引擎异常 | 网络错误 + 远端报文，经**归一化层**映射到本地标准 |
| 部署形态 | 进程内加载模型 | 进程外服务，经网络通信 |
| 协议语义 | 引擎实现自带 | **框架不带**：具体协议（如 OpenAI 兼容）由 `DCEngines` 适配器实现 |

### 1.2 两条硬性要求（来自定位）

1. **本地配置形状规则**：每个网络算子的端口 Schema（类型、形状、锚定、默认值）
   在本地声明，不依赖远端提供元数据。
2. **转化为本地标准报错模式**：网络错误（超时/断连/拒绝）与对方机器传来的错误
   报文，统一映射为 DCinfer 既有标准（`Node::Result` + `NodeStatus` +
   `ErrorTracker` 诊断），图级语义与本地引擎节点完全一致，InferGraph / DCIr /
   上层调度对 DCNet 节点零特判。

---

## 2. 核心设计决策

### 2.1 ADR-1：契约内置，协议外置（自带协议 vs 纯契约的裁决）

**问题**：算子本身是否自带通信协议？

- 自带协议 → 对方非 DCinfer 时必须参考我们的协议对接，锁死生态；
- 纯契约 → 开发者按"与我们核心服务的对接契约"手动适配对方协议。

**结论**：两者按两层拆开，不二选一：

- **对接契约必须内置**——这是算子与 DCinfer Runtime 的接缝，稳定、版本化；
- **线上通信协议一律外置**为可注入适配器，不硬编码进算子/框架；
- **DCinfer 原生协议（DCNet.Native）仅作为可选适配器之一**，用于 DCinfer↔DCinfer
  直连，不是默认要求。

理由：

1. **不锁死生态**。README 定位是"让模型节点自由放置在本地/云端/局域网/不同引擎"，
   远端生态已是 vLLM / TGI / llama.cpp / Triton / 云 API 的天下。强制自带协议等于
   要求第三方反过来对接我们。
2. **版本责任隔离**。自有协议一旦铺开，升级就是永久兼容包袱；协议外置后，版本
   问题被隔离在单个适配器内部。
3. **架构一致性**。现有形态中 `converter`、`preRun/onError/releaseEngine` 全部是
   可注入钩子（`EngineDescriptor`，`DCinfer/include/Graph/EngineRegistry.h`），
   通信协议本应属于同一类可注入策略。

### 2.2 ADR-2：用 EngineDescriptor，不用 registerOperator

- `EngineRegistry::registerOperator`（`EngineRegistry.h`）是无状态轻量路径，
  承载不了连接池、健康检查、重连等实例状态；
- DCNet 网络算子有连接生命周期，且"与 ONNX 适配器形态类似"本身就指向
  `EngineDescriptor` 形态（`createEngine` / 端口推导 / `factory` / 运行时钩子）。
- 纯编解码类辅助（payload 组装/拆解）若确需独立算子形态，另行评估，不进 DCNet 核心。

### 2.3 ADR-3：开发者的心智模型是"双向翻译器"

**不是**"下载对方 SDK，把 SDK 输出作为契约对象的数据源"。

正确的模型：

```
对方服务（vLLM / 云API / 私有服务）
   │ ① 获取接口定义：C++ SDK / OpenAPI / 文档
   ▼
[适配器开发者] —— 唯一要写代码的地方
   │ ② 实现 NetTransport：发出请求、收回响应
   │     · 变体 A：内部直接调对方 C++ 同步 SDK（SDK 内嵌在 transport 里）
   │     · 变体 B：裸协议实现（对方非 C++ SDK / 无 SDK，走 REST/gRPC/私有二进制）
   │ ③ 实现 NetCodec：请求 = 本地端口 → 对方报文；响应 = 对方报文 → 本地端口（校验形状规则）
   ▼
DCNet 契约对象（schema + transport + codec 注入）
   ▼
DCinfer 运行时 —— RunFn / NodeStatus / ErrorTracker，图级语义统一
```

要点：

- SDK / 接口文档是**翻译素材**；契约是**翻译器输出端**；运行时**零接触 SDK**；
- SDK 返回类型与 DC 端口（`Tensor` / `Data`）无自动映射，翻译代码必须写；
- SDK 是适配器的**实现细节**，契约保持 SDK 无关（可换实现不动契约）；
- SDK 异步/线程模型不得泄漏：DCinfer RunFn 在线程池阻塞式执行，适配器对运行时
  必须暴露**同步、线程安全**接口；
- SDK 的强依赖（重量级库、自有生命周期）是适配器级代价，不进核心。

### 2.4 ADR-6：I/O 执行上下文策略——默认不派生，核心桥 + transport 级可选 worker

**问题**：网络 I/O 特化算子，是否默认以子进程/子线程形式启动监听/常驻执行体，
以减轻"把异步 SDK 包装成同步接口"的负担？

**结论**：

1. **默认不派生任何子进程/子线程**：
   - RunFn 本就是阻塞契约，跑在 System 池 worker 上，阻塞被池吸收
     （Connector 节点即先例）；"为了不阻塞而开线程"是伪需求；
   - 节点单任务串行（`NodeException::Reentrant` 拒绝重入），"每算子一个 I/O 线程
     实现节点内并发"的收益不存在；多路复用的真正收益点在**多节点共享同一引擎
     实例**（`EngineRegistry` 按端点缓存 transport，全图共用连接）。
2. **异步包装负担在核心消除一次**：核心提供通用 async→sync 桥
   `DcNetSync::syncAwait`，适配器开发者不再手写 promise/condvar。
3. **transport 级可选 worker**：异步原生 SDK / gRPC / 流式 / 多路复用场景下，
   transport 内部可自持 I/O 线程 + 事件循环；对运行时仍呈现同步接口（契约不变，
   换策略不动 RunFn / 图 / 契约）。
4. **子进程仅限外来运行时 / 崩溃隔离**（FreeToken LocalSpawn 先例），非 DCNet
   出站算子默认形态。
5. **真正需要"监听"的场景**（`DCNet.Native` 接收端 / 本地代理端点）属服务端组件，
   单独设计，不属于出站算子职责（见 §2.5 ADR-7：节点服务化）。

判定矩阵：

| 场景 | 策略 | 为什么 |
|---|---|---|
| 简单 HTTP/JSON（OpenAI 兼容） | RunFn 内直接阻塞调用 | 池线程已吸收；I/O 线程零增益 |
| 异步原生 SDK / gRPC / 流式 / 多路复用 | transport 内常驻 I/O 线程（可选） | SDK 异步模型原生可用；连接跨节点复用 |
| Python / 外来运行时 / 崩溃隔离 | 子进程（FreeToken 先例） | 无法内嵌，进程即边界 |
| 附着既有服务（RemoteAttach） | 纯客户端，无任何监听 | 服务端在别处（本端被远程驱动的场景见 ADR-7 / M-server） |

### 2.5 ADR-7：服务端 / 入站组件（节点服务化）

**问题**：把本地 DCinfer 节点暴露为可被现有出站 `send→recv` 远程驱动的监听服务
（节点服务化），补齐分布式部署中「算力端被图端驱动」的闭环。

**结论**：单列独立服务端组件，不与出站协议（`DCNet.Native`）捆绑；以现有
`DCNet.Tensor`（HTTP + 张量/文本 JSON）承载，零新增依赖（POCO 已在）。
设计决策点裁决如下：

| # | 决策点 | 裁决 |
|---|---|---|
| 1 | 组件形态归属 | **独立服务端组件**：`DcNetListener` + `registerDcNetServerAdapter`（不注册 EngineDescriptor，不动出站契约，§8 只增不改）；ADR-6(5)「单独设计」落于此 |
| 2 | 鉴权分级 | **P1 仅 Bearer token**：`NetServerEndpoint::authToken` 非空时启用 Authorization 头校验（镜像出站 `authToken` 注入语义，裸 key / `Bearer` 前缀等价）；mTLS 与服务端证书配置后置 |
| 3 | `RemoteMalformed` 的 wire 取值 | **415**（未列举状态码 → 对端兜底 RemoteMalformed → ExecutionFailed + dcnet 诊断）；错误体 `malformed_frame` 仅为诊断细化 |
| 4 | `bind` 错误出口 | **配置期抛 `NodeException`**（对齐 `createEngine` 先例与 DESIGN.md §6「配置/编译期」约定）；start 后运行期错误不抛出，一律 wire 应答（不崩溃、不静默丢弃） |
| 5 | RunContext 生命周期 / 并发隔离 | **一请求一节点实例**（`EngineRegistry::createNode` 每请求构造，实例级隔离）；引擎实例按 `engineType + localModelRef` 缓存复用，本地执行互斥串行（引擎单任务语义）；**server codec 不暴露 `RunContext`**，以「端口名 ↔ 张量」为界 |
| 6 | 装配入口命名 | 采用 **`registerDcNetServerAdapter(reg, DcNetServerAdapterDesc)`**；本地模型标识命名 **`localModelRef`**，避免与 modelPath=远端端点的全局约定冲突 |
| 7 | 服务端配置结构 | **派生独立结构 `NetServerEndpoint`**（listenHost/port/basePath/requestPath/authToken/backlog/maxInFlight/requestTimeout），不复用出站 `NetEndpoint` 全套 |

**wire 逆向映射原则（语义一致性）**：归一化两段式的第一段由服务端产出 wire
应答；本地执行结果状态 → HTTP 状态码的映射归核心统一维护（ADR-4），即
`wireHttpStatusFor` / `wireCodeFor`（§6.1），使对端 `normalizeHttpResponse` 归一化
结果等于该失败在本地执行时的 status。鉴权 401/403、过载 429、wire 级垃圾报文
415 无本地对应物，由监听/装配层直接应答（§6.1）。

**已知解析限度**：非鉴权 `InternalError` 无忠实 wire 表示（对端 401/403 →
RemoteAuth → InternalError 仅覆盖鉴权路径；未列举状态码 → RemoteMalformed →
ExecutionFailed），按「本地执行失败 → 5xx」应答 500 → 对端 ExecutionFailed。
若集成实测要求严格一致，可选扩表方案：新增已知错误体 code → RemoteMalformed
（finalize 为 ExecutionFailed + dcnet 诊断）——**暂不采纳**。

**输入边界 schema 校验**：服务端在执行前对请求张量按节点本地 schema 校验端口名 /
类型 / 形状（-1 动态维），违例 → 400 → 对端 InvalidInput（§6.1；镜像出站
`decodeResponse`「校验本地形状规则」职责，§3.2/§3.4）。

---

## 3. 对接契约（核心交付物）

契约 = 4 个组成部分：传输抽象、协议映射、错误归一化、本地形状规则。

### 3.1 NetTransport —— 传输抽象

```cpp
struct DcNetTransport {
    virtual ~DcNetTransport() = default;
    /// 连接建立 / 就绪探测（createEngine 时调用；失败抛 NodeException）
    virtual NetError connect(const NetEndpoint&) = 0;
    /// 发送请求载荷（阻塞，遵守超时；失败返回非 Ok 的 NetError）
    virtual NetError send(const Payload&) = 0;
    /// 接收响应载荷（阻塞，遵守超时；失败时 remoteDetail 保留原始错误报文）
    virtual NetError recv(Payload&) = 0;
    /// 健康判定（进程存活 / 心跳 / 连接可用）
    virtual bool alive() const = 0;
    /// 释放连接与资源（releaseEngine 调用）
    virtual void close() = 0;
};
```

契约约束（ADR-6）：

- **对运行时永远呈现同步接口**；I/O 线程 / 事件循环 / 子进程全部是 transport
  **内部实现细节**，换策略不动 RunFn / 图 / 契约；
- 需要承载异步 SDK 时，transport 内部可自持 I/O 线程，配合核心桥
  `DcNetSync::syncAwait`（§3.5）将完成回调转为 RunFn 所在线程的阻塞等待；
- 返回的 `NetError` 须已归一化（`localStatus/localMessage` 已填，§3.3）。

### 3.2 NetCodec —— 协议映射

```cpp
struct DcNetCodec {
    /// 本地端口 → 对方请求报文（读 ctx.peek(port)，拼请求体）
    virtual Payload encodeRequest(const Node::RunContext&) = 0;
    /// 对方响应报文 → 本地端口（校验本地形状规则后 ctx.output）
    virtual void decodeResponse(Payload&, Node::RunContext&) = 0;
};
```

错误提取/归一化**不属 codec 职责**：HTTP 场景由 transport 在 `recv` 时按状态码 +
报文调用核心归一化函数（`NetError.h` 的 `normalizeHttpResponse` /
`normalizeRemoteBody`）；协议特有错误码需要精确映射时，codec 内部调用同一组
核心函数即可，分类规则仍归核心统一维护。

### 3.3 NetError —— 归一化中间结构

```cpp
struct NetError {
    NetErrorCategory category;  // None/Timeout/Unreachable/RemoteRejected/RemoteAuth/
                                // RemoteRateLimited/RemoteServer/RemoteMalformed/Other
    std::string code;           // 对方原始错误码（如 "invalid_api_key"）
    bool retryable;             // 超时/5xx/429 → true；4xx → false
    std::string remoteDetail;   // 对方原始报文摘要（保留回溯现场）
    // 契约保证的映射出口（由 DCNet 核心统一计算，适配器不自行映射）：
    Node::Status localStatus;   // → InvalidInput / ExecutionFailed / InternalError
    std::string localMessage;   // → Node::Result.message + ErrorTracker
};
```

**归一化原则：只做"翻译"，不做"发明"。** 图级语义（status）与本地引擎节点一致；
网络/远端细节全部收进 message 与诊断，不新增图级状态。完整映射表见 §6。

### 3.4 本地形状规则（端口 Schema）

复用 `Node::Port` 既有能力（`DCinfer/include/Node/Node.h`），由
`getInputPorts/getOutputPorts` 返回静态端口表：

- 静态形状：`NodePort::in<float>("data", {1,3,224,224})`
- 动态维度：形状中直接写 `-1`（`Tensor::Shape = vector<int64_t>`，与 ONNX 动态维
  语义一致；DCIr 序列化已 int64_t 直通，见 `DCIr/include/Ir/GraphCompiler.h`）
- 形状锚定：`NodePort::anchored<T>("out", "in_x")`（输出形状跟随输入端口）
- 可选端口 + 默认值：`NodePort::optional<T>(...)`
- 文本/JSON 载荷：`TensorType::Data` 约定（typeSize=1，UTF-8 字节；`SlotType`
  校验器不比对 typeSize，文本通路安全——FreeToken DESIGN.md §7 已核实）

```cpp
// 示例：OpenAI 兼容 chat 算子的本地形状规则
static Node::Schema chatSchema() {
    Node::Schema s;
    s.inputs  = { NodePort::in<std::vector<char>>("prompt"),
                  NodePort::optional<std::vector<char>>("system", {}),
                  NodePort::optional<std::vector<char>>("params", {}) };
    s.outputs = { NodePort::out<std::vector<char>>("response"),
                  NodePort::optional<std::vector<char>>("usage", {}) };
    return s;
}
```

### 3.5 NetSync —— 核心 async→sync 桥（ADR-6）

```cpp

/// submit 内可将任务投递到 transport 自持的 I/O 线程 / 事件循环。
template <typename R>
R DcNetSync::syncAwait(const std::function<void(std::function<void(R)>)>& submit);
```

### 3.6 服务端契约（M-server；ADR-7）

服务端组件 = 4 个部分：监听端生命周期、服务端协议映射、服务端端点配置、装配入口：

```cpp
struct DcNetListener {                       // 监听端生命周期（DcNetTransport 的服务端镜像）
    virtual void bind(const NetServerEndpoint&) = 0;   // 配置期；失败抛 NodeException（ADR-7）
    virtual void start(RequestHandler) = 0;  // 内部自持 I/O 线程 / accept 循环（ADR-6(3)）
    virtual void stop() = 0;                 // graceful drain（受 requestTimeout 约束）
    virtual bool alive() const = 0;          // 服务端健康镜像
    virtual int port() const = 0;            // 实际端口（port=0 时 bind 后回读）
};

struct DcNetServerCodec {                    // 服务端协议映射（载荷复用 NetCodec_Tensor）
    // 报文 → 本地输入端口张量映射；抛异常 = wire 级垃圾报文（→ 415；ADR-7：不暴露 RunContext）
    virtual std::unordered_map<std::string, Tensor> decodeRequest(const Payload&) = 0;
    virtual Payload encodeResponse(const std::unordered_map<std::string, Tensor>&) = 0;
    virtual std::string requestPath() const; // 与出站 DcNetCodec::requestPath 对称
};

std::shared_ptr<DcNetServerService>          // 注册并启动「节点服务化」监听端
registerDcNetServerAdapter(EngineRegistry& reg, DcNetServerAdapterDesc desc);
// desc：engineType + localModelRef + codec + NetServerEndpoint
// 请求路径：decodeRequest → createNode（一请求一实例）→ setInput → tryExecute
//         → collectOutputs → encodeResponse；失败经 wireStatusFor 逆向映射（§6.1）
```

闸门顺序（§6.1）：accept → 读请求 → 过载 429（`maxInFlight`，探测连接不入计）
→ 方法 405 → 鉴权 401 → 路径 404 → 业务处理。请求体仅支持 Content-Length
（不支持 chunked）；逐请求应答后关闭。

---

## 4. 模块结构（框架 + 适配器）

```
┌─ DCNet 核心（内置，稳定契约）─────────────────────────────┐
│  · 契约接口：NetTransport / NetCodec / NetEndpoint / NetSync │
│  · 数据格式：张量 JSON（数值 base64 / Data 文本 UTF-8）    │
│  · NetError 归一化（远端报文 → 本地标准报错）              │
│  · 端口 Schema 声明辅助（本地形状规则）                    │
├─ 内置传输实现（传输层，与协议无关）───────────────────────┤
│  · NetTransport_Http   HTTP/1.1（POCO，跨平台，keep-alive） │
├─ 协议级适配器（DCEngines，具体后端）──────────────────────┤
│  · DCEngine::OpenAI    OpenAI 兼容 /v1/chat/completions    │
│  · 开发者自定义：实现 NetTransport + NetCodec → registerDcNetAdapter │
└────────────────────────────────────────────────────────────┘
```

| 层 | 何时用 | 代价 |
|---|---|---|
| `DCNet.Tensor`（HTTP 传输 + 张量/文本格式） | 对方是任意"张量进/张量出"的远端服务 | 对方须接受我们的 JSON 报文格式 |
| `DCEngine::OpenAI`（协议适配器，DCEngines） | 对方是 OpenAI 兼容服务 | 文本级 Data 端口，无 tensor 保真 |
| 自定义适配器 | 对方是私有协议 / 云 API | 我们写几百行适配，对方零改动 |

关键判断：**"对方不是 DCinfer"从来不是问题——对接责任始终在我们这一侧，且每个
协议只写一次**。

---

## 5. 与 EngineDescriptor 的衔接

每个协议族注册一个 engineType（如 `"DCNet.Tensor"` / `"OpenAI"`），组装 `EngineDescriptor`（执行钩子经 `ExecutionPhases phases` 嵌套携带；相位契约：preRun → RunFn → synchronize → postRun，任一相位失败 → onError）：

| 钩子 | 实现 |
|---|---|
| `createEngine(modelPath)` | `modelPath` 即远端端点（URL / host:port）；创建 transport + 连接 + 就绪探测；失败抛 `NodeException`（配置期错误） |
| `getInputPorts/getOutputPorts` | 返回静态本地形状规则表（§3.4），不依赖远端推导 |
| `factory` | 构造节点：`ThreadPoolAffinity::System` + RunFn + 绑定实例 |
| `converter` | 不需要（文本经 `TensorType::Data` 承载）；tensor 级原生协议另行评估 |
| `RunFn` | 见 §5.1 请求流程 |
| `synchronize` | 留空（阻塞式 HTTP 同步返回） |
| `preRun` | 留空 |
| `postRun` | 留空（响应已在 RunFn 内解析） |
| `onError` | 留空 |
| `releaseEngine` | 留空（transport 随实例共享句柄析构释放连接） |

### 5.1 RunFn 请求流程

1. `ctx.engine()` 取 transport（缺失直接失败）
2. codec.encodeRequest：读输入端口 → 拼对方请求报文
3. transport.send / transport.recv（超时控制）
4. 非成功 → NetError → 核心归一化 → `ctx.failure(...)`
5. codec.decodeResponse：校验本地形状规则 → 写输出端口
6. `ctx.success()`；耗时等指标按需输出

---

## 6. 错误归一化设计（映射表）

`NetError.cpp` 内的纯函数映射（无 I/O，可单测）：

| 错误来源 | category | retryable | localStatus | 消息前缀 |
|---|---|---|---|---|
| 超时 | Timeout | true | `ExecutionFailed` | `net:timeout` |
| 连接拒绝 / DNS / 重置 / TLS | Unreachable | true | `ExecutionFailed` | `net:unreachable` |
| HTTP 400 / 422 | RemoteRejected | false | `InvalidInput` | `remote:invalid_request` |
| HTTP 401 / 403 | RemoteAuth | false | `InternalError` | `remote:auth` |
| HTTP 404 | RemoteRejected | false | `InvalidInput` | `remote:not_found` |
| HTTP 408 | Timeout | true | `ExecutionFailed` | `net:timeout` |
| HTTP 429 | RemoteRateLimited | true | `ExecutionFailed` | `remote:rate_limited` |
| HTTP 5xx | RemoteServer | true | `ExecutionFailed` | `remote:server_error` |
| 远端错误体 `{"error":{code,message}}` | 已知 code 精确映射（如 `invalid_api_key`→Auth、`rate_limit_exceeded`→RateLimited）；未知 code 按类别兜底 | 见 code | 见映射 | `remote:<code>` |
| 报文不可解析 | RemoteMalformed | false | `ExecutionFailed` | `remote:malformed` |

出口统一：

- 运行期：`Node::Result{status, message}`（`NodeStatus` 定义见 `Node.h`）；
- 诊断：`ErrorTracker::recordError(taskId, nodeName, source, message)` 保留远端原始
  报文与本地归一化结果的对应关系；
- 配置/编译期：非法端点、缺 transport/codec → 抛 `NodeException` / `GraphException`
  （携带 ErrorType 枚举，见 `NodeException.h`）。

### 6.1 入站 wire 逆向映射（M-server；两段式的第一段）

服务端把本地执行结果状态翻译为 wire 应答，使对端出站节点经 `normalizeHttpResponse`
归一化后得到与本地执行一致的 status（语义一致性）。映射归核心统一维护
（ADR-4）：`wireHttpStatusFor` / `wireCodeFor`（`NetError.h/.cpp`），与 §6 正向表逐行
对偶：

| 本地执行结果 | wire 应答 | 对端归一化（§6 表） | 错误体 code |
|---|---|---|---|
| `Ok` | 200 | 2xx 直接成功 | （无） |
| `InvalidInput` | 400 | RemoteRejected → `InvalidInput` | `invalid_input` |
| `SchemaMismatch`（预留行，本地当前不产出） | 422 | RemoteRejected → `InvalidInput` | `schema_mismatch` |
| `ExecutionFailed` | 500 | RemoteServer → `ExecutionFailed` | `execution_failed` |
| `InternalError` | 500 | RemoteServer → `ExecutionFailed` | `internal_error` |

无本地对应物的失败由监听/装配层直接应答（不参与「远程 == 本地」对拍）：

| 入站失败 | wire 应答 | 对端归一化 |
|---|---|---|
| 鉴权失败 | 401 / 403 | RemoteAuth → `InternalError`（`remote:auth`） |
| wire 级垃圾报文（codec decodeRequest 抛异常） | 415（未列举，[ADR-7 #3]） | RemoteMalformed → `ExecutionFailed`（`remote:malformed`） |
| 过载（在途超 `maxInFlight`） | 429 | RemoteRateLimited → `ExecutionFailed`（retryable） |
| 未知路径 / 非 POST | 404 / 405 | RemoteRejected → `InvalidInput` / RemoteMalformed → `ExecutionFailed` |
| 请求中止（对端已断开） | 无应答 | 对端自行归一化 `net:timeout` |

> **解析限度（ADR-7）**：非鉴权 `InternalError` 无忠实 wire 表示（500 → 对端
> ExecutionFailed），按「本地执行失败 → 5xx」采纳；严格一致的扩表方案
> 见 ADR-7，暂不采纳。
> **SchemaMismatch 备注**：本地当前无产出点（`Node.h` 预留），对端按 422
> 归一化为 InvalidInput 与本地形状违例现行行为一致；本地改产后本表无需变更
> （`SchemaMismatch` 与 `InvalidInput` 同映 422/400 → InvalidInput，若需区分再扩表）。
> 建议 `NetErrorTest` 的入站类用例与出站映射表同构维护（已落地：`wireRoundTripParity`）。

---

## 7. 并发模型

执行上下文（ADR-6）：默认无任何派生——RunFn 直接阻塞调用 transport；需要时
transport 内部自持 I/O 线程 / 事件循环（异步 SDK、流式、多路复用场景），
对运行时仍是同步接口；子进程仅用于外来运行时 / 崩溃隔离（FreeToken 先例）。

- 节点归属 `ThreadPoolAffinity::System`（README：System Pool 承担 I/O、网络传输），
  阻塞式 HTTP 调用天然适合；
- 适配器内不加锁：连接池线程安全由 transport 实现保证（如 libcurl easy handle
  每线程独立创建 / 互斥保护共享连接）；
- 并发上限控制用既有 `registerGroupLimit(tag, N)`（组信号量跨池共享、全局互斥），
  适配器不自行串行化；
- transport 内部 `std::mutex` 仅保护状态字段（连接、健康、重连计数），不保护请求路径。

---

## 8. 模块结构与文件布局

```
DCNet/
├── CMakeLists.txt                 # 静态库 DCNet + DCNet::DCNet（构建接线见 §9）
├── DESIGN.md                      # 本文档
├── include/DCNet/
│   ├── NetAdapter.h               # 公共注册入口：registerDcNetAdapter / DcNetAdapterDesc
│   ├── NetEndpoint.h              # 端点配置（host/port/protocol/timeout/retry/auth/TLS）
│   ├── NetError.h                 # NetError 分类枚举 + 归一化声明
│   ├── NetTransport.h             # 传输抽象接口（§3.1）
│   ├── NetCodec.h                 # 协议映射接口（§3.2，含 requestPath/schema 声明）
│   ├── NetSync.h                  # 核心 async→sync 桥（§3.5）
│   ├── NetTransport_Http.h        # 内置 HTTP transport（POCO，跨平台）
│   ├── NetCodec_Tensor.h          # 内置数据格式工厂：张量/文本 JSON codec
│   ├── DcNetHttp.h                # registerDcNetHttp 接线（HTTP 传输 + 任意 codec）
│   ├── MockServer.h               # 测试基础设施：极简 mock HTTP 服务（POCO，跨平台）
│   ├── NetServerEndpoint.h        # 服务端监听端点配置（M-server，ADR-7）
│   ├── NetServerCodec.h           # 服务端协议映射接口（M-server，ADR-7）
│   ├── NetListener.h              # 监听端生命周期抽象（M-server，ADR-7）
│   └── NetServerAdapter.h         # registerDcNetServerAdapter 接线（M-server）
├── src/
│   ├── NetAdapter.cpp             # 组装 EngineDescriptor（§5）
│   ├── NetEndpoint.cpp            # 端点描述串解析（URL → host/port/basePath/TLS）
│   ├── NetError.cpp               # 归一化映射表（纯函数）
│   ├── NetTransport_Http.cpp      # HTTP 后端（POCO；单一实现覆盖 Windows/POSIX）
│   ├── NetCodec_Tensor.cpp        # 张量 JSON codec（数值 base64 + Data 文本直传）
│   ├── DcNetHttp.cpp              # registerDcNetHttp 接线
│   ├── NetBase64.h                # 内部 base64 工具（仅头）
│   ├── NetWire.h                  # 内部共享：入站 wire 错误体/状态短语（M-server）
│   ├── NetListener_Http.cpp       # HTTP 监听器（POCO ServerSocket；闸门/drain）
│   └── NetServerAdapter.cpp       # registerDcNetServerAdapter 装配（M-server）
└── test/
    ├── NetErrorTest.cpp           # 映射表纯单测（含入站 wire 逆向映射 §6.1）
    ├── NetAdapterTest.cpp         # 契约实现测试（FakeTransport，不依赖真实远端）
    ├── ServerAdapterTest.cpp      # 节点服务化端到端：本地执行 vs 远程驱动对拍 + 闸门
    └── HttpTransportTest.cpp      # 真实 HTTP：传输/归一化/张量/文本端到端
```

协议级适配器（基于 DCNet 契约开发，位于 DCEngines）：

```
DCEngines/OpenAI/
├── include/DCEngine/OpenAiEngine.h   # registerOpenAiEngine / OpenAiOptions
├── src/OpenAiEngine.cpp              # OpenAI 兼容 chat codec（NetCodec 契约）+ 注册接线
├── test/OpenAiEngineTest.cpp         # chat 端到端（MockHttpServer，复用 DCNet 测试设施）
└── CMakeLists.txt                    # 静态库 DCEngine_OpenAI（DCEngine::OpenAI）
```

---

## 9. 依赖与构建接线

**依赖现状**：`nlohmann-json` 已在核心依赖树中——JSON 编解码
零新增；`poco[netssl]` 承载 HTTP/TLS 传输与测试 MockServer。

**CMake 接线**：根 `CMakeLists.txt` 以 `_dcinfer_option()` 注册
`DCINFER_BUILD_DCNET`（旧 `BUILD_DCNET` 兼容映射；默认 OFF，与 vcpkg feature
'net' 对应）；`DCNet/CMakeLists.txt` 提供静态库 `DCNet::DCNet`（依赖
`DCinfer::DCinfer` + `nlohmann_json` + POCO，含安装导出），测试随
`DCINFER_BUILD_TESTS` 启用。

DCIr 兼容：DCNet 节点是普通引擎节点（`engineType` 已注册），`modelPath` 字段承载
远端端点，`GraphCompiler` JSON/.dcg 序列化无需改动（引擎节点路径已支持）。

**外部消费注意事项**：

- vcpkg manifest 模式只安装清单声明的包：外部 `add_subdirectory` 消费方必须在
  自己的 `vcpkg.json` 声明 `nlohmann-json` 与 `poco[netssl]`（DCNet 及
  DCEngine::OpenAI 的 `find_package` 依赖），否则配置失败；
- 不消费 DCNet 的零依赖消费方（如 DCinfer-test 的 `smoke/`）应显式
  `set(DCINFER_BUILD_DCNET OFF CACHE BOOL "" FORCE)`，避免拉入 nlohmann-json
  （`BUILD_ENGINE_OPENAI=ON` 缺 `DCNet::DCNet` 时 CMake 直接 FATAL_ERROR）；
- 外部引用目标：`DCNet::DCNet`（静态库 alias，同构建树可用）；协议适配器
  `DCEngine::OpenAI`（DCEngines/OpenAI，依赖 `DCNet::DCNet`）；
- 外部开发者接入"对方服务"的完整最小范例见 `DCinfer-test/src/net_smoke.cpp`
  （自定义 transport + codec → `registerDcNetAdapter` → `createNode` 组图）；
  直接对接 OpenAI 兼容服务用 `DCEngine::OpenAI` 的 `registerOpenAiEngine`。

---

## 10. 测试策略（不依赖真实远端）

1. **MockServer**：`DCNet/include/DCNet/MockServer.h`（C++ 内嵌极简 HTTP 服务，
   POCO，跨平台；DCNet 与 DCEngines 适配器测试共用），实现 `/v1/infer` 与
   `/v1/chat/completions` 面，可注入错误响应（5xx / 错误体 / 畸形报文 / 超时）；
2. **单元测试**：
   - `NetErrorTest`：映射表纯单测（每类网络错误 / HTTP 状态 / 远端错误体 → 期望的
     category / retryable / localStatus / 消息前缀）；
   - 端口 Schema：文本/数值 Tensor 编解码往返；形状规则校验（含 -1 动态维、anchored）；
   - 契约实现：MockServer + transport + codec → RunFn 输出断言
     （`HttpTransportTest` 张量/文本端到端；`OpenAiEngineTest` chat 端到端）；
   - 服务端/入站（M-server）：`ServerAdapterTest` 端到端——本地执行 vs 远程驱动
     逐项比对（status + 输出值）、鉴权 401 / wire 级垃圾报文 415 / schema 违例
     400 / 过载 429 / 404 闸门、生命周期与配置期错误；`NetErrorTest` 入站类
     （`wireRoundTripParity`）与出站映射表同构维护；
3. **集成测试（可选，需真实环境）**：连真实 OpenAI 兼容服务验证端到端
   prompt → response（`DCEngines/OpenAI`），标记可选，不在默认 CI 中。

---

## 附：设计决策记录

| 编号 | 决策 | 备选 | 理由 |
|---|---|---|---|
| ADR-1 | 契约内置、协议外置；原生协议可选 | 框架自带协议 / 纯契约 | 不锁死生态；版本责任隔离；架构一致性 |
| ADR-2 | EngineDescriptor 形态 | registerOperator | 连接生命周期需实例状态；与 ONNX 形态一致 |
| ADR-3 | 双向翻译器心智模型 | SDK 输出即数据源 | SDK 非即插即用；契约须 SDK 无关 |
| ADR-4 | 错误归一化归核心统一执行 | 各适配器自行映射 | 图级语义保证一致；纯函数可单测 |
| ADR-5 | System affinity + 阻塞式 + 不加锁 | 自建并发控制 | 复用既有线程池与限流；与 README 分工一致 |
| ADR-6 | 默认不派生；核心 async→sync 桥；transport 级可选 worker；子进程仅限外来运行时 | 默认子进程 / 默认子线程 | 池已吸收阻塞；节点串行（Reentrant）；契约保持同步接口 |
| ADR-7 | 服务端/入站组件单列 M-server（节点服务化）：独立服务组件、一请求一节点实例、wire 逆向映射归核心 | 与原生协议阶段（DCNet.Native）捆绑 | 只增不改；语义一致性需映射归核心统一维护 |
