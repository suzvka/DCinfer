# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.2] - 2026-09-16

### Fixed

- **CORE-01（High）同一输出端口二次 `connect()` 静默丢数据 + 任务永久挂起**：
  新增构图期拦截——`(srcNode, srcPort)` 已有出边时抛
  `GraphException(DuplicateEdge)`（新增错误类型），错误消息指引正确姿势；
  1:N 分发必须显式创建 `Connector.Broadcast(N)`（README 增补示例与 FAQ）。
- **IR-01（High）两节点共享同一模型文件时 .dcg 序列化互相覆盖**：序列化时
  按源路径复用同一 archive 名（模型只入包一份、所有引用节点写回同一相对
  路径）——共享权重场景不再产生悬空引用导致必然编译失败。
- **CORE-02 `Tensor::View` 破坏 const 正确性**：`const Tensor` 的
  `operator[]` 改回返回只读 `ConstView`（仅链式索引 + read/readScalar），
  删除 `const_cast` 构造——const 路径的写入在编译期被禁止。
- **CORE-03 `TensorData` 双参构造除零 UB**：shape 乘积为 0（含 0 维）时
  前置抛 `std::invalid_argument`。
- **CORE-04 `exportNode` 悬垂契约**：新增子图生命周期哨兵（weak 检测）——
  子图先析构再执行导出节点从段错误降级为显式 `ExecutionFailed`；
  `exportNode` 注释与 README 显著标注生命周期契约（best-effort 检测，
  不替代契约）。
- **IR-02 .dcg 反序列化安全不变量旁路**：object 形状 `nodes` 显式拒绝；
  受限模式下所有节点 `modelPath` 经 PathGuard 校验（拒绝 `../`、绝对路径、
  盘符等越界声明）——.json（本地可信输入）保持宽松语义。
- **IR-03 PathGuard 未拒 Windows 罪名字形**：新增拒绝保留设备名
  （CON/PRN/AUX/NUL/COM1-9/LPT1-9，含扩展名形式）、ADS 冒号、尾点尾空格。
- **IR-04/05 zip 预算单条目化**：新增解包条目数上限（256）、归档全局条目数
  上限（4096）、累计解压总量预算（4 GiB）与 graph.json 专用上限（64 MiB）；
  `readGraphJson` 预分配不再信任 ZIP 声明体积。
- **IR-06 >4 GiB 模型静默截断**：`addModelFile` 改为 64 KiB 分块流式写入
  （不再整模型读入内存），超过 minizip 单条目 unsigned 上限（4 GiB）的
  文件显式拒绝（不再静默截断）。
- **IR-07 连接失败 fail-open**：`rebuildEdges` 连接失败从 stderr 告警继续
  改为直接抛 `GraphException`（fail-fast，不再产生孤儿连接器/残缺图）。
- **IR-08 typeSize 负数穿透**：端口 `typeSize` 负值（穿透为 SIZE_MAX）在
  反序列化期显式拒绝；0（Void 不校验语义）保持合法。

### Changed

- `connect()` 行为修正：同一输出端口二次调用从“静默丢数据”改为抛
  `GraphException(DuplicateEdge)`；反序列化畸形边（无效端口/未知节点）
  从“告警 + 图残缺”改为编译失败。
- .dcg 反序列化收紧：`nodes` 必须为数组；`modelPath` 必须为解压目录内的
  安全相对路径（.json 不受影响）。
- `Tensor::operator[]` const 重载返回类型由 `View`（可写）改为
  `ConstView`（只读）——const 张量的只读用法兼容，写入用法编译期报错。
- 回归测试增补：`GraphNodeTest`（DuplicateEdge）、`TensorTest`/`TensorDataTest`
  （ConstView/除零）、`NestedGraphCancelTest`（子图生命周期哨兵）、
  `GraphCompilerTest`（共享模型/.dcg 校验/fail-fast/typeSize）、
  `DcgArchiveSecurityTest`（罪名字形/聚合预算）。

### Notes

- 明确边界：不支持 >4 GiB 的单个模型文件入库（zip64 写入未启用），超限
  显式拒绝。
- 本次未包含 DCNet/DCEngines 审查项（NET-01~05、ENG-01~04）的修复。

## [0.5.1] - 2026-09-16

### Fixed

- **B-1 `Tensor::getData<T>()` 堆越界写（内存安全）**：此前 `sizeof(T) < typeSize()`
  时按整块字节数 `memcpy` 进按元素数分配的缓冲区即溢出（如 float 张量取
  `getData<uint8_t>()`）。现改为"字节堆按 T 重解释"语义：容量与拷贝均以
  `sizeof(T)` 为基准，元素数 = ceil(总字节/sizeof(T))（末元素零填充、全部字节
  无损），不做类型校验；`typeSize()` 退出计算（同时消除空张量的除零）。
- **H-1 多输入节点双触发 / 节点闸竞争伪失败**：TaskBuffer 写入与就绪判定合并
  到单一临界区（`setInputAndCheckReady`），并发传播仅"最后写入者"触发提交；
  节点执行闸从"拒绝即错"改为排队重投（`enqueueRetry` + 释放时全量投递）——
  共享图上并发任务的节点竞争败者经重投执行，不再被 Reentrant 记为 Error 判死；
  NotReady 降级为 Warning 跳过。
- **H-2 非 NodeException 逃逸导致任务永久 Running**：引擎调度层 catch 扩展为
  NodeException 分流 + `std::exception`/`...` 统一记录 Error 诊断，由耗尽检测
  收束 Failed；执行流水线闸租约改为 RAII（异常路径含完成回调二次抛出不再泄漏租约）。
- **H-3 终态发布先于收尾完成（跨轮污染）**：同 ID 复用/释放准入收紧为
  "终态 + 结果可读（waitForResult 返回）"；收尾窗口内 submit / releaseTask /
  detachTask / feedInput 一律拒绝（detach 登记自动回收）——旧轮结果抢救与
  执行态清理不再污染新轮。
- **H-5 并发同 ID 提交 TOCTOU**：提交改为引擎侧单临界区原子事务（准入检查 +
  声明清理/写入 + 执行态捕获 + 轮次登记），并发同 ID 恰一方成功、败者
  DuplicateTask 零副作用。
- **H-4 嵌套子图无限期挂起 / 取消不跨边界**：exportNode 的子图任务 ID 改为
  父任务 ID（taskId 空间贯穿父子边界）；RunFn 分段等待（100ms）轮询父轮终止，
  父任务取消/收束/TTL 后主动取消子图任务并解围返回——不再永久占住父池线程。

### Changed

- `ExecutionEngine::submit` 签名携带输出声明参数（`declarations`）——声明清理/
  写入并入提交事务（内部 API，InferGraph 同步迁移）。
- 任务完成回调约束补充：回调内不得 submit / feedInput / releaseTask /
  detachTask（收尾窗口内均被拒绝）；新一轮提交请在 waitForResult 返回后进行。
- `Node::RunContext` 新增 `taskId()` / `isCancellationRequested()`（协作式取消
  感知，供长等待节点轮询解围；单节点路径恒 false）。
- 新增回归测试：`ExecutionConcurrencyTest`（H-1/H-2）、`NestedGraphCancelTest`
  （H-4）；`TaskLifecycleRegressionTest` 增补收尾窗口拒绝 / 弃置自动回收 /
  并发同 ID 提交用例；`TensorTest` 增补 getData 字节重解释用例。

## [0.5.0] - 2026-09-16

### Added

- **类型化输入访问器 `Node::RunContext::input<T>()`**：组合 peek + 类型标签校验 +
  空值检查，替代手写 "peek → as<T> → 判空" 三步样板——失败返回 nullptr 并可选
  输出原因（端口不存在 / 数据未到达 / 类型不匹配 / 空值）；类型不匹配从
  `Value::as<T>()` 的未定义行为降为可处理的失败。内置算子（Builtin）、
  OnnxRuntime / OpenAI 适配器已迁移为按此写法。
- **自定义节点教程示例 `examples/03_custom_node`**：NodePort 工厂声明 Schema →
  `ctx.input<Tensor>` 类型化读取 → `registerOperator` 注册 → 建图执行的完整
  可运行教程；README 新增"常见问题"段（自定义节点入口 + 任务资源释放）。
- **图公开接口 `GraphInterface`（按别名喂数据 / 取结果）**：`graph.interface()`
  冻结图并一次性解析公开绑定（alias → (nodeName, portName)）后返回；
  `api.createTask()` 取得任务句柄（析构按状态回收：终态即释放 / 在飞弃置即
  请求取消并回收 / 未提交即释放输入），
  `task.feed(alias, data)` / `task.run()`（同步）/ `task.take(alias)` 只按公开
  别名操作；未知别名抛 `GraphException(InvalidBinding)` 并列出全部可用别名；
  绑定坐标在创建接口时校验（NodeNotFound / PortNotFound）。内核寻址不变——
  `feedInput` / `takeOutput` 等仍唯一按 (nodeName, portName) 坐标。
- **统一任务句柄 `GraphInterface::Task` 同步/异步同级**：新增 `submit()`（异步
  启动，不等待）、`wait()` / `wait(timeout)`、`status()`、`cancel()`、
  `has(alias)`、`errors()` 与 `run(timeout)`；`feed` 支持链式返回 `Task&`。
  同步 `run()` 与异步 `submit + wait` 是同一句柄上的同级表达——宿主层仅一套
  API（两种节奏演示见 examples/04_task_lifecycle），无需 taskId / 坐标。
  原 `TaskScope` 的能力（超时 run / 取消 / 状态查询 / 结构化等待）全部并入。
- **任务生命周期回归测试 `TaskLifecycleRegressionTest`**（8 用例）：取消后复用同
  taskId 的旧轮次消费竞态、失败后输出、必需输出缺失、终止回调异常、活动任务
  释放拒绝（并发幂等）、未提交句柄析构释放输入、弃置即取消并回收、不可达提交
  回滚后重试——覆盖 F04–F10 各项缺陷的复现路径。
- **DCIr 归档安全回归测试 `DcgArchiveSecurityTest`**：路径校验单元（空/内嵌
  NUL/绝对路径/盘符/UNC/父目录跳转）、`extractOne` 越界写入拒绝端到端、
  符号链接祖先目录防御（无权限环境自动跳过）、截断归档明确报错、高压缩比
  条目预算拒绝、正常归档往返回归。
- **`InferGraph::detachTask` / `InferGraph::discardUnsubmitted`**：任务句柄析构
  路径配套——在飞任务弃置先请求取消（协作式），再经 detachTask 兜底回收
  （已终态立即释放；取消竞态窗口由完成收尾自动回收状态/结果/诊断）、
  已喂数据但从未成功提交的任务立即释放输入。

### Changed

- **任务句柄析构：“弃置即取消”（协作式）**：`GraphInterface::Task` 析构按下述
  三路回收——已终止 → 立即释放全部资源；在飞弃置 → 先请求 `cancel()`
  （不中断在飞节点执行，传播链在下一检查点停止），再回收；未提交 → 立即释放
  已喂入输入。取代原“弃置不取消、完成后自动回收”语义；长工作流下弃置任务不再
  空耗算力，资源随终态同步释放。外部经坐标 API（`submitBound`）提交的任务不受
  句柄析构影响。
- **OutputZone 搬运补充终止复查（检查点 2）**：传播第二步（输出搬运）在
  `takeOutput` / `append` 前复查 `round->terminated`，关闭与并发取消 / 同 ID
  复用相关的窄写入竞态窗口。
- **寻址模型定案（单栈）**：运行时数据注入与取用唯一按 `(nodeName, portName)`
  复合坐标寻址，无名称解析、无回退；删除 `feedBoundInput`（0.3.0 引入、
  0.4.0 收窄的别名寻址路径），`feedInput` / `takeOutput` / `takeOutputTensor` /
  `hasOutput` 统一仅按内部坐标寻址。`alias` 定位为图级签名的序列化/内省元数据，
  不参与运行时寻址（取代 0.4.0 条目中的别名寻址描述）。
- **破坏性 API 收窄**：移除 `GraphBuilder::store()` 非 const 重载（唯一使用点
  `InferGraph::node()` 改走 `GraphBuilder::node()`）。冻结前泄漏可变
  `GraphStore&` 的路径在编译期不可达；`store() const` 保留且冻结后返回快照持有的
  源图。
- **View 写入接口统一为 `set()`**：删除 `Tensor::View::item`——同名
  `Tensor::item<T>()` 为读取语义，View 侧写入方法造成读写命名不对称
  （全仓库零调用者，直接移除而非保留别名）；View 写入用 `set()` / `operator=`。
- **Schema 声明统一迁移 NodePort 工厂**：示例与测试中的手写聚合初始化
  （如 `{"x", Tensor::TensorType::Float, sizeof(float), {}}`）迁移为
  `Node::Port::in<T>` / `out<T>` / `optional<T>` 工厂形式，消除类型标签与
  sizeof 双写风险；DCIr 测试的通用 makePort 辅助删除（Void 端口无 C++ 类型
  载体，保留语义收窄后的 voidPort）。
- **`02_lowering_benchmark` 循环示范句柄回收**：大量短任务循环无需手工
  `releaseTask`——每轮局部任务句柄析构即自动回收资源；README 常见问题同步点名。
- **`releaseTask` / `waitForResult` 语义定案**：`InferGraph::releaseTask` 收窄为
  仅终态可释放——活动任务被拒绝、结果/诊断保持不动；`waitForResult` 等待未
  满足（含终态已迁移、结果尚未就绪的收尾窗口）一律如实报告
  `{status=Running}`，`wait` 返回成功才保证结果可读。

### Removed

- **内核坐标层作用域句柄 `TaskScope` 下线**：其能力已并入
  `GraphInterface::Task`（同步/异步同级）；examples/04_task_scope 迁往
  examples/04_task_lifecycle。

### Fixed

- **首次惰性冻结的数据竞争（冻结事务一次性发布）**：`GraphRuntimeState` 的冻结快照
  由"公开可变成员 + 无同步读写"改为发布协议——快照所有权私有（`_graph`，仅冻结
  线程写一次），全部派生状态（节点执行闸表）先完成，最后以 release 语义置位
  `_frozen` 发布；读取方经 `snapshot()` acquire 读，只能观察到"尚未发布（nullptr）"
  或"完整初始化"两种状态。此前 `_ensureFrozen` 快路径（以及 `_ensureNotFrozen` /
  `_topology` / 绑定视图 / `ExecutionEngine` 六处读取）在锁外读同一普通
  `shared_ptr`、锁内写——并发首次冻结构成 C++ 内存模型数据竞争；`attachGraph`
  先发布 `graph` 再初始化闸表的顺序还可能提前暴露未完成初始化的状态。现在
  并发首次 `freeze`/`feedInput`/`submit`（含 exportNode 子图由父图执行线程触发
  的首次冻结）恰执行一次初始化且无竞争。
- **冻结后仍可经泄漏引用修改拓扑/节点配置**：`Node` 新增冻结门——`compile()`
  封印全部节点（`_sealForExecution`），此后 8 个公开可变入口（`bindEngine` /
  `setTag` / `setConnector` / `bindSignal` / `setBlockedOverride` /
  `setReadyOverride` / `setModelPath` / `setCompletionCallback`）抛
  `NodeException(Frozen)`；`GraphStore` 新增封印（`seal`）——冻结后
  `addNode`/`connect`/`connectRaw`/`bindInput` 抛 `GraphException(Frozen)`。
  此前在冻结前保存的 `Node&` / `Node*`（含 `InferGraph::node()` 构建期可写指针）
  可在冻结后静默修改影响调度/执行的配置，与执行流水线读取构成竞争。
- **构建与冻结并发**：`GraphBuilder` 全部公开方法持内部互斥锁，构建 API 与
  `compile()` 串行化——每个构建操作要么先于编译完成（纳入快照），要么在冻结后
  确定抛 Frozen，消除"检查通过后被冻结插入"的 TOCTOU 窗口（原 `_ensureMutable`
  读 `_snapshot` 与 `compile` 写无同步）；`compile()` 内部改为先封印（拓扑 +
  节点配置）再只读遍历，签名构建 / lowering / 运行期读取不再可能与写入并发。
- **构建器冻结后内省崩溃**：`GraphBuilder` 的 `store()/node()/nodeCount()/
  edgeCount()/nodeNames()/edges()/inputBindings()/outputBindings()` 在
  `compile()` 后曾空指针解引用（注释误称"返回空/零"，实际 `_store` 已被移走）——
  现持锁并委托快照读取（源图视角与 facade `_topology` 一致）。
- **exportNode 端口唯一性校验（单栈命名法则）**：子图导出时端口名在接口层扁平化
  （父图按该名寻址），同名端口（跨节点同名/同一端口重复绑定）此前静默折叠至同一
  槽位（数据串扰）；现拒绝导出并抛 `GraphException(DuplicatePort)`，消息含端口名
  与来源绑定（`node.port`）。运行时复合坐标寻址在普通拓扑下天然消歧，仅此扁平化
  时刻需要接口端口名唯一。
- **任务生命周期竞态加固（F04–F10 批次）**：`ExecutionEngine` 的任务状态表与
  活动闸表合并为自包含"轮次"对象——提交（成功登记）时捕获本轮执行态，等待
  协议与状态发布共用同一把锁。修复：取消/终止后复用同 taskId 时，旧轮次在飞
  节点不再消费新轮次输入（F04）；节点失败优先于输出存在判定，无输出记
  InternalError 且不传播（F05）；终止回调异常被隔离、结果发布与唤醒由 RAII
  收尾保证完成（F06/F07）；`releaseTask` 仅终态可释放、并发幂等（F08）；
  句柄析构按状态三路回收（F09，见 Changed）；提交期拓扑守卫前移、登记失败
  无残留可重试（F10）。
- **DCIr 归档路径穿越与读取健壮性（F01 批次）**：`extractOne` / `compileFile`
  入口拒绝 `..` / 绝对路径 / 盘符 / UNC 条目（归一化后组件级前缀比对，防前缀
  误判），写入前拒绝符号链接祖先目录逃逸；zip 条目改 64 KiB 分块流式读取，
  累计字节与声明不符即报错、关闭时校验 CRC；新增单条目体积（1 GiB）与压缩比
  （200:1）预算拦截 zip 炸弹；临时解包目录随机化命名 + 独占创建 + 尽力设置
  私有权限（POSIX 0700）。
- **注册表并发加固**：`EnvRegistry` 容器加锁、factory/cleanup 钩子移出锁外
  调用（并发创建先到者胜出）；`ValidatorRegistry` register/find 加锁，契约
  明确为"启动期注册、运行期并发读取"；`EngineRegistry::releaseEngine` /
  `releaseAllEngines` 把待析构句柄移入局部容器、锁外释放——用户
  `releaseEngine` 钩子不再持锁运行。
- **引擎集成测试注册开关修复（F11）**：OpenAI / OnnxRuntime 测试目录此前以
  未定义的旧名 `BUILD_TESTS` 作开关，仅传规范名 `DCINFER_BUILD_TESTS=ON` 时
  测试静默不注册；改随规范开关启用。CI 同步：TSan 用例扩围至
  GraphInterfaceTest / TaskLifecycleRegressionTest，dcnet-openai 作业新增
  "OpenAiEngineTest 已注册"断言。

## [0.4.0] - 2026-09-10

### Changed

- **移除执行时间看门狗（时间语义归节点实现方）**：`submit` / `submitBound` 不再接受
  执行超时参数，`TaskStatus::TimedOut` 状态删除（破坏性 API 变更）。理由：通用执行
  引擎跨平台无一致的时钟语义，且模型运行时长只有节点实现方有解释权（如 DCNet 的
  `NetEndpoint::timeout` + `ctx.failure(ExecutionFailed, msg, Diagnostic)` 自报范式），
  调度器设定无意义的超时。配套语义补齐：
  - **节点失败闭环**：传播耗尽且存在 Error 级诊断时，任务终止为 `Failed`（不再依赖
    看门狗/挂起）。实现：`TaskGate` 增加在飞计数 `inflight`，节点执行 lambda 提交即
    +1、RAII 收尾 -1，归零触发 `_exhaustedCheck`——取代原“最后持有者析构”方案
    （活动门控表强持有 gate 至 `_terminate`，该析构路径实际不触发）；纯信号停滞
    （无 Error）保持挂起，由宿主 `waitForResult(timeout)` + `cancel()` 解围
  - **提交期拓扑守卫**：`submit` 时对本次声明做纯拓扑可达性检测（忽略信号阻断，
    避免误伤合法挂起构图），构图/断链错误无需再等待异步挂起与宿主兜底，
    在提交时刻即抛 `GraphException(UnreachableDeclaration)`；
    `SignalProbe` 新增 `canSatisfyTopologically`（纯拓扑版，原信号语义函数保留供
    exportNode 使用）
  - `waitForResult(taskId, timeout)` 保留但重新定位为**宿主护栏**：只放弃等待，
    不参与图时间语义、不取消任务
  - 删除 `TimerService`（引擎级共享定时器）及看门狗注册/失效/到点仲裁链；
    `_timerHandles` / `_timer` 成员移除，`ExecutionEngine` 析构顺序简化（定时器先停约束消失）
- **图级绑定统一为强制公共别名（消除同名重载的语义分叉）**：删除
  `bindInput(nodeName, portName)` / `bindOutput(nodeName, portName)` 两参重载，
  仅保留 `(alias, nodeName, portName)` —— 原两参/三参重载的第一参数含义不同
  （节点名 vs 别名），是隐性歧义签名。新增 `GraphException::InvalidBinding`
  （别名为空）；别名唯一性校验不变。连带收敛：`feedBoundInput` / `takeOutput` /
  `takeOutputTensor` / `hasOutput` 的 name 参数统一**仅按公共别名寻址**
  （此寻址语义已由上方 Unreleased「寻址模型定案（单栈）」取代：最终定案为仅按
  内部坐标寻址，`feedBoundInput` 已删除），
  删除“唯一绑定端口名回退”分支与跨绑定歧义 `FeedFailed` 错误。
  DCIr 序列化格式同步：绑定 JSON 新增 `alias` 字段，反序列化对旧文件回退
  `alias = portName`（行为等价于原回退路径）。
  迁移：`bindInput("adder", "a")` → `bindInput("a", "adder", "a")`（别名取端口名同名即可）
- **等待 API 统一为 `waitForResult` 单轨**：删除 `InferGraph::wait` 两个重载，
  仅保留 `waitForResult(taskId)`（默认无限等待）与 `waitForResult(taskId, timeout)`
  （超时未终止时 status 为 Running）。消除 `wait` 返回 bool 的双义
  （false = 超时还是 taskId 未知不可辨）与 `timeout=0` 在 `wait`/`submit` 中
  含义相反的隐性陷阱。`ExecutionEngine::wait` 保留（内部被 waitForResult 使用），
  `exportNode` 内部同步等待改走引擎内部路径
- **接线 API 收敛为 `connect` 单轨**：删除 `InferGraph::connectRaw` /
  `connectAll` 与 `GraphBuilder::connectRaw` / `connectAll`。`connect` 的
  “自动插入直通导线”语义覆盖全部常规构图；裸拓扑（纯 wire 链/环）仅存于
  lowering 验证，改由 `GraphStore::connectRaw`（标注 internal）+
  `buildRuntimeView` 单元级直测覆盖，`DirectConnect` 守卫不变量保留并有回归


- **EngineDescriptor 执行钩子收编 `ExecutionPhases`（接口契约显式化）**：`preRun` /
  `synchronize` / `postRun` / `onError` 四个平铺钩子收编为嵌套结构
  `EngineDescriptor::ExecutionPhases phases`——类型名承载"顺序即契约"的相位语义；
  公开头补齐成功/失败路径矩阵（任一相位失败 → onError，后续相位跳过）、
  逐钩子可空性与组合约束（postRun 依赖 synchronize 已执行）；修正 `preRun`
  注释越权承诺（"I/O 绑定"需要 task 输入访问，实际输入绑定发生在 RunFn 内经
  TensorConverter）。源级破坏：`desc.X → desc.phases.X`
  （createEngine / 端口推导 / factory / releaseEngine 保持顶层不变）
- **onError 覆盖任一执行相位失败**：原实现仅 RunFn 失败触发 onError，
  preRun/synchronize/postRun 自身抛出会绕过复位直通外层 catch。现统一经
  `safeTriggerOnError`（吞掉 onError 自身异常，不传播次生异常）触发复位后
  按原语义 rethrow / 返回 NodeResult；RunFn 失败路径行为不变。
  失败路径矩阵以 EngineRegistryTest Test 16–19 固化
- **超时看门狗 → 引擎级共享 TimerService**：`ExecutionEngine` 不再为每条带超时的
  submit 创建看门狗线程（原 100ms 轮询 `jthread` + per-task 注册/回收），改为单定时器
  线程 + deadline 最小堆：每任务仅登记一个 deadline 条目，终止路径 O(1) 作废、零 join。
  超时触发从 ~100ms 轮询粒度变为精确 deadline 唤醒；引擎析构时定时器线程先于
  线程池停止（原看门狗晚于池析构，存在池关闭期间触发超时的窗口）；
  每 task 成本从"一线程 + 周期轮询"降为"一个堆条目"。
  同 ID 复用后旧条目到点时经活动门控身份校验失配退出，不会误杀新任务
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
### Removed

- **`InferGraph::declareSubgraph` 与 tag 分组调度机制**：子图互斥能力由
  `exportNode`（独立引擎、部署边界）覆盖；连带摘除死代码链：
  `ExecutionEngine::registerGroupLimit`、`ThreadPool::registerGroupLimit`、
  `GroupSemaphoreRegistry` 跨池信号量表、`PoolConfig::groupLimits`、
  分组信号量获取/释放与活跃计数、worker 轮询退避（`kThrottledRetryInterval`）、
  `ThreadPool::submit` 的 nodeTag 参数与 `_dispatchToPool` 的 tag 参数。
  `Node::tag` 保留为纯序列化元数据（DCIr JSON/.dcg 的 tag 字段往返），
  无调度语义。纯 wire 环防挂起回归经 GraphStore 单元测试保留
- **`NodeExecutor::peekOutput` / `TaskBuffer::peekOutput`**：零调用者的
  非破坏式预览链路；消费式语义统一由 `takeOutput` 表达。图级 API 从未暴露
  预览入口（`InferGraph.h` 注释引用同步删除）

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

- **引擎创建异常在 GCC/Clang 上静默丢失（Linux CI 全红根因）**：
  `EngineRegistry::getOrCreateEngine` 失败路径先 `promise.set_exception(std::move(error))`
  再判定 `if (error)` 重抛——`std::exception_ptr` 移动后源对象按标准被置空
  （libstdc++ 严格执行，MSVC 实现宽松、移动后源仍非空，故 Windows 侥幸通过）。
  GCC/Clang 上领导者线程判定恒 false，创建异常不再透传（跟随者仍经 shared_future
  收到异常，更隐蔽）。改为拷贝进 promise（`set_exception(error)`）后用原 `error` 重抛。
  EngineRegistryTest Test 15 回归覆盖（此处自 03a8041 single-flight 重构引入，
  修复后 Linux GCC/Clang 与 Windows MSVC 基础测试均 13/13 通过）
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
  （注：此别名寻址语义为 0.3.0 历史行为，后经 0.4.0 收窄、Unreleased 定案单栈
  后已整体移除；最终寻址唯一按 (nodeName, portName) 复合坐标）
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

[0.5.1]: https://github.com/suzvka/DCinfer/releases/tag/v0.5.1
[0.5.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.5.0
[0.4.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.4.0
[0.3.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.3.0
[0.2.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.2.0
[0.1.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.1.0
