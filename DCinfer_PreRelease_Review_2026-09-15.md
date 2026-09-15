# DCinfer 发布上线前审查报告

审查日期：2026-09-15。结论：**不建议将当前提交直接上线。**

本次确认 12 项需要处理的问题，其中 7 项 P1、5 项 P2。P1 指相关功能上线前必须修复；P2 指应在发布前解决或明确限制使用范围。未认定 P0。安全问题的实际暴露取决于是否接受外部归档、是否启用 DCNet 服务端以及部署边界。

> **修复状态（追加，2026-09-16）**：本报告所列问题已按发布前修复批次处理完毕——
> F01 与 F04–F11 已修复并新增回归守护（`TaskLifecycleRegressionTest` /
> `DcgArchiveSecurityTest`），通过本机 Debug/Release 全量与加严验证；第 5 节
> 注册表并发风险项同步加固；F02 / F03 / F12 涉及的 DCNet 服务端与传输加固不随
> 本次交付（DCNet 标注为实验性、禁止网络暴露部署，见 [README.md](README.md)
> 发布状态）。变更明细见 [CHANGELOG.md](CHANGELOG.md) 的 Unreleased 段。
> 本报告正文保留为 2026-09-15 审查时点快照。

## 1. 版本与新鲜度

| 项目 | 核实结果 |
|---|---|
| 仓库 | https://github.com/suzvka/DCinfer |
| 远端默认分支 | master |
| 审查提交 | `79d8919b3cbc1c3f142cd6dae5b63a43957db570` |
| 提交作者时间、提交者时间 | 均为 `2026-09-15T04:16:44+08:00`，即 UTC 2026-09-14 20:16:44 |
| 最新提交内容 | 统一任务句柄同步/异步同级，下线 TaskScope，宿主层仅一套 API |
| 最新版本标签 | `v0.4.0`，指向 `da2c4e65c6279ecf90644c775d0faae9adcc9526`，其提交时间为北京时间 2026-09-10 21:07:19 |
| 标签与审查代码差异 | 当前 HEAD 比 v0.4.0 多 5 个提交；本次没有使用旧标签代码 |
| 依赖锁定 | vcpkg 子模块固定为 `9e53836916341b64a14162a78ba9802488034a5f`；本次没有更新该子模块指针 |

通过实时 `git ls-remote --symref` 核实远端 HEAD，完整克隆主仓后 detached checkout 到上述完整 SHA。审查期间再次查询远端，结果相同。这里记录的是 Git 提交日期；未将搜索引擎缓存日期或 GitHub 页面更新时间冒充源码版本证据。最终复核时间见文末。

## 2. 已执行验证与边界

环境：Linux、GCC 13.3.0、CMake 4.4.3、Ninja 1.13.2。构建工具在隔离工作目录安装。

| 验证 | 结果 |
|---|---|
| 核心 + Builtin + 示例，Debug 配置/编译 | 成功 |
| Debug 核心 CTest（14 个测试程序） | **13 通过、1 失败**：GraphNodeTest 的分支子图用例 |
| 对 GraphNodeTest 单独重复 5 次 | 5 次通过，说明第一次失败不能当成稳定失败，也不能忽略 |
| 在原测试副本中仅补充错误打印 | 再次失败，诊断为子图任务 `FanOutGraph` 的 NotReady；父图 sink 无输出 |
| 核心 + Builtin + 示例，Release 配置/编译 | 成功 |
| Release 核心 CTest | 14/14 通过；单次通过不能消除已复现竞态 |
| 安装核心库后独立 find_package、链接、运行 | 成功，输出 `install smoke OK` |
| hello_graph、task_lifecycle 示例 | 成功，得到预期加法结果 |
| 6 个独立核心异常/生命周期场景 | 均观察到报告描述的缺陷行为 |
| DcgArchive 路径穿越 | 使用原仓库 DcgArchive.cpp 编译运行，在隔离目录复现越界写入 |

归档复现使用临时拉取的上游 zlib/minizip 源码及系统 zlib，验证的是本仓库路径处理代码，并非声称通过了仓库固定依赖版本的完整 DCIr 构建。归档测试只写入本次审查工作目录。

已静态检查顶层与模块 CMake、预设、CI、核心任务接口、调度/等待/取消/回收、节点执行流水线、张量与校验、注册表、DCIr 归档及图编译入口、DCNet 服务端/客户端、OpenAI/ONNX 适配器。

未完成：Windows/MSVC、Clang、GCC 11 最低版本、完整 vcpkg 依赖矩阵、DCNet/OpenAI/ONNX 全模块编译及集成测试、真实 HTTPS/云 API/GPU 模型、ASan/TSan 实跑、外部依赖 CVE 数据库扫描、GitHub 当前 CI 运行结果及分支保护设置、生产配置和负载验证。因此不能把本报告视为这些项目已通过。没有修改或推送原仓库源码。

## 3. P1：上线阻断项

### F01：归档 modelPath 可逃出解包目录，写入或覆盖其他文件【实测】

位置：[DCIr/src/DcgArchive.cpp:154](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCIr/src/DcgArchive.cpp#L154)；调用入口 [DCIr/src/GraphCompiler.cpp:477](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCIr/src/GraphCompiler.cpp#L477)。

`extractOne()` 直接执行 `_tempDir / archivePath`，随后创建父目录并用普通 `ofstream` 写入。没有拒绝 `..`、绝对路径、盘符或验证规范化后的路径仍在临时目录内。`compileFile()` 又直接把 `.dcg` 的 graph.json 中的 modelPath 传入此函数。

复现：归档内保存条目 `../escaped-model.bin` 后调用 `extractOne()`。输出文件实际位于随机解包子目录的父目录，而非解包子目录内。

影响：若加载第三方或用户上传的 .dcg，归档可利用进程权限写入/覆盖指定位置；实际影响受操作系统权限约束。这里没有把文件写入直接等同于已验证远程代码执行。

修复：只接受安全的归档相对路径，处理 Windows/Unix 路径语义，拒绝绝对路径及父目录跳转；规范化后做目录包含关系校验，并防范符号链接逃逸。使用唯一且私有的临时目录和安全文件创建方式。加入父目录跳转、绝对路径、盘符和符号链接回归用例。

### F02：HTTP 分离线程的生命周期超过监听器/服务对象，存在释放后访问【源码确认，未实跑】

位置：[DCNet/src/NetListener_Http.cpp:145](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCNet/src/NetListener_Http.cpp#L145)、[DCNet/src/NetListener_Http.cpp:103](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCNet/src/NetListener_Http.cpp#L103)、[DCNet/src/NetServerAdapter.cpp:173](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCNet/src/NetServerAdapter.cpp#L173)。

每个连接建立 `std::thread([this,...]).detach()`。`stop()` 只等 `_inFlight`；该计数直到整个 HTTP 请求读完才增加。因此，一个正在慢速读取请求头/体的 worker 未被排水计数覆盖，stop 可以直接返回，析构之后 worker 仍会访问 `_endpoint`、`_inFlight`、`_handler`。已进入 handler 的线程也可能超过排水 deadline，而代码仍允许对象被销毁。socket 读超时并不限制模型执行时间。

`ServerService` 还把 `_execMutex`、`_taskSeq` 声明在 `_listener` 之后；默认析构时，它们会先于监听器析构。服务没有显式析构函数先停止并收回所有 worker。另有普通 bool `_stopped` 的跨线程无同步读写。

影响：连接未读完、慢推理或服务停止期间可能发生数据竞争、崩溃或内存破坏。

修复：使用可 join 的有界 worker 管理或独立共享状态，从 accept 开始登记所有连接；停止时关闭/中断活动连接并 join 全部 worker，再释放 handler 状态。ServerService 显式先完成停止。停止标志使用原子或统一互斥。用慢头部、慢请求体、慢 handler 和停服并发用例验证，配合 ASan/TSan。

### F03：maxInFlight 限制位于完整读取请求之后，无法约束连接线程和读取内存【源码确认，未实跑】

位置：[DCNet/src/NetListener_Http.cpp:145](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCNet/src/NetListener_Http.cpp#L145)、[DCNet/src/NetListener_Http.cpp:165](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCNet/src/NetListener_Http.cpp#L165)、[DCNet/src/NetListener_Http.cpp:176](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCNet/src/NetListener_Http.cpp#L176)。

每个连接先创建独立线程并完整读入最大 64 MiB 请求体，之后才判断 `_inFlight` 并进行鉴权。大量未鉴权连接或持续缓慢发送数据的连接不受此业务计数保护；逐次 socket 超时也不是整个请求的绝对截止时间。

影响：在网络可达的部署中，线程数和累计内存可远超 maxInFlight 预期，造成拒绝服务。请求体单次 64 MiB 上限并不能限制总资源消耗。

修复：accept 前后即执行连接并发准入，采用有界线程池；设置整个头部/请求体的绝对读取期限、全局字节预算，并尽早拒绝非法鉴权。外层反向代理可降低暴露，但不能替代库内生命周期修复。

### F04：已终止任务的排队节点仍进入执行，且可能读取同 ID 新一轮状态【实测】

位置：[DCinfer/src/ExecutionEngine.cpp:101](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCinfer/src/ExecutionEngine.cpp#L101)。

池中 lambda 在检查 `gate->terminated` 之前，先用 taskId 获取/创建当前 TaskExecutionState，并调用执行流水线。若旧任务已经取消或其一个分支已满足输出声明，后续排队节点仍会进入这段代码。重新使用 taskId 时，旧 lambda 可以读取并消费新一轮输入，之后才发现旧 gate 已终止并丢弃结果；新的 lambda 随后收到 NotReady。

确定性复现：单线程池先用另一个节点阻塞；提交 b，再取消 b；重新给 b 喂数据并提交；解除阻塞。结果是新任务 `Failed`，记录 NotReady 和未满足输出声明，尽管合法输入存在。

这也与现有 GraphNodeTest 间歇失败一致：只有一个分支输出被声明，另一个排队分支在任务终止后执行并留下 NotReady；exportNode 读取到错误后，父图 sink 无输出。该具体竞态根因仍应在修复中用确定性调度进一步锁定。

修复：调度任务捕获所属轮次的执行态和 generation，不在开始执行时按可复用 ID 重新寻址；取消/终止与准入建立同步协议；旧轮次禁止执行、写诊断、传播和清理新轮次。仅在函数前再加一次无锁 if 不足以关闭全部检查后竞态。

### F05：节点已返回失败，但只要有输出就被当作成功传播【实测】

位置：[DCinfer/src/ExecutionPipeline.cpp:106](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCinfer/src/ExecutionPipeline.cpp#L106)、[DCinfer/src/ExecutionEngine.cpp:133](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCinfer/src/ExecutionEngine.cpp#L133)。

执行流水线即使收到失败 Result，仍可保存已经产生的输出。调度层仅在“完全没有输出”时记录 result.message/diagnostic，没有先判定 `result.ok()`；因此输出存在就传播，并可能满足整图成功条件。

复现：RunFn 先 `ctx.output("y",...)`，随后 `ctx.failure(ExecutionFailed,"deliberate failure after output")`。高层 task.run 返回 `Succeeded`，errors 数量为 0。

影响：失败状态和错误诊断被吞掉，下游或调用方把失败推理的部分结果视为成功；缺失必需输出导致的 InternalError 也可能走相同路径。

修复：显式检查并记录 NodeResult 失败，再决定失败闭环；如果允许部分输出，设计单独的部分成功契约，不能静默覆盖失败。加入先产出后失败、必需输出缺失和相位异常测试。

### F06：条件变量的等待锁与修改谓词的锁不同，存在丢失唤醒窗口【源码确认，未强制复现】

位置：[DCinfer/src/ExecutionEngine.cpp:524](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCinfer/src/ExecutionEngine.cpp#L524)、[DCinfer/src/ExecutionEngine.cpp:410](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCinfer/src/ExecutionEngine.cpp#L410)。

`wait()` 持 `_completionMutex` 检查谓词，状态却在 `_terminationMutex` 下更新，生产者在不持 `_completionMutex` 的情况下 notify。可以出现：等待线程读到谓词为 false → 生产者完成更新并通知 → 等待线程才进入睡眠。普通条件变量不会保存这次通知。

影响：任务实际已结束，但默认无限等待的 run/wait 可能挂起；有限等待可能平白耗尽超时。状态自身使用另一把锁并不能避免该通知竞态。

修复：用同一把 mutex 保护等待谓词及条件变量协议，或改用绑定每轮任务的 future/可靠完成事件。设计精确控制谓词检查与终止时序的测试，避免仅以多跑几次替代协议验证。

### F07：任务完成回调抛异常会中断结果发布与清理，有限 wait 又忽略等待失败【实测】

位置：[DCinfer/src/ExecutionEngine.cpp:379](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCinfer/src/ExecutionEngine.cpp#L379)、[DCinfer/src/InferGraph.cpp:154](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCinfer/src/InferGraph.cpp#L154)。

`_terminate()` 先设置 terminal status，再直接调用用户完成回调；回调抛异常后，剩余信号清理、执行态回收、resultsReady 发布和 notify 被跳过。ThreadPool 最外层只打印异常，不能恢复终止事务。`waitForResult()` 又忽略 `_engine->wait()` 的 bool 返回值，按早已发布的状态返回结果。

复现：完成回调抛 `runtime_error`，调用 waitForResult(200ms)，实际等满 200ms 后却返回 `Succeeded`。源码路径显示无期限 wait 缺少完成条件；本次用有限等待避免留下挂死进程。

影响：调用方无法从返回状态判断结果发布是否完成，资源回收和后续生命周期也可能不一致。即使回调不抛异常，回调慢于 wait 超时也存在终态与结果可读时点不一致问题。

修复：隔离用户回调异常，使用 RAII 保证终止收尾；明确回调重入规则，避免回调中等待同一任务造成自锁。将终态可读与资源清理协议统一，结构化返回等待超时，不能忽略内部 wait 返回值。

## 4. P2：发布前修复或明确限制

### F08：releaseTask 对活动任务仍会删除声明、结果和诊断【实测】

位置：[DCinfer/src/InferGraph.cpp:162](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCinfer/src/InferGraph.cpp#L162)。

ExecutionEngine::releaseTask 对 Running 返回不释放，但 InferGraph::releaseTask 随后无条件 clear 输出区和错误区。复现中对一个阻塞运行中的任务 release 后，解除阻塞，200ms 等待仍为 Running，虽已有输出却无法正常完成。

修复：让引擎返回是否真正获得释放资格；只有成功完成一致的状态迁移才清除其余资源。测试 Running/Unknown/已终止和并发 release 场景。

### F09：只 feed 未 submit 的高层句柄析构不回收输入；异步弃置也没有后续自动回收【实测前半段，后半段源码确认】

位置：[DCinfer/src/GraphInterface.cpp:131](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCinfer/src/GraphInterface.cpp#L131)、[DCinfer/src/InferGraph.cpp:35](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCinfer/src/InferGraph.cpp#L35)。

Task 析构仅回收 Succeeded/Failed/Cancelled。feed 已创建任务执行态，但未 submit 的状态是 Unknown，因此析构不会回收。复现用自定义 Tensor deleter：Task 析构后 deleter 未运行；只有整个 InferGraph 析构后才运行。长寿命图中请求装配失败、输入校验异常等路径会逐步保留资源。

README 已说明在飞任务不随句柄析构而取消，因此这部分不作为“必须取消”的违约。问题在于句柄丢弃后没有登记终态回收机制；除非调用者事先保留 taskId 并用坐标 API 清理，否则终态记录/结果会继续留存。

修复：明确 Created/Running/Terminal/Detached 的拥有关系；Created 析构清除输入，异步弃置采用完成后回收或显式 detached 所有权，保留不强制取消的语义。

### F10：不可达提交先登记 Running 和 gate，再抛异常，没有回滚【实测】

位置：[DCinfer/src/ExecutionEngine.cpp:176](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCinfer/src/ExecutionEngine.cpp#L176)、[DCinfer/src/ExecutionEngine.cpp:203](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCinfer/src/ExecutionEngine.cpp#L203)。

任务状态和活动 gate 已插入表中，之后拓扑守卫才抛 UnreachableDeclaration。复现中 submit 抛异常后 taskStatus 仍为 Running。后续重试会被当作重复提交，句柄析构也不会回收这一状态。

修复：将可能失败的验证放在登记之前，或用事务式回滚统一清理状态、gate、声明和输入。测试提交抛异常后的 status、可重试性及回收。

### F11：OpenAI/ONNX 测试检查旧 BUILD_TESTS，CI 可漏跑测试【源码确认】

位置：[DCEngines/OpenAI/CMakeLists.txt:35](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCEngines/OpenAI/CMakeLists.txt#L35)、[DCEngines/OnnxRuntime/CMakeLists.txt:49](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCEngines/OnnxRuntime/CMakeLists.txt#L49)；CI：[.github/workflows/ci.yml:100](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/.github/workflows/ci.yml#L100)。

顶层使用 DCINFER_BUILD_TESTS，兼容宏只把旧名传值到新名，不会自动生成旧 BUILD_TESTS。两个引擎子目录仍然写 `if(BUILD_TESTS)`。当前 dcnet-openai CI 开启引擎但未设置 BUILD_TESTS，因此可以编译适配器、运行其他 CTest，却不注册 OpenAiEngineTest。ONNX 也有同样的开关问题，且现有 CI 未启用 ONNX EP 作完整验证。include(CTest) 产生的是 BUILD_TESTING，不会修复这个问题。

修复：统一改为 DCINFER_BUILD_TESTS，并在 CI 中校验 `ctest -N` 确实包含预期引擎测试。增加 DCIr、ONNX CPU、安装消费路径及新高层 API 的测试；TSan 当前正则未包含 GraphInterfaceTest。

### F12：HTTP send 出错将 transport 永久标为 failed，后续“重试”未重连【源码确认，未实跑】

位置：[DCNet/src/NetTransport_Http.cpp:109](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCNet/src/NetTransport_Http.cpp#L109)、[DCNet/src/NetAdapter.cpp:54](https://github.com/suzvka/DCinfer/blob/79d8919b3cbc1c3f142cd6dae5b63a43957db570/DCNet/src/NetAdapter.cpp#L54)。

send 的异常路径置 `_failed=true`，后续 send 一进入就直接返回错误。默认重试循环只是再次 send/recv，并未 connect/reset；同 endpoint 的 transport 又被 EngineRegistry 缓存。因此一次瞬态发送故障可能使本次重试以及后续请求都无法恢复，直到外部释放/重建实例。

修复：明确可恢复错误的状态机，在重试前重建/重置连接；按错误类别决定重试，不重试鉴权/参数等永久错误，并处理 POST 重放可能带来的重复推理或计费。加入先失败后恢复的本地 mock 测试。

## 5. 其他发布风险与检查结果

- **归档完整性与大文件**：readEntryToMemory/extractOne 按未压缩长度一次性分配，未设体积/压缩比上限；只调用一次 unzReadCurrentFile，size 转 unsigned，且忽略 unzCloseCurrentFile 的 CRC 返回值。大模型/损坏归档可能截断、占用大量内存或未被拒绝。这些未进行 >4 GiB 实测；应改为有预算的流式读取并验证完整性。
- **HTTP 写出**：respond 只调用一次 sendBytes，未按返回字节数补发；大响应或发送缓冲受限时应验证完整发送。客户端 readBody 没有响应体预算；recv 的异常与 send 的错误归一化不对称。未实跑传输截断测试。
- **共享引擎并发**：不同节点可以共享同一个缓存 transport/引擎实例，而执行闸是每节点的。多个 System worker 或多个图共享实例时，需要确认共享 transport 的 send/recv 是否串行；当前 HttpTransport 的 session/_response 没有实例级互斥。默认单 System worker 不等于所有部署都安全。
- **注册表线程与释放回调**：EnvRegistry、ValidatorRegistry 的 map 未同步；如只允许启动阶段注册，应明确契约。EngineRegistry 在锁内 erase 缓存可能运行最后一个 EngineInstance 的用户释放钩子；回调若重入注册表存在死锁风险，适合移出锁外销毁。
- **包版本与文档**：最新代码仍自报 0.4.0，但已在旧标签后发生公开 API 变更。发布时应给新版本号/标签并记录迁移，避免相同版本对应不同 API。README 中 `find_package(DCinfer 0.3 ...)` 与安装包 SameMinorVersion/0.4.0 不匹配；安装示例还缺少 configure 后 build 再 install 的必要步骤。Presets schema v3 的使用要求高于 README 裸构建的 CMake 3.17 最低要求，应分开说明。
- **安装交付边界**：核心安装消费已实测通过。Builtin 有安装规则；OpenAI/ONNX 当前仅有源码树目标，没有对应 install/export 规则。若发布承诺所有引擎都能通过安装包消费，需补齐；若只支持源码集成，应明确。
- **供应链**：已有固定 vcpkg 子模块，不能声称完全无锁；但本次没有验证固定依赖的已知漏洞与实际解析结果。CI actions 使用版本 tag，未固定不可变 SHA；可增加依赖/许可证清单、SBOM、发布物校验和及依赖安全扫描。
- **生产可运维性**：仓库是可嵌入运行时，不应因为缺少容器文件就判定核心库不合格。使用 DCNet 服务化的发布还需验证 TLS/鉴权配置、容量限制、故障恢复、停止排水、指标和回滚；本次未取得真实部署配置。

## 6. 建议的发布门槛

1. 先修复 F01–F07；若本次仅发布 core，DCNet/DCIr 的阻断项可限定为不交付/不启用相关模块，但 F04–F07 仍阻断核心上线。
2. 修复 F08–F12，至少将相关限制写入公开 API/部署约束，并提供回归测试。建议以结构化每轮任务状态集中解决取消、等待、结果发布和回收，不要逐处追加松散 if。
3. Debug 与 Release 的核心测试全部通过；分支子图竞态加入可控调度回归用例，新任务接口纳入 sanitizer 验证。单次 Release 绿色不能覆盖已发生的 Debug 失败。
4. 按实际交付模块建立 DCIr、DCNet/OpenAI、ONNX CPU 的测试与安装消费矩阵；真实 TLS、至少一个真实模型和目标平台执行上线验收。
5. 为修复后的代码更新版本号、CHANGELOG、迁移示例；核对远端完整 SHA 后打新标签并生成可校验构建物，保留回滚版本。

## 7. 复现方法与原始证据

核心代码保持未修改。以下独立程序链接本次 Debug 构建的原始 libDCinfer.a。状态编号：Unknown=0、Running=1、Succeeded=2、Failed=3、Cancelled=4。程序退出码为 0 表示复现程序正常运行，并不表示所观察的库行为正确。

构建命令（在仓库父目录，按本地 CMake/Ninja 安装方式调整可执行文件路径）：

```bash
cmake -S DCinfer -B build-review -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_ENGINE_BUILTIN=ON
cmake --build build-review -j 4
ctest --test-dir build-review --output-on-failure --timeout 40 -j 2
g++ -std=c++20 -pthread \
  -I DCinfer/DCinfer/include -I DCinfer/DCinfer/include/Graph \
  -I DCinfer/DCinfer/include/Node -I DCinfer/DCinfer/include/Tensor \
  -I DCinfer/DCinfer/include/Tools \
  review_repro.cpp build-review/lib/libDCinfer.a -o review_repro
for mode in failure unreachable reuse callback release unsubmitted; do
  ./review_repro "$mode"
done
```

### 核心复现输出

```json
{
  "failure": {
    "exit_code": 0,
    "stdout": "status=2 errors=0 output=1\n",
    "stderr": ""
  },
  "unreachable": {
    "exit_code": 0,
    "stdout": "submit threw=[ExecutionEngine::submit] Error: - Unreachable Output Declaration - declared output is topologically unreachable from any fed input node (cycle/break in graph construction)\nstatus after throw=1\n",
    "stderr": ""
  },
  "reuse": {
    "exit_code": 0,
    "stdout": "reused status=3 errors=2 output=1\nNodeException in tryExecute: [ExecutionPipeline::execute] Error: - Not Ready - task 'b' is not ready\ndeclared output 'n:y' not satisfied (expected 1, got 0); reason: propagation chain exhausted with error-level diagnostics (node-reported failure)\n",
    "stderr": ""
  },
  "callback": {
    "exit_code": 0,
    "stdout": "status=2 elapsed_ms=200\n",
    "stderr": "ThreadPool: exception in task: callback error\n"
  },
  "release": {
    "exit_code": 0,
    "stdout": "status after release active=1 output=1\n",
    "stderr": ""
  },
  "unsubmitted": {
    "exit_code": 0,
    "stdout": "input freed after task destructor=0\ninput freed after graph destructor=1\n",
    "stderr": ""
  }
}
```

### 核心复现源码：review_repro.cpp

```cpp
#include "InferGraph.h"
#include <iostream>
#include <future>
using namespace DC;
using namespace std::chrono_literals;
Tensor val(float x) { auto t=Tensor::Create<float>(); t=x; return t; }
Node::Schema schema() {Node::Schema s; s.inputs={Node::Port::in<float>("x")}; s.outputs={Node::Port::out<float>("y")}; return s;}
auto pass=[](Node::RunContext& c){ c.output("y",c.pop("x")); return c.success();};
int main(int argc,char** argv){
 std::string mode=argc>1?argv[1]:"failure";
 if(mode=="failure") {
  InferGraph g; g.addNode(std::make_unique<Node>("test","n",schema(),[](Node::RunContext& c){c.output("y",c.pop("x")); return c.failure(Node::Status::ExecutionFailed,"deliberate failure after output");}));
  g.bindInput("x","n","x"); g.bindOutput("y","n","y"); auto api=g.interface();auto t=api.createTask();t.feed("x",val(4));auto r=t.run(1s); std::cout<<"status="<<int(r.status)<<" errors="<<r.errors.size()<<" output="<<t.has("y")<<std::endl;
 } else if(mode=="unsubmitted") {
  bool destroyed=false;{InferGraph g;g.addNode(std::make_unique<Node>("test","n",schema(),pass));g.bindInput("x","n","x");g.bindOutput("y","n","y");auto api=g.interface();{auto t=api.createTask();t.feed("x",Value(new Tensor(val(1)),[&](Tensor* p){destroyed=true;delete p;}));}std::cout<<"input freed after task destructor="<<destroyed<<std::endl;}std::cout<<"input freed after graph destructor="<<destroyed<<std::endl;
 } else if(mode=="unreachable") {
  InferGraph g;for(auto name:{"a","b"})g.addNode(std::make_unique<Node>("test",name,schema(),pass));g.feedInput("t","a","x",val(1));try{g.submit("t","b","y");}catch(const std::exception& e){std::cout<<"submit threw="<<e.what()<<std::endl;}std::cout<<"status after throw="<<int(g.taskStatus("t"))<<std::endl;g.cancel("t");
 } else if(mode=="reuse") {
  std::promise<void> entered,go;auto ready=go.get_future().share();InferGraph g;
  g.addNode(std::make_unique<Node>("test","block",schema(),[&](Node::RunContext& c){entered.set_value();ready.wait();return pass(c);}));
  g.addNode(std::make_unique<Node>("test","n",schema(),pass));
  g.feedInput("a","block","x",val(1));g.submit("a","block","y");entered.get_future().wait();
  g.feedInput("b","n","x",val(2));g.submit("b","n","y");g.cancel("b");
  g.feedInput("b","n","x",val(3));g.submit("b","n","y");go.set_value();
  auto r=g.waitForResult("b",1s); std::cout<<"reused status="<<int(r.status)<<" errors="<<r.errors.size()<<" output="<<g.hasOutput("b","n","y")<<std::endl;for(auto& e:r.errors)std::cout<<e.message<<std::endl;if(r.status==TaskStatus::Running)g.cancel("b");
 } else if(mode=="callback") {
  InferGraph g;g.addNode(std::make_unique<Node>("test","n",schema(),pass));g.setTaskCompleteCallback([](auto&){throw std::runtime_error("callback error");});g.feedInput("t","n","x",val(1));g.submit("t","n","y");auto start=std::chrono::steady_clock::now();auto r=g.waitForResult("t",200ms);std::cout<<"status="<<int(r.status)<<" elapsed_ms="<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count()<<std::endl;
 } else if(mode=="release") {
  std::promise<void> entered,go;auto ready=go.get_future().share();InferGraph g;g.addNode(std::make_unique<Node>("test","n",schema(),[&](Node::RunContext& c){entered.set_value();ready.wait();return pass(c);}));g.feedInput("t","n","x",val(1));g.submit("t","n","y");entered.get_future().wait();g.releaseTask("t");go.set_value();auto r=g.waitForResult("t",200ms);std::cout<<"status after release active="<<int(r.status)<<" output="<<g.hasOutput("t","n","y")<<std::endl;if(r.status==TaskStatus::Running)g.cancel("t");
 }
}

```

### 归档路径穿越复现源码

链接原仓库 `DCIr/src/DcgArchive.cpp` 与 minizip/zlib；运行时将 TMPDIR 指向本次隔离测试目录。程序只创建带审查标记的文件。

```cpp
#include "Ir/DcgArchive.h"
#include <fstream>
#include <iostream>
int main(){
 std::ofstream("model.bin")<<"review-only marker";
 {auto a=DC::Ir::DcgArchive::openWrite("traversal.dcg");a->writeGraphJson("{}");a->addModelFile("../escaped-model.bin","model.bin");a->finalize();}
 auto a=DC::Ir::DcgArchive::openRead("traversal.dcg");auto p=a->extractOne("../escaped-model.bin");std::cout<<"tempDir="<<a->tempDir()<<" extracted="<<p.lexically_normal()<<" exists="<<std::filesystem::exists(p)<<std::endl;
}

```

实际输出（工作目录已固定，随机后缀只用于隔离）：

```text
tempDir="/workspace/scratch/11a49f57d6f1/archive-test/tmp/dcg_traversal_1789485244826636861"
extracted="/workspace/scratch/11a49f57d6f1/archive-test/tmp/escaped-model.bin"
exists=1
```

### 原生 GraphNodeTest 的诊断复验

仅在独立测试副本的 `CHECK(parent.hasOutput(...))` 前添加 taskErrors 打印，未修改库：

```text
Node execution failed: NodeException in tryExecute:
[ExecutionPipeline::execute] Error: - Not Ready - task 'FanOutGraph' is not ready
declared output 'sink:y' not satisfied (expected 1, got 0)
node was never reached during propagation
FAIL: sink should have output
```

原始 Debug 首轮：93% tests passed, 1 tests failed out of 14。Release 首轮：100% tests passed, 0 tests failed out of 14。

## 8. 交付前最终版本复核

最终实时核对时间：`2026-09-15T15:19:41Z`。远端默认分支仍为 master，HEAD 仍为 `79d8919b3cbc1c3f142cd6dae5b63a43957db570`；本地 `git status --porcelain` 为空。报告对应截至该核对时刻的最新远端默认分支代码。
