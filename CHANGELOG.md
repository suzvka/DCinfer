# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.6] - 2026-10-10

引擎/模型分层（破坏性）：`EngineDescriptor` 的引擎生命周期钩子拆分为显式两层——
引擎级一次性初始化（`createEngineCore`，缓存键 = `engineType`，每类型恰好一次，
产物 `EngineCore` 由该类型全部模型实例共享持有）与模型级加载（`loadModel`，
缓存键 = `engineType:modelPath`，每组合恰好一次）。此前 `createEngine` 名义上
"创建引擎"、实际做模型加载，且与真正的引擎初始化（如 `Ort::Env`）混在一起；
拆分后"引擎初始化一次、加载多个模型"成为一等语义。

### Changed

- **破坏性（0.x）钩子四名更换（旧名全部退场，迁移点由编译期强制暴露）**：
  `createEngine` 拆分为 `createEngineCore()` → `EngineCore`（引擎级；无引擎级
  资源的适配器留空，框架合成空核心）与 `loadModel(core, modelPath)` →
  `EngineInstance`（模型级；core 供绑定引擎级资源，如 Session 绑定 Env）；
  `releaseEngine` 更名 `releaseModel`（模型级释放，语义不变），新增
  `releaseEngineCore`（引擎核心释放）。适配器迁移：OnnxRuntime 将
  `Ort::Env` 归引擎核心（删除进程级 Env 单例）、`Ort::Session` 归模型加载；
  DCNet/OpenAI 无引擎级共享资源（不注册核心钩子，忽略 core 参数）。
- **`EngineRegistry` 公开方法语义更新（保留现名）**：`getOrCreateEngine`
  内部先确保引擎核心就绪（`getOrCreateEngineCore`，single-flight 每类型一次）
  再加载模型；`releaseAllEngines` 同清两级缓存（模型实例 + 引擎核心）；
  待析构句柄仍锁外释放；实例经共享句柄持有核心（核心存活期覆盖全部实例，
  原生资源析构顺序：模型级对象先于引擎核心）。
- `EngineInstance` 新增 `attachCore` / `core()`：框架在实例发布前注入核心句柄；
  适配器执行期可经 `ctx.engineInstance()->core()` 访问引擎级对象，无需全局状态。

### Added

- `EngineCore` / `EngineCoreHandle`（类型擦除的引擎级运行时句柄）；
  `EngineDescriptor::createEngineCore` / `releaseEngineCore`；
  `EngineRegistry::getOrCreateEngineCore` / `releaseEngineCore`。

### Removed

- **破坏性（0.x）`EnvRegistry` 设施整体移除**（`EnvRegistry.h` / `EnvRegistry.cpp` /
  `EnvRegistryTest`）：该设施与引擎核心机制概念重复且语义更弱，框架内无生产
  使用方；其"引擎运行时环境"职责已由本版本的 `createEngineCore` / `EngineCore` /
  `releaseEngineCore` 完整承接（每 engineType single-flight 一次、句柄链保证环境
  晚于全部模型实例销毁、清理钩子在真实析构时恰好一次）。迁移：适配器经
  `EngineDescriptor::createEngineCore` 提供引擎级共享资源（如 `Ort::Env` /
  CUDA context），执行期经 `ctx.engineInstance()->core()` 访问。

## [0.7.5] - 2026-10-09

引擎节点编译语义修正（破坏性）：GraphCompiler 不再于编译期急切创建引擎
实例——引擎节点改为延迟物化（不加载模型、不缓存实例），`modelPath` 收束
为引擎自定义不透明信息（原样透传：不拼接、不校验、不做文件系统解释），
`.dcg` 编译不再解压模型文件。编译机不再需要持有模型文件；模型加载由宿主
经 `getOrCreateEngine` + `Node::bindEngine` 显式完成，编译机/执行机分离
部署成为一等公民。

### Changed

- **破坏性（0.x）引擎节点延迟物化**：引擎节点反序列化不再调用
  `getOrCreateEngine`——编译期零加载、零实例缓存，先前「模型缺失/不可达
  → 编译中断」的路径不再存在（加载失败语义回归 `createEngine` 钩子，在
  宿主调用 `getOrCreateEngine` 时暴露）。节点经新增公开接口
  `EngineRegistry::createLazyNode` 物化：schema 取 JSON 声明（不做实例
  推导、不被覆盖），工厂提供引擎 RunFn；未注册工厂回退骨架
  （RunFn=nullptr）并输出警告，JSON 声明 schema 为空亦报告警告。宿主须
  在冻结前经 `getOrCreateEngine(engineType, modelPath)` +
  `Node::bindEngine` 注入实例（典型：执行机侧解析 modelPath 后加载）；
  实例缓存键为该字符串本身。
- **破坏性（0.x）`modelPath` 语义收束为透传**：不再做基目录拼接、路径
  校验或任何文件系统解释——相对路径、URL、模型标识符均原样保留
  （与 `engineConfig` 的「一字段一语义」哲学对齐）；依赖编译期路径解析、
  拼接或越界拒绝的行为不再存在。
- **`GraphCompiler::compileString` 删除 `baseDir` 参数**（透传后不再需要
  基目录锚点）；`compileFile(.dcg)` 只读取 `graph.json`，不再收集/校验/
  解压 modelPath 引用文件——资源就绪由宿主经 `DcgArchive::extractOne`
  自行处理（归档读取防御与 `addModelFile` 写入侧保护不变）。

## [0.7.4] - 2026-10-09

发布前安全审查批次（23 项）与输出取数不变量修复。要点：DCNet 请求面与
生命周期加固（未认证内存 DoS、明文凭据边界、并发 stop / 租约 / 错误脱敏），
核心线程池生命周期硬契约，DCIr 归档读取与临时目录安全，OpenAI `params`
白名单，发布链路定型（exact-SHA CI 门禁、归档规范化、双构建差异记录）与
CI 覆盖补齐；绑定端口收束为终端端口不变量。

### Security

- **P0 NET-1 鉴权前读取完整请求体（未认证内存 DoS）**：有界头部解析后先做
  认证/方法/路径/配额检查，再预留正文预算——默认单体上限 8 MiB、聚合
  32 MiB；未认证请求在读取正文前即被拒绝（提前 401/413/429），预算恢复
  与共享头/正文 deadline 均有回归；正常认证请求行为不变。
- **P0 NET-2 明文凭据边界**：明文监听仅允许解析后回环；出站按最终 URI
  拒绝在明文 HTTP 上发送敏感凭据（开发例外为显式开关、默认关闭）；
  Windows 原生信任/主机名策略在发送任何 HTTP 字节前验证。
- **NET-8 错误与日志入口脱敏**：非法 header 只输出序号/安全名称（密钥
  子串与原始 CR/LF 不再进入错误串）；控制符清洗与长度上限统一。
- **NET-9 内部错误外泄**：内部 500 仅返回稳定错误码与关联 ID（文件路径/
  配置/请求片段不外泄）；受控 sink 只接收阶段标签，不接收任意异常/请求
  原文；远端诊断保留 dcnet 通道但限长并清洗控制符。
- **AI-1 OpenAI `params` 采样白名单**：仅 `temperature`/`top_p`/
  `max_tokens`/`presence_penalty`/`frequency_penalty` 等采样参数经逐项
  类型与范围检查后生效；`model`/`messages`/`stream` 等协议字段与未知
  字段一律拒绝（非法表驱动配置零实际请求有回归）。
- **IR-2 `.dcg` 临时目录安全创建**：POSIX 原子 0700 + `openat` no-follow；
  Windows 当前用户私有 DACL、祖先句柄防目录替换、reparse 点拒绝、独占
  创建——消除权限非原子的 TOCTOU 窗口。

### Fixed

- **CORE-1 池线程内关闭/析构线程池**：worker 身份检测（TLS 自标识）——
  池线程内 `shutdown()`/`resetInstance()`/最后所有者析构改为明确诊断的
  fail-fast，替代原先的 `resource deadlock`/挂死/进程 abort；不 detach
  规避（防 UAF）；外部并发 shutdown 与幂等补救有回归。
- **CORE-2 节点/回调内析构引擎**：无分配 TLS 活动引擎栈自检（覆盖节点、
  同步 cancel、回调），命中即明确诊断；8 个隔离 death case 覆盖禁止重入
  析构契约与有界结束。
- **CORE-4 TensorSlot 类型擦除数据泄漏**：析构不释放 `store()` 存储的
  运行时数据、移动不置空源——未经 `take()`/`clear()` 的输入值随槽位
  销毁泄漏（LeakSanitizer 在常规执行路径复现）；析构改为 RAII 释放，
  移动显式转移所有权（源仅清空、不二次释放）。
- **CORE-5 fail-fast 诊断在重定向下丢失**：禁止析构路径的诊断文本在
  glibc `freopen` 重定向后（全流缓冲）随 `_Exit` 未 flush 丢失，death
  case 只观测到退出码、无法核验诊断内容；三处防护显式 flush 兜底。
- **NET-3 并发 `stop()`**：完整 stop 流程串行化并排水（第二调用等待首次
  完成，不再提前返回引发 UAF）；handler 内同步 stop 在锁前拒绝。
- **NET-4 `_activeThreads` 记账事务性**：先登记容器再增加活跃计数
  （RAII 回滚）；分配失败注入验证计数与容器一致——`stop()` 不再永等。
- **NET-5 `recv()` 租约 RAII**：全出口回收租约、异常 abort 会话；截断读
  后其他线程请求仍能在有界时间内完成。
- **NET-6 `start()` 发布时序**：accept 线程建立后才发布 started；创建
  失败回滚状态与 handler，可成功重试。
- **NET-7 `readBody` 短读**：区分 stream failure 与 EOF，并校验固定
  Content-Length——截断正文必须失败而非进入 codec。
- **IR-1 `.json` 无界读取**：流式读取至 32 MiB+1 哨兵即拒绝（峰值内存
  有界）；精确边界与正常图回归。
- **ORT-1 Windows ORT 预设测试路径**：Release/单配置直接注册
  `TARGET_FILE`（仅多配置 Debug 保留 DLL wrapper）——官方 Windows ORT
  预设 CTest 不再必败。
- **BUILD-1 安装接口路径**：目标 `INSTALL_INTERFACE` 统一 GNUInstallDirs
  （`${CMAKE_INSTALL_INCLUDEDIR}`）——自定义包含目录（如 `inc`）的安装
  树可被消费；四个包的非默认安装与搬迁后消费均验证通过。
- **BUILD-2 探测上下文隔离**：libatomic 探测显式 C++20 并隔离/恢复父工程
  check context（重新计算自身缓存）；`find_package(Threads)` 同域隔离
  （CMake 3.17 的 FindThreads 会继承宿主 `CMAKE_REQUIRED_*` 哨兵值导致
  pthread.h 误判 not found）；C++17 宿主 add_subdirectory 前后上下文保持
  不变（CI 用 probe-only 库哨兵断言）。
- **BUILD-3 clang-14 测试编译**：`make_shared<MockSession>(path)` 依赖
  C++20 P0960 括号聚合初始化（clang 16 前未实现），最低工具链 clang-14
  构建测试报 `construct_at` 无匹配；改为显式聚合构造
  （`MockSession{path}`，与同文件其余 mock 一致）。
- **输出取数端口不变量（High）绑定端口带出边会截断下游数据流**：
  `bindOutput` 的端口在图中存在出边时，传播期「OutputZone 搬运」与
  「出边搬运」共享同一消费槽——取数截走数据后下游永久饿死：任务或
  挂起（下游声明永不满足、无 Error 级诊断、不自行终止，直至宿主
  cancel 释放），或静默跳过下游（Succeeded 但下游从未执行、无任何
  诊断）。现收束为：**绑定端口必须是终端端口（无出边）**——冻结期
  （compile，失败可修正重试）校验，违规抛新增的
  `GraphException(NonTerminalPort)`。中间结果既要对外可见又要继续参与
  下游时显式插入分支：把该端口 connect 到直通节点、绑定挂分支叶子
  （同源口再次 connect 自动扩容扇出）。取数语义不变——「输出 = 把
  张量 move 到目的地」，无隐式拷贝/共享；提交期显式声明保持既有
  完成条件语义（循环计数/部分求值依赖其自由度，不受本约束）。

### Changed

- **破坏性（0.x）**：以下入口由静默降级/延迟暴露收束为显式拒绝——非回环
  明文监听、明文 HTTP 承载 Bearer 凭据（默认关闭，可显式开关）、
  `params` 越界键、绑定带出边（非终端）端口。
- **NET-10 部署边界成文**：README/DESIGN 明确服务端 TLS 与 chunked 未
  实现，远端明文部署须经 TLS 代理回环上游；Windows 网络/原生 TLS 分支
  纳入 CI 矩阵。
- **CORE-3 背压示例**：README 给出宿主 semaphore admission 示例（许可
  持续到真实排水）；无界队列与「submit 成功即会执行」契约不变。

### CI / Release

- **REL-1 发布门禁与真实测试**：发布流程执行完整 CTest 套件
  （`--no-tests=error`，非枚举）；exact-SHA CI 门禁——tag 必须指向
  master/main push 上整条 ci.yml 全矩阵成功的提交；API 不可达、无 run、
  运行中、失败、取消一律 fail-closed 拒绝。consumer smoke 注入期望版本
  强校验安装树。
- **REL-2 归档规范化与可追溯性**：`SOURCE_DATE_EPOCH`；归档条目排序 +
  固定 mtime/owner（tar）/统一 mtime（zip），gzip 不写时间戳；同安装树
  两次打包字节级比对；双干净构建的安装树逐文件 SHA256 清单差异如实
  记录随产物归档（未宣称二进制可重现）；runner 与工具链显式 pin
  （windows-2022、CMake 3.31.6、MSVC 14.44、g++-13）。
- **CI-1 防悬挂**：全部测试注册 TIMEOUT 120，CI 全局 `--timeout`/
  `--no-tests=error`，关键筛选带名称断言——挂起的 fixture 在 120s 被判
  失败而非耗尽作业。
- **CI-2 覆盖补齐**：新增 Debug、ASan+UBSan（core+DCNet+OpenAI）、最低
  工具链（GCC 11 / Clang 14 / CMake 3.17.3）、GCC 32-bit atomic probe、
  Windows DCNet/OpenAI/TLS/ORT 作业；README quickstart 在 Windows/Linux
  逐字执行校验。
- **install_smoke 版本守护**：期望版本升为 0.7.4，过期安装前缀/错位 tag
  不再静默通过。

## [0.7.3] - 2026-10-06

SDK 发布链路定型与消费侧防御加固：release workflow 切换 `sdk-release` 预设
并加装安装边界断言；DCNet 张量帧解码与 TensorData 形状入口补防御校验；
ORT 1.28 导出 target 缺失的 include 路径消费侧补齐。

### Security

- **P1 DCNet 张量帧解码防御加固**：`NetCodec` 的 `decodeTensor` 面向不可信
  远端帧的线级限制（不改动公开 Tensor API）——帧必须为 JSON 对象、
  `dtype` 必须为字符串；`shape` 必须为数组且秩 ≤ 64，维度必须为非负整数
  （0 元素 = 空载荷张量合法，与 TensorData metadata-only 语义对齐），
  元素计数乘法溢出检查；单张量字节上限 1 GiB（按 dtype typeSize 折算）。
  载荷完整性：`data` 必须为字符串（数值 dtype 经 base64 解码，Data 文本
  UTF-8 直传），字节数必须精确等于 元素数×typeSize——形状/dtype/数据
  三者不一致在解码期拒绝，防解码期无界分配与整型回绕。

### Fixed

- **ORT 1.28 导出 target 缺失 include 路径**：上游导出 target 不再携带
  `INTERFACE_INCLUDE_DIRECTORIES`（头文件平铺安装至
  `<prefix>/include/onnxruntime`）——1.23 的动态导入 target 自带该属性，
  1.28 的静态导入 target（m64-linux 默认 static linkage）完全缺失，
  `#include <onnxruntime_cxx_api.h>` 无法解析。消费侧按需补齐：
  `get_target_property` 检测为空则 `find_path`（`NO_DEFAULT_PATH` +
  `REQUIRED`）定位并 SYSTEM 注入——两种 linkage 下均可编译。

### Changed

- **TensorData 形状入口校验**：元素计数与元素数×typeSize 字节数溢出拒绝
  （`std::invalid_argument`）；零维保留既有语义——0 元素张量是空文本/空
  集合的合法表示（OpenAI 空响应文本、wire 空文本帧），三参构造空载荷
  metadata-only 语义不变（带载荷则 expectedBytes=0 必然 mismatch 拒绝），
  `loadData` 稠密直通路径继续拒绝零维（入口不变量不变）；
  `TensorData(shape, DataBlock&&)` 的 typeSize 推导改用溢出检查的逐维
  乘积（原 `std::accumulate` 链乘无溢出防护）。
- **install_smoke 消费者版本守护**：`SMOKE_EXPECT_VERSION` 显式期望版本
  （EXACT 请求 + DCinfer/DCIr 安装版本双校验），过期安装前缀/错位 tag
  不再静默通过；`SMOKE_SDK` 形态与 Builtin/DCNet 互斥——SDK 冒烟不得
  误验引擎包。
- **破坏性（0.x）**：`NetCodec` 张量帧解码与 `TensorData` 构造新增拒绝
  路径——缺失/类型不符字段、负/超秩维度、超限载荷在入口显式拒绝
  （零维空载荷张量保持合法）。

### Added

- **`sdk-release` CMake 预设**：Core + DCIr 的 SDK 发布配置，显式排除
  引擎/DCNet/测试/示例——发布产物边界由预设固定。

### CI / Release

- release.yml SDK 发布链路定型：双平台切换 `sdk-release` 预设；tag 版本
  严格校验（`vMAJOR.MINOR.PATCH` 格式 + 与 `project()` 版本一致性比对）；
  SDK 安装边界断言——核心/IR 头文件与 Config 齐备，安装树含
  engine/onnx 制品即失败；sdk-release 关测试（全矩阵由 ci.yml 覆盖），
  改为 CTest 注册核验；SBOM 明确扫描最终 SDK 安装树；消费者冒烟
  （`SMOKE_SDK` + `SMOKE_WITH_IR` + 期望版本注入）。
- README 发布消费说明同步：SDK 包仅含 `DCinfer::DCinfer` 与可选
  `DCIr::DCIr`，不夹带具体引擎适配器（Builtin 需
  `-DBUILD_ENGINE_BUILTIN=ON` 单独构建）；测试运行说明补充（先构建再
  `ctest --output-on-failure`，`ctest -N` 核对注册）。

## [0.7.2] - 2026-09-30

v0.7.1 发布审查（release-blocker issue）修复：完成回调双调用、schema/typeSize
入口校验缺失、DCNet 端点解析与部署边界加固、响应发送完整性、并发边界收窄、
安装消费者覆盖与发布链路（CI 供应链 / release workflow）补齐。

### Security

- **P1 DCNet 端点解析严格化**：`NetEndpoint::parse` scheme 大小写不敏感识别
  `http/https`，显式未知 scheme（如 `ftp://`）入口拒绝——不再静默降级为明文
  HTTP；端口替换 `atoi` 为严格解析（仅十进制、全串消费、≤65535），非法/负数/
  超范围一律拒绝。无 scheme 简写（`host[:port][/path]`）兼容不变。
- **P1 DCNet 请求头注入防御**：`headers`/`authToken`/`contentType` 中的
  CR/LF/内嵌 NUL/其余控制字符在 `connect()`（配置期）拒绝，不发起网络 I/O——
  防请求序列化时报文行注入。
- **P1 HTTPS 显式 TLS 校验**：`HTTPSClientSession` 改用显式客户端上下文——
  宿主已初始化 `SSLManager` 时按宿主管辖，否则框架兜底 `VERIFY_STRICT`（证书
  链 + 主机名校验 + 默认 CA，禁用 SSLv2/v3/TLS1.0/1.1）；HTTPS 不再依赖宿主
  全局配置（未初始化宿主上原本不可用）。宿主自定义信任锚：首个 connect 前
  `initializeClient(context)` 即可接管。Windows SChannel 分支未经 CI 验证，
  需人工确认。`HttpTlsTest`（内嵌自签证书 + SecureServerSocket）覆盖：不受信
  证书拒绝 / 信任锚接管正向交换 / hostname 不匹配拒绝。
- **P1 服务端部署边界**：`DcNetListener::bind` 配置期全量校验——非回环
  `listenHost`（非 127.0.0.1/::1/localhost）且 `authToken` 为空 → 拒绝（服务端
  TLS 未实现前，无认证对外监听不允许）；空白 token（`Bearer ` 前缀剥后为空
  或全空白）→ 拒绝（堵空凭据绕过）；负 `maxInFlight`/`maxConnections`、
  `backlog ≤ 0`、`port` 越界、负 `requestTimeout` → 拒绝。

### Fixed

- **P1 完成回调至多一次**：正常路径回调自身抛异常时，外层 catch 会再次调用
  同一回调——重复提交状态/通知/释放资源，第二次抛出还覆盖原始错误。改门闩
  （`completed` 标志）统一正常与异常路径：回调恰好一次，原始异常原样重抛。
  `ExecutionConcurrencyTest` H-2 增回调计数断言。
- **P1 Node schema 入口校验缺失**：`NodeSchema::valid()` 已定义但从未强制。
  现在 `Node` 构造与 `GraphStore::addNode` 均拒绝非法 schema（重复端口名/
  typeSize=0 的非 Void 端口/默认值 type+typeSize 不一致）——缺陷前置暴露
  （`NodeException(SchemaError)`）。
- **P1 运行时 Tensor typeSize 校验缺失**：`TaskBuffer::drainInputsTo` 原只比
  较逻辑类型，逻辑类型相同但元素宽度不同的张量（schema 声明与实际内存布局
  不符）会流入后端。现追加 `typeSize` 比对 → `NodeException(TypeMismatch)`。
  `NodeTest` 新增构造期正反例 + 执行期 typeSize mismatch 用例。
- **P2 响应发送完整性**：`HttpListener::respond` 单次 `sendBytes` 不查返回值
  即关闭连接——阻塞 socket 部分发送（发送缓冲满/超时窗口不足）时静默截断，
  客户端实际字节数与 `Content-Length` 不符。改循环补发，失败/超时仍静默
  关闭（对端归一化）。`MockServer` 同步修复。`HttpTransportTest` 新增 4 MiB
  大响应逐字节完整性用例。
- **P2 Content-Length 严格解析**：`strtoull` 不查全串/溢出/重复 header——
  "100abc" 被静默解析为 100、负数回绕为巨大值（行为安全但语义错）、重复
  header 首值静默生效。改 `std::from_chars` 全串校验（畸形 → 400）+ 重复
  `content-length` 显式 400（多值混淆走私向量）。`ServerAdapterTest` 新增
  raw socket 畸形 CL 用例（负数/部分数字/重复 → 400，合法仍 200）。

### Changed

- **OutputDeclaration.count=0 入口拒绝**：count=0 的声明立即视为满足，无
  诊断价值——"无需该输出"的正确语义是不声明。`OutputZone::declare` 两个
  重载均拒绝（校验先于写入，拒绝路径零副作用）。
- **GraphStore 容器引用收窄**：`InferGraph::edges()` / `GraphBuilder::edges()`
  改为按值返回——内部容器引用永不外泄（`edgesToJson` 曾把 range-for 元素
  指针存入跨语句索引表，值语义下临时容器析构即悬垂，已同步修正）。`GraphStore`
  内部引用版 `nodes()/edges()` 保留并注释"仅限持锁或封印后调用"。
- **并发 shutdown 串行化**：`ThreadPool::_drainWorkers` 新增 `_drainMutex`、
  `ResourceScheduler::shutdown` 新增 `_shutdownMutex`——并发重复 shutdown
  对同一 worker 的 joinable+join 竞态（UB）消除；顺序重复调用幂等性不变。
  `ThreadPoolTest` 新增并发 shutdown ×8 用例。
- **EngineRegistry 实例缓存容量上限**：唯一 modelPath 槽位只增不减（长驻
  服务场景内存无界增长）。达到上限（常量 64）后按 LRU（lastAccess 最旧）
  驱逐非 loading 槽位；仍被节点持有的实例由共享句柄保活。`EngineRegistryTest`
  新增容量驱逐回归（驱逐重建 / 最新条目保持命中）。
- **GraphCompiler 图定义预算**：`compileString`/`compileFile(.dcg)` 入口
  JSON 大小上限（32 MiB）+ `buildGraph` 节点数上限（4096）——不可信/异常
  来源的图定义在解析前拒绝，防无界分配。
- **破坏性（0.x）**：`NetEndpoint::parse`/`DcNetListener::bind`/`Node` 构造
  新增拒绝路径（非法输入在入口显式拒绝）；`InferGraph::edges()`/
  `GraphBuilder::edges()` 返回类型变更（值拷贝）。

### Dependencies

- vcpkg artifact registry 固定到具体 commit（`vcpkg-configuration.json`）——
  `refs/heads/main.zip` 会随上游漂移，工具获取不可复现。
- ORT overlay portfile 平台条件化：`/Zc:preprocessor` `/wd4996`（MSVC 专用）
  仅在 Windows 追加，Linux GCC/Clang CUDA 路径不再直接失败；CUDA 架构经
  `DCINFER_CUDA_ARCHS`（env/triplet 变量）可配置，默认 `89-real`（RTX 4070）。
- vcpkg submodule 升级（OpenSSL ≥3.6.5 / Poco 1.15.x overlay）作为后续项：
  需全量重建验证，本版未移动 submodule 固定提交。

### CI / Release

- ci.yml 供应链加固：顶层 `permissions: contents: read`；第三方 action 全部
  pin 到完整 commit SHA；所有 checkout 关闭 `persist-credentials`；新增
  `ort-cpu` job（hosted runner 无 GPU，CUDA 构建不进 CI，限文档声明组合的
  本地验证）。
- 新增 `release.yml`（`v*` tag 触发）：Linux/Windows 全量构建+测试 → 安装树
  打包（tar.gz/zip）→ SHA256SUMS → SPDX SBOM（syft）→ SLSA v1 build
  provenance attestation → 发布 GitHub Release。
- 安装冒烟覆盖扩展：`examples/install_smoke` 新增 `SMOKE_WITH_BUILTIN`
  （真实跑一次 Builtin 算子图）与 `SMOKE_WITH_NET`（回环监听 bind）两个
  消费者形态；CI install-smoke 覆盖 DCinfer/DCIr/Builtin/DCNet 四类。
- README 依赖表修正：DCIr 行补 nlohmann_json/minizip，DCNet 行补
  nlohmann_json（与各包 Config 的 `find_dependency` 一致）。

## [0.7.1] - 2026-09-29

v0.7.0 发布后收尾审查的调度器/引擎生命周期与 DCNet 传输层并发缺陷修复，
张量数据块语义修正；类型注册表死 API 收窄（破坏性，0.x）。

### Fixed

- **P1 引擎析构逃逸路径 use-after-free 竞态（发布前审查二轮发现）**：
  原自排水等待以 `ResourceScheduler::isStopped()` 为逃逸条件，而 `shutdown()`
  先置位关停标志、后逐池 join——两者之间存在窗口，此时析构引擎会越过排水
  等待与在飞任务的引擎回访竞态（UAF）。排水计数改票据制（`DrainTicket`）：
  每次派发随任务 lambda 签发票据，执行完成（worker）与被池弃置（shutdown
  清队析构 function）两条路径均经票据析构回收计数——析构可无条件等待归零，
  逃逸路径整体删除。`GraphOperatorTest` 起首的默认预算回归用例顺带覆盖
  `resetInstance()` 关停路径。
- **P1 默认预算下 GraphOperator 父子嵌套开箱自死锁（发布前审查二轮发现）**：
  默认 `SchedulerConfig{1,1,1}` 且组合节点默认亲和 Operator 时，父图等待型
  节点占住 Operator 类唯一槽位，子图节点排队同一池永不执行——无诊断永久
  挂起（旧版每图私有池时代默认安全的场景回归）。修复分两层：
  `makeNode` 默认亲和改为 `System`（等待型编排节点归基础设施类，与子图
  业务节点默认 Operator 类分离）；因 System 类同时承载图连接器，单独改
  亲和会让子图内连接器与嵌套等待链在 System 默认 1 槽位下同类自锁
  （回归用例实测暴露），故同步将 `SchedulerConfig` 默认 System 预算
  放宽为 4（Compute/Operator 保持 1）——开箱覆盖 ≤3 层嵌套 + 并发连接器，
  显式收紧或指定其它亲和时仍需按叠加规则规划。`GraphOperatorTest`
  新增默认预算嵌套回归用例（早于显式预算放宽运行）。
- **P2 调度器杂项**：`_poolFor` 对非法枚举值加范围防御（与 `workersFor`
  “未知资源类按拒绝处理”契约一致，防越界下标）；`DCEngines` 缺 DCNet 的
  FATAL_ERROR 提示改用主名 `-DDCINFER_BUILD_DCNET=ON`（原引导旧别名）。
- **P0 引擎排水票据生命周期窗口**：票据析构原为“先递减 `_pendingRuns` 后
  拿排水锁唤醒”——归零与拿锁之间析构者可越过排水等待走完整析构（含成员
  销毁），worker 再触碰已销毁的 `_drainMutex/_drainCv`。改为锁内递减（递减
  与 notify 原子于排水锁）；同时票据登记改为“临界区外构造 + 登记成功才
  armed”——票据分配失败不再泄漏已加的排水计数，未登记路径析构零副作用。
  `ResourceSchedulerTest` 新增析构与调度器 shutdown 并发压力回归。
- **P0 调度器 shutdown 持锁 join 死锁**：`shutdown()` 原持 `_initMutex`
  逐池 join worker，而 worker 正在执行的任务 lambda 经 `_poolFor` 阻塞在
  同一把锁上（已过第一道关停快检后关停插入）——循环等待。改为锁内仅快照
  现有池指针、锁外逐池关停：置位后双检查均拒绝新池创建，锁外 join 期间
  被阻塞 worker 按拒绝收尾退出。`ResourceSchedulerTest` 新增任务内递归
  提交 × 并发 shutdown 的压力回归（shared_future 看门狗）。
- **P1 HttpTransport close 与 recv 竞态（use-after-free）**：close 等交换
  收尾 5s 超时后强收并销毁 session，而 recv 正在读的响应流指向 session
  内部——悬垂。会话改 `shared_ptr`：交换方（send/recv）持本地副本保活
  对象，强收仅 abort 中断阻塞读；成员访问（`_response/_session`）全部
  收敛到 `_ioMutex` 临界区；被中断的读取返回归一化错误而非静默短读。
  `HttpTransportTest` 新增慢响应 × 并发 close 用例（慢 body 测试服务）。
- **P1 `TensorData::crop` 多维截断语义错误**：原实现扁平 resize 到新元素
  总数，对 `{2,3}→crop({2,2})` 得 `1,2,3,4`（`[[1,2],[3,4]]`）而非每维
  前缀的 `1,2,4,5`（`[[1,2],[4,5]]`）。重写为按行主序逐块 memcpy 的多维
  前缀裁剪（一维行为不变），并失效稀疏视图保持 cache/view 一致。
- **P1 `TensorData::expand` 不扩已有块**：原实现只填充缺失块，`{2}→{4}`
  时既有 root 块不扩容、形状停留 `{2}`。改为遍历全部块路径：缺失块整块
  填充（原语义），已有块扩容到目标块长且仅新区域按 fillData 填充（旧值
  保留），并补登记 catalog 使 `getCurrentShape()` 与目标一致。
- **P2 `TensorData::loadData` 缺 shape/尺寸自洽性校验**：原直接接受任意
  shape/typeSize/bytes 组合，声明元素数大于实际缓冲时下游按 shape 寻址
  越界读。加与三参构造同一不变量校验（拒绝 0 维、防连乘/乘积溢出、精确
  字节数匹配）；`read<T>` 单元素路径补越界检查（前缀路径已有）。校验对象
  是调用方声明的自洽性（元数据），字节载荷仍原样进入 cache 不做解释。
- **P2 `TensorData::write(element)` 整数溢出边界**：溢出检查由
  `elementIndex > SIZE_MAX/typeSize` 收紧为 `>=`——旧条件放行的边界值
  `(elementIndex+1)*typeSize` 恰好回绕（typeSize 为 2 的幂时归零），写入
  偏移仍是天文数字，后续 memcpy 越界写；检查同时移到 `updateCatalog`
  之前，巨大/回绕负索引不再先污染 catalog 再被拒。
- **P2 ThreadPool 构造期线程创建失败触发 std::terminate**：构造循环中
  任一 `std::thread` 抛出时，构造未完成、已启动 worker 随成员析构对
  joinable 线程 terminate。提取 `_drainWorkers()` 供 shutdown 与构造
  catch 共用：失败时回收已启动 worker 后传播异常。`ThreadPoolTest`
  经注入钩子（仅测试 TU 可设）覆盖中途失败用例。
- **P2 HttpTransport 3xx 状态码与文档不符**：文档承诺“2xx 成功、非 2xx
  归一化”，实现仅 `status >= 400` 走错误路径——3xx 被当成功且响应体当
  payload。改为 2xx 窗口判定（不做自动重定向跟随，头文件注释显式化）。
- **P2 HttpTransport 响应体无大小上限**：`copyToString` 无界拷贝，异常/
  恶意远端可致客户端无界分配。`NetEndpoint` 新增 `maxResponseBody`
  （默认 512 MiB，0 = 宿主显式豁免），读取改分块循环：成功体超限报错、
  非 2xx 错误体仅诊断允许截断。

### Changed

- **类型注册表死 API 收窄（破坏性，0.x）**：`Tools/DCtype.h` 删除全库零调用的
  接口——`setFallback`/`tryGetFallback` 与 fallback 查询分支（从未有任何调用方
  设置过 fallback）；显式 `freeze()`/`isFrozen()`（冻结由首次读查询自动触发，
  语义已注释于 `ensureFrozen`）；`getTypeOr`/`tryGetType` 全部重载；按 C++
  类型查询的 `getSize<T>()`/`getSizeOr<T>()`（文档声称“未找到返回 0/备用值”，
  实现恒返回 `sizeof(T)`，与注册表语义无关）。`ITypeRegistry` 同步收缩为纯
  类型擦除基类。保留在用接口：`registerType` / `getType` / `getSize(enum)`。
- **文档清理**：`DCNet/DESIGN.md` 与 `DCEngines/OpenAI/README.md` 移除指向根
  README“发布状态”章节的悬空引用（该章节已不存在）及“本次交付”快照式措辞，
  改为自洽的实验性声明；`GraphStore.cpp` 删除串行化汇聚 TODO 演进注释（限制
  已由 README“已知限制”与异常消息承载）；`Tensor.hpp` 类头合并重复 `@brief`
  并移除空壳“典型用法”段。
- **输入/协议边界行为收紧（有意为之）**：`TensorData::loadData` 拒绝
  shape/尺寸不一致的声明、非 2xx（含 3xx）一律归一化、成功响应体超限
  报错——三者均为“声明与实际不一致/超预期输入在入口显式拒绝”，不影响
  合法调用；“载荷字节原样传递、不解释内容”的哲学不变（校验对象是调用方
  自身声明的元数据自洽性，非数据含义）。

## [0.7.0] - 2026-09-28

发布前审查发现的张量视图公共 API 语义缺陷与 README 功能虚宣修正；
资源调度器升级为进程级共享模型（破坏性，0.x）。

### Fixed

- **P0 `Tensor::View` / `ConstView` 分叉二次索引丢前缀**：`operator[]` 原以
  `std::move(_shape)` 把视图内部路径直接搬运到返回对象，自身 `_shape` 处于
  moved-from 置空状态——从同一命名视图二次分叉
  （`auto row = t[0]; row[1]; row[2];`）时第二次路径丢失前缀
  （`[0]` → `[]` → `[2]` 而非 `[0, 2]`），轻则写入错误偏移，重则触发越界访问。
  改为“拷贝前缀 + 追加”的值语义派生，使视图可从任意位置多次分叉得到独立
  路径（NumPy 风格基本用法）；同步去除 `_shape` 的 `mutable` 修饰（不再修改
  自身）。`ConstView` 同修。`TensorTest` 新增用例 #13 覆盖：View 分叉写 /
  循环分叉写 / ConstView 分叉读——此前测试仅走一次性线性链，无法暴露。

### Changed

- README “类 NumPy 链式视图索引”段改写：明确当前 `View` 仅支持逐维标量索引
  与分叉（仍为零拷贝），区间切片/重塑/转置**未实现**、属规划特性——原描述自
  v0.1.0 起即与实现对齐不符（`git log --all -S 'Tensor::reshape|transpose|slice'`
  零命中），本次直接取消该描述。
- README 新增顶层“已知限制”节，上提以下项目到显眼位置：N:1 串行化汇聚未支持
  （原仅 FAQ 末尾小字注）；视图区间切片/重塑/转置未实现；单个 >4 GiB 模型文件
  不能入 .dcg（zip64 未启用）；macOS 不在 CI 验证矩阵；图拓扑保持扁平，子图能力
  由 `GraphOperator` 封装。
- **资源调度器升级为进程级共享模型（破坏性，0.x）**：资源隔离不再依赖“每图
  自建三线程池”的线程副作用，改由进程级 `ResourceScheduler` 承载。
  `ThreadPoolAffinity` 更名 `ResourceClass`（无兼容别名；`Node::affinity()`
  返回类型与 `makeNode` 默认参数/`NodeMeta` 同步，JSON wire 字符串
  "Compute"/"Operator"/"System" 不变，图档兼容）；`InferGraph` 构造由池
  配置改为调度器注入（缺省 `nullptr` = `ResourceScheduler::instance()`，多图
  默认共享进程预算；默认各类 1 槽位，对齐旧单图默认），`ExecutionEngine`
  构造改为调度器注入（空指针抛 `std::invalid_argument`）。等待型节点
  （`GraphOperator`）等待期间占住资源类槽位：进程预算需覆盖全部并发等待
  节点数，嵌套等待链按“同类槽位叠加”规划，否则可能自锁。执行引擎析构改为
  自排水（停止新派发 → 标记轮次终止 → 等待在飞任务完成；调度器已关闭时
  走放弃等待逃逸路径）。

### Added

- `ResourceScheduler` / `SchedulerConfig`：进程级共享调度器与用户可控预算
  （每资源类 worker 数，非法配置抛 `std::invalid_argument`）；每类惰性创建
  独立 `ThreadPool`，`submit`/`shutdown`/`isStopped` 契约与池一致；
  `instance()` / `configureInstance()`（仅首次创建前有效）/ `resetInstance()`
  静态接口支撑进程默认实例与测试隔离；`InferGraph` 支持注入自定义实例。
- `Graph/ResourceClass.h`：`ResourceClass {Compute, Operator, System}` 自
  `Node.h` 迁出为独立头文件（含语义文档注释）。
- `ResourceSchedulerTest`：调度器串行/并发上限、资源类隔离、配置校验、
  关闭拒绝、全局实例语义（幂等/首配/重置）与引擎析构排水回归用例。

## [0.6.2] - 2026-09-19

发布前全库审查发现的 DCNet 并发/内存安全与发布链路缺陷修复。0.6.1 未打 tag、
未对外发布，其变更随本版本一并发布。

### Fixed

- **P0-1 监听器停止期 use-after-free**：`HttpListener` 排水改以**工作线程存活计数**
  `_activeThreads` 为准，且计数在 accept 期与连接入册同一临界区内完成（**先于起线程**，
  故“线程已创建但尚未调度”也在账内；worker 退出时回收），不再依赖“请求完整读入
  后”才自增的 `_inFlight`——半开/慢速连接（请求未读完、不计在途）不再存在
  “`stop()` 已返回、分离 worker 仍访问已析构成员”的窗口。`stop()` 返回即蕴含全部
  worker 已退出，紧随其后的析构安全；线程创建失败时同步回滚该账（不致挂起）。
- **P1-2 `HttpTransport` 共享实例并发交换竞态**：引擎实例按 `engineType:modelPath`
  缓存复用，而执行互斥粒度在节点级——两节点可并发进入同一 transport。现以
  「交换权」（标志 + 条件变量，非跨线程解锁互斥）把一次 `send → recv` 整体串行化：
  `send` 领取占用，`recv`/失败/异常/`close` 释放，`_session`/`_response`/`_failed` 的
  访问均受占用保护，消除请求/响应错配与数据竞争（DESIGN.md §2.3 “适配器必须
  呈现同步、线程安全接口”契约达成）。
- **P1-3 监听器状态标志数据竞争**：`_started`/`_stopped` 改 `std::atomic<bool>`——
  此前 `acceptLoop` 与 `alive()` 的无锁读与 `stop()` 的锁内写构成 data race（UB）。
- **P1-4 慢速连接致线程耗尽（DoS）**：新增连接级闸门 `maxConnections`（默认 32，
  0 = 不限制），配额检查与连接入册同一临界区（无超限窗口），超限连接在 accept 期
  就地关闭、不起工作线程；同时 `readRequest` 引入**单请求读总预算**（每次 receive 前
  把超时收紧到剩余时间），滴灌连接到点自行释放线程。
- **P2-5 tensor/text codec 违反错误分类契约**：`decodeResponse` 的 JSON 解析与 dtype
  解码异常统一改抛 `DcCodecRemoteError`（此前直接漏出 `nlohmann::parse_error` /
  `std::runtime_error`，被标准 RunFn 归入 `InternalError` 并丢失 dcnet 诊断码），
  现与 OpenAI codec 一致，映射为 `ExecutionFailed` + `Diagnostic{dcnet, RemoteMalformed}`。
- **P2-6 安装冒烟与文档版本请求在 0.6.x 下必然失败**：`examples/install_smoke` 与
  README 引入示例的 `find_package(DCinfer 0.5)` 与本包 `SameMinorVersion` 兼容策略
  冲突（请求的 major.minor 必须与已安装版本一致），升 minor 后按文档执行即报
  “not compatible”。两处均改为 `0.6`，并说明每次 minor 升级需同步、或可省略版本请求。
- **P2-7 安装版本元数据可被宿主版本污染**：四个模块的 `DCINFER_PKG_VERSION` 此前
  无条件取 `PROJECT_VERSION`，宿主直接 `add_subdirectory` 单模块目录时会写入宿主
  版本号；现仅当 `PROJECT_NAME` 为 DCinfer 时采用 `PROJECT_VERSION`，否则用字面版本。
- **P2-8 CI 覆盖缺口**：`DCINFER_BUILD_IR` 默认转 OFF 后，所有 release 预设均不再
  构建 DCIr——`DcgArchiveSecurityTest` 等反序列化安全用例在 CI 中零覆盖。三个
  release 预设（gcc/clang/msvc）显式置 `DCINFER_BUILD_IR=ON`；TSan job 并发过滤集
  补入 v0.6.1 新增的 `ExecutionConcurrencyTest|ThreadPoolTest|ValueSharingTest|FailureClosureTest`。

### Added

- `NetServerEndpoint::maxConnections`：服务端并发连接（工作线程）上限，与
  `maxInFlight`（在途请求 429 闸门）分层——前者约束连接级资源（含半开连接），
  后者约束请求级过载。
- DCNet 回归用例：`stopDrainsHalfOpenConnection`（P0 排水）、
  `connectionCapRejectsExcess`（连接闸门）、
  `transportSerializesSharedEndpointExchanges`（共享 transport 并发不串号）、
  `malformedRemoteResponseMapsToRemoteMalformed`（codec 分类契约）。

### Changed

- **`DcNetListener::stop()` 语义收紧**：由“等待在途请求（受 requestTimeout 约束）”
  改为“等待全部已接受连接的工作线程退出；grace（`max(requestTimeout + 1s, 5s)`）
  到期后强制关闭在册连接再等其退净”。返回后对象可安全析构。socket 层面的阻塞
  由单请求读预算与强制关闭双重有界；但本地引擎若在 handler 内永久挂起，stop()
  会随之挂起——以挂起换内存安全（不再提前放行排水）。
- **`HttpTransport` 并发行为**：同一实例上的并发 `send/recv` 由未定义行为改为
  串行执行（连接复用、交换串行）。`send` 与 `recv` 应成对在同一线程调用（RunFn
  契约）；跨线程未收尾的占用由同线程下一次 `send` / `close` / 析构回收。

## [0.6.1] - 2026-09-17

### Fixed

- **#2（P0）僵尸重试污染已完成/已复用任务**：`_terminate()` 终态迁移成功处统一
  置 `round->terminated`（确立"已收尾 ⇒ 已终止"不变量）——经节点执行门排队
  （`enqueueRetry`，不计 `inflight`）的迟到重试与迟到的在飞传播在提交/执行/
  传播全部拦截点被丢弃，不再事后覆盖已发布结果，也不再向同 ID 复用轮次写入
  计数与结果。
- **#3（P0）收尾抢救与传播线程竞态跳过 `clearTaskState`**：`TaskBuffer` 新增
  `tryTakeOutput`（一次加锁完成"检查+取数"，无输出返回 `nullopt`，不抛异常）
  替换 `_terminate` 抢救段与 `_propagateFrom` 第二/三步的 check-then-act 序列；
  抢救段移入 `round->m` 临界区并以 RAII 守卫保证 `clearTaskState` 必然执行
  ——`cancel()` 不再有异常逃逸路径，同 ID 复用不再继承陈旧执行态（残留输入
  重放被根除）。
- **#8-12 `feedInput` 收尾窗口输入丢失**：`ExecutionEngine` 新增
  `tryWriteTaskState`（持 `round->m` 检查"终态已发布、resultsReady 未置"的
  收尾窗口并拒绝写入），`InferGraph::feedInput` 经其执行 taskState + setInput
  写入，拒绝时统一抛 `DuplicateTask`——Running→finalizing 竞态窗口内输入
  不再静默丢失，同 ID 复用（重试）语义完整。
- **#4（P1）`~ExecutionEngine` 析构期跨池提交 UB**：析构起始显式按
  `_computePool` → `_operatorPool` → `_systemPool` 顺序 `shutdown()` 全部
  join 后再移出轮次表；配合 `submit` 的关闭检查（#8-1），残余跨池竞态收敛
  为良性丢弃（不再持已销毁互斥、写已销毁队列）。
- **#5（P1）`TensorSlot::store` 构造抛出致悬垂指针（UAF/双释放）**：改为先在
  局部完成新值构造、成功后再释放旧值并接手（强异常安全）——构造失败时旧值
  完好，`peek`/`take` 不再读到已释放内存。
- **#6（P1）`GraphOperator` 子任务资源泄漏 + 绑定输出缺失静默**：RunFn 全流程
  套 RAII 守卫（析构 `detachTask`），含取数抛出在内的任何退出路径都不再滞留
  子图结果与 OutputZone 条目；绑定输出缺失由静默跳过改 fail-fast
  （`ExecutionFailed` 并携带缺失端口别名列表）。
- **#7（P1）派发失败致 `inflight` 永久泄漏、任务挂起**：`ThreadPool::submit`
  改返回 `bool`（池已关闭或入队失败 → false，含内存压力下 `bad_alloc` 不再
  抛异常）；`_submitNodeRun` 对"拒绝 + 异常"统一回滚计数并按 Failed 收尾
  （与 RunDone 归零语义一致），`_dispatchToPool` 透传三池结果——不再绕过
  maxHops 与宿主护栏无限 Running。
- **#8（P2 清理 17 项）**：`InferGraph::submit` 声明遍历补节点/端口存在性校验
  （拼写错误不再致任务悬挂，#8-13）；`EngineRegistry::registerOperator` 检查
  与插入合并单临界区（并发同名不再静默覆盖，#8-2）；`GraphStore::nodeNames`
  补锁（#8-3）；`TensorData::expand` 补空 shape/秩不等校验并修复 1D 单块填充
  路径（#8-5）；`TensorData::write(element)` 删除死代码预计算（#8-6）；
  `buildCache` 越界截断路径补断言诊断（debug 构建暴露数据不一致，#8-7）；
  `buildView` 标量/1D 降级路径补 `setViewFlag()`（#8-8）；`TaskBuffer::taskCount`
  改并集口径（`_taskInputs ∪ _taskOutputs`）与 `hasTask` 一致（#8-17）。

### Changed

- **`ThreadPool::submit` 返回值契约**：`void` → `bool`（false = 池已关闭或
  入队失败，任务未被接受）；任务队列无界设计意图注释显式声明。
- **`EnvRegistry::getOrCreate` 句柄化**：`void*` → `std::shared_ptr<void>`
  ——`release`/`releaseAll` 仅移除缓存，外部持有句柄期间实例存活；`nullptr`
  语义保留（#8-15）。
- **`Value::isPublished` 粘性语义**：`share()` 同时置位源句柄"曾发布"标记
  ——别名消亡后源句柄仍报告已发布；`takeOutput` 克隆路径据此收紧（共享/
  冻结载荷不逃逸，方向更安全，#8-9）。
- **绑定查询改值副本返回**：`InferGraph`/`GraphStore`/`GraphBuilder`/
  `InputZone` 的 `bindings`/`inputBindings` 由锁外引用改返回值副本（#8-4）。
- **`NodeExecutionGate` 清理**：删除死状态 `_currentTaskId` 与
  `setCurrentTask`/`clearCurrentTask`（连带 `ExecutionPipeline` 调用点）；
  `enqueueRetry` 同 key 去重改"后到覆盖"（消除 TTL 轻微漂移，#8-10）。
- **`GraphOperator` childTid 转义**：taskId/节点名中的 `\` 与 `|` 转义后拼接
  （单射不碰撞，#8-14）。
- **`ExecutionEngine` 移动操作显式 `= delete`**（原 `= default` 实为删除），
  头文件注释修正（`_isTerminated` → `_exhaustedCheck`，#8-11）。
- README："灵活的图拓扑"节声明同口多上游串行化汇聚（N:1）为规划特性
  （当前未实现，#8-16）。
- 测试：新增 `ThreadPoolTest`（submit 返回值语义）与
  `TaskLifecycleRegressionTest`（门重试×收尾交叉、cancel×count>1 完成竞态
  循环 + 同 ID 复用、feed×finalizing 竞态）；`TensorDataTest`、
  `GraphOperatorTest`、`TensorSlotTest`、`EnvRegistryTest`、`ValueSharingTest`、
  `NodeTest` 增补用例与断言。

## [0.6.0] - 2026-09-17

### Added

- **组合算子 `GraphOperator`（新增 `include/Compose/GraphOperator.h`）**：把一整张
  推理图包装成普通 Node 的算子工厂（构造 → `makeNode` → `addNode`），仅使用
  公开 API、不新增任何 InferGraph 特判。构造取 `shared_ptr<InferGraph>` 接管
  共享所有权并立即 freeze（构造即定型：消灭懒冻结竞争与 schema 漂移窗口）；
  按绑定推导 Schema（端口名 = 绑定 alias，类型/形状/required 从目标端口拷贝；
  节点/端口缺失与连接器目标构造期 fail-fast）。执行语义：子任务 ID 按
  "父任务ID|实例号|节点名" 命名空间隔离——同一子图被多个组合节点/多个父图
  并发复用无 DuplicateTask 限制（旧设计公开缺陷解除）；内建协作式取消联动
  （`Options.pollInterval` 默认 100ms 轮询父轮取消 → 取消子图任务解围，
  等待型节点不再因内层信号停滞永久占住池线程）；终态自动回收子任务资源；
  内层诊断带上下文转发（block 名 + 子任务 ID + 首条错误消息）。

- **零拷贝广播与值共享发布语义（数据面重构）**：`Value` 载荷改为 `shared_ptr`
  承载（保留类型删除器，独占投递路径零开销），新增 `share()`（共享只读别名）/
  `isShared()`/`isPublished()`（发布位）/`cloneOwned()`（take 语义：出口取得
  独立可变所有权）；引用计数仅在显式 `share()` 时产生。`TensorData`/`Tensor`
  新增冻结门（`freeze()`/`isFrozen()`：冻结后写路径抛 `TensorException(Frozen)`；
  有效标志原子化与惰性物化串行化；拷贝/移动产出非冻结副本）；
  `ValueCloneRegistry` 按 `SlotDataType` 注册深拷贝函数（未注册类型的共享载荷
  只读消费）。`Broadcast(N>1)` 改为发布时一次性 freeze、share N 份零拷贝只读
  分发（替代拷贝 N-1 份 + move 最后一份）。

### Removed

- **`InferGraph::exportNode` 及核心侧子图特判整组移除**（破坏性变更）：组合/
  复用推理图改由算子层 `GraphOperator` 实现，核心图语义保持扁平。连带移除：
  生命周期哨兵 `_lifeToken` 与"子图必须存活于导出节点使用期"契约（shared_ptr
  接管后悬垂不可表示，CORE-04 整类问题消失）；`Node::setBlockedOverride`
  与信号感知静态预演 `canSatisfyDeclarations`（运行期阻塞语义保持单点实现；
  `canSatisfyTopologically` 保留，submit 提交期守卫在用）；
  `GraphException::DuplicatePort`（alias 推导后接口层命名碰撞不可表示）；
  README "嵌套子图（exportNode）的生命周期约定" 一节。

### Changed

- **同一输出端口二次 `connect()` 升级为自动扩容**（v0.5.2 `DuplicateEdge`
  fail-fast 语义的可用性修正）：既有连接由广播导线承载（自动导线 / 显式
  `Broadcast` 的 in 接线）时原地扩容为 N 路扇出——增加连接即扩扇出、返回同一
  连接器引用，无需手写 `Broadcast(N)`；非广播拓扑（不含连接器的直连构造等）
  保持构图期 `DuplicateEdge` fail-fast；扇入守卫改为按输入口去重（同一输入口
  二次驱动仍构图期拒绝）。README 1:N 分发示例与 FAQ 同步更换。
- `take` 边界语义：`InferGraph::takeOutput`/`takeOutputTensor` 对发布残留载荷
  （广播共享/冻结）自动克隆为独立可变副本——take 即得可变所有权，冻结共享
  载荷不逃逸出图边界。
- DCIr 序列化穿透连接器链展开逻辑边：1:1 链（自动导线 / 包裹导线）折叠为
  单条直连边；链上出现多路分发 Broadcast 时展开边标 `mode=broadcast`
  （重建还原为同一分发组）；防环（节点+入口去重）与悬空连接器丢弃。
- README：FAQ 一节改写为"如何组合/复用一张推理图？"（GraphOperator 用法、
  shared_ptr 生命周期、等待型节点占池语义、序列化待遇）。
- 测试重组：新增 `GraphOperatorTest`（13 用例：基本往返/分支/三层嵌套/环+TTL/
  链式/Schema 推导/空接口拒绝/构造即冻结/父取消解围（自 NestedGraphCancelTest
  迁移）/同一子图并发复用/共享所有权生命周期/诊断转发/构造校验）；
  `GraphNodeTest` 的通用用例（绑定内省/wait/并发压力/CORE-01 扇出扩容/
  扇入守卫）迁入 `InferGraphTest`；删除 `GraphNodeTest`、`NestedGraphCancelTest`；
  `FreezeBoundaryTest` 移除 setBlockedOverride 冻结断言；新增 `ValueSharingTest`
  （冻结门/共享别名/clone 语义/广播零拷贝分发/跨边界所有权）；`GraphCompilerTest`
  增补扩容往返稳定性；`LoweringTest` 增补扩容导线保留与双分支值分发。
- CI：until-fail 重复名单与 TSan 子集改用 `GraphOperatorTest`（原 `GraphNodeTest`）。

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

[Unreleased]: https://github.com/suzvka/DCinfer/compare/v0.7.1...HEAD
[0.7.1]: https://github.com/suzvka/DCinfer/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.7.0
[0.6.2]: https://github.com/suzvka/DCinfer/releases/tag/v0.6.2
[0.6.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.6.0
[0.5.2]: https://github.com/suzvka/DCinfer/releases/tag/v0.5.2
[0.5.1]: https://github.com/suzvka/DCinfer/releases/tag/v0.5.1
[0.5.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.5.0
[0.4.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.4.0
[0.3.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.3.0
[0.2.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.2.0
[0.1.0]: https://github.com/suzvka/DCinfer/releases/tag/v0.1.0
