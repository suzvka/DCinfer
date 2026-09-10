# DCEngine::OpenAI — OpenAI 兼容远端服务引擎适配器

> 把任何 OpenAI 兼容服务（vLLM / TGI / llama.cpp server / 云 API）以统一形态
> 接入 DCinfer 图运行时。基于 [DCNet](../../DCNet/DESIGN.md) 张量网络传输框架：
> HttpTransport（POCO，跨平台）+ chat codec（`DC::Net::DcNetCodec` 契约实现）+
> NetError 归一化。与 OnnxRuntime（本地模型后端）对称并列，同属
> EngineDescriptor 家族。

## 使用

```cpp
#include "DCEngine/OpenAiEngine.h"

// 注册（可多次注册不同 engineType / model / 鉴权变体；同 engineType 保留首次注册）
DC::OpenAI::registerOpenAiEngine(EngineRegistry::instance(), {.model = "gpt-4o"});

// 建节点：modelPath 即远端端点
auto node = EngineRegistry::instance().createNode(
    "OpenAI", "llm", "http://192.168.1.10:8080/v1");

// 端口（本地形状规则，不依赖远端）：
//   in  prompt（Data，必填）
//   in  system（Data，可选）
//   in  params（Data，可选：请求级采样参数 JSON，逐请求覆盖；
//               非法 JSON → InvalidInput）
//   out response（Data；响应缺 choices[0].message.content → ExecutionFailed，
//               附 dcnet 领域诊断 code=RemoteMalformed）
```

### 鉴权 / 超时 / 重试（云 API 接入）

```cpp
DC::OpenAI::registerOpenAiEngine(EngineRegistry::instance(), {
    .model = "gpt-4o",
    .engineType = "OpenAI.Cloud",
    .authToken = "sk-...",            // 裸 key 自动补 "Bearer " 前缀；不写入日志/错误信息
    // .tokenProvider = [] {          // 动态取 token（注册时求值一次，优先于 authToken）
    //     return std::getenv("OPENAI_API_KEY");
    // },
    .headers = {"X-Custom: v1"},      // 附加请求头
    .connectTimeout = std::chrono::milliseconds{5000},
    .requestTimeout = std::chrono::milliseconds{30000},
    .maxRetries = 2,                  // 传输级失败（超时/拒连/5xx/429）退避重试
});
auto node = EngineRegistry::instance().createNode(
    "OpenAI.Cloud", "llm", "https://api.openai.com/v1");
```

错误分类：非法 params JSON → `InvalidInput`；远端响应非 JSON 或缺
`choices[0].message.content` → `ExecutionFailed`（附 dcnet 领域诊断
`code=RemoteMalformed`；核心枚举保持通用）；网络/服务错误 → `ExecutionFailed`
（HTTP 语义经 `NetError` 归一化）。`content: ""` 是合法空内容，正常成功返回。

请求路径：`{basePath}/chat/completions`（OpenAI 兼容协议面）。

## 构建

```bash
# OpenAI 适配器依赖 DCNet（HTTP 传输）；未启用 DCNet 时 CMake 直接 FATAL_ERROR（不再静默跳过）
cmake -B build -S . -DDCINFER_BUILD_DCNET=ON -DBUILD_ENGINE_OPENAI=ON
```

依赖：`DCinfer::DCinfer` + `DCNet::DCNet` + `nlohmann-json`
（vcpkg feature `net` 已含 nlohmann-json 与 poco[netssl]，随 `DCINFER_BUILD_DCNET=ON` 自动安装）。

## 测试

`OpenAiEngineTest`（MockHttpServer 假远端，真实 HTTP 传输）：

- chat 端到端：prompt/system/params → response，校验 model / messages / stream / 参数覆盖
- 鉴权：Bearer Token 注入 Authorization 头（服务端可断言）
- 错误分类：非法 params → InvalidInput；响应缺 content / 非 JSON → ExecutionFailed
  （dcnet 诊断 code=RemoteMalformed）；content="" 合法空内容成功返回
- 重试：500 一次后成功（maxRetries=1，服务端恰好收到 2 次请求）
- 失败路径：远端 500 → NetError 归一化（`ExecutionFailed` / `remote:server_error`）

## 设计要点

- **双向翻译器**（DCNet DESIGN.md ADR-3）：本适配器是"DC 端口 ↔ OpenAI 报文"
  的翻译器；协议错误码精确映射可调用 DCNet 核心归一化函数
  （`NetError.h`）扩展，分类规则归核心统一维护
- **本地形状规则**：端口 Schema 静态声明，不依赖远端元数据（DCNet DESIGN.md §1.2）
- 流式输出 / embedding 端口 / mTLS 属展望（DCNet DESIGN.md §12）
