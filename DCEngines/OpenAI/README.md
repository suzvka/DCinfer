# DCEngine::OpenAI — OpenAI 兼容远端服务引擎适配器

> 把任何 OpenAI 兼容服务（vLLM / TGI / llama.cpp server / 云 API）以统一形态
> 接入 DCinfer 图运行时。基于 [DCNet](../../DCNet/DESIGN.md) 张量网络传输框架：
> HttpTransport（POCO，跨平台）+ chat codec（`DC::Net::DcNetCodec` 契约实现）+
> NetError 归一化。与 OnnxRuntime（本地模型后端）对称并列，同属
> EngineDescriptor 家族。

## 使用

```cpp
#include "DCEngine/OpenAiEngine.h"

// 注册（可多次注册不同 engineType / model 变体）
DC::OpenAI::registerOpenAiEngine(EngineRegistry::instance(), {.model = "gpt-4o"});

// 建节点：modelPath 即远端端点
auto node = EngineRegistry::instance().createNode(
    "OpenAI", "llm", "http://192.168.1.10:8080/v1");

// 端口（本地形状规则，不依赖远端）：
//   in  prompt（Data，必填）
//   in  system（Data，可选）
//   in  params（Data，可选：请求级采样参数 JSON，逐请求覆盖）
//   out response（Data）
```

请求路径：`{basePath}/chat/completions`（OpenAI 兼容协议面）。

## 构建

```bash
cmake -B build -S . -DBUILD_ENGINE_OPENAI=ON   # 默认 ON；依赖 BUILD_DCNET=ON
```

依赖：`DCinfer::DCinfer` + `DCNet::DCNet` + `nlohmann-json`（已在核心依赖树）。
外部消费方需在自己的 `vcpkg.json` 声明 `nlohmann-json`。

## 测试

`OpenAiEngineTest`（MockHttpServer 假远端，真实 HTTP 传输）：

- chat 端到端：prompt/system/params → response，校验 model / messages / stream / 参数覆盖
- 失败路径：远端 500 → NetError 归一化（`ExecutionFailed` / `remote:server_error`）

## 设计要点

- **双向翻译器**（DCNet DESIGN.md ADR-3）：本适配器是"DC 端口 ↔ OpenAI 报文"
  的翻译器；协议错误码精确映射可调用 DCNet 核心归一化函数
  （`NetError.h`）扩展，分类规则归核心统一维护
- **本地形状规则**：端口 Schema 静态声明，不依赖远端元数据（DCNet DESIGN.md §1.2）
- 流式输出 / embedding 端口 / mTLS 属展望（DCNet DESIGN.md §12）
