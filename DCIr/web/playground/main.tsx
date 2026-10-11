/**
 * Playground — 编辑器最小演示宿主（二次开发参考样板）
 *
 * 演示四个宿主职责（编辑器本体一概不管）：
 * 1. 调色板源装配：catalogSources 桥接旧快照目录 + counterSource 自定义视图程序
 *   （协议一元化：数据源 = browse 组合子的 fetch 取样，自定义 = 直接交 program）；
 *   不传 sources 时缺省由 catalogProvider prop 桥接生成（旧接入零改动）；
 * 2. side-band 存取：EditorValue（含 layout 坐标）落 localStorage 草稿；
 * 3. 图 JSON 出入档：serializeGraph 导出 / applyImport 导入（.dcg 打包属宿主侧，
 *   对应 C++ DcgArchive；编辑器只面对 graph.json 本体）。
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

import {
  applyImport,
  emptyGraph,
  fPort,
  fillLayout,
  sanitizeGraph,
  serializeGraph,
  staticCatalogProvider,
  type CatalogModelItem,
  type CatalogView,
  type EditorIssue,
  type EditorValue,
} from "../src/core/index";
import InferGraphEditor, { catalogSources, ioSource, type EditorTheme, type PaletteSource } from "../src/ui/index";

// ── 1. 静态目录（示例条目；真实场景由注册表导出或 HTTP 提供器接入）──

const DEMO_CATALOG: CatalogView = {
  groups: [],
  endpointCount: 0,
  labels: ["基础算子", "模型"],
  models: [
    {
      modelKey: "builtin/add",
      engineType: "Builtin",
      endpoints: [],
      labels: ["基础算子"],
      io: { inputs: [fPort("a", [-1]), fPort("b", [-1])], outputs: [fPort("out", [-1])] },
    },
    {
      modelKey: "builtin/mul",
      engineType: "Builtin",
      endpoints: [],
      labels: ["基础算子"],
      io: { inputs: [fPort("a", [-1]), fPort("b", [-1])], outputs: [fPort("out", [-1])] },
    },
    {
      modelKey: "models/mnist.onnx",
      engineType: "OnnxRuntime",
      endpoints: [],
      labels: ["模型"],
      io: { inputs: [fPort("data", [-1, 1, 28, 28])], outputs: [fPort("prob", [-1, 10])] },
    },
  ],
};

// 模块级常量：provider 引用稳定（旧快照契约；此处经 catalogSources 桥接为源组，
// 也可不传 sources 而直接传 catalogProvider prop 走编辑器内置缺省装配）
const catalogProvider = staticCatalogProvider(DEMO_CATALOG);

// ── 2. 自定义源示例：完全自持交互面（不用行/详情卡/搜索框），产出仍经 ctx.place ──

const COUNTER_ITEM: CatalogModelItem = {
  modelKey: "demo/counter",
  engineType: "Builtin",
  endpoints: [],
  labels: ["自定义"],
  io: { inputs: [fPort("tick", [-1])], outputs: [fPort("value", [-1])] },
};

const counterSource: PaletteSource = {
  id: "counter",
  label: "自定义示例",
  icon: "✎",
  program: ({ ctx }) => {
    // 程序体可用 hooks（ActiveProgram 以组件语义调用，切组重挂载）
    const [used, setUsed] = useState(0);
    return (
      <div style={{ padding: 14, display: "grid", gap: 10, alignContent: "start", color: "inherit" }}>
        <p style={{ margin: 0, fontSize: 13, lineHeight: 1.6 }}>
          自定义视图程序：不受行/详情卡形态约束，写通道仍是唯一的 ctx.place
          （窄校验 + 级联落点 + 撤销栈全部自动生效）。
        </p>
        <button type="button" className="ig-btn ig-primary" onClick={() => { ctx.place(COUNTER_ITEM); setUsed((u) => u + 1); }}>
          放置 counter 节点（已放置 {used} 次）
        </button>
        <p style={{ margin: 0, fontSize: 12, opacity: 0.7 }}>画布当前节点数：{ctx.value.graph.nodes.length}（只读快照）</p>
      </div>
    );
  },
};

// ── 3. 草稿存取（side-band 与图 JSON 一并落 localStorage）──

const DRAFT_KEY = "dcir-graph-editor-playground";

function emptyValue(): EditorValue {
  return fillLayout({ graph: emptyGraph(), layout: {}, remote: {} });
}

function loadDraft(): EditorValue {
  try {
    const raw = localStorage.getItem(DRAFT_KEY);
    if (!raw) return emptyValue();
    const o = JSON.parse(raw) as Partial<EditorValue> | null;
    if (typeof o !== "object" || o === null || !o.graph) return emptyValue();
    const { graph } = sanitizeGraph(o.graph);
    return fillLayout({ graph, layout: o.layout ?? {}, remote: o.remote ?? {}, io: o.io });
  } catch {
    return emptyValue();
  }
}

// ── 4. 宿主页 ──

function App() {
  const [value, setValue] = useState<EditorValue>(loadDraft);
  const [issues, setIssues] = useState<EditorIssue[]>([]);
  const [theme, setTheme] = useState<EditorTheme>("system");
  const fileRef = useRef<HTMLInputElement | null>(null);

  // 源装配：内置编排组 + 目录桥接组 + 自定义示例（引用稳定：模块级常量 + 一次 memo）
  const sources = useMemo(
    () => [ioSource, ...catalogSources(catalogProvider, DEMO_CATALOG.labels), counterSource],
    [],
  );

  // 宿主 chrome 跟随主题：同步 <html> 的 color-scheme（light-dark() 只认生效方案，
  // 编辑器在 .ig-root 局部设的方案不向上传播；这段即宿主自适配的标准写法）
  useEffect(() => {
    document.documentElement.style.colorScheme = theme === "system" ? "light dark" : theme;
  }, [theme]);

  useEffect(() => {
    try {
      localStorage.setItem(DRAFT_KEY, JSON.stringify(value));
    } catch {
      // 隐私模式 / 配额不足：草稿降级为内存态，不阻塞编辑
    }
  }, [value]);

  const errorCount = issues.filter((i) => i.level === "error").length;

  const exportJson = () => {
    const blob = new Blob([JSON.stringify(serializeGraph(value), null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "graph.json";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const importFile = (file: File) => {
    void file.text().then((text) => {
      const r = applyImport(value, text);
      setValue(fillLayout(r.value));
      if (r.issues.length > 0) {
        window.alert(`导入清洗 ${r.issues.length} 项：\n${r.issues.map((i) => `[${i.level}] ${i.detail}`).join("\n")}`);
      }
    });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div className="pg-bar">
        <strong>DCIr 图编辑器 · Playground</strong>
        <span className="pg-status">{errorCount > 0 ? `${errorCount} 个错误（不建议导出）` : "就绪"}</span>
        <span className="spacer" />
        <label>
          主题
          <select value={theme} onChange={(e) => setTheme(e.target.value as EditorTheme)}>
            <option value="system">跟随系统</option>
            <option value="dark">深色</option>
            <option value="light">浅色</option>
          </select>
        </label>
        <button type="button" onClick={exportJson}>导出 graph.json</button>
        <button type="button" onClick={() => fileRef.current?.click()}>载入 .json</button>
        <input
          ref={fileRef}
          type="file"
          accept=".json,application/json"
          style={{ display: "none" }}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) importFile(f);
            e.target.value = "";
          }}
        />
        <button type="button" onClick={() => setValue(emptyValue())}>清空</button>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <InferGraphEditor
          value={value}
          onChange={setValue}
          onIssues={setIssues}
          sources={sources}
          theme={theme}
          height="100%"
        />
      </div>
    </div>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
