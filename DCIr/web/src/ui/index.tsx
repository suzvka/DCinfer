/**
 * InferGraphEditor — 可视化推理图编辑器（一站式嵌入式组件）
 *
 * 契约：受控组件（value / onChange），零宿主框架耦合（仅 React + @xyflow/react），
 * 可嵌入任意页面。分层：DCIr JSON 为唯一权威（core 层纯函数）；React Flow 仅为
 * 投影与交互壳。坐标（layout）、远程意图（remote）、IO 控件元素（io）为编辑器侧
 * side-band，序列化时剥离。
 *
 * 扩展注入点：sources（调色板源，一元协议见 palette-protocol.ts；旧
 * catalogProvider 快照经内置桥接降解为缺省源组）/ layout（布局策略）/
 * theme（dark/light/system，token 层见 styles.css 头部）。
 * 交互规则收口见 core/rules.ts；引用完整性见 core/model.ts 各操作函数。
 */

import {
  Background,
  BackgroundVariant,
  ConnectionMode,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  applyEdgeChanges,
  applyNodeChanges,
  useReactFlow,
  useStoreApi,
  type Connection,
  type EdgeChange,
  type NodeChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import "./styles.css";
import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent, type MouseEvent as ReactMouseEvent } from "react";

import {
  IO_ELEM_HEIGHT,
  IO_ELEM_WIDTH,
  NODE_WIDTH,
  autoLayoutStrategy,
  cascadePlace,
  fillLayout,
  nodeHeight,
  type LayoutStrategy,
  type ViewportRect,
} from "../core/layout";
import {
  addEdge,
  addIoElement,
  addModelNode,
  addSignalLogicInput,
  addSignalNode,
  applyImport,
  bindIoElement,
  edgeIdOf,
  ioElementIdOfEdge,
  isIoEdgeId,
  parseHandle,
  patchNode,
  removeEdge,
  removeIoElement,
  removeNode,
  removeSignalLogicInput,
  renameNode,
  serializeGraph,
  setRemote,
  setSignalConstValue,
  setSignalSourceName,
  toFlowModel,
  unbindIoElement,
  type InferNodeData,
} from "../core/model";
import { IG_MODEL_MIME, parseCatalogItem, sanitizeCatalogItem, type CatalogModelItem, type CatalogProvider } from "../core/catalog";
import { checkConnection, issueLevelByNode, validateEditorValue } from "../core/rules";
import type { EditorIssue, EditorValue, SignalLogicKind, XY } from "../core/types";
import type { AnyRfNode, RfEdge } from "./flow-types";
import { InferIoNode, InferNode } from "./nodes";
import { IG_IO_MIME, IG_MOBILE_QUERY, IG_SIGNAL_MIME, type PaletteHost, type PaletteSource } from "./palette-protocol";
import { Palette } from "./palette";
import { Inspector, IoInspector, IssuePanel, JsonDrawer } from "./panels";
import { defaultSources } from "./sources/catalog";
import { useCatalog } from "./use-catalog";

// 模块级稳定引用（React Flow 警告项：nodeTypes 不得在渲染内重建）
const NODE_TYPES = { inferNode: InferNode, inferIoNode: InferIoNode };

/** 主题（system = 跟随宿主 prefers-color-scheme） */
export type EditorTheme = "dark" | "light" | "system";

// 画布装饰色板（随解析主题切换；表面/边线由 styles.css token 层接管）
const CANVAS_ACCENTS = {
  dark: { dot: "rgba(120, 145, 190, 0.25)", mask: "rgba(10, 14, 23, 0.72)", ioIn: "#2f9e8f", ioOut: "#b9832f", node: "#4f8cff" },
  light: { dot: "rgba(70, 95, 140, 0.3)", mask: "rgba(222, 230, 241, 0.72)", ioIn: "#0d9464", ioOut: "#b45309", node: "#3b76e8" },
} as const;

/** system 主题解析（matchMedia 监听实时切换；无 matchMedia 环境按暗色 = 组件缺省色板） */
function useResolvedTheme(theme: EditorTheme): "dark" | "light" {
  const [systemDark, setSystemDark] = useState<boolean>(
    () => typeof window === "undefined" || window.matchMedia?.("(prefers-color-scheme: dark)").matches === true,
  );
  useEffect(() => {
    if (theme !== "system") return;
    const mq = typeof window !== "undefined" ? window.matchMedia?.("(prefers-color-scheme: dark)") : undefined;
    if (!mq) return;
    const onChange = (e: MediaQueryListEvent) => setSystemDark(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [theme]);
  return theme === "system" ? (systemDark ? "dark" : "light") : theme;
}

export interface InferGraphEditorProps {
  value: EditorValue;
  onChange: (next: EditorValue) => void;
  /** 只读模式：禁编辑/连线/删除/调色板/检查器，保留平移缩放与 JSON 查看 */
  readOnly?: boolean;
  /** 容器高度（默认 560；宽度撑满父级） */
  height?: number | string;
  className?: string;
  /** 校验问题外送（host 用作提交门禁；组件内同时展示计数） */
  onIssues?: (issues: EditorIssue[]) => void;
  /** 调色板业务分组骨架（合法标签；缺省由注入目录的 labels 驱动） */
  paletteLabelWhitelist?: readonly string[];
  /** 调色板源列表（注入点；缺省由 catalogProvider 桥接生成，两者皆缺 = 仅系组提示态） */
  sources?: readonly PaletteSource[];
  /** 目录数据源（旧快照契约；仅在未传 sources 时经内置桥接降解为缺省源组） */
  catalogProvider?: CatalogProvider | null;
  /** 布局策略（注入点；缺省 autoLayoutStrategy = dagre 分层自动布局） */
  layout?: LayoutStrategy;
  /** 主题（缺省 "system" 跟随宿主；data-ig-theme 落根元素，token 层见 styles.css） */
  theme?: EditorTheme;
}

/** 值等价（提交去重：同一事件内多次变更派发时防重复历史/丢失合并） */
function valueEquals(a: EditorValue, b: EditorValue): boolean {
  return (
    JSON.stringify(a.graph) === JSON.stringify(b.graph) &&
    JSON.stringify(a.layout) === JSON.stringify(b.layout) &&
    JSON.stringify(a.remote) === JSON.stringify(b.remote) &&
    JSON.stringify(a.io) === JSON.stringify(b.io)
  );
}

/** 选择名单内容等价（顺序敏感） */
function namesEqual(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((n, i) => n === b[i]);
}

/** 单条边移除（IO 投影边删除 = 解绑，元素保留；普通边 = 从图移除） */
function applyEdgeRemoval(cur: EditorValue, id: string): EditorValue {
  if (isIoEdgeId(id)) return unbindIoElement(cur, ioElementIdOfEdge(id));
  const edge = cur.graph.edges.find((e) => edgeIdOf(e) === id);
  return edge ? removeEdge(cur, edge) : cur;
}

export default function InferGraphEditor(props: InferGraphEditorProps) {
  return (
    <ReactFlowProvider>
      <EditorInner {...props} />
    </ReactFlowProvider>
  );
}

function EditorInner({
  value,
  onChange,
  readOnly = false,
  height = 560,
  className,
  onIssues,
  paletteLabelWhitelist,
  sources,
  catalogProvider = null,
  layout = autoLayoutStrategy,
  theme = "system",
}: InferGraphEditorProps) {
  const resolvedTheme = useResolvedTheme(theme);
  const { screenToFlowPosition, fitView } = useReactFlow();
  const storeApi = useStoreApi();
  const rootRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLDivElement | null>(null);

  // 最新值快照：同一事件内多次 op 派发（如 RF 级联删除同时走 nodes/edges change）
  // 时基于本地最新值串行变换，避免陈旧闭包互相覆盖。
  const latest = useRef(value);
  useEffect(() => {
    latest.current = value;
  }, [value]);

  const [selectedNames, setSelectedNamesRaw] = useState<string[]>([]);
  /**
   * 选择名单更新（内容相等保持原引用）：
   * 防 echo ⇄ adopt 反馈回环——内容无变化时若仍产生新引用，重渲染会持续触发回灌。
   */
  const setSelectedNames = useCallback((compute: string[] | ((prev: string[]) => string[])) => {
    setSelectedNamesRaw((prev) => {
      const next = typeof compute === "function" ? compute(prev) : compute;
      return namesEqual(prev, next) ? prev : next;
    });
  }, []);
  const [jsonOpen, setJsonOpen] = useState(false);
  const [issuesPanelOpen, setIssuesPanelOpen] = useState(false);
  // 组件面板开合：移动端挂载默认折叠（惰性初始化避免首帧闪烁；jsdom 无 matchMedia 时按桌面处理）
  const [paletteOpen, setPaletteOpen] = useState<boolean>(
    () => !(typeof window !== "undefined" && window.matchMedia?.(IG_MOBILE_QUERY)?.matches === true),
  );
  // 撤销栈可用性外提为 state（禁止渲染期读 ref）
  const [histFlags, setHistFlags] = useState({ canUndo: false, canRedo: false });
  const history = useRef<{ past: EditorValue[]; future: EditorValue[] }>({ past: [], future: [] });
  const syncHistFlags = useCallback(() => {
    setHistFlags({ canUndo: history.current.past.length > 0, canRedo: history.current.future.length > 0 });
  }, []);

  const issues = useMemo(() => validateEditorValue(value), [value]);
  useEffect(() => {
    onIssues?.(issues);
  }, [issues, onIssues]);
  const levelByNode = useMemo(() => issueLevelByNode(issues), [issues]);
  const errors = issues.filter((i) => i.level === "error");
  const warns = issues.filter((i) => i.level === "warn");
  const selectedName = selectedNames.length === 1 ? selectedNames[0] : null;
  const selectedNode = selectedName !== null ? value.graph.nodes.find((n) => n.name === selectedName) ?? null : null;
  const selectedIo = selectedName !== null && !selectedNode ? value.io.find((e) => e.id === selectedName) ?? null : null;

  // ── 提交（快照式撤销栈；内容等价去重）──
  const commit = useCallback(
    (next: EditorValue) => {
      if (next === latest.current || valueEquals(next, latest.current)) return;
      history.current.past.push(latest.current);
      if (history.current.past.length > 64) history.current.past.shift();
      history.current.future.length = 0;
      latest.current = next;
      syncHistFlags();
      setSelectedNames((prev) =>
        prev.filter((n) => next.graph.nodes.some((x) => x.name === n) || next.io.some((e) => e.id === n)),
      );
      onChange(next);
    },
    [onChange, syncHistFlags, setSelectedNames],
  );

  const undo = useCallback(() => {
    const prev = history.current.past.pop();
    if (!prev) return;
    history.current.future.push(latest.current);
    latest.current = prev;
    syncHistFlags();
    setSelectedNames((names) =>
      names.filter((n) => prev.graph.nodes.some((x) => x.name === n) || prev.io.some((e) => e.id === n)),
    );
    onChange(prev);
  }, [onChange, syncHistFlags, setSelectedNames]);

  const redo = useCallback(() => {
    const next = history.current.future.pop();
    if (!next) return;
    history.current.past.push(latest.current);
    latest.current = next;
    syncHistFlags();
    setSelectedNames((names) =>
      names.filter((n) => next.graph.nodes.some((x) => x.name === n) || next.io.some((e) => e.id === n)),
    );
    onChange(next);
  }, [onChange, syncHistFlags, setSelectedNames]);

  // 键盘撤销/重做（编辑器容器内；输入控件内让位原生文本撤销）
  useEffect(() => {
    if (readOnly) return;
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT" || target.isContentEditable)
      ) {
        return;
      }
      const el = rootRef.current;
      if (el && target && target !== document.body && !el.contains(target)) return;
      const key = e.key.toLowerCase();
      if (key === "z" && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if ((key === "z" && e.shiftKey) || key === "y") {
        e.preventDefault();
        redo();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo, readOnly]);

  // ── React Flow 本地态（交互瞬态）+ 外部值回声同步 ──
  const model = useMemo(
    () => toFlowModel(value, { issueLevelByNode: levelByNode, selectedNames }),
    [value, levelByNode, selectedNames],
  );
  const [flowNodes, setFlowNodes] = useState<AnyRfNode[]>(model.nodes as unknown as AnyRfNode[]);
  const [flowEdges, setFlowEdges] = useState<RfEdge[]>(model.edges as unknown as RfEdge[]);
  // 最近一次回灌时的选择名单（区分「选择权威回灌」与「结构回灌」）
  const echoedSelRef = useRef<readonly string[]>([]);
  useEffect(() => {
    // 选中位单一权威 = RF 事件 → onNodesChange → flowNodes：
    // 仅当选择名单变化（用户选择或程序性设置）时以模型为准回灌；结构回灌一律保留
    // 现行选中位。双写（回灌与 SelectionListener 报告各写一拍）会形成 adoptUserNodes
    // ⇄ SelectionListener 的选中位翻转环（真实浏览器带测量数据时必现 Maximum
    // update depth exceeded，2026-10-11 实测）。
    const selAuthoritative = !namesEqual(echoedSelRef.current, selectedNames);
    echoedSelRef.current = selectedNames;
    // core 结构投影 → RF 运行时类型（单点断言；见 flow-types.ts）
    const nextNodes = model.nodes as unknown as AnyRfNode[];
    const nextEdges = model.edges as unknown as RfEdge[];
    setFlowNodes((prev) => {
      if (selAuthoritative) return nextNodes;
      const live = new Map(prev.map((n) => [n.id, n.selected] as const));
      return nextNodes.map((n) => {
        const s = live.get(n.id);
        return s === undefined || s === n.selected ? n : { ...n, selected: s };
      });
    });
    setFlowEdges((prev) => {
      // 边选中位是纯交互瞬态（不属受控 value）：回灌一律保留现行选中位，防止
      // 点击连线后的选中高亮被下一次回灌清除。
      const live = new Map(prev.map((e) => [e.id, e.selected] as const));
      return nextEdges.map((e) => {
        const s = live.get(e.id);
        return s === undefined || s === e.selected ? e : { ...e, selected: s };
      });
    });
  }, [model, selectedNames]);

  const onNodesChange = useCallback(
    (changes: NodeChange<AnyRfNode>[]) => {
      setFlowNodes((ns) => applyNodeChanges(changes, ns));
      if (readOnly) return;
      const removed = changes.filter((c) => c.type === "remove");
      if (removed.length === 0) return;
      let next = latest.current;
      for (const c of removed) {
        const id = (c as { id: string }).id;
        // IO 元素与图节点同处画布空间，按 id 归属分派
        if (next.io.some((e) => e.id === id)) next = removeIoElement(next, id);
        else next = removeNode(next, id);
      }
      commit(next);
    },
    [readOnly, commit],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange<RfEdge>[]) => {
      setFlowEdges((es) => applyEdgeChanges(changes, es));
      if (readOnly) return;
      const removed = changes.filter((c) => c.type === "remove");
      if (removed.length === 0) return;
      let next = latest.current;
      for (const c of removed) next = applyEdgeRemoval(next, (c as { id: string }).id);
      commit(next);
    },
    [readOnly, commit],
  );

  const isValidConnection = useCallback(
    (conn: Connection | RfEdge) =>
      checkConnection(value, {
        source: conn.source,
        sourceHandle: conn.sourceHandle ?? null,
        target: conn.target,
        targetHandle: conn.targetHandle ?? null,
      }).ok,
    [value],
  );

  const onConnect = useCallback(
    (conn: Connection) => {
      if (readOnly) return;
      const sh = parseHandle(conn.sourceHandle);
      const th = parseHandle(conn.targetHandle);
      if (!sh || !th) return;
      const cur = latest.current;
      // IO 元素参与：输入元素（源）→ 节点输入口；节点输出口 → 输出元素（目标）
      const srcEl = cur.io.find((e) => e.id === conn.source);
      const dstEl = cur.io.find((e) => e.id === conn.target);
      if (srcEl) {
        if (srcEl.kind !== "input" || sh.direction !== "out" || th.direction !== "in" || dstEl) return;
        commit(bindIoElement(cur, srcEl.id, conn.target, th.port));
        return;
      }
      if (dstEl) {
        if (dstEl.kind !== "output" || th.direction !== "in" || sh.direction !== "out") return;
        commit(bindIoElement(cur, dstEl.id, conn.source, sh.port));
        return;
      }
      if (sh.direction !== "out" || th.direction !== "in") return;
      commit(addEdge(cur, { srcNode: conn.source, srcPort: sh.port, dstNode: conn.target, dstPort: th.port }));
    },
    [readOnly, commit],
  );

  const onNodeDragStop = useCallback(() => {
    if (readOnly) return;
    const layout: Record<string, XY> = { ...latest.current.layout };
    let changed = false;
    for (const n of flowNodes) {
      const x = Math.round(n.position.x);
      const y = Math.round(n.position.y);
      const prev = layout[n.id];
      if (!prev || prev.x !== x || prev.y !== y) {
        layout[n.id] = { x, y };
        changed = true;
      }
    }
    if (changed) commit({ ...latest.current, layout });
  }, [flowNodes, readOnly, commit]);

  const onSelectionChange = useCallback((params: { nodes: AnyRfNode[] }) => {
    setSelectedNames(params.nodes.map((n) => n.id));
  }, [setSelectedNames]);

  // ── 连线删除（点击连线）：高亮选中由库 .selected 类驱动；删除按钮浮于命中点附近 ──
  const [edgeMenu, setEdgeMenu] = useState<{ id: string; x: number; y: number } | null>(null);
  const onEdgeClick = useCallback(
    (event: ReactMouseEvent, edge: RfEdge) => {
      if (readOnly) return;
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;
      // 命中点右下偏移；坐标收敛在画布容器内（触屏无 hover，按钮是主要删除路径）
      const x = Math.min(Math.max(event.clientX - rect.left + 12, 8), Math.max(rect.width - 42, 8));
      const y = Math.min(Math.max(event.clientY - rect.top + 12, 8), Math.max(rect.height - 42, 8));
      setEdgeMenu({ id: edge.id, x, y });
    },
    [readOnly],
  );
  // 平移 / 缩放后按钮锚点（容器相对坐标）与画布内容错位 → 收起
  const onMoveStart = useCallback(() => setEdgeMenu(null), []);

  // ── 单击连线（connectOnClick）：起点光圈由库类 clickconnecting 驱动；根类驱动候选对端提示 ──
  const [linkingFrom, setLinkingFrom] = useState<"source" | "target" | null>(null);
  const onConnectClickStart = useCallback((_: unknown, params: { handleType: "source" | "target" | null }) => {
    setLinkingFrom(params.handleType === "target" ? "target" : "source");
  }, []);
  const onConnectClickEnd = useCallback(() => {
    setLinkingFrom(null);
  }, []);
  // 点空白取消待定连线：v12.12 单击待定态由库内部键 connectionClickStartHandle 持有，
  // 点空白 / Esc 均不会自行清除（已按 dist 源码验证）——升级 @xyflow/react 时需复核。
  const onPaneClick = useCallback(() => {
    setLinkingFrom(null);
    setEdgeMenu(null);
    if (storeApi.getState().connectionClickStartHandle) {
      storeApi.setState({ connectionClickStartHandle: null });
    }
  }, [storeApi]);

  // ── 画布视口（流坐标）与级联落位（「添加到画布」缺省坐标）──
  const viewportRect = useCallback((): ViewportRect | null => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return null;
    const tl = screenToFlowPosition({ x: rect.left, y: rect.top });
    const br = screenToFlowPosition({ x: rect.right, y: rect.bottom });
    return { x: tl.x, y: tl.y, width: br.x - tl.x, height: br.y - tl.y };
  }, [screenToFlowPosition]);

  const cascadeFor = useCallback(
    (size: { width: number; height: number }): XY => {
      const vp = viewportRect();
      if (!vp) return { x: 0, y: 0 };
      return cascadePlace({ viewport: vp, size, taken: Object.values(latest.current.layout) });
    },
    [viewportRect],
  );

  // ── 调色板添加（IO 元素 / 目录条目）──
  const addIo = useCallback(
    (kind: "input" | "output", position?: XY) => {
      const pos = position ?? cascadeFor({ width: IO_ELEM_WIDTH, height: IO_ELEM_HEIGHT });
      const { value: next, id } = addIoElement(latest.current, kind, pos);
      commit(next);
      setSelectedNames([id]);
    },
    [cascadeFor, commit, setSelectedNames],
  );
  
  // 目录条目建节点（远程模型）：io 驱动端子 + 自动落远程意图
  const addModel = useCallback(
    (item: CatalogModelItem, position?: XY) => {
      const pos =
        position ??
        cascadeFor({
          width: NODE_WIDTH,
          height: nodeHeight({ inputs: item.io?.inputs ?? [], outputs: item.io?.outputs ?? [] }),
        });
      const { value: next, name } = addModelNode(latest.current, item, pos);
      commit(next);
      setSelectedNames([name]);
    },
    [cascadeFor, commit, setSelectedNames],
  );

  // 信号编排逻辑节点：只有信号端子（缺省落点 = 视口中心级联避让）
  const addSignal = useCallback(
    (logic: SignalLogicKind, position?: XY) => {
      const pos = position ?? cascadeFor({ width: NODE_WIDTH, height: nodeHeight({ inputs: [], outputs: [] }) });
      const { value: next, name } = addSignalNode(latest.current, logic, pos);
      commit(next);
      setSelectedNames([name]);
    },
    [cascadeFor, commit, setSelectedNames],
  );

  const onDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      if (readOnly) return;
      e.preventDefault();
      const p = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      const pos = { x: Math.round(p.x), y: Math.round(p.y) };
      const ioKind = e.dataTransfer.getData(IG_IO_MIME);
      if (ioKind === "input" || ioKind === "output") {
        addIo(ioKind, pos);
        return;
      }
      const sigLogic = e.dataTransfer.getData(IG_SIGNAL_MIME);
      if (sigLogic) {
        addSignal(sigLogic as SignalLogicKind, pos);
        return;
      }
      const modelRaw = e.dataTransfer.getData(IG_MODEL_MIME);
      if (modelRaw) {
        const item = parseCatalogItem(modelRaw);
        if (item) addModel(item, pos);
        return;
      }
    },
    [readOnly, screenToFlowPosition, addIo, addSignal, addModel],
  );

  const onDragOver = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      if (readOnly) return;
      if (
        e.dataTransfer.types.includes(IG_IO_MIME) ||
        e.dataTransfer.types.includes(IG_SIGNAL_MIME) ||
        e.dataTransfer.types.includes(IG_MODEL_MIME)
      ) {
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
      }
    },
    [readOnly],
  );

  // 调色板唯一写通道：窄校验 → 建节点（缺省落点 = 视口中心级联避让）
  const place = useCallback(
    (item: CatalogModelItem, position?: XY) => {
      const safe = sanitizeCatalogItem(item);
      if (safe) addModel(safe, position);
    },
    [addModel],
  );

  // 源装配：显式 sources 优先；缺省由旧目录快照桥接（labels 到达前亦可渲染空分组）
  const { catalog } = useCatalog(catalogProvider);
  const paletteHost: PaletteHost = useMemo(
    () => ({ place, placeIo: addIo, placeSignalNode: addSignal, viewport: viewportRect, value }),
    [place, addIo, addSignal, viewportRect, value],
  );
  const effectiveSources = useMemo(
    () => sources ?? defaultSources(catalogProvider, catalog?.labels ?? paletteLabelWhitelist ?? []),
    [sources, catalogProvider, catalog, paletteLabelWhitelist],
  );

  const relayout = useCallback(() => {
    const cur = latest.current;
    commit({ ...cur, layout: layout.compute(cur.graph, cur.io, cur.layout) });
    // 排布后自动取景（等一帧让 layout 回灌到 flowNodes）
    requestAnimationFrame(() => void fitView({ padding: 0.18, duration: 300 }));
  }, [commit, fitView, layout]);

  const applyImportText = useCallback(
    (text: string): EditorIssue[] => {
      const r = applyImport(latest.current, text);
      if (r.value === latest.current) return r.issues;
      commit(fillLayout(r.value));
      setSelectedNames([]);
      return r.issues;
    },
    [commit, setSelectedNames],
  );

  const json = useMemo(() => (jsonOpen ? JSON.stringify(serializeGraph(value), null, 2) : ""), [jsonOpen, value]);
  const canUndo = histFlags.canUndo;
  const canRedo = histFlags.canRedo;
  const isEmpty = value.graph.nodes.length === 0 && value.io.length === 0;

  return (
    <div
      ref={rootRef}
      data-ig-theme={resolvedTheme}
      className={`ig-root${className ? ` ${className}` : ""}${paletteOpen ? "" : " ig-palette-collapsed"}${
        linkingFrom ? ` ig-linking ig-linking-from-${linkingFrom}` : ""
      }`}
      style={{ height }}
    >
      <div className="ig-toolbar">
        <div className="ig-brand-lockup">
          <span className="ig-brand-mark" aria-hidden="true">✦</span>
          <div>
            <span className="ig-title">Inference Studio</span>
            <span className="ig-subtitle">推理图编辑器 · 工作台</span>
          </div>
        </div>
        <span className="ig-save-status"><i />本地草稿</span>
        <div className="ig-toolbar-divider" />
        {!readOnly && (
          <button
            type="button"
            className="ig-btn ig-panel-toggle"
            aria-expanded={paletteOpen}
            title="折叠 / 展开左侧组件面板"
            onClick={() => setPaletteOpen((v) => !v)}
          >
            <span aria-hidden="true">＋</span> 添加组件
          </button>
        )}
        {readOnly && (
          <button type="button" className={`ig-issue-chip error${errors.length > 0 ? " on" : ""}`} onClick={() => setIssuesPanelOpen((v) => !v)}>
            <i />
            {errors.length > 0 ? `${errors.length} 个错误` : "无错误"}
          </button>
        )}
        {readOnly && (
          <button type="button" className={`ig-issue-chip warn${warns.length > 0 ? " on" : ""}`} onClick={() => setIssuesPanelOpen((v) => !v)}>
            <i />
            {warns.length > 0 ? `${warns.length} 个提醒` : "无提醒"}
          </button>
        )}
        <span className="ig-spacer" />
        {!readOnly && (
          <button type="button" className="ig-btn ig-btn-quiet" onClick={relayout} title="自动整理节点布局">
            <span aria-hidden="true">⌘</span> 自动整理
          </button>
        )}
        {!readOnly && (
          <button type="button" className="ig-icon-action" onClick={undo} disabled={!canUndo} title="撤销（Ctrl+Z）" aria-label="撤销">↶</button>
        )}
        {!readOnly && (
          <button type="button" className="ig-icon-action" onClick={redo} disabled={!canRedo} title="重做（Ctrl+Shift+Z）" aria-label="重做">↷</button>
        )}
        <button type="button" className="ig-btn ig-btn-quiet" onClick={() => void fitView({ padding: 0.18, duration: 200 })}>
          适应画布
        </button>
        <button type="button" className={`ig-btn${jsonOpen ? " ig-primary" : " ig-btn-quiet"}`} onClick={() => setJsonOpen((v) => !v)}>
          <span aria-hidden="true">{`{ }`}</span> 数据视图
        </button>
      </div>

      <IssuePanel
        issues={issues}
        visible={issuesPanelOpen}
        onClose={() => setIssuesPanelOpen(false)}
        onFocusNode={(nodeName) => {
          setSelectedNames([nodeName]);
          void fitView({ nodes: [{ id: nodeName }], padding: 0.3, duration: 300 });
        }}
      />

      <div className="ig-main">
        {!readOnly && <Palette sources={effectiveSources} host={paletteHost} />}
        {!readOnly && (
          <button
            type="button"
            className="ig-rail-handle"
            aria-expanded={paletteOpen}
            title={paletteOpen ? "折叠组件面板" : "展开组件面板"}
            aria-label={paletteOpen ? "折叠组件面板" : "展开组件面板"}
            onClick={() => setPaletteOpen((v) => !v)}
          >
            <span aria-hidden="true">{paletteOpen ? "‹" : "›"}</span>
          </button>
        )}
        <div className="ig-canvas" ref={canvasRef} onDrop={onDrop} onDragOver={onDragOver}>
          <ReactFlow
            className="ig-flow"
            nodes={flowNodes}
            edges={flowEdges}
            nodeTypes={NODE_TYPES}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            isValidConnection={isValidConnection}
            onSelectionChange={onSelectionChange}
            onNodeDragStop={onNodeDragStop}
            connectOnClick={!readOnly}
            onClickConnectStart={onConnectClickStart}
            onClickConnectEnd={onConnectClickEnd}
            onPaneClick={onPaneClick}
            onEdgeClick={onEdgeClick}
            onMoveStart={onMoveStart}
            nodeDragThreshold={4}
            connectionMode={ConnectionMode.Strict}
            nodesDraggable={!readOnly}
            nodesConnectable={!readOnly}
            elementsSelectable={!readOnly}
            deleteKeyCode={readOnly ? null : ["Backspace", "Delete"]}
            fitView
            fitViewOptions={{ padding: 0.18, maxZoom: 1 }}
            minZoom={0.15}
            maxZoom={2}
            colorMode={resolvedTheme}
          >
            <Background variant={BackgroundVariant.Dots} gap={22} size={1} color={CANVAS_ACCENTS[resolvedTheme].dot} />
            <Controls showInteractive={false} />
            <MiniMap
              pannable
              zoomable
              nodeColor={(n) => {
                const d = n.data as InferNodeData & { kind?: "input" | "output" };
                const accents = CANVAS_ACCENTS[resolvedTheme];
                if (d.kind) return d.kind === "input" ? accents.ioIn : accents.ioOut;
                return accents.node;
              }}
              maskColor={CANVAS_ACCENTS[resolvedTheme].mask}
            />
          </ReactFlow>
          {isEmpty && !jsonOpen && (
            <div className="ig-empty-canvas">
              <div className="ig-empty-orb" aria-hidden="true">✦</div>
              <h2>从一个推理图开始</h2>
              <p>把输入、模型和输出连接起来，构建你的第一条推理流程。</p>
              {!readOnly && (
                <button type="button" className="ig-btn ig-primary" onClick={() => setPaletteOpen(true)}>
                  <span aria-hidden="true">＋</span> 浏览组件
                </button>
              )}
              <div className="ig-empty-hints">
                <span><b>1</b> 添加输入</span>
                <span><b>2</b> 选择模型</span>
                <span><b>3</b> 连线运行</span>
              </div>
            </div>
          )}
          {!readOnly && edgeMenu && flowEdges.some((e) => e.id === edgeMenu.id && e.selected) && (
            <button
              type="button"
              className="ig-edge-del"
              style={{ left: edgeMenu.x, top: edgeMenu.y }}
              aria-label="删除连线"
              title="删除连线"
              onPointerDown={(e) => e.stopPropagation()}
              onClick={() => {
                commit(applyEdgeRemoval(latest.current, edgeMenu.id));
                setEdgeMenu(null);
              }}
            >
              ×
            </button>
          )}
        </div>
        {!readOnly && selectedNode && (
          <Inspector
            key={selectedNode.name}
            value={value}
            node={selectedNode}
            onClose={() => setSelectedNames([])}
            onRename={(newName) => {
              const r = renameNode(latest.current, selectedNode.name, newName);
              if (r.error) return r.error;
              commit(r.value);
              setSelectedNames([newName]);
              return null;
            }}
            onPatchNode={(patch) => commit(patchNode(latest.current, selectedNode.name, patch))}
            onRemoveNode={() => {
              commit(removeNode(latest.current, selectedNode.name));
              setSelectedNames([]);
            }}
            onSetRemote={(modelKey) => commit(setRemote(latest.current, selectedNode.name, modelKey))}
            onSetSignalSourceName={(n) => commit(setSignalSourceName(latest.current, selectedNode.name, n))}
            onSetSignalConstValue={(v) => commit(setSignalConstValue(latest.current, selectedNode.name, v))}
            onAddLogicInput={() => commit(addSignalLogicInput(latest.current, selectedNode.name))}
            onRemoveLogicInput={(port) => commit(removeSignalLogicInput(latest.current, selectedNode.name, port))}
          />
        )}
        {!readOnly && selectedIo && !selectedNode && (
          <IoInspector
            key={selectedIo.id}
            element={selectedIo}
            onClose={() => setSelectedNames([])}
            onDelete={() => {
              commit(removeIoElement(latest.current, selectedIo.id));
              setSelectedNames([]);
            }}
          />
        )}
        {!readOnly && (
          <div className="ig-status-dock" role="status" aria-live="polite">
            <button type="button" className={`ig-issue-chip error${errors.length > 0 ? " on" : ""}`} title={errors.length > 0 ? `${errors.length} 个错误` : "无错误"} onClick={() => setIssuesPanelOpen((v) => !v)}>
              <i />
              <span className="ig-chip-label">{errors.length > 0 ? `${errors.length} 个错误` : "无错误"}</span>
            </button>
            <button type="button" className={`ig-issue-chip warn${warns.length > 0 ? " on" : ""}`} title={warns.length > 0 ? `${warns.length} 个提醒` : "无提醒"} onClick={() => setIssuesPanelOpen((v) => !v)}>
              <i />
              <span className="ig-chip-label">{warns.length > 0 ? `${warns.length} 个提醒` : "无提醒"}</span>
            </button>
          </div>
        )}
      </div>

      {jsonOpen && (
        <JsonDrawer json={json} readOnly={readOnly} onApply={applyImportText} onClose={() => setJsonOpen(false)} />
      )}
    </div>
  );
}

// ── 公共导出面（宿主与二次开发消费；core 层全量 API 见 core/index.ts）──

export { validateEditorValue, hasBlockingIssues, issueLevelByNode, shapeCompatible } from "../core/rules";
export { serializeGraph, sanitizeGraph, remoteNodesOf, localInputsOf, applyImport, emptyGraph } from "../core/model";
export { fillLayout, autoLayout, autoLayoutStrategy, keepLayoutStrategy } from "../core/layout";
export type { LayoutStrategy, LayoutMap } from "../core/layout";
export { staticCatalogProvider, httpCatalogProvider } from "../core/catalog";
export type { CatalogProvider, CatalogView, CatalogModelItem } from "../core/catalog";
// 调色板一元协议与内置组合子（自定义源入口）
export type { PaletteSource, PaletteHost, SourceCtx, SourceProgram, SourceFetch, PaletteItem } from "./palette-protocol";
export { browse, staticSource, querySource } from "./sources/browse";
export { ioSource } from "./sources/io";
export { catalogSources, defaultSources } from "./sources/catalog";
export type { EditorValue, EditorIssue, DcirGraphJson, DcirNode, DcirPort, IssueLevel, XY } from "../core/types";
