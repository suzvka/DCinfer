/**
 * InferGraphEditor — 可视化推理图编辑器（一站式嵌入式组件）
 *
 * 重设计原则（面向用户而非测试人员）：
 * - 工具栏：只保留用户日常操作（撤销/重做、整理布局、适应视图、问题提示）；
 *   JSON 查看、原始数据等调试功能移入「更多」（⋮）溢出菜单；
 * - 品牌文案：精简为 "推理图编辑器"；
 * - 问题面板：使用 i18n 友好文案；
 * - 空画布引导：优化文案，强调「从模型库拖入节点」的流程。
 *
 * 契约：受控组件（value / onChange），零宿主框架耦合（仅 React + @xyflow/react）。
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

const CANVAS_ACCENTS = {
  dark: { dot: "rgba(120, 145, 190, 0.25)", mask: "rgba(10, 14, 23, 0.72)", ioIn: "#2f9e8f", ioOut: "#b9832f", node: "#4f8cff" },
  light: { dot: "rgba(70, 95, 140, 0.3)", mask: "rgba(222, 230, 241, 0.72)", ioIn: "#0d9464", ioOut: "#b45309", node: "#3b76e8" },
} as const;

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
  readOnly?: boolean;
  height?: number | string;
  className?: string;
  onIssues?: (issues: EditorIssue[]) => void;
  paletteLabelWhitelist?: readonly string[];
  sources?: readonly PaletteSource[];
  catalogProvider?: CatalogProvider | null;
  layout?: LayoutStrategy;
  theme?: EditorTheme;
}

function valueEquals(a: EditorValue, b: EditorValue): boolean {
  return (
    JSON.stringify(a.graph) === JSON.stringify(b.graph) &&
    JSON.stringify(a.layout) === JSON.stringify(b.layout) &&
    JSON.stringify(a.remote) === JSON.stringify(b.remote) &&
    JSON.stringify(a.io) === JSON.stringify(b.io)
  );
}

function namesEqual(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((n, i) => n === b[i]);
}

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

  const latest = useRef(value);
  useEffect(() => { latest.current = value; }, [value]);

  const [selectedNames, setSelectedNamesRaw] = useState<string[]>([]);
  const setSelectedNames = useCallback((compute: string[] | ((prev: string[]) => string[])) => {
    setSelectedNamesRaw((prev) => {
      const next = typeof compute === "function" ? compute(prev) : compute;
      return namesEqual(prev, next) ? prev : next;
    });
  }, []);

  const [jsonOpen, setJsonOpen] = useState(false);
  const [issuesPanelOpen, setIssuesPanelOpen] = useState(false);
  const [moreMenuOpen, setMoreMenuOpen] = useState(false);
  const moreMenuRef = useRef<HTMLDivElement | null>(null);

  const [paletteOpen, setPaletteOpen] = useState<boolean>(
    () => !(typeof window !== "undefined" && window.matchMedia?.(IG_MOBILE_QUERY)?.matches === true),
  );
  const [histFlags, setHistFlags] = useState({ canUndo: false, canRedo: false });
  const history = useRef<{ past: EditorValue[]; future: EditorValue[] }>({ past: [], future: [] });
  const syncHistFlags = useCallback(() => {
    setHistFlags({ canUndo: history.current.past.length > 0, canRedo: history.current.future.length > 0 });
  }, []);

  const issues = useMemo(() => validateEditorValue(value), [value]);
  useEffect(() => { onIssues?.(issues); }, [issues, onIssues]);
  const levelByNode = useMemo(() => issueLevelByNode(issues), [issues]);
  const errors = issues.filter((i) => i.level === "error");
  const warns = issues.filter((i) => i.level === "warn");
  const selectedName = selectedNames.length === 1 ? selectedNames[0] : null;
  const selectedNode = selectedName !== null ? value.graph.nodes.find((n) => n.name === selectedName) ?? null : null;
  const selectedIo = selectedName !== null && !selectedNode ? value.io.find((e) => e.id === selectedName) ?? null : null;

  // 「更多」菜单点击外部关闭
  useEffect(() => {
    if (!moreMenuOpen) return;
    const onPointerDown = (e: PointerEvent) => {
      if (moreMenuRef.current && e.target instanceof Node && moreMenuRef.current.contains(e.target)) return;
      setMoreMenuOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [moreMenuOpen]);

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

  useEffect(() => {
    if (readOnly) return;
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT" || target.isContentEditable)) return;
      const el = rootRef.current;
      if (el && target && target !== document.body && !el.contains(target)) return;
      const key = e.key.toLowerCase();
      if (key === "z" && !e.shiftKey) { e.preventDefault(); undo(); }
      else if ((key === "z" && e.shiftKey) || key === "y") { e.preventDefault(); redo(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo, readOnly]);

  const model = useMemo(
    () => toFlowModel(value, { issueLevelByNode: levelByNode, selectedNames }),
    [value, levelByNode, selectedNames],
  );
  const [flowNodes, setFlowNodes] = useState<AnyRfNode[]>(model.nodes as unknown as AnyRfNode[]);
  const [flowEdges, setFlowEdges] = useState<RfEdge[]>(model.edges as unknown as RfEdge[]);
  const echoedSelRef = useRef<readonly string[]>([]);
  useEffect(() => {
    const selAuthoritative = !namesEqual(echoedSelRef.current, selectedNames);
    echoedSelRef.current = selectedNames;
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

  const [edgeMenu, setEdgeMenu] = useState<{ id: string; x: number; y: number } | null>(null);
  const onEdgeClick = useCallback(
    (event: ReactMouseEvent, edge: RfEdge) => {
      if (readOnly) return;
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;
      const x = Math.min(Math.max(event.clientX - rect.left + 12, 8), Math.max(rect.width - 42, 8));
      const y = Math.min(Math.max(event.clientY - rect.top + 12, 8), Math.max(rect.height - 42, 8));
      setEdgeMenu({ id: edge.id, x, y });
    },
    [readOnly],
  );
  const onMoveStart = useCallback(() => setEdgeMenu(null), []);

  const [linkingFrom, setLinkingFrom] = useState<"source" | "target" | null>(null);
  const onConnectClickStart = useCallback((_: unknown, params: { handleType: "source" | "target" | null }) => {
    setLinkingFrom(params.handleType === "target" ? "target" : "source");
  }, []);
  const onConnectClickEnd = useCallback(() => { setLinkingFrom(null); }, []);
  const onPaneClick = useCallback(() => {
    setLinkingFrom(null);
    setEdgeMenu(null);
    if (storeApi.getState().connectionClickStartHandle) {
      storeApi.setState({ connectionClickStartHandle: null });
    }
  }, [storeApi]);

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

  const addIo = useCallback(
    (kind: "input" | "output", position?: XY) => {
      const pos = position ?? cascadeFor({ width: IO_ELEM_WIDTH, height: IO_ELEM_HEIGHT });
      const { value: next, id } = addIoElement(latest.current, kind, pos);
      commit(next);
      setSelectedNames([id]);
    },
    [cascadeFor, commit, setSelectedNames],
  );

  const addModel = useCallback(
    (item: CatalogModelItem, position?: XY) => {
      const pos = position ?? cascadeFor({
        width: NODE_WIDTH,
        height: nodeHeight({ inputs: item.io?.inputs ?? [], outputs: item.io?.outputs ?? [] }),
      });
      const { value: next, name } = addModelNode(latest.current, item, pos);
      commit(next);
      setSelectedNames([name]);
    },
    [cascadeFor, commit, setSelectedNames],
  );

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
      if (ioKind === "input" || ioKind === "output") { addIo(ioKind, pos); return; }
      const sigLogic = e.dataTransfer.getData(IG_SIGNAL_MIME);
      if (sigLogic) { addSignal(sigLogic as SignalLogicKind, pos); return; }
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

  const place = useCallback(
    (item: CatalogModelItem, position?: XY) => {
      const safe = sanitizeCatalogItem(item);
      if (safe) addModel(safe, position);
    },
    [addModel],
  );

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
  const hasErrors = errors.length > 0;
  const hasWarns = warns.length > 0;

  return (
    <div
      ref={rootRef}
      data-ig-theme={resolvedTheme}
      className={`ig-root${className ? ` ${className}` : ""}${paletteOpen ? "" : " ig-palette-collapsed"}${
        linkingFrom ? ` ig-linking ig-linking-from-${linkingFrom}` : ""
      }`}
      style={{ height }}
    >
      {/* ── 工具栏：精简为用户日常操作 ── */}
      <div className="ig-toolbar">
        <div className="ig-brand-lockup">
          <span className="ig-brand-mark" aria-hidden="true">✦</span>
          <div>
            <span className="ig-title">推理图编辑器</span>
          </div>
        </div>

        <div className="ig-toolbar-divider" />

        {!readOnly && (
          <button
            type="button"
            className="ig-btn ig-panel-toggle"
            aria-expanded={paletteOpen}
            title={paletteOpen ? "折叠组件库" : "展开组件库"}
            onClick={() => setPaletteOpen((v) => !v)}
          >
            <span aria-hidden="true">＋</span> 组件库
          </button>
        )}

        <span className="ig-spacer" />

        {/* 撤销/重做 */}
        {!readOnly && (
          <>
            <button type="button" className="ig-icon-action" onClick={undo} disabled={!canUndo} title="撤销（Ctrl+Z）" aria-label="撤销">↶</button>
            <button type="button" className="ig-icon-action" onClick={redo} disabled={!canRedo} title="重做（Ctrl+Shift+Z）" aria-label="重做">↷</button>
          </>
        )}

        {/* 整理布局 */}
        {!readOnly && (
          <button type="button" className="ig-btn ig-btn-quiet" onClick={relayout} title="自动整理节点布局">
            整理布局
          </button>
        )}

        {/* 适应视图 */}
        <button type="button" className="ig-btn ig-btn-quiet" onClick={() => void fitView({ padding: 0.18, duration: 200 })} title="缩放以显示全部节点">
          适应视图
        </button>

        {/* 问题 chip */}
        {(hasErrors || hasWarns || readOnly) && (
          <>
            {(hasErrors || readOnly) && (
              <button type="button" className={`ig-issue-chip error${hasErrors ? " on" : ""}`} onClick={() => setIssuesPanelOpen((v) => !v)}>
                <i />
                <span className="ig-chip-label">{hasErrors ? `${errors.length} 个错误` : "无错误"}</span>
              </button>
            )}
            {(hasWarns || readOnly) && (
              <button type="button" className={`ig-issue-chip warn${hasWarns ? " on" : ""}`} onClick={() => setIssuesPanelOpen((v) => !v)}>
                <i />
                <span className="ig-chip-label">{hasWarns ? `${warns.length} 个提示` : "无提示"}</span>
              </button>
            )}
          </>
        )}

        {/* 「更多」溢出菜单（JSON 查看等调试功能） */}
        <div className="ig-more-menu-wrap" ref={moreMenuRef}>
          <button
            type="button"
            className={`ig-icon-action${moreMenuOpen ? " active" : ""}`}
            title="更多选项"
            aria-label="更多选项"
            aria-expanded={moreMenuOpen}
            onClick={() => setMoreMenuOpen((v) => !v)}
          >
            ⋮
          </button>
          {moreMenuOpen && (
            <div className="ig-more-menu" role="menu">
              <button
                type="button"
                className="ig-more-item"
                role="menuitem"
                onClick={() => { setJsonOpen((v) => !v); setMoreMenuOpen(false); }}
              >
                <span aria-hidden="true">{ `{ }` }</span>
                查看图结构 JSON
              </button>
              {!readOnly && (
                <button
                  type="button"
                  className="ig-more-item"
                  role="menuitem"
                  onClick={() => { setIssuesPanelOpen((v) => !v); setMoreMenuOpen(false); }}
                >
                  <span aria-hidden="true">⚠</span>
                  查看所有问题（{issues.length}）
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* 问题面板 */}
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
            title={paletteOpen ? "折叠组件库" : "展开组件库"}
            aria-label={paletteOpen ? "折叠组件库" : "展开组件库"}
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

          {/* 空画布引导 */}
          {isEmpty && !jsonOpen && (
            <div className="ig-empty-canvas">
              <div className="ig-empty-orb" aria-hidden="true">✦</div>
              <h2>开始构建推理图</h2>
              <p>从左侧组件库拖入模型节点，然后连接端口，构建完整的推理流程。</p>
              {!readOnly && (
                <button type="button" className="ig-btn ig-primary" onClick={() => setPaletteOpen(true)}>
                  <span aria-hidden="true">＋</span> 打开组件库
                </button>
              )}
              <div className="ig-empty-hints">
                <span><b>1</b> 添加图输入</span>
                <span><b>2</b> 拖入模型</span>
                <span><b>3</b> 连接端口</span>
                <span><b>4</b> 添加图输出</span>
              </div>
            </div>
          )}

          {/* 连线删除按钮 */}
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

        {/* 右侧检查器 */}
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

        {/* 底部状态 dock（仅编辑模式，保留问题指示器） */}
        {!readOnly && (
          <div className="ig-status-dock" role="status" aria-live="polite">
            <button type="button" className={`ig-issue-chip error${hasErrors ? " on" : ""}`} title={hasErrors ? `${errors.length} 个错误，点击查看` : "没有错误"} onClick={() => setIssuesPanelOpen((v) => !v)}>
              <i />
              <span className="ig-chip-label">{hasErrors ? `${errors.length} 个错误` : "无错误"}</span>
            </button>
            <button type="button" className={`ig-issue-chip warn${hasWarns ? " on" : ""}`} title={hasWarns ? `${warns.length} 个提示，点击查看` : "没有提示"} onClick={() => setIssuesPanelOpen((v) => !v)}>
              <i />
              <span className="ig-chip-label">{hasWarns ? `${warns.length} 个提示` : "无提示"}</span>
            </button>
          </div>
        )}
      </div>

      {/* JSON 抽屉（从「更多」菜单打开） */}
      {jsonOpen && (
        <JsonDrawer json={json} readOnly={readOnly} onApply={applyImportText} onClose={() => setJsonOpen(false)} />
      )}
    </div>
  );
}

// ── 公共导出面 ──

export { validateEditorValue, hasBlockingIssues, issueLevelByNode, shapeCompatible } from "../core/rules";
export { serializeGraph, sanitizeGraph, remoteNodesOf, localInputsOf, applyImport, emptyGraph } from "../core/model";
export { fillLayout, autoLayout, autoLayoutStrategy, keepLayoutStrategy } from "../core/layout";
export type { LayoutStrategy, LayoutMap } from "../core/layout";
export { staticCatalogProvider, httpCatalogProvider } from "../core/catalog";
export type { CatalogProvider, CatalogView, CatalogModelItem } from "../core/catalog";
export type { PaletteSource, PaletteHost, SourceCtx, SourceProgram, SourceFetch, PaletteItem } from "./palette-protocol";
export { browse, staticSource, querySource } from "./sources/browse";
export { ioSource } from "./sources/io";
export { catalogSources, defaultSources } from "./sources/catalog";
export type { EditorValue, EditorIssue, DcirGraphJson, DcirNode, DcirPort, IssueLevel, XY } from "../core/types";
