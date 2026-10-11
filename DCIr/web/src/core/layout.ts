/**
 * 推理图编辑器 — 自动布局（dagre 分层，LR 方向）
 *
 * 用途：模板载入 / JSON 导入后的坐标种子；工具条「自动排布」全量重排。
 * 反馈回路（环）由 dagre 内部破环处理（仅影响布局视觉，不改变拓扑）。
 * IO 元素按绑定锚点推断：输入元素置于锚点左侧、输出元素右侧（同锚点堆叠）。
 * 坐标是编辑器侧元数据，绝不进入 DCIr 图 JSON。
 *
 * 2026-10-11 移动端适配批：新增 cascadePlace 级联落位（「添加到画布」缺省坐标）。
 */

import dagre from "@dagrejs/dagre";
import { ioFromBindings } from "./model";
import type { DcirGraphJson, DcirNode, EditorValue, IoElement, XY } from "./types";

/** 节点宽 + 高度估算（与 styles.css 的 .ig-node 尺寸约定一致） */
export const NODE_WIDTH = 236;
const HEADER_H = 46;
const PORT_ROW_H = 22;
const PAD_BOTTOM = 12;
/** 信号条带高（顶部 gate 入 / 底部状态出，各一条） */
const SIG_BAND_H = 18;

/** IO 元素尺寸/间距估算（与 styles.css 的 .ig-io-node 约定一致；级联落位复用） */
export const IO_ELEM_WIDTH = 132;
export const IO_ELEM_HEIGHT = 44;
const IO_GAP_X = 64;
const IO_GAP_Y = 12;

export function nodeHeight(node: Pick<DcirNode, "inputs" | "outputs">): number {
  const sigIn = node.inputs.some((p) => p.kind === "signal");
  const sigOut = node.outputs.some((p) => p.kind === "signal");
  // tensor 行数仅按数据端口计；纯信号逻辑节点无 tensor 行
  const tensorRows = Math.max(
    node.inputs.filter((p) => p.kind !== "signal").length,
    node.outputs.filter((p) => p.kind !== "signal").length,
    sigIn || sigOut ? 0 : 1,
  );
  const bands = (sigIn ? SIG_BAND_H : 0) + (sigOut ? SIG_BAND_H : 0);
  return HEADER_H + tensorRows * PORT_ROW_H + bands + PAD_BOTTOM;
}

/** 全量分层布局（node name / io element id → 左上角坐标） */
export function autoLayout(graph: DcirGraphJson, io: readonly IoElement[] = []): Record<string, XY> {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "LR", nodesep: 36, ranksep: 76, marginx: 24, marginy: 24 });
  g.setDefaultEdgeLabel(() => ({}));

  const names = new Set(graph.nodes.map((n) => n.name));
  for (const n of graph.nodes) {
    g.setNode(n.name, { width: NODE_WIDTH, height: nodeHeight(n) });
  }
  const paired = new Set<string>();
  for (const e of graph.edges) {
    if (!names.has(e.srcNode) || !names.has(e.dstNode)) continue;
    const key = `${e.srcNode}\u0000${e.dstNode}`;
    if (paired.has(key)) continue; // 同对节点多重边对布局等价，去重
    paired.add(key);
    g.setEdge(e.srcNode, e.dstNode);
  }

  dagre.layout(g);

  const layout: Record<string, XY> = {};
  for (const n of graph.nodes) {
    const pos = g.node(n.name) as { x: number; y: number } | undefined;
    if (!pos) {
      layout[n.name] = { x: 0, y: 0 };
      continue;
    }
    layout[n.name] = {
      x: Math.round(pos.x - NODE_WIDTH / 2),
      y: Math.round(pos.y - nodeHeight(n) / 2),
    };
  }

  if (io.length === 0) return layout;

  // IO 元素：绑定锚点外侧排列（同锚点同向堆叠）；未绑定元素靠左/右缘堆叠
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  for (const pos of Object.values(layout)) {
    minX = Math.min(minX, pos.x);
    maxX = Math.max(maxX, pos.x + NODE_WIDTH);
    minY = Math.min(minY, pos.y);
  }
  if (!Number.isFinite(minX)) {
    minX = 0;
    maxX = 0;
    minY = 0;
  }
  const slotByAnchor = new Map<string, number>();
  const loose = { input: 0, output: 0 };
  for (const el of io) {
    const anchor = el.node !== undefined ? layout[el.node] : undefined;
    if (anchor) {
      const slotKey = `${el.node}\u0000${el.kind}`;
      const slot = slotByAnchor.get(slotKey) ?? 0;
      slotByAnchor.set(slotKey, slot + 1);
      layout[el.id] = {
        x: Math.round(el.kind === "input" ? anchor.x - IO_GAP_X - IO_ELEM_WIDTH : anchor.x + NODE_WIDTH + IO_GAP_X),
        y: Math.round(anchor.y + slot * (IO_ELEM_HEIGHT + IO_GAP_Y)),
      };
    } else {
      const idx = loose[el.kind];
      loose[el.kind] += 1;
      layout[el.id] = {
        x: Math.round(el.kind === "input" ? minX - IO_GAP_X - IO_ELEM_WIDTH : maxX + IO_GAP_X),
        y: Math.round(minY + idx * (IO_ELEM_HEIGHT + IO_GAP_Y)),
      };
    }
  }
  return layout;
}

/** fillLayout 入参：io 缺省时从 bindings 派生（模板/存量装配路径） */
export interface EditorValueInput {
  graph: DcirGraphJson;
  layout: Record<string, XY>;
  remote: Record<string, string>;
  io?: IoElement[];
}

/** 规范化：补 io（缺省从 bindings 派生）+ 缺失坐标补种 + 孤儿坐标清理；已有坐标原样保留 */
export function fillLayout(value: EditorValueInput): EditorValue {
  const io = value.io ?? ioFromBindings(value.graph);
  const auto = autoLayout(value.graph, io);
  const layout: Record<string, XY> = {};
  for (const n of value.graph.nodes) {
    layout[n.name] = value.layout[n.name] ?? auto[n.name] ?? { x: 0, y: 0 };
  }
  for (const el of io) {
    layout[el.id] = value.layout[el.id] ?? auto[el.id] ?? { x: 0, y: 0 };
  }
  return { graph: value.graph, layout, remote: value.remote, io };
}

// ── 级联落位（「添加到画布」缺省坐标；Windows 式连续开窗堆叠）──

/** 级联步进（流坐标；向右下偏移量） */
export const CASCADE_STEP = 32;
/** 左上角重合容差（双轴 |Δ| 均小于该值即视为重位；小范围判定，不必完全重合） */
export const CASCADE_TOLERANCE = 12;
/** 迭代上限（防死循环；耗尽返回视口左上兜底） */
export const CASCADE_MAX_ATTEMPTS = 64;
/** 越界兜底内边距（相对视口左上角内缩） */
export const CASCADE_FALLBACK_PAD = 24;

/** 视口矩形（流坐标；左上角 + 尺寸） */
export interface ViewportRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface CascadePlaceOptions {
  viewport: ViewportRect;
  /** 待放置项尺寸估算（图节点 NODE_WIDTH × nodeHeight / IO 元素 132×44） */
  size: { width: number; height: number };
  /** 已放置项左上角坐标（图节点 ∪ IO 元素的 layout 值） */
  taken: readonly XY[];
}

/**
 * 级联落位（纯函数）：
 * - 初始候选 = 视口中心点 − 尺寸/2；
 * - 与任一已放置项左上角小范围重位（双轴 |Δ| < CASCADE_TOLERANCE）→ 右下步进
 *   （CASCADE_STEP）后重查；
 * - 候选右边/下边触达视口右/下边界（含相切）→ 重置到视口左上（+内边距）继续
 *   重查（Windows 连续开窗触边重启级联语义）；
 * - 迭代上限耗尽 → 返回视口左上兜底。
 */
export function cascadePlace(opts: CascadePlaceOptions): XY {
  const { viewport, size, taken } = opts;
  const right = viewport.x + viewport.width;
  const bottom = viewport.y + viewport.height;
  const fallback: XY = {
    x: Math.round(viewport.x + CASCADE_FALLBACK_PAD),
    y: Math.round(viewport.y + CASCADE_FALLBACK_PAD),
  };
  let x = Math.round(viewport.x + (viewport.width - size.width) / 2);
  let y = Math.round(viewport.y + (viewport.height - size.height) / 2);
  const collides = (px: number, py: number): boolean =>
    taken.some((p) => Math.abs(p.x - px) < CASCADE_TOLERANCE && Math.abs(p.y - py) < CASCADE_TOLERANCE);
  for (let i = 0; i < CASCADE_MAX_ATTEMPTS; i += 1) {
    if (x + size.width >= right || y + size.height >= bottom) {
      // 越界兜底：回到视口左上重启级联
      x = fallback.x;
      y = fallback.y;
    } else if (!collides(x, y)) {
      return { x, y };
    } else {
      x += CASCADE_STEP;
      y += CASCADE_STEP;
    }
  }
  return fallback;
}

// ── 布局策略注入点（二次开发自定义布局）──

/** 坐标表：node name / io element id → 左上角画布坐标（side-band，不进图 JSON） */
export type LayoutMap = Record<string, XY>;

/**
 * 布局策略契约：从图拓扑 + IO 元素全量重算坐标；prev 为现行坐标表（策略自行
 * 决定参考或忽略）。实现要求：纯函数、无副作用、不修改入参。
 */
export interface LayoutStrategy {
  compute(graph: DcirGraphJson, io: readonly IoElement[], prev?: LayoutMap): LayoutMap;
}

/** 默认策略：dagre 分层自动布局（全量重排，忽略 prev） */
export const autoLayoutStrategy: LayoutStrategy = {
  compute: (graph, io) => autoLayout(graph, io),
};

/** 保守策略：原样保留现行坐标（宿主完全自管坐标的场景；新增节点由 fillLayout 补种） */
export const keepLayoutStrategy: LayoutStrategy = {
  compute: (_graph, _io, prev) => (prev ? { ...prev } : {}),
};
