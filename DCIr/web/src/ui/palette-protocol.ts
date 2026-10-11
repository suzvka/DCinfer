/**
 * 调色板协议 — 一元原语（视图程序 × 唯一写通道）
 *
 * 第一性还原后的形态（三轮讨论收口）：
 * - 源的本质 = 一个消费上下文、产出交互面的程序（SourceProgram，按 React 组件
 *   方式调用，分组切换重挂载，程序体内可用 hooks）；
 * - 「静态快照 / 惰性查询 / 自定义面板」不是三种机制，只是「枚举策略 × 呈现归属」
 *   两个正交轴上的取样点——数据源降为编辑器自带组合子 browse 的参数（sources/）；
 * - 写通道唯一：一切候选经 ctx.place / ctx.placeIo 落画布（内部 sanitizeCatalogItem
 *   → addModelNode/addIoElement → commit），值语义与撤销栈的权威不容源直接触碰；
 * - 呈现自由：编辑器对 program 返回的交互面零假设（不接全局搜索是诚实的降级；
 *   要接搜索的源用 browse 组合子）。
 */

import type { ReactNode } from "react";
import type { CatalogModelItem } from "../core/catalog";
import type { ViewportRect } from "../core/layout";
import type { EditorValue, SignalLogicKind, XY } from "../core/types";

/** 拖拽载荷 MIME（IO 控件元素 / 信号逻辑节点 / 目录条目；与 core/catalog 的模型 MIME 同族） */
export const IG_IO_MIME = "application/x-infer-io";
export const IG_SIGNAL_MIME = "application/x-infer-signal";
export { IG_MODEL_MIME } from "../core/catalog";

/** 移动端断点（与 styles.css 媒体查询同源） */
export const IG_MOBILE_QUERY = "(max-width: 640px)";

/** 窄屏判定（jsdom 无 matchMedia 时按桌面处理） */
export function isMobileViewport(): boolean {
  return typeof window !== "undefined" && window.matchMedia?.(IG_MOBILE_QUERY)?.matches === true;
}

/**
 * 源上下文：编辑器提供给 program 的全部能力（能力即协议，除此之外零攻击面）。
 * value 为渲染期快照（只读；写一律经 place*）。
 */
export interface SourceCtx {
  /** 唯一模型产出通道：窄校验 → addModelNode（缺省落点 = 视口中心级联避让） */
  place(item: CatalogModelItem, position?: XY): void;
  /** IO 控件元素产出通道（编排类源使用） */
  placeIo(kind: "input" | "output", position?: XY): void;
  /** 信号编排逻辑节点产出通道 */
  placeSignalNode(logic: SignalLogicKind, position?: XY): void;
  /** 只读受控值（自定义视图可据此置灰已用条目等） */
  value: EditorValue;
  /** 画布视口矩形（流坐标；级联/定位辅助） */
  viewport(): ViewportRect | null;
  /** 收起浮层（源自愿退出舞台） */
  close(): void;
}

/** Palette 容器注入的主机半边（close 由容器补全成完整 SourceCtx） */
export type PaletteHost = Omit<SourceCtx, "close">;

/** 源程序：按组件方式调用（props = { ctx }），体内可用 hooks */
export type SourceProgram = (props: { ctx: SourceCtx }) => ReactNode;

/**
 * 默认浏览器（browse 组合子）的惰性列举函数：查询参数 + 中止信号。
 * 快照源 = 常函数；动态逻辑源 = 实现内部自行检索/分页/远程调用。
 */
export type SourceFetch = (query: string, ctrl: AbortController) => Promise<CatalogModelItem[]>;

/** 导轨挂载记录：一组 = 一个程序（label 即导轨按钮文案） */
export interface PaletteSource {
  id: string;
  label: string;
  /**
   * 分组图标：面板折叠（导轨收窄为方形磁贴、隐藏文字）时唯一可视的区分标识。
   * 缺省时回退 label 首字，保证内建与自定义源在折叠态都能互相区分。
   */
  icon?: string;
  program: SourceProgram;
}

// ── 调色板条目词汇（行渲染 / 详情卡 / 拖拽载荷的共用形态）──

export type PaletteItem =
  | { kind: "io"; ioKind: "input" | "output"; label: string; desc: string }
  | { kind: "signal"; logic: SignalLogicKind; label: string; desc: string }
  | { kind: "model"; item: CatalogModelItem };

/** 条目稳定键（列表 key / 详情卡开关判定） */
export function paletteItemKey(item: PaletteItem): string {
  if (item.kind === "io") return `io:${item.ioKind}`;
  if (item.kind === "signal") return `signal:${item.logic}`;
  return `model:${item.item.modelKey}`;
}

/** 条目检索文本（browse 搜索框用） */
export function paletteItemText(item: PaletteItem): string {
  if (item.kind === "io") return `${item.label} ${item.ioKind} ${item.desc}`;
  if (item.kind === "signal") return `${item.label} signal ${item.logic} ${item.desc}`;
  return `${item.item.modelKey} ${item.item.endpoints.join(" ")} ${item.item.labels.join(" ")}`;
}

/** 条目 → 写通道分派（place / placeIo / placeSignalNode 的唯一消费者） */
export function placePaletteItem(ctx: SourceCtx, item: PaletteItem): void {
  if (item.kind === "io") ctx.placeIo(item.ioKind);
  else if (item.kind === "signal") ctx.placeSignalNode(item.logic);
  else ctx.place(item.item);
}
