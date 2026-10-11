/**
 * 推理图编辑器 — 节点目录（core 层：数据契约 + 提供器抽象，无 React 依赖）
 *
 * 解耦纪律：编辑器对目录数据来源零假设。二次开发经 CatalogProvider 注入自己的
 * 数据源（静态清单 / HTTP / 本地注册表导出均可）；ui 层 useCatalog 只消费
 * load() 返回的 CatalogView，拉取失败一律降级离线（不阻塞本地编辑）。
 *
 * 调色板静态资产：IO 控件元素条目 + 端口工厂（基础算子/模型条目一律经目录进入画布，
 * 编辑期不设手编模板）。
 */

import type { DcirPort, SignalLogicKind } from "./types";

// ── IO 控件元素调色板（编辑器侧构造：投影为图级输入/输出绑定）──

export interface IoPaletteSpec {
  kind: "input" | "output";
  label: string;
  desc: string;
}

export const IO_PALETTE: IoPaletteSpec[] = [
  { kind: "input", label: "输入元素", desc: "图级输入（外部供数）：右侧接线点连向节点输入端口" },
  { kind: "output", label: "输出元素", desc: "图输出（结果取回）：节点输出端口连向它左侧接线点" },
];

// ── 信号编排逻辑节点调色板（只有信号端子的图内控制流原语）──

export interface SignalPaletteSpec {
  logic: SignalLogicKind;
  label: string;
  desc: string;
}

export const SIGNAL_PALETTE: SignalPaletteSpec[] = [
  { logic: "source", label: "信号源", desc: "对外信号入口：宿主经 setSignal(名) 驱动，输出布尔" },
  { logic: "const", label: "常量信号", desc: "固定布尔输出（true/false），用于恒定放行/阻断" },
  { logic: "not", label: "信号非", desc: "单输入取反" },
  { logic: "and", label: "信号与", desc: "多输入全 true 则 true（入度可增减）" },
  { logic: "or", label: "信号或", desc: "多输入任一 true 则 true（入度可增减）" },
];

/** Float 端口工厂（tensorType/typeSize 单点书写） */
export function fPort(name: string, shape: number[], required = true): DcirPort {
  return { name, tensorType: "Float", typeSize: 4, shape, required };
}

// ── 目录数据契约 ──

/** 目录条目拖拽载荷 MIME（载荷 = 条目 JSON） */
export const IG_MODEL_MIME = "application/x-infer-model";

/** 目录模型条目（模型 / 算子统一形态） */
export interface CatalogModelItem {
  modelKey: string;
  /** 条目自报引擎类型（模型=推理引擎名；算子=引擎名）；null = 未自报 */
  engineType: string | null;
  endpoints: string[];
  labels: string[];
  /** 条目端口声明（io 驱动建节点端口；null = 未声明） */
  io: { inputs: DcirPort[]; outputs: DcirPort[] } | null;
}

/** 目录视图（CatalogProvider.load 的返回契约） */
export interface CatalogView {
  groups: { id: string; label: string; models: CatalogModelItem[] }[];
  models: CatalogModelItem[];
  endpointCount: number;
  /** 合法标签全量列表（含暂无条目者；= 调色板业务分组的白名单骨架来源） */
  labels: string[];
}

/** 条目结构窄校验（任意 unknown → 合法条目或 null；内存对象 / 拖拽载荷共用同一防线） */
export function sanitizeCatalogItem(raw: unknown): CatalogModelItem | null {
  if (typeof raw !== "object" || raw === null) return null;
  const o = raw as Record<string, unknown>;
  if (typeof o.modelKey !== "string" || o.modelKey.length === 0) return null;
  return {
    modelKey: o.modelKey,
    engineType: typeof o.engineType === "string" && o.engineType.length > 0 ? o.engineType : null,
    endpoints: Array.isArray(o.endpoints) ? o.endpoints.filter((x): x is string => typeof x === "string") : [],
    labels: Array.isArray(o.labels) ? o.labels.filter((x): x is string => typeof x === "string") : [],
    io: (o.io as CatalogModelItem["io"] | undefined) ?? null,
  };
}

/** 拖拽载荷窄校验（载荷来自 DOM dataTransfer，不可信；解析后转共用防线） */
export function parseCatalogItem(raw: string): CatalogModelItem | null {
  try {
    return sanitizeCatalogItem(JSON.parse(raw));
  } catch {
    return null;
  }
}

/** 端口声明摘要（条目行 / 详情卡） */
export function ioSummary(item: CatalogModelItem): string {
  if (!item.io) return "端口未声明（建节点后在检查器补）";
  return `in ${item.io.inputs.length} / out ${item.io.outputs.length}`;
}

// ── 提供器抽象（数据源注入点）──

/**
 * 目录提供器契约：编辑器唯一的目录接入面。
 * 实现要求：load() 可重复调用、失败 reject（ui 层据此降级离线，不抛入渲染路径）。
 * 宿主传入的实例应保持引用稳定（模块级常量或 useMemo），避免重复拉取。
 */
export interface CatalogProvider {
  load(): Promise<CatalogView>;
}

/** 静态清单提供器（本地开发 / 离线场景：注册表导出、内嵌常量均可） */
export function staticCatalogProvider(view: CatalogView): CatalogProvider {
  return { load: () => Promise.resolve(view) };
}

/**
 * HTTP 提供器：GET url → CatalogView JSON。
 * token 为惰性读取（每次拉取时求值），有值附 Bearer；非 2xx / 网络失败 reject。
 */
export function httpCatalogProvider(
  url: string,
  opts: { token?: () => string | null } = {},
): CatalogProvider {
  return {
    load: async () => {
      const token = opts.token?.() ?? null;
      const r = await fetch(url, token ? { headers: { authorization: `Bearer ${token}` } } : undefined);
      if (!r.ok) throw new Error(`catalog HTTP ${r.status}`);
      return (await r.json()) as CatalogView;
    },
  };
}
