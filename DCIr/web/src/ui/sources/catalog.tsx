/**
 * 目录桥 — CatalogProvider（一次性快照契约）→ PaletteSource 语法糖族
 *
 * 标签分组语言改为用户友好：
 * - 「基础算子」分组保留，标签文案不变（业务约定）；
 * - 「查看全部」改为「全部模型」；
 * - 分组图标池保持用于折叠态区分。
 */

import type { CatalogModelItem, CatalogProvider, CatalogView } from "../../core/catalog";
import type { PaletteSource, SourceFetch } from "../palette-protocol";
import { browse } from "./browse";
import { ioSource } from "./io";

/** 系统保留标签（由系统组承载，不入业务标签展开） */
const RESERVED_LABELS: ReadonlySet<string> = new Set(["基础算子"]);

const loadCache = new WeakMap<CatalogProvider, Promise<CatalogView>>();

/** provider 级共享加载（快照语义：同一 provider 只拉一次；失败清缓存允许重试） */
export function loadShared(provider: CatalogProvider): Promise<CatalogView> {
  let p = loadCache.get(provider);
  if (!p) {
    p = provider.load().catch((e: unknown) => {
      loadCache.delete(provider);
      throw e;
    });
    loadCache.set(provider, p);
  }
  return p;
}

/** 标签过滤 fetch：load 共享快照 → 按标签切片 → 查询词本地过滤 */
function labelFetch(provider: CatalogProvider | null | undefined, label: string | null): SourceFetch {
  return async (query, ctrl) => {
    if (!provider) return [];
    const view = await loadShared(provider);
    if (ctrl.signal.aborted) throw new Error("aborted");
    const base: CatalogModelItem[] = label === null ? view.models : view.models.filter((m) => m.labels.includes(label));
    const q = query.trim().toLowerCase();
    if (!q) return base;
    return base.filter((m) => `${m.modelKey} ${m.labels.join(" ")} ${m.endpoints.join(" ")}`.toLowerCase().includes(q));
  };
}

/** 标签分组图标池 */
const TAG_ICONS: readonly string[] = ["⚙", "⇄", "▤", "◈", "⬡", "▦", "◉", "⬢"];

function tagIcon(label: string): string {
  let h = 0;
  for (let i = 0; i < label.length; i += 1) h = (h * 31 + label.charCodeAt(i)) >>> 0;
  return TAG_ICONS[h % TAG_ICONS.length];
}

/** 目录快照 → 标签分组源 + 「全部模型」兜底 */
export function catalogSources(provider: CatalogProvider | null | undefined, labels: readonly string[]): PaletteSource[] {
  const out: PaletteSource[] = [
    {
      id: "operators",
      label: "内置算子",
      icon: "⚙",
      program: browse(labelFetch(provider, "基础算子"), { searchable: false, placeholder: "搜索算子…" }),
    },
  ];
  for (const label of labels) {
    if (RESERVED_LABELS.has(label)) continue;
    out.push({
      id: `tag:${label}`,
      label,
      icon: tagIcon(label),
      program: browse(labelFetch(provider, label), { searchable: false }),
    });
  }
  out.push({
    id: "all",
    label: "全部模型",
    icon: "▦",
    program: browse(labelFetch(provider, null), { placeholder: "搜索模型…" }),
  });
  return out;
}

/** 缺省源装配：流程控制恒在 + 目录展开 */
export function defaultSources(provider: CatalogProvider | null | undefined, labels: readonly string[]): PaletteSource[] {
  return [ioSource, ...catalogSources(provider, labels)];
}
