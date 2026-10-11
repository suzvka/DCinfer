/**
 * 目录桥 — CatalogProvider（一次性快照契约）→ PaletteSource 语法糖族
 *
 * 兼容纪律：core 的 CatalogProvider 不动（它是一批条目的一次性快照，真实存在的
 * 业务形态），此处把它降解为「一个/多个 browse 源的取数后端」：
 * - loadShared：同一 provider 共享一次 load（WeakMap 缓存，失败清缓存可重试）；
 * - catalogSources：labels（或宿主白名单）逐个展开为标签分组 + 「查看全部」兜底；
 * - defaultSources：编排节点恒在 + 目录展开（provider 缺省 = 只剩编排）。
 * 旧 PaletteView 的「基础算子」保留标签语义在此原样继承（系统组、不入业务展开）。
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

/** 标签过滤 fetch：load 共享快照 → 按标签切片 → 查询词本地过滤（provider 缺省 = 空列表，空分组降级） */
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

/** 标签分组图标池（折叠态磁贴靠此区分不同标签组；按 label 稳定取模，同一标签恒定同图） */
const TAG_ICONS: readonly string[] = ["⚙", "⇄", "▤", "◈", "⬡", "▦", "◉", "⬢"];

function tagIcon(label: string): string {
  let h = 0;
  for (let i = 0; i < label.length; i += 1) h = (h * 31 + label.charCodeAt(i)) >>> 0;
  return TAG_ICONS[h % TAG_ICONS.length];
}

/** 目录快照 → 标签分组源 + 「查看全部」作为最后防线（labels 缺省用宿主白名单；provider 缺省 = 空分组渲染） */
export function catalogSources(provider: CatalogProvider | null | undefined, labels: readonly string[]): PaletteSource[] {
  const out: PaletteSource[] = [
    // 基础算子（系统预置标签组）：恒在、不依赖白名单
    { id: "operators", label: "基础算子", icon: "⚙", program: browse(labelFetch(provider, "基础算子"), { searchable: false }) },
  ];
  for (const label of labels) {
    if (RESERVED_LABELS.has(label)) continue;
    out.push({ id: `tag:${label}`, label, icon: tagIcon(label), program: browse(labelFetch(provider, label), { searchable: false }) });
  }
  out.push({ id: "all", label: "查看全部", icon: "▦", program: browse(labelFetch(provider, null)) });
  return out;
}

/** 缺省源装配：编排恒在 + 目录展开（provider 缺省 = 各组空态，与旧离线降级同口径） */
export function defaultSources(provider: CatalogProvider | null | undefined, labels: readonly string[]): PaletteSource[] {
  return [ioSource, ...catalogSources(provider, labels)];
}
