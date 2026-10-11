/**
 * browse 组合子 — 「数据源即视图程序」的标准实现
 *
 * 一元化后编辑器不再有「三级源接口」：静态快照与惰性查询都只是 fetch 的退化
 * 形态（常数函数 / 参数化函数），呈现由本组合子的默认浏览器承担。二次开发
 * 只要默认交互形态，给 fetch 就行；不够再自写 program（见 palette-protocol.ts）。
 *
 * fetch 纪律：
 * - 每次查询携带 AbortController，切换/重查时旧请求作废（aborted 不置错误态）；
 * - 失败/异常 = 视图内提示，绝不 throw 进渲染路径；
 * - fetch 引用参与重查依赖，宿主应保持稳定（staticSource/querySource 工厂已保证）。
 */

import { useEffect, useMemo, useState } from "react";
import type { CatalogModelItem } from "../../core/catalog";
import {
  paletteItemText,
  type PaletteItem,
  type PaletteSource,
  type SourceCtx,
  type SourceFetch,
  type SourceProgram,
} from "../palette-protocol";
import { RowExplorer, modelPaletteItem } from "./parts";

export type { SourceFetch };

/** 搜索防抖（ms）：连续键入合并为一次 fetch */
const BROWSE_DEBOUNCE = 200;

/** 默认浏览器交互体：搜索框（可选）+ 行列表 + 详情卡 */
function BrowseBody({
  fetch,
  ctx,
  searchable,
  placeholder,
}: {
  fetch: SourceFetch;
  ctx: SourceCtx;
  searchable: boolean;
  placeholder: string;
}) {
  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");
  const [phase, setPhase] = useState<"loading" | "ready" | "error">("loading");
  const [items, setItems] = useState<CatalogModelItem[]>([]);

  useEffect(() => {
    const t = window.setTimeout(() => setDebounced(query), BROWSE_DEBOUNCE);
    return () => window.clearTimeout(t);
  }, [query]);

  useEffect(() => {
    const ctrl = new AbortController();
    let alive = true;
    setPhase("loading");
    fetch(debounced, ctrl)
      .then((x) => {
        if (!alive || ctrl.signal.aborted) return;
        setItems(x);
        setPhase("ready");
      })
      .catch(() => {
        if (alive && !ctrl.signal.aborted) setPhase("error");
      });
    return () => {
      alive = false;
      ctrl.abort();
    };
  }, [fetch, debounced]);

  const paletteItems: PaletteItem[] = useMemo(() => items.map(modelPaletteItem), [items]);
  const hint =
    phase === "error"
      ? "条目加载失败（数据源异常；不阻塞本地编辑）"
      : phase === "loading"
        ? "加载条目中…"
        : items.length === 0
          ? "暂无条目"
          : null;

  return (
    <>
      {searchable && (
        <div className="ig-fly-search">
          <input
            className="ig-input"
            placeholder={placeholder}
            value={query}
            spellCheck={false}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
      )}
      <RowExplorer items={paletteItems} ctx={ctx} hint={hint} />
    </>
  );
}

/**
 * 组合子：fetch → 默认浏览器程序。
 * opts.searchable = false 关闭搜索框（结果集由过滤逻辑预切片的小分组适用）。
 */
export function browse(
  fetch: SourceFetch,
  opts: { searchable?: boolean; placeholder?: string } = {},
): SourceProgram {
  const searchable = opts.searchable !== false;
  const placeholder = opts.placeholder ?? "搜索条目…";
  return ({ ctx }) => <BrowseBody fetch={fetch} ctx={ctx} searchable={searchable} placeholder={placeholder} />;
}

/** 语法糖：静态清单源（fetch = 常函数 + 本地过滤；等价旧 L0 快照） */
export function staticSource(id: string, label: string, items: readonly CatalogModelItem[]): PaletteSource {
  const snapshot = [...items];
  const fetch: SourceFetch = async (query) => {
    const q = query.trim().toLowerCase();
    if (!q) return snapshot;
    return snapshot.filter((m) => `${m.modelKey} ${m.labels.join(" ")}`.toLowerCase().includes(q));
  };
  return { id, label, program: browse(fetch) };
}

/** 语法糖：查询函数源（任意动态逻辑；等价旧 L1；等价 abort 由实现自愿使用） */
export function querySource(
  id: string,
  label: string,
  fetch: SourceFetch,
  opts: { searchable?: boolean; placeholder?: string } = {},
): PaletteSource {
  return { id, label, program: browse(fetch, opts) };
}

/** 条目检索文本再导出（宿主写 fetch 本地过滤时可用） */
export { paletteItemText };
