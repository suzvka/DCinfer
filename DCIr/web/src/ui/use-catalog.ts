/**
 * ui 层 — 目录拉取 hook（CatalogProvider 消费面）
 *
 * provider 引用变化才重拉（宿主应保持实例稳定：模块级常量或 useMemo）；
 * 未注入 / load() 失败 → 离线降级，不阻塞本地编辑。
 */

import { useEffect, useState } from "react";
import type { CatalogProvider, CatalogView } from "../core/catalog";

export function useCatalog(provider: CatalogProvider | null | undefined): {
  catalog: CatalogView | null;
  offline: boolean;
} {
  const [catalog, setCatalog] = useState<CatalogView | null>(null);
  const [offline, setOffline] = useState(false);
  useEffect(() => {
    let alive = true;
    if (!provider) {
      setCatalog(null);
      setOffline(true);
      return;
    }
    setOffline(false);
    provider
      .load()
      .then((view) => {
        if (alive) setCatalog(view);
      })
      .catch(() => {
        if (alive) setOffline(true);
      });
    return () => {
      alive = false;
    };
  }, [provider]);
  return { catalog, offline };
}
