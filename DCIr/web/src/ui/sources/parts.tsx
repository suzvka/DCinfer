/**
 * 调色板共享部件：条目行 / 详情卡 / 行探索器（RowExplorer）
 *
 * browse 组合子与 ioSource 共用本模块——默认呈现只在这里定义一份，二次开发可
 * 只换数据（browse(fetch)）或整体换视图（自定义 program），两个轴互不牵连。
 * RowExplorer 承接原 PaletteView 的浮层内部逻辑（行点击 → 锚定详情卡 → 添加）。
 */

import { useMemo, useState, type DragEvent, type MouseEvent } from "react";
import { ioSummary, type CatalogModelItem } from "../../core/catalog";
import type { DcirPort } from "../../core/types";
import {
  IG_IO_MIME,
  IG_MODEL_MIME,
  IG_SIGNAL_MIME,
  isMobileViewport,
  paletteItemKey,
  placePaletteItem,
  type PaletteItem,
  type SourceCtx,
} from "../palette-protocol";

/** 端子类型标签（与节点卡片口径一致：f32 简写；信号端子 = signal） */
export function detailPortTag(p: DcirPort): string {
  if (p.kind === "signal") return "signal";
  const type = p.tensorType === "Float" && p.typeSize === 4 ? "f32" : `${p.tensorType}:${p.typeSize}`;
  return `${type}[${(p.shape ?? []).join(",")}]`;
}

/** 子菜单条目行：拖拽 = 落点添加；点击 = 详情卡（回调上抛点击事件，宿主计算锚点） */
export function PaletteRow({
  item,
  active,
  onClick,
}: {
  item: PaletteItem;
  active: boolean;
  onClick: (e: MouseEvent<HTMLDivElement>) => void;
}) {
  const onDragStart = (e: DragEvent<HTMLDivElement>) => {
    if (item.kind === "io") {
      e.dataTransfer.setData(IG_IO_MIME, item.ioKind);
    } else if (item.kind === "signal") {
      e.dataTransfer.setData(IG_SIGNAL_MIME, item.logic);
    } else {
      e.dataTransfer.setData(IG_MODEL_MIME, JSON.stringify(item.item));
    }
    e.dataTransfer.effectAllowed = "copy";
  };
  const tip =
    item.kind === "model"
      ? `${item.item.modelKey}\n${item.item.endpoints.length > 0 ? `提供端点：${item.item.endpoints.join(", ")}\n` : ""}${ioSummary(item.item)}\n拖拽 = 放置到画布；点击 = 查看详情`
      : `${item.desc}\n拖拽 = 放置到画布；点击 = 查看详情`;
  return (
    <div
      className={`ig-pal-item${item.kind === "io" ? ` ig-pal-io io-${item.ioKind}` : ""}${item.kind === "signal" ? " ig-pal-signal" : ""}${active ? " active" : ""}`}
      draggable
      onDragStart={onDragStart}
      onClick={onClick}
      title={tip}
    >
      {(item.kind === "io" || item.kind === "signal") && (
        <>
          <div className="ig-pal-head">
            <span className="ig-pal-label">{item.label}</span>
          </div>
          <div className="ig-pal-desc">{item.desc}</div>
        </>
      )}
      {item.kind === "model" && (
        <>
          <div className="ig-pal-head">
            <span className="ig-pal-label ig-mono">{item.item.modelKey}</span>
          </div>
          <div className="ig-pal-desc">
            {item.item.labels.length > 0 ? item.item.labels.join(" / ") : item.item.modelKey}
          </div>
        </>
      )}
    </div>
  );
}

/** 详情卡「添加到画布」主按钮 */
function DetailAddButton({ onAdd }: { onAdd: () => void }) {
  return (
    <button type="button" className="ig-btn ig-primary ig-detail-add" onClick={onAdd}>
      添加到画布
    </button>
  );
}

/** 详情卡：IO 元素（描述）/ 目录条目（端点 / 标签 / 端子声明） */
export function PaletteDetail({ item, onAdd }: { item: PaletteItem; onAdd: () => void }) {
  if (item.kind === "io") {
    return (
      <>
        <div className="ig-detail-head">
          <span className="ig-detail-title">{item.label}</span>
          <span className={`ig-aff aff-${item.ioKind === "input" ? "operator" : "compute"}`}>
            {item.ioKind === "input" ? "图输入" : "图输出"}
          </span>
        </div>
        <p className="ig-detail-desc">{item.desc}</p>
        <DetailAddButton onAdd={onAdd} />
        <p className="ig-detail-note">点击「添加到画布」落位（桌面端亦可拖拽）；线段连到节点端子即完成绑定。</p>
      </>
    );
  }
  if (item.kind === "signal") {
    return (
      <>
        <div className="ig-detail-head">
          <span className="ig-detail-title">{item.label}</span>
          <span className="ig-aff aff-system">信号</span>
        </div>
        <p className="ig-detail-desc">{item.desc}</p>
        <DetailAddButton onAdd={onAdd} />
        <p className="ig-detail-note">点击「添加到画布」落位；信号端子间拖线即连接（布尔）。</p>
      </>
    );
  }
  const m = item.item;
  return (
    <>
      <div className="ig-detail-head">
        <span className="ig-detail-title ig-mono">{m.modelKey}</span>
      </div>
      <div className="ig-detail-sub">{`标签：${m.labels.join(" / ") || "（无标签）"}`}</div>
      <DetailAddButton onAdd={onAdd} />
      {m.io ? (
        <>
          <DetailPorts title="输入端子（条目声明）" ports={m.io.inputs} />
          <DetailPorts title="输出端子（条目声明）" ports={m.io.outputs} />
        </>
      ) : (
        <p className="ig-detail-note">端子未声明：建节点后在检查器补端子 Schema（声明即执行面）。</p>
      )}
    </>
  );
}

/** 端子表（详情卡内） */
function DetailPorts({ title, ports }: { title: string; ports: readonly DcirPort[] }) {
  return (
    <div className="ig-detail-ports">
      <div className="ig-detail-ports-title">{`${title}（${ports.length}）`}</div>
      {ports.length === 0 ? (
        <div className="ig-detail-port-none">（空）</div>
      ) : (
        ports.map((p) => (
          <div key={p.name} className="ig-detail-port">
            <span className="ig-detail-port-name">{`${p.name}${p.required === false ? "?" : ""}`}</span>
            <span className="ig-detail-port-tag">{detailPortTag(p)}</span>
          </div>
        ))
      )}
    </div>
  );
}

/**
 * 行探索器（浮层交互体）：条目列表 + 锚定详情卡 + 添加分派（经唯一写通道）。
 * 宿主程序负责供给条目与提示文案（搜索框 / 加载态在 browse 组合子侧）。
 */
export function RowExplorer({ items, ctx, hint }: { items: readonly PaletteItem[]; ctx: SourceCtx; hint?: string | null }) {
  const [detailKey, setDetailKey] = useState<string | null>(null);
  const [detailTop, setDetailTop] = useState(0);

  // 条目集合变化时详情卡跟随；条目消失自动关闭
  const detailItem = useMemo(
    () => (detailKey !== null ? (items.find((it) => paletteItemKey(it) === detailKey) ?? null) : null),
    [items, detailKey],
  );

  // 条目点击：开关详情卡（锚点 = 行顶部相对浮层顶部的偏移，钳制在可视区内；窄屏 top = 0）
  const openRow = (item: PaletteItem, e: MouseEvent<HTMLDivElement>) => {
    const key = paletteItemKey(item);
    if (detailKey === key) {
      setDetailKey(null);
      return;
    }
    setDetailKey(key);
    const fly = e.currentTarget.closest(".ig-flyout") as HTMLElement | null;
    if (!fly || isMobileViewport()) {
      setDetailTop(0);
      return;
    }
    const flyRect = fly.getBoundingClientRect();
    const rowRect = e.currentTarget.getBoundingClientRect();
    setDetailTop(Math.max(0, Math.min(rowRect.top - flyRect.top, flyRect.height - 96)));
  };

  // 「添加到画布」：唯一写通道落图 + 关详情卡；窄屏连关浮层（露出画布中心落点）
  const addItem = (item: PaletteItem) => {
    placePaletteItem(ctx, item);
    setDetailKey(null);
    if (isMobileViewport()) ctx.close();
  };

  return (
    <>
      <div className="ig-fly-scroll">
        {items.map((it) => {
          const key = paletteItemKey(it);
          return <PaletteRow key={key} item={it} active={detailKey === key} onClick={(e) => openRow(it, e)} />;
        })}
        {hint != null && <p className="ig-hint">{hint}</p>}
      </div>
      {detailItem && (
        <div className="ig-detail" style={{ top: detailTop }}>
          <button type="button" className="ig-iconbtn ig-detail-close" title="关闭详情" onClick={() => setDetailKey(null)}>
            ×
          </button>
          <PaletteDetail item={detailItem} onAdd={() => addItem(detailItem)} />
        </div>
      )}
    </>
  );
}

/** 目录条目 → 调色板条目形态（browse 组合子用） */
export function modelPaletteItem(item: CatalogModelItem): PaletteItem {
  return { kind: "model", item };
}
