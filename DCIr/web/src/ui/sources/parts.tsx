/**
 * 调色板共享部件：条目行 / 详情卡 / 行探索器（RowExplorer）
 *
 * 重设计原则（面向用户）：
 * - 模型条目显示短名（modelKey 末段）+ 可读标签，不显示完整路径；
 * - 详情卡端口用友好类型名（f32 / i32），形状用自然语言格式（[批次, 通道, 高, 宽]）；
 * - IO 元素和信号节点文案用任务语言。
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
import { friendlyPortShape, friendlyPortType, shortModelName } from "../i18n";

/** 端子类型标签（详情卡、检查器共用口径；信号端子 = signal） */
export function detailPortTag(p: DcirPort): string {
  if (p.kind === "signal") return "signal";
  const type = p.tensorType === "Float" && p.typeSize === 4 ? "f32" : `${p.tensorType}:${p.typeSize}`;
  return `${type}[${(p.shape ?? []).join(",")}]`;
}

/** 条目行：拖拽 = 落点添加；点击 = 详情卡 */
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

  // 条目行标题：模型用短名，IO/信号用友好标签
  const rowTitle =
    item.kind === "model"
      ? shortModelName(item.item.modelKey)
      : item.label;

  // 条目行副标题
  const rowSub =
    item.kind === "model"
      ? item.item.labels.length > 0
        ? item.item.labels.join(" / ")
        : ioSummary(item.item)
      : item.desc;

  const tip =
    item.kind === "model"
      ? `${item.item.modelKey}\n${item.item.endpoints.length > 0 ? `端点：${item.item.endpoints.join(", ")}\n` : ""}${ioSummary(item.item)}\n拖拽放置或点击查看详情`
      : `${item.desc}\n拖拽放置或点击查看详情`;

  return (
    <div
      className={`ig-pal-item${item.kind === "io" ? ` ig-pal-io io-${item.ioKind}` : ""}${item.kind === "signal" ? " ig-pal-signal" : ""}${active ? " active" : ""}`}
      draggable
      onDragStart={onDragStart}
      onClick={onClick}
      title={tip}
    >
      <div className="ig-pal-head">
        <span className="ig-pal-label">{rowTitle}</span>
      </div>
      {rowSub && <div className="ig-pal-desc">{rowSub}</div>}
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

/** 端子表（详情卡内）——使用用户友好类型名 */
function DetailPorts({ title, ports }: { title: string; ports: readonly DcirPort[] }) {
  return (
    <div className="ig-detail-ports">
      <div className="ig-detail-ports-title">{`${title}（${ports.length}）`}</div>
      {ports.length === 0 ? (
        <div className="ig-detail-port-none">（无）</div>
      ) : (
        ports.map((p) => {
          const typeStr = friendlyPortType(p);
          const shapeStr = friendlyPortShape(p);
          return (
            <div key={p.name} className="ig-detail-port">
              <span className="ig-detail-port-name">{`${p.name}${p.required === false ? "（可选）" : ""}`}</span>
              <span className="ig-detail-port-tag">
                {p.kind === "signal" ? "信号" : `${typeStr}${shapeStr ? " " + shapeStr : ""}`}
              </span>
            </div>
          );
        })
      )}
    </div>
  );
}

/** 详情卡：IO 元素 / 信号节点 / 目录条目 */
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
        <p className="ig-detail-note">点击「添加到画布」后，把接线点拖到节点端口即完成绑定。桌面端也可直接拖拽到画布。</p>
      </>
    );
  }
  if (item.kind === "signal") {
    return (
      <>
        <div className="ig-detail-head">
          <span className="ig-detail-title">{item.label}</span>
          <span className="ig-aff aff-system">控制流</span>
        </div>
        <p className="ig-detail-desc">{item.desc}</p>
        <DetailAddButton onAdd={onAdd} />
        <p className="ig-detail-note">信号端口之间用橙色虚线连接，传递布尔值控制节点是否执行。</p>
      </>
    );
  }

  const m = item.item;
  const displayName = shortModelName(m.modelKey);
  return (
    <>
      <div className="ig-detail-head">
        <span className="ig-detail-title">{displayName}</span>
      </div>
      {m.modelKey !== displayName && (
        <div className="ig-detail-sub" title="完整模型标识">
          {m.modelKey}
        </div>
      )}
      {m.labels.length > 0 && (
        <div className="ig-detail-tags">
          {m.labels.map((l) => <span key={l} className="ig-tag">{l}</span>)}
        </div>
      )}
      <DetailAddButton onAdd={onAdd} />
      {m.io ? (
        <>
          <DetailPorts title="输入端口" ports={m.io.inputs} />
          <DetailPorts title="输出端口" ports={m.io.outputs} />
        </>
      ) : (
        <p className="ig-detail-note">端口信息未声明。添加到画布后可在右侧属性面板查看和编辑。</p>
      )}
    </>
  );
}

/**
 * 行探索器（浮层交互体）：条目列表 + 锚定详情卡 + 添加分派（经唯一写通道）。
 */
export function RowExplorer({ items, ctx, hint }: { items: readonly PaletteItem[]; ctx: SourceCtx; hint?: string | null }) {
  const [detailKey, setDetailKey] = useState<string | null>(null);
  const [detailTop, setDetailTop] = useState(0);

  const detailItem = useMemo(
    () => (detailKey !== null ? (items.find((it) => paletteItemKey(it) === detailKey) ?? null) : null),
    [items, detailKey],
  );

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
          <button type="button" className="ig-iconbtn ig-detail-close" title="关闭" onClick={() => setDetailKey(null)}>
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
