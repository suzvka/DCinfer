/**
 * ioSource — 编排节点源（IO 控件元素的调色板程序）
 *
 * 与 browse 同为一元原语的取样点：静态两行 + 详情卡，写通道 = ctx.placeIo。
 * 编辑器内置恒在（系统组），宿主可整体替换（自写一个同 id 程序覆盖语义）。
 */

import { IO_PALETTE, SIGNAL_PALETTE } from "../../core/catalog";
import type { PaletteItem, PaletteSource } from "../palette-protocol";
import { RowExplorer } from "./parts";

const IO_ITEMS: PaletteItem[] = IO_PALETTE.map((spec) => ({
  kind: "io",
  ioKind: spec.kind,
  label: spec.label,
  desc: spec.desc,
}));

const SIGNAL_ITEMS: PaletteItem[] = SIGNAL_PALETTE.map((spec) => ({
  kind: "signal",
  logic: spec.logic,
  label: spec.label,
  desc: spec.desc,
}));

export const ioSource: PaletteSource = {
  id: "orchestration",
  label: "编排节点",
  icon: "⇄",
  program: ({ ctx }) => <RowExplorer items={[...IO_ITEMS, ...SIGNAL_ITEMS]} ctx={ctx} hint={null} />,
};
