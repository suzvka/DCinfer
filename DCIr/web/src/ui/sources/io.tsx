/**
 * ioSource — 编排 / 控制流节点调色板源
 *
 * 面向用户的友好标签：IO 控件元素 + 信号逻辑节点。
 * 「编排节点」改名为「流程控制」；信号节点用简洁的任务语言描述。
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

// 信号节点用更直白的描述
const SIGNAL_LABEL_MAP: Record<string, string> = {
  source: "信号触发器",
  const: "常量信号",
  not: "取反",
  and: "全部通过（AND）",
  or: "任一通过（OR）",
};

const SIGNAL_DESC_MAP: Record<string, string> = {
  source: "由外部代码触发的布尔开关，控制节点是否执行",
  const: "始终输出固定的 true 或 false",
  not: "将信号取反（true→false，false→true）",
  and: "所有输入都为 true 时才输出 true",
  or: "任意一个输入为 true 时输出 true",
};

const SIGNAL_ITEMS: PaletteItem[] = SIGNAL_PALETTE.map((spec) => ({
  kind: "signal",
  logic: spec.logic,
  label: SIGNAL_LABEL_MAP[spec.logic] ?? spec.label,
  desc: SIGNAL_DESC_MAP[spec.logic] ?? spec.desc,
}));

export const ioSource: PaletteSource = {
  id: "orchestration",
  label: "流程控制",
  icon: "⇄",
  program: ({ ctx }) => <RowExplorer items={[...IO_ITEMS, ...SIGNAL_ITEMS]} ctx={ctx} hint={null} />,
};
