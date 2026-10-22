/**
 * i18n — 用户友好的文案映射
 *
 * 将 core 层的工程化 issue code / detail 转为面向用户的自然语言，
 * 隐藏内部术语（modelKey / affinity / typeSize / sig.gate 等）。
 * core 层 detail 仍保留（调试 / JSON 抽屉用），UI 渲染时经此模块转译。
 *
 * 纪律：本模块只做文案映射，不触碰值模型；core 层零改动。
 */

import { SIGNAL_NODE_TYPES } from "../core/model";
import type { DcirPort, EditorIssue } from "../core/types";

// ── 问题文案 ──

/** 从 detail 中提取 node.port 形式的端口引用（容错：匹配失败返回 null） */
function extractNodePort(detail: string): { node: string; port: string } | null {
  const m = detail.match(/([A-Za-z0-9_.-]+)\.([A-Za-z0-9_.-]+)/);
  if (!m) return null;
  return { node: m[1], port: m[2] };
}

/**
 * 用户友好的问题文案（按 code 分派；未覆盖的 code 回退原始 detail）。
 * 导入清洗类（import-*）保留原始 detail——已含具体条目信息且仅在 JSON 抽屉展示。
 */
export function friendlyIssueDetail(issue: EditorIssue): string {
  const { code, detail, node } = issue;
  switch (code) {
    // 节点命名
    case "name-pattern":
      return `节点名含不允许的字符（只能用字母、数字、下划线、点和短横线）`;
    case "node-type-missing":
      return `节点缺少引擎类型，请从左侧模型库重新选择`;
    case "port-name-pattern":
      return `端口名含不允许的字符`;
    case "duplicate-port":
      return `同一节点存在同名端口`;

    // 连线
    case "dangling-edge":
      return `有一条连线指向已删除的节点或端口`;
    case "self-loop":
      return `不支持节点连接到自己`;
    case "duplicate-edge":
      return `两个端口之间已存在连线`;
    case "input-multi-source": {
      const np = extractNodePort(detail);
      return np
        ? `「${np.node}」的「${np.port}」只能接一条线，请先断开原有连线`
        : `输入端口只能接一条线，请先断开原有连线`;
    }
    case "edge-type-mismatch":
      return `连线两端的数据形状或类型不匹配`;

    // 绑定
    case "dangling-binding":
      return `图输入/输出绑定指向的端口已不存在`;
    case "binding-edge-conflict": {
      const np = extractNodePort(detail);
      return np
        ? `「${np.node}」的「${np.port}」已有连线，不能再设为图输入`
        : `该端口已有连线，不能再设为图输入`;
    }

    // IO 元素
    case "io-unbound":
      return `输入/输出元素尚未连接到任何节点`;
    case "io-dangling":
      return `输入/输出元素的连接目标已不存在`;

    // 远程意图
    case "remote-orphan":
      return node
        ? `节点「${node}」的模型标识已失效（节点可能已被删除）`
        : `模型标识指向的节点已不存在`;

    // 必需输入
    case "required-unconnected": {
      const np = extractNodePort(detail);
      return np
        ? `「${np.node}」的「${np.port}」需要连接一个数据来源`
        : `有必需的输入端口尚未连接`;
    }

    // 信号
    case "signal-source-name":
      return node
        ? `信号源「${node}」需要设置一个对外信号名`
        : `信号源需要设置一个对外信号名`;
    case "signal-const-value":
      return node
        ? `常量信号「${node}」需要设置一个布尔值`
        : `常量信号需要设置一个布尔值`;

    // 导入清洗类（保留原始 detail）
    default:
      return detail;
  }
}

// ── 节点类型友好名 ──

const SIGNAL_TYPE_LABELS: Record<string, string> = {
  [SIGNAL_NODE_TYPES.source]: "信号源",
  [SIGNAL_NODE_TYPES.const]: "常量信号",
  [SIGNAL_NODE_TYPES.not]: "信号非",
  [SIGNAL_NODE_TYPES.and]: "信号与",
  [SIGNAL_NODE_TYPES.or]: "信号或",
};

/** 节点 type → 用户可读标签（信号节点映射友好名；算子/模型保留引擎名） */
export function friendlyNodeType(type: string): string {
  if (SIGNAL_TYPE_LABELS[type]) return SIGNAL_TYPE_LABELS[type];
  return type || "未设置";
}

// ── 端口类型友好展示 ──

/** 端口数据类型简写（f32 / i32 等；信号端子 = "信号"） */
export function friendlyPortType(p: DcirPort): string {
  if (p.kind === "signal") return "信号";
  return p.tensorType === "Float" && p.typeSize === 4
    ? "f32"
    : `${p.tensorType ?? "?"}:${p.typeSize ?? "?"}`;
}

/** 端口形状友好展示（空 = "标量"；信号 = ""） */
export function friendlyPortShape(p: DcirPort): string {
  if (p.kind === "signal") return "";
  const shape = p.shape ?? [];
  if (shape.length === 0) return "标量";
  return `[${shape.join(", ")}]`;
}

/** 端口完整 tooltip（hover 时展示全部技术细节） */
export function portTooltip(p: DcirPort, extra = ""): string {
  if (p.kind === "signal") return `信号端口（布尔）${extra}`;
  const shape = (p.shape ?? []).join(", ");
  return `${p.name}：${p.tensorType ?? "?"}（${p.typeSize ?? "?"} 字节）形状 [${shape}]${p.required !== false ? " · 必需" : ""}${extra}`;
}

// ── 节点显示名 ──

/** 从 modelKey 提取末段作为用户可读短名（models/mnist.onnx → mnist.onnx） */
export function shortModelName(modelKey: string): string {
  if (!modelKey) return "";
  const seg = modelKey.includes("/") ? modelKey.split("/").pop()! : modelKey;
  return seg;
}
