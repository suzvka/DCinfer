/**
 * 推理图编辑器 — 交互规则与校验（纯函数）
 *
 * 规则收口：
 * - 连接单向（out → in）；输入口单源 = 替换式（允许连入，addEdge 负责替换旧边）；
 * - 重复四元组 = 拒绝；自环（src === dst）= 拒绝；环 / 反馈回路 = 放行（不做 DAG 限制）；
 * - IO 元素：输入元素仅可作源（→ 节点输入口）、输出元素仅可作目标（节点输出口 →），
 *   元素↔元素直连拒绝；元素绑定不检查类型（外部数据形状由对接端子决定）；
 * - 类型兼容 = tensorType + typeSize 相等 且 shape 兼容（-1 动态维兼容，与 DCIr
 *   声明层口径一致）；
 * - 名字白名单 = warn 级（不阻断提交，DCIr 只要求非空）；error 级才阻断提交。
 */

import { SIGNAL_NODE_TYPES, edgeIdOf, ioElementOf, parseHandle } from "./model";
import type { DcirGraphJson, DcirNode, DcirPort, EditorIssue, EditorValue, IssueLevel } from "./types";

/** shape 兼容：秩相等且逐维相等或含 -1 动态维（声明层口径：-1 仅存在于声明层） */
export function shapeCompatible(graphShape: readonly number[], declaredShape: readonly number[]): boolean {
  if (graphShape.length !== declaredShape.length) return false;
  return graphShape.every((g, i) => {
    const d = declaredShape[i];
    return g === -1 || d === -1 || g === d;
  });
}

/** 名字白名单（对象键安全子集：node/port 名会进入总线对象键段） */
export const NAME_PATTERN = /^[A-Za-z0-9_.-]+$/;

export interface ConnectionInput {
  source: string | null;
  sourceHandle: string | null;
  target: string | null;
  targetHandle: string | null;
}

export interface ConnectionCheck {
  ok: boolean;
  reason?: string;
}

function portOf(ports: readonly DcirPort[], name: string): DcirPort | undefined {
  return ports.find((p) => p.name === name);
}

function nodeOf(graph: DcirGraphJson, name: string): DcirNode | undefined {
  return graph.nodes.find((n) => n.name === name);
}

/** 端子对兼容：先判 kind（信号/数据不可互连）；signal↔signal 布尔恒兼容；tensor↔tensor 走形状校验 */
export function portsCompatible(out: DcirPort, input: DcirPort): string | null {
  const outKind = out.kind ?? "tensor";
  const inKind = input.kind ?? "tensor";
  if (outKind !== inKind) return "信号端子与数据端子不可互连";
  if (outKind === "signal") return null;
  if (out.tensorType !== input.tensorType) return `类型不匹配（${out.tensorType} → ${input.tensorType}）`;
  if (out.typeSize !== input.typeSize) return `typeSize 不匹配（${out.typeSize} → ${input.typeSize}）`;
  if (!shapeCompatible(out.shape ?? [], input.shape ?? [])) {
    return `shape 不兼容（[${(out.shape ?? []).join(",")}] → [${(input.shape ?? []).join(",")}]）`;
  }
  return null;
}

/** 连接校验（渲染层 isValidConnection 钩子面） */
export function checkConnection(value: EditorValue, conn: ConnectionInput): ConnectionCheck {
  const { source, target, sourceHandle, targetHandle } = conn;
  if (!source || !target) return { ok: false, reason: "连接缺少源/目标" };
  if (source === target) return { ok: false, reason: "不支持自环（同一节点自连）" };
  const sh = parseHandle(sourceHandle);
  const th = parseHandle(targetHandle);
  if (!sh) return { ok: false, reason: "源接线点缺失" };
  if (!th) return { ok: false, reason: "目标接线点缺失" };

  // IO 元素分支（单接线点；不检查类型——外部数据形状由对接端子决定）
  const srcEl = ioElementOf(value, source);
  const dstEl = ioElementOf(value, target);
  if (srcEl) {
    if (srcEl.kind !== "input") return { ok: false, reason: "输出元素不可作为连线起点" };
    if (sh.direction !== "out") return { ok: false, reason: "输入元素仅右侧接线点" };
    if (dstEl) return { ok: false, reason: "元素之间不支持直接连线" };
    if (th.direction !== "in") return { ok: false, reason: "目标必须是输入端子" };
    const dstNode = nodeOf(value.graph, target);
    if (!dstNode) return { ok: false, reason: "节点不存在" };
    if (!portOf(dstNode.inputs, th.port)) return { ok: false, reason: `输入端子 '${th.port}' 不存在` };
    return { ok: true };
  }
  if (dstEl) {
    if (dstEl.kind !== "output") return { ok: false, reason: "输入元素不可作为连线终点" };
    if (th.direction !== "in") return { ok: false, reason: "输出元素仅左侧接线点" };
    if (sh.direction !== "out") return { ok: false, reason: "源必须是输出端子" };
    const srcNode = nodeOf(value.graph, source);
    if (!srcNode) return { ok: false, reason: "节点不存在" };
    if (!portOf(srcNode.outputs, sh.port)) return { ok: false, reason: `输出端子 '${sh.port}' 不存在` };
    return { ok: true };
  }

  // 节点 ↔ 节点
  if (sh.direction !== "out") return { ok: false, reason: "源必须是输出端子" };
  if (th.direction !== "in") return { ok: false, reason: "目标必须是输入端子" };
  const srcNode = nodeOf(value.graph, source);
  const dstNode = nodeOf(value.graph, target);
  if (!srcNode || !dstNode) return { ok: false, reason: "节点不存在" };
  const outPort = portOf(srcNode.outputs, sh.port);
  const inPort = portOf(dstNode.inputs, th.port);
  if (!outPort) return { ok: false, reason: `输出端子 '${sh.port}' 不存在` };
  if (!inPort) return { ok: false, reason: `输入端子 '${th.port}' 不存在` };
  if (
    value.graph.edges.some(
      (e) => e.srcNode === source && e.srcPort === sh.port && e.dstNode === target && e.dstPort === th.port,
    )
  ) {
    return { ok: false, reason: "重复边" };
  }
  const reason = portsCompatible(outPort, inPort);
  if (reason) return { ok: false, reason };
  return { ok: true };
}

// ── 结构校验（提交门禁 / 节点标记）──

/** 环检测（Tarjan SCC，迭代版；自环天然表达）→ 参与环的节点集合 */
export function findCycleNodes(graph: DcirGraphJson): Set<string> {
  const adj = new Map<string, string[]>();
  for (const n of graph.nodes) adj.set(n.name, []);
  const selfLoops = new Set<string>();
  for (const e of graph.edges) {
    if (e.srcNode === e.dstNode) selfLoops.add(e.srcNode);
    const list = adj.get(e.srcNode);
    if (list && adj.has(e.dstNode) && !list.includes(e.dstNode)) list.push(e.dstNode);
  }

  const index = new Map<string, number>();
  const low = new Map<string, number>();
  const onStack = new Set<string>();
  const stack: string[] = [];
  const cycles = new Set<string>(selfLoops);
  let counter = 0;

  for (const root of graph.nodes.map((n) => n.name)) {
    if (index.has(root)) continue;
    const work: { node: string; childIdx: number }[] = [{ node: root, childIdx: 0 }];
    while (work.length > 0) {
      const frame = work[work.length - 1];
      const v = frame.node;
      if (frame.childIdx === 0) {
        index.set(v, counter);
        low.set(v, counter);
        counter += 1;
        stack.push(v);
        onStack.add(v);
      }
      const children = adj.get(v) ?? [];
      let recursed = false;
      while (frame.childIdx < children.length) {
        const w = children[frame.childIdx];
        frame.childIdx += 1;
        if (!index.has(w)) {
          work.push({ node: w, childIdx: 0 });
          recursed = true;
          break;
        }
        if (onStack.has(w)) low.set(v, Math.min(low.get(v) as number, index.get(w) as number));
      }
      if (recursed) continue;
      if (low.get(v) === index.get(v)) {
        const component: string[] = [];
        for (;;) {
          const w = stack.pop() as string;
          onStack.delete(w);
          component.push(w);
          if (w === v) break;
        }
        if (component.length > 1) for (const w of component) cycles.add(w);
      }
      work.pop();
      if (work.length > 0) {
        const parent = work[work.length - 1].node;
        low.set(parent, Math.min(low.get(parent) as number, low.get(v) as number));
      }
    }
  }
  return cycles;
}

/** 全量校验：error 阻断提交（host 门禁）；warn 仅提示 */
export function validateEditorValue(value: EditorValue): EditorIssue[] {
  const issues: EditorIssue[] = [];
  const { graph } = value;
  const nodeByName = new Map(graph.nodes.map((n) => [n.name, n]));

  // 节点：命名 / 重名 / 端子重名 / 缺 type
  for (const n of graph.nodes) {
    if (!NAME_PATTERN.test(n.name)) {
      issues.push({ level: "error", code: "name-pattern", node: n.name, detail: `节点名 '${n.name}' 含非法字符（仅允许 A-Za-z0-9_.-）` });
    }
    if (!n.type) {
      issues.push({ level: "error", code: "node-type-missing", node: n.name, detail: `节点 '${n.name}' 缺少引擎类型` });
    }
    for (const [direction, ports] of [
      ["inputs", n.inputs],
      ["outputs", n.outputs],
    ] as const) {
      const seen = new Set<string>();
      for (const p of ports) {
        if (!NAME_PATTERN.test(p.name)) {
          issues.push({ level: "error", code: "port-name-pattern", node: n.name, detail: `端子 '${n.name}.${p.name}' 含非法字符（仅允许 A-Za-z0-9_.-）` });
        }
        if (seen.has(p.name)) {
          issues.push({ level: "error", code: "duplicate-port", node: n.name, detail: `节点 '${n.name}' ${direction} 端子 '${p.name}' 重复` });
        }
        seen.add(p.name);
      }
    }
  }

  // 边：悬垂 / 自环 / 重复 / 单源 / 类型兼容（warn）
  const edgeSeen = new Set<string>();
  const dstOccupied = new Set<string>();
  for (const e of graph.edges) {
    const src = nodeByName.get(e.srcNode);
    const dst = nodeByName.get(e.dstNode);
    if (!src || !dst) {
      issues.push({ level: "error", code: "dangling-edge", node: e.srcNode, detail: `边 ${edgeIdOf(e)} 指向不存在的节点` });
      continue;
    }
    const outPort = portOf(src.outputs, e.srcPort);
    const inPort = portOf(dst.inputs, e.dstPort);
    if (!outPort || !inPort) {
      issues.push({ level: "error", code: "dangling-edge", node: e.dstNode, detail: `边 ${edgeIdOf(e)} 指向不存在的端子` });
      continue;
    }
    if (e.srcNode === e.dstNode) {
      issues.push({ level: "error", code: "self-loop", node: e.srcNode, detail: `自环边 ${edgeIdOf(e)} 不支持` });
    }
    const id = edgeIdOf(e);
    if (edgeSeen.has(id)) {
      issues.push({ level: "error", code: "duplicate-edge", node: e.dstNode, detail: `重复边 ${id}` });
    }
    edgeSeen.add(id);
    const dstKey = `${e.dstNode}\u0000${e.dstPort}`;
    if (dstOccupied.has(dstKey)) {
      issues.push({ level: "error", code: "input-multi-source", node: e.dstNode, detail: `输入口 ${e.dstNode}.${e.dstPort} 被多条边驱动（单源约束）` });
    }
    dstOccupied.add(dstKey);
    const mismatch = portsCompatible(outPort, inPort);
    if (mismatch) {
      issues.push({ level: "warn", code: "edge-type-mismatch", node: e.dstNode, detail: `边 ${id}：${mismatch}` });
    }
  }

  // 绑定：悬垂 / 与入边互斥
  for (const b of graph.inputBindings ?? []) {
    const node = nodeByName.get(b.nodeName);
    if (!node || !node.inputs.some((p) => p.name === b.portName)) {
      issues.push({ level: "error", code: "dangling-binding", node: b.nodeName, detail: `图输入绑定 ${b.nodeName}.${b.portName} 指向不存在的输入端子` });
      continue;
    }
    if (dstOccupied.has(`${b.nodeName}\u0000${b.portName}`)) {
      issues.push({ level: "error", code: "binding-edge-conflict", node: b.nodeName, detail: `输入口 ${b.nodeName}.${b.portName} 同时被入边驱动与声明为图输入（互斥）` });
    }
  }
  for (const b of graph.outputBindings ?? []) {
    const node = nodeByName.get(b.nodeName);
    if (!node || !node.outputs.some((p) => p.name === b.portName)) {
      issues.push({ level: "error", code: "dangling-binding", node: b.nodeName, detail: `图输出绑定 ${b.nodeName}.${b.portName} 指向不存在的输出端子` });
    }
  }

  // IO 元素：悬垂绑定（防御式；绑定经操作层同步，正常不悬垂）/ 未连接提示
  for (const el of value.io) {
    if (el.node === undefined || el.port === undefined) {
      issues.push({ level: "warn", code: "io-unbound", detail: `IO 元素 '${el.id}'（${el.kind === "input" ? "输入" : "输出"}）未连接（不参与提交）` });
      continue;
    }
    const node = nodeByName.get(el.node);
    const ports = node ? (el.kind === "input" ? node.inputs : node.outputs) : null;
    if (!node || !ports || !ports.some((p) => p.name === el.port)) {
      issues.push({ level: "error", code: "io-dangling", detail: `IO 元素 '${el.id}' 绑定 ${el.node}.${el.port} 指向不存在的端子` });
    }
  }

  // 远程意图：孤儿键
  for (const [name, key] of Object.entries(value.remote)) {
    if (!key) continue;
    if (!nodeByName.has(name)) {
      issues.push({ level: "warn", code: "remote-orphan", node: name, detail: `远程意图指向不存在的节点 '${name}'（提交时已忽略）` });
    }
  }

  // required 输入未供数（边或绑定）
  const boundIn = new Set((graph.inputBindings ?? []).map((b) => `${b.nodeName}\u0000${b.portName}`));
  for (const n of graph.nodes) {
    for (const p of n.inputs) {
      if (p.required === false) continue;
      const key = `${n.name}\u0000${p.name}`;
      if (!dstOccupied.has(key) && !boundIn.has(key)) {
        issues.push({ level: "warn", code: "required-unconnected", node: n.name, detail: `输入 ${n.name}.${p.name}（required）无入边且未声明为图输入` });
      }
    }
  }

  // 信号编排逻辑节点：参数完整性（error 阻断提交）
  for (const n of graph.nodes) {
    if (n.type === SIGNAL_NODE_TYPES.source) {
      const sn = typeof n.signalName === "string" ? n.signalName.trim() : "";
      if (!sn) {
        issues.push({ level: "error", code: "signal-source-name", node: n.name, detail: `信号源 '${n.name}' 缺少对外信号名（宿主 setSignal 入口）` });
      } else if (!NAME_PATTERN.test(sn)) {
        issues.push({ level: "error", code: "signal-source-name", node: n.name, detail: `信号源 '${n.name}' 对外名 '${sn}' 含非法字符（仅允许 A-Za-z0-9_.-）` });
      }
    } else if (n.type === SIGNAL_NODE_TYPES.const) {
      if (typeof n.signalValue !== "boolean") {
        issues.push({ level: "error", code: "signal-const-value", node: n.name, detail: `常量信号 '${n.name}' 缺少布尔值` });
      }
    }
  }

  // 环（放行；不做 DAG 限制）
  // 注：原 cycle-remote-group 警告已删除——所有算力均为远程，环内分组约束无意义

  return issues;
}

export function hasBlockingIssues(issues: readonly EditorIssue[]): boolean {
  return issues.some((i) => i.level === "error");
}

/** 供 UI 标记：节点 → 最高问题级别 */
export function issueLevelByNode(issues: readonly EditorIssue[]): Map<string, IssueLevel> {
  const map = new Map<string, IssueLevel>();
  for (const i of issues) {
    if (!i.node) continue;
    const prev = map.get(i.node);
    if (i.level === "error" || prev === undefined) map.set(i.node, i.level);
  }
  return map;
}
