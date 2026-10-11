/**
 * 推理图编辑器 — 领域层（纯函数，无 React / 渲染框架运行时依赖）
 *
 * DCIr JSON 为唯一权威；渲染框架仅为投影（toFlowModel 产出框架无关的结构类型，
 * ui 层负责桥接到 React Flow）。所有变更经本层纯函数产出新的 EditorValue
 * （值语义，未知字段原样保留 → 无损往返）。
 *
 * 交互规则收口（与 DCinfer 构图期检查同口径）：
 * - 输入口单源：新边落向已占用输入口 = 替换旧边（DCinfer 构图期 DuplicateEdge 的前置收口）；
 * - 重复四元组边 = 无操作；自环（src === dst）= 连接层拒绝；
 * - 图输入绑定 ↔ 入边互斥（互相清理，二者语义均为「外部供数一个端口」）；
 * - IO 控件元素（io side-band）：输入/输出绑定的可视化载体；绑定经 bindIoElement
 *   单项落图，syncBindings 从元素表重算 bindings（同 key 原对象复用保 unknown）。
 */

import type {
  DcirBinding,
  DcirEdge,
  DcirGraphJson,
  DcirNode,
  DcirPort,
  EditorIssue,
  EditorValue,
  IoElement,
  IssueLevel,
  SignalLogicKind,
  XY,
} from "./types";

// ── 句柄命名（handle id = in:<port> / out:<port>；防止输入/输出端口同名歧义）──

export const HANDLE_IN_PREFIX = "in:";
export const HANDLE_OUT_PREFIX = "out:";

export const inHandle = (port: string): string => `${HANDLE_IN_PREFIX}${port}`;
export const outHandle = (port: string): string => `${HANDLE_OUT_PREFIX}${port}`;

/** IO 元素固定 handle id（单接线点：输入元素仅右侧出点 / 输出元素仅左侧入点） */
export const IO_HANDLE_IN = "in:io";
export const IO_HANDLE_OUT = "out:io";

/** 解析 handle id；非法（缺前缀）返回 null */
export function parseHandle(handleId: string | null | undefined): { direction: "in" | "out"; port: string } | null {
  if (typeof handleId !== "string") return null;
  if (handleId.startsWith(HANDLE_IN_PREFIX)) {
    return { direction: "in", port: handleId.slice(HANDLE_IN_PREFIX.length) };
  }
  if (handleId.startsWith(HANDLE_OUT_PREFIX)) {
    return { direction: "out", port: handleId.slice(HANDLE_OUT_PREFIX.length) };
  }
  return null;
}

// ── 信号端子 / 信号编排节点（控制面：布尔连线，独立于张量数据流）──

/** 信号端子名保留前缀（与 tensor 端口名不相交，避免 React Flow handle id 冲突） */
export const SIG_PREFIX = "sig.";
/** 计算节点信号输入口（门控）：无入边 = 放行（fail-open），有入边且上游 false = 阻断 */
export const SIG_GATE = "sig.gate";
/** 计算节点状态输出端子（运行期由节点背后算法侧赋值） */
export const SIG_STATE_OUTPUTS: readonly string[] = ["sig.idle", "sig.running", "sig.blocked", "sig.error", "sig.done"];
/** 逻辑节点输出端子 */
export const SIG_OUT = "sig.out";
/** 逻辑节点输入端子前缀（sig.in0 / sig.in1 …） */
export const SIG_IN_PREFIX = "sig.in";
/** AND/OR 入度上限（防无限增输入） */
export const SIG_LOGIC_MAX_INPUTS = 8;

/** 信号逻辑节点 type 常量（保留名） */
export const SIGNAL_NODE_TYPES: Record<SignalLogicKind, string> = {
  source: "dc:SignalSource",
  const: "dc:SignalConst",
  not: "dc:SignalNot",
  and: "dc:SignalAnd",
  or: "dc:SignalOr",
};

/** 信号逻辑节点缺省名基（uniqueNodeName 用；避开 type 里 ':' 非法字符） */
const SIGNAL_NODE_NAME_BASE: Record<SignalLogicKind, string> = {
  source: "signalSource",
  const: "signalConst",
  not: "signalNot",
  and: "signalAnd",
  or: "signalOr",
};

/** 是否信号端口 */
export function isSignalPort(p: DcirPort): boolean {
  return p.kind === "signal";
}

/** 是否信号编排逻辑节点（按 type 保留名判定） */
export function isSignalNodeType(type: string): boolean {
  return (Object.values(SIGNAL_NODE_TYPES) as string[]).includes(type);
}

/** 信号端口工厂（布尔端子：无张量形状，非 required） */
export function sigPort(name: string): DcirPort {
  return { name, kind: "signal", required: false };
}

/** 去掉信号端子名的保留前缀（渲染/展示用） */
export function sigLabel(portName: string): string {
  return portName.startsWith(SIG_PREFIX) ? portName.slice(SIG_PREFIX.length) : portName;
}

/**
 * 幂等注入标准信号端子：计算节点补 sig.gate 输入 + 5 个状态输出；
 * 信号逻辑节点自带端子，原样返回。已存在同名端子保留（不覆盖 unknown 字段）。
 */
export function withSignalTerminals(node: DcirNode): DcirNode {
  if (isSignalNodeType(node.type)) return node;
  const inputs = node.inputs.some((p) => p.name === SIG_GATE) ? node.inputs : [...node.inputs, sigPort(SIG_GATE)];
  const missing = SIG_STATE_OUTPUTS.filter((s) => !node.outputs.some((p) => p.name === s));
  const outputs = missing.length > 0 ? [...node.outputs, ...missing.map((s) => sigPort(s))] : node.outputs;
  if (inputs === node.inputs && outputs === node.outputs) return node;
  return { ...node, inputs, outputs };
}

// ── 边 ID 确定性规则（用于 React Flow edge.id；集合比对 / 撤销栈友好）──

export function edgeIdOf(e: Pick<DcirEdge, "srcNode" | "srcPort" | "dstNode" | "dstPort">): string {
  return `${e.srcNode}.${e.srcPort}→${e.dstNode}.${e.dstPort}`;
}

/** IO 绑定投影边的 id（前缀隔离：不与图内边四元组 id 混淆） */
export const ioEdgeId = (elementId: string): string => `io:${elementId}`;
export const isIoEdgeId = (edgeId: string): boolean => edgeId.startsWith("io:");
export const ioElementIdOfEdge = (edgeId: string): string => edgeId.slice(3);

// ── 值语义克隆（JSON 域数据；unknown 字段原样保留）──

export function cloneGraph(graph: DcirGraphJson): DcirGraphJson {
  return JSON.parse(JSON.stringify(graph)) as DcirGraphJson;
}

export function emptyGraph(): DcirGraphJson {
  return { version: "1.0", nodes: [], edges: [], inputBindings: [], outputBindings: [] };
}

function withGraph(value: EditorValue, graph: DcirGraphJson): EditorValue {
  return { ...value, graph };
}

// ── 防御式解析（JSON 导入 / 模板载入）：结构窄化 + 悬垂引用清理 + 问题上报 ──

function sanitizePorts(raw: unknown, nodeName: string, issues: EditorIssue[]): DcirPort[] {
  if (!Array.isArray(raw)) return [];
  const ports: DcirPort[] = [];
  const seen = new Set<string>();
  for (const pRaw of raw) {
    if (typeof pRaw !== "object" || pRaw === null) continue;
    const pObj = pRaw as Record<string, unknown>;
    const name = typeof pObj.name === "string" ? pObj.name : "";
    if (!name) {
      issues.push({ level: "warn", code: "import-port", node: nodeName, detail: `节点 '${nodeName}' 存在缺少 name 的端口，已丢弃` });
      continue;
    }
    if (seen.has(name)) {
      issues.push({ level: "warn", code: "import-port", node: nodeName, detail: `节点 '${nodeName}' 端口 '${name}' 重复，保留首个` });
      continue;
    }
    seen.add(name);
    if (pObj.kind === "signal") {
      const sig: DcirPort = {
        ...pObj,
        name,
        kind: "signal",
        required: typeof pObj.required === "boolean" ? pObj.required : false,
      };
      delete sig.tensorType;
      delete sig.typeSize;
      delete sig.shape;
      ports.push(sig);
      continue;
    }
    ports.push({
      ...pObj,
      name,
      tensorType: typeof pObj.tensorType === "string" ? pObj.tensorType : "Float",
      typeSize: typeof pObj.typeSize === "number" ? pObj.typeSize : 4,
      shape: Array.isArray(pObj.shape) ? (pObj.shape.filter((n) => typeof n === "number") as number[]) : [],
      required: typeof pObj.required === "boolean" ? pObj.required : true,
    });
  }
  return ports;
}

export interface SanitizeResult {
  graph: DcirGraphJson;
  issues: EditorIssue[];
}

/** 防御式读取 opaque JSON → DcirGraphJson；被丢弃的条目逐项上报 */
export function sanitizeGraph(raw: unknown): SanitizeResult {
  const issues: EditorIssue[] = [];
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    issues.push({ level: "error", code: "import-shape", detail: "图谱必须是 JSON 对象" });
    return { graph: emptyGraph(), issues };
  }
  const rawObj = raw as Record<string, unknown>;
  const graph = { ...rawObj } as DcirGraphJson;
  if (typeof rawObj.version !== "string" || rawObj.version.length === 0) {
    graph.version = "1.0";
    issues.push({ level: "warn", code: "import-version", detail: "缺少 version，已补默认 1.0" });
  }

  // 节点
  const nodes: DcirNode[] = [];
  const names = new Set<string>();
  if (!Array.isArray(rawObj.nodes)) {
    issues.push({ level: "warn", code: "import-nodes", detail: "nodes 缺失或非数组，已按空处理" });
  }
  for (const nRaw of Array.isArray(rawObj.nodes) ? rawObj.nodes : []) {
    if (typeof nRaw !== "object" || nRaw === null) continue;
    const nObj = nRaw as Record<string, unknown>;
    const name = typeof nObj.name === "string" ? nObj.name : "";
    if (!name) {
      issues.push({ level: "warn", code: "import-node", detail: "存在缺少 name 的节点，已丢弃" });
      continue;
    }
    if (names.has(name)) {
      issues.push({ level: "warn", code: "import-node", node: name, detail: `节点名 '${name}' 重复，保留首个` });
      continue;
    }
    names.add(name);
    const node = {
      ...nObj,
      name,
      type: typeof nObj.type === "string" ? nObj.type : "",
      inputs: sanitizePorts(nObj.inputs, name, issues),
      outputs: sanitizePorts(nObj.outputs, name, issues),
    } as DcirNode;
    if (typeof nObj.type !== "string" || nObj.type.length === 0) {
      issues.push({ level: "warn", code: "import-node-type", node: name, detail: `节点 '${name}' 缺少 type（引擎名），请在检查器补齐` });
    }
    nodes.push(withSignalTerminals(node));
  }
  graph.nodes = nodes;
  const nodeByName = new Map(nodes.map((n) => [n.name, n]));

  // 边
  const edges: DcirEdge[] = [];
  const edgeSeen = new Set<string>();
  const dstOccupied = new Set<string>();
  if (!Array.isArray(rawObj.edges)) {
    issues.push({ level: "warn", code: "import-edges", detail: "edges 缺失或非数组，已按空处理" });
  }
  for (const eRaw of Array.isArray(rawObj.edges) ? rawObj.edges : []) {
    if (typeof eRaw !== "object" || eRaw === null) continue;
    const eObj = eRaw as Record<string, unknown>;
    const { srcNode, srcPort, dstNode, dstPort } = eObj;
    if (
      typeof srcNode !== "string" || typeof srcPort !== "string" ||
      typeof dstNode !== "string" || typeof dstPort !== "string"
    ) {
      issues.push({ level: "warn", code: "import-edge", detail: "存在字段缺失的边，已丢弃" });
      continue;
    }
    const src = nodeByName.get(srcNode);
    const dst = nodeByName.get(dstNode);
    if (!src || !dst) {
      issues.push({ level: "error", code: "import-dangling-edge", node: srcNode, detail: `边 ${srcNode}.${srcPort}→${dstNode}.${dstPort} 指向不存在的节点，已丢弃` });
      continue;
    }
    if (!src.outputs.some((p) => p.name === srcPort) || !dst.inputs.some((p) => p.name === dstPort)) {
      issues.push({ level: "error", code: "import-dangling-edge", node: dstNode, detail: `边 ${srcNode}.${srcPort}→${dstNode}.${dstPort} 指向不存在的端口，已丢弃` });
      continue;
    }
    if (srcNode === dstNode) {
      issues.push({ level: "error", code: "import-self-loop", node: srcNode, detail: `自环边 ${srcNode}.${srcPort}→${dstNode}.${dstPort} 不支持，已丢弃` });
      continue;
    }
    const id = edgeIdOf({ srcNode, srcPort, dstNode, dstPort });
    if (edgeSeen.has(id)) {
      issues.push({ level: "warn", code: "import-duplicate-edge", node: dstNode, detail: `重复边 ${id}，保留首个` });
      continue;
    }
    const dstKey = `${dstNode}\u0000${dstPort}`;
    if (dstOccupied.has(dstKey)) {
      issues.push({ level: "warn", code: "import-input-multi-source", node: dstNode, detail: `输入口 ${dstNode}.${dstPort} 被多条边驱动（单源约束），保留首个` });
      continue;
    }
    edgeSeen.add(id);
    dstOccupied.add(dstKey);
    edges.push({ ...eObj, srcNode, srcPort, dstNode, dstPort });
  }
  graph.edges = edges;

  // 绑定（图输入与入边互斥：保留边、丢弃绑定）
  const inboundPorts = new Set(edges.map((e) => `${e.dstNode}\u0000${e.dstPort}`));
  const inputBindings: DcirBinding[] = [];
  for (const bRaw of Array.isArray(rawObj.inputBindings) ? rawObj.inputBindings : []) {
    if (typeof bRaw !== "object" || bRaw === null) continue;
    const bObj = bRaw as Record<string, unknown>;
    const nodeName = typeof bObj.nodeName === "string" ? bObj.nodeName : "";
    const portName = typeof bObj.portName === "string" ? bObj.portName : "";
    const node = nodeByName.get(nodeName);
    if (!node || !node.inputs.some((p) => p.name === portName)) {
      issues.push({ level: "error", code: "import-dangling-binding", node: nodeName, detail: `图输入绑定 ${nodeName}.${portName} 指向不存在的输入端口，已丢弃` });
      continue;
    }
    if (inboundPorts.has(`${nodeName}\u0000${portName}`)) {
      issues.push({ level: "warn", code: "import-binding-edge-conflict", node: nodeName, detail: `输入口 ${nodeName}.${portName} 已有入边，图输入绑定已丢弃（二者互斥）` });
      continue;
    }
    inputBindings.push({ ...bObj, nodeName, portName });
  }
  graph.inputBindings = inputBindings;

  const outputBindings: DcirBinding[] = [];
  for (const bRaw of Array.isArray(rawObj.outputBindings) ? rawObj.outputBindings : []) {
    if (typeof bRaw !== "object" || bRaw === null) continue;
    const bObj = bRaw as Record<string, unknown>;
    const nodeName = typeof bObj.nodeName === "string" ? bObj.nodeName : "";
    const portName = typeof bObj.portName === "string" ? bObj.portName : "";
    const node = nodeByName.get(nodeName);
    if (!node || !node.outputs.some((p) => p.name === portName)) {
      issues.push({ level: "error", code: "import-dangling-binding", node: nodeName, detail: `图输出绑定 ${nodeName}.${portName} 指向不存在的输出端口，已丢弃` });
      continue;
    }
    outputBindings.push({ ...bObj, nodeName, portName });
  }
  graph.outputBindings = outputBindings;

  return { graph, issues };
}

// ── 流投影（框架无关结构类型；ui 层桥接到 React Flow）──

/** 投影节点的最小形状（与 React Flow Node 结构兼容；字段为其子集） */
export interface FlowNode<TData> {
  id: string;
  type: string;
  position: XY;
  selected?: boolean;
  data: TData;
}

/** 投影边的最小形状（与 React Flow Edge 结构兼容） */
export interface FlowEdge {
  id: string;
  source: string;
  sourceHandle?: string;
  target: string;
  targetHandle?: string;
  className?: string;
  selected?: boolean;
}

/** 自定义节点的 data（渲染所需全部信息；连接性在投影期预计算，节点组件零状态） */
export interface InferNodeData extends Record<string, unknown> {
  name: string;
  type: string;
  affinity: string;
  inputs: DcirPort[];
  outputs: DcirPort[];
  remoteKey?: string;
  /** 已有供数的输入口（入边 ∪ 图输入绑定） */
  fedInputs: string[];
  boundInputs: string[];
  boundOutputs: string[];
  issueLevel?: IssueLevel;
}

export type InferFlowNode = FlowNode<InferNodeData>;
export type InferFlowEdge = FlowEdge;

/** IO 元素节点的 data（单接线点；已绑定时展示 node.port） */
export interface InferIoNodeData extends Record<string, unknown> {
  kind: "input" | "output";
  node?: string;
  port?: string;
}

export type InferIoFlowNode = FlowNode<InferIoNodeData>;

/** 画布节点并集（图节点 + IO 元素） */
export type AnyFlowNode = InferFlowNode | InferIoFlowNode;

export interface FlowModel {
  nodes: AnyFlowNode[];
  edges: InferFlowEdge[];
}

export function toFlowModel(
  value: EditorValue,
  opts: { selectedNames?: readonly string[]; issueLevelByNode?: ReadonlyMap<string, IssueLevel> } = {},
): FlowModel {
  const fed = new Map<string, Set<string>>();
  const touch = (name: string): Set<string> => {
    let s = fed.get(name);
    if (!s) {
      s = new Set<string>();
      fed.set(name, s);
    }
    return s;
  };
  for (const e of value.graph.edges) touch(e.dstNode).add(e.dstPort);
  const boundInputsByNode = new Map<string, string[]>();
  for (const b of value.graph.inputBindings ?? []) {
    touch(b.nodeName).add(b.portName);
    const list = boundInputsByNode.get(b.nodeName) ?? [];
    list.push(b.portName);
    boundInputsByNode.set(b.nodeName, list);
  }
  const boundOutputsByNode = new Map<string, string[]>();
  for (const b of value.graph.outputBindings ?? []) {
    const list = boundOutputsByNode.get(b.nodeName) ?? [];
    list.push(b.portName);
    boundOutputsByNode.set(b.nodeName, list);
  }

  const selectedSet = new Set(opts.selectedNames ?? []);
  const nodes: InferFlowNode[] = value.graph.nodes.map((n) => {
    const remoteKey = value.remote[n.name];
    const issueLevel = opts.issueLevelByNode?.get(n.name);
    return {
      id: n.name,
      type: "inferNode",
      position: value.layout[n.name] ?? { x: 0, y: 0 },
      selected: selectedSet.has(n.name),
      data: {
        name: n.name,
        type: n.type,
        affinity: n.affinity ?? "Operator",
        inputs: n.inputs,
        outputs: n.outputs,
        ...(remoteKey ? { remoteKey } : {}),
        fedInputs: [...(fed.get(n.name) ?? [])],
        boundInputs: boundInputsByNode.get(n.name) ?? [],
        boundOutputs: boundOutputsByNode.get(n.name) ?? [],
        ...(issueLevel ? { issueLevel } : {}),
      },
    };
  });

  const sigPorts = new Set<string>();
  for (const n of value.graph.nodes) {
    for (const p of n.inputs) if (isSignalPort(p)) sigPorts.add(`${n.name}\u0000${p.name}`);
    for (const p of n.outputs) if (isSignalPort(p)) sigPorts.add(`${n.name}\u0000${p.name}`);
  }
  const edges: InferFlowEdge[] = value.graph.edges.map((e) => {
    const isSig = sigPorts.has(`${e.srcNode}\u0000${e.srcPort}`) || sigPorts.has(`${e.dstNode}\u0000${e.dstPort}`);
    return {
      id: edgeIdOf(e),
      source: e.srcNode,
      sourceHandle: outHandle(e.srcPort),
      target: e.dstNode,
      targetHandle: inHandle(e.dstPort),
      ...(isSig ? { className: "ig-signal-edge" } : {}),
    };
  });
  // IO 绑定投影边（不参与图拓扑/环检测；删除 = 解绑）
  for (const el of value.io) {
    if (!el.node || !el.port) continue;
    if (el.kind === "input") {
      edges.push({
        id: ioEdgeId(el.id),
        source: el.id,
        sourceHandle: IO_HANDLE_OUT,
        target: el.node,
        targetHandle: inHandle(el.port),
        className: "ig-io-edge",
      });
    } else {
      edges.push({
        id: ioEdgeId(el.id),
        source: el.node,
        sourceHandle: outHandle(el.port),
        target: el.id,
        targetHandle: IO_HANDLE_IN,
        className: "ig-io-edge",
      });
    }
  }

  const ioNodes: InferIoFlowNode[] = value.io.map((el) => ({
    id: el.id,
    type: "inferIoNode",
    position: value.layout[el.id] ?? { x: 0, y: 0 },
    selected: selectedSet.has(el.id),
    data: {
      kind: el.kind,
      ...(el.node !== undefined && el.port !== undefined ? { node: el.node, port: el.port } : {}),
    },
  }));

  return { nodes: [...nodes, ...ioNodes], edges };
}

// ── 节点操作 ──

export interface NewNodeSpec {
  type: string;
  affinity: string;
  inputs: DcirPort[];
  outputs: DcirPort[];
  /** 唯一名生成基（缺省用 type；信号逻辑节点用友好名基避开 ':' 非法字符） */
  nameBase?: string;
}

/** 唯一名生成：引擎名首字母小写 + `-N` 去重（满足名字白名单；与 IO 元素 id 一并避让） */
export function uniqueNodeName(graph: DcirGraphJson, base: string, extraTaken?: Iterable<string>): string {
  const root = (base.charAt(0).toLowerCase() + base.slice(1)) || "node";
  const taken = new Set<string>([...graph.nodes.map((n) => n.name), ...(extraTaken ?? [])]);
  if (!taken.has(root)) return root;
  for (let i = 2; ; i += 1) {
    const candidate = `${root}-${i}`;
    if (!taken.has(candidate)) return candidate;
  }
}

export function addNode(value: EditorValue, spec: NewNodeSpec, position: XY): { value: EditorValue; name: string } {
  const graph = cloneGraph(value.graph);
  const name = uniqueNodeName(graph, spec.nameBase ?? spec.type ?? "node", value.io.map((e) => e.id));
  const node: DcirNode = withSignalTerminals({
    name,
    type: spec.type,
    affinity: spec.affinity,
    inputs: JSON.parse(JSON.stringify(spec.inputs)) as DcirPort[],
    outputs: JSON.parse(JSON.stringify(spec.outputs)) as DcirPort[],
  });
  graph.nodes = [...graph.nodes, node];
  return {
    value: { graph, layout: { ...value.layout, [name]: position }, remote: value.remote, io: value.io },
    name,
  };
}

export function removeNode(value: EditorValue, name: string): EditorValue {
  const graph = cloneGraph(value.graph);
  graph.nodes = graph.nodes.filter((n) => n.name !== name);
  graph.edges = graph.edges.filter((e) => e.srcNode !== name && e.dstNode !== name);
  // IO 元素联动：解绑（元素保留；bindings 由 syncBindings 重算剔除）
  const io = value.io.map((e) => (e.node === name ? unbound(e) : e));
  const layout = { ...value.layout };
  delete layout[name];
  const remote = { ...value.remote };
  delete remote[name];
  return syncBindings({ graph, layout, remote, io });
}

/** 改名：级联边 / 绑定 / remote / layout 键（重名或空名拒绝） */
export function renameNode(value: EditorValue, oldName: string, newName: string): { value: EditorValue; error?: string } {
  const name = newName.trim();
  if (name === oldName) return { value };
  if (!name) return { value, error: "名称不能为空" };
  if (value.graph.nodes.some((n) => n.name === name)) return { value, error: `节点名 '${name}' 已存在` };
  if (value.io.some((e) => e.id === name)) return { value, error: `名称 '${name}' 与 IO 元素 id 冲突` };
  const graph = cloneGraph(value.graph);
  for (const n of graph.nodes) if (n.name === oldName) n.name = name;
  for (const e of graph.edges) {
    if (e.srcNode === oldName) e.srcNode = name;
    if (e.dstNode === oldName) e.dstNode = name;
  }
  for (const b of graph.inputBindings ?? []) if (b.nodeName === oldName) b.nodeName = name;
  for (const b of graph.outputBindings ?? []) if (b.nodeName === oldName) b.nodeName = name;
  const layout: Record<string, XY> = {};
  for (const [k, v] of Object.entries(value.layout)) layout[k === oldName ? name : k] = v;
  const remote: Record<string, string> = {};
  for (const [k, v] of Object.entries(value.remote)) remote[k === oldName ? name : k] = v;
  const io = value.io.map((e) => (e.node === oldName ? { ...e, node: name } : e));
  return { value: syncBindings({ graph, layout, remote, io }) };
}

/** 编辑节点属性（type / affinity）；未知字段保留 */
export function patchNode(value: EditorValue, name: string, patch: Partial<Pick<DcirNode, "type" | "affinity">>): EditorValue {
  const graph = cloneGraph(value.graph);
  for (const n of graph.nodes) if (n.name === name) Object.assign(n, patch);
  return withGraph(value, graph);
}

/** 信号逻辑节点端子集（按种类）：source/const 无输入，not 单输入，and/or 双输入；均单 sig.out 输出 */
function signalLogicPorts(logic: SignalLogicKind): { inputs: DcirPort[]; outputs: DcirPort[] } {
  const outputs = [sigPort(SIG_OUT)];
  if (logic === "not") return { inputs: [sigPort(`${SIG_IN_PREFIX}0`)], outputs };
  if (logic === "and" || logic === "or") return { inputs: [sigPort(`${SIG_IN_PREFIX}0`), sigPort(`${SIG_IN_PREFIX}1`)], outputs };
  return { inputs: [], outputs };
}

/**
 * 建信号编排逻辑节点（affinity 固定 System，不落远程意图）：
 * source 附 signalName（宿主对外入口，初始空），const 附 signalValue（初始 true）。
 */
export function addSignalNode(value: EditorValue, logic: SignalLogicKind, position: XY): { value: EditorValue; name: string } {
  const { inputs, outputs } = signalLogicPorts(logic);
  const { value: created, name } = addNode(
    value,
    { type: SIGNAL_NODE_TYPES[logic], affinity: "System", inputs, outputs, nameBase: SIGNAL_NODE_NAME_BASE[logic] },
    position,
  );
  if (logic !== "source" && logic !== "const") return { value: created, name };
  const graph = cloneGraph(created.graph);
  for (const n of graph.nodes) {
    if (n.name !== name) continue;
    if (logic === "source") n.signalName = "";
    else n.signalValue = true;
  }
  return { value: withGraph(created, graph), name };
}

/** 设置 SignalSource 对外信号名（宿主 setSignal 入口） */
export function setSignalSourceName(value: EditorValue, name: string, signalName: string): EditorValue {
  const graph = cloneGraph(value.graph);
  for (const n of graph.nodes) if (n.name === name) n.signalName = signalName.trim();
  return withGraph(value, graph);
}

/** 设置 SignalConst 常量布尔值 */
export function setSignalConstValue(value: EditorValue, name: string, signalValue: boolean): EditorValue {
  const graph = cloneGraph(value.graph);
  for (const n of graph.nodes) if (n.name === name) n.signalValue = signalValue;
  return withGraph(value, graph);
}

/** AND/OR 增一个信号输入（sig.inN，上限 SIG_LOGIC_MAX_INPUTS） */
export function addSignalLogicInput(value: EditorValue, name: string): EditorValue {
  const node = value.graph.nodes.find((n) => n.name === name);
  if (!node) return value;
  const sigInputs = node.inputs.filter(isSignalPort);
  if (sigInputs.length >= SIG_LOGIC_MAX_INPUTS) return value;
  const taken = new Set(sigInputs.map((p) => p.name));
  let idx = sigInputs.length;
  while (taken.has(`${SIG_IN_PREFIX}${idx}`)) idx += 1;
  return addPort(value, name, "inputs", sigPort(`${SIG_IN_PREFIX}${idx}`));
}

/** AND/OR 减一个信号输入（至少保留一个；级联清边由 removePort 负责） */
export function removeSignalLogicInput(value: EditorValue, name: string, portName: string): EditorValue {
  const node = value.graph.nodes.find((n) => n.name === name);
  if (!node) return value;
  if (node.inputs.filter(isSignalPort).length <= 1) return value;
  return removePort(value, name, "inputs", portName);
}

// ── 端口操作（引用完整性：边 / 绑定联动清理）──

export type PortDirection = "inputs" | "outputs";

export function addPort(value: EditorValue, nodeName: string, direction: PortDirection, port: DcirPort): EditorValue {
  const graph = cloneGraph(value.graph);
  for (const n of graph.nodes) {
    if (n.name !== nodeName) continue;
    n[direction] = [...n[direction], JSON.parse(JSON.stringify(port)) as DcirPort];
  }
  return withGraph(value, graph);
}

export function removePort(value: EditorValue, nodeName: string, direction: PortDirection, portName: string): EditorValue {
  const graph = cloneGraph(value.graph);
  for (const n of graph.nodes) {
    if (n.name !== nodeName) continue;
    n[direction] = n[direction].filter((p) => p.name !== portName);
  }
  graph.edges = graph.edges.filter((e) =>
    direction === "inputs"
      ? !(e.dstNode === nodeName && e.dstPort === portName)
      : !(e.srcNode === nodeName && e.srcPort === portName),
  );
  // 绑定联动：解绑指向该端口的 IO 元素（bindings 由 syncBindings 重算剔除）
  const io = value.io.map((e) => (e.node === nodeName && e.port === portName ? unbound(e) : e));
  return syncBindings({ ...value, graph, io });
}

/** 端口改名：级联边 / 绑定（同名冲突或空名拒绝） */
export function renamePort(
  value: EditorValue,
  nodeName: string,
  direction: PortDirection,
  oldName: string,
  newName: string,
): { value: EditorValue; error?: string } {
  const name = newName.trim();
  if (name === oldName) return { value };
  if (!name) return { value, error: "端口名不能为空" };
  const node = value.graph.nodes.find((n) => n.name === nodeName);
  if (!node) return { value, error: "节点不存在" };
  if (node[direction].some((p) => p.name === name)) return { value, error: `端口 '${name}' 已存在` };
  const graph = cloneGraph(value.graph);
  for (const n of graph.nodes) {
    if (n.name !== nodeName) continue;
    for (const p of n[direction]) if (p.name === oldName) p.name = name;
  }
  for (const e of graph.edges) {
    if (direction === "inputs") {
      if (e.dstNode === nodeName && e.dstPort === oldName) e.dstPort = name;
    } else if (e.srcNode === nodeName && e.srcPort === oldName) {
      e.srcPort = name;
    }
  }
  const bindingsKey = direction === "inputs" ? "inputBindings" : "outputBindings";
  for (const b of graph[bindingsKey] ?? []) {
    if (b.nodeName === nodeName && b.portName === oldName) b.portName = name;
  }
  const io = value.io.map((e) => (e.node === nodeName && e.port === oldName ? { ...e, port: name } : e));
  return { value: syncBindings({ ...value, graph, io }) };
}

/** 端口 Schema 补丁（tensorType / typeSize / shape / required）；不涉及引用 */
export function updatePort(
  value: EditorValue,
  nodeName: string,
  direction: PortDirection,
  portName: string,
  patch: Partial<Pick<DcirPort, "tensorType" | "typeSize" | "shape" | "required">>,
): EditorValue {
  const graph = cloneGraph(value.graph);
  for (const n of graph.nodes) {
    if (n.name !== nodeName) continue;
    for (const p of n[direction]) if (p.name === portName) Object.assign(p, patch);
  }
  return withGraph(value, graph);
}

// ── 边操作（输入口单源 = 替换式；重复去重；绑定互斥）──

export function addEdge(value: EditorValue, e: DcirEdge): EditorValue {
  const nodesByName = new Map(value.graph.nodes.map((n) => [n.name, n]));
  const src = nodesByName.get(e.srcNode);
  const dst = nodesByName.get(e.dstNode);
  if (!src || !dst) return value;
  if (e.srcNode === e.dstNode) return value; // 自环：连接层已拒，防御式无操作
  if (!src.outputs.some((p) => p.name === e.srcPort) || !dst.inputs.some((p) => p.name === e.dstPort)) return value;
  const graph = cloneGraph(value.graph);
  const id = edgeIdOf(e);
  if (graph.edges.some((x) => edgeIdOf(x) === id)) return value; // 重复去重（无操作）
  graph.edges = [
    // 单源替换：清除同一输入口上的既有边
    ...graph.edges.filter((x) => !(x.dstNode === e.dstNode && x.dstPort === e.dstPort)),
    { srcNode: e.srcNode, srcPort: e.srcPort, dstNode: e.dstNode, dstPort: e.dstPort },
  ];
  // 绑定互斥：入边落位即解绑该口输入元素（syncBindings 同步投影清绑定）
  const io = value.io.map((el) =>
    el.kind === "input" && el.node === e.dstNode && el.port === e.dstPort ? unbound(el) : el,
  );
  return syncBindings({ ...value, graph, io });
}

export function removeEdge(value: EditorValue, e: Pick<DcirEdge, "srcNode" | "srcPort" | "dstNode" | "dstPort">): EditorValue {
  const id = edgeIdOf(e);
  const graph = cloneGraph(value.graph);
  graph.edges = graph.edges.filter((x) => edgeIdOf(x) !== id);
  return withGraph(value, graph);
}

// ── 远程意图 / 图级绑定 ──

export function setRemote(value: EditorValue, nodeName: string, modelKey: string): EditorValue {
  const key = modelKey.trim();
  const remote = { ...value.remote };
  if (key) remote[nodeName] = key;
  else delete remote[nodeName];
  return { ...value, remote };
}

// ── 目录条目（模型 / 算子节点）──

/** 目录条目建节点入参（与 CatalogModelItem 结构兼容子集） */
export interface CatalogModelSpec {
  modelKey: string;
  /** 条目自报引擎类型（模型=引擎名；算子=引擎名）；缺省/未自报 → 建节点落 unknown */
  engineType?: string | null;
  /** 条目声明的端口（io 驱动；null/缺省 = 未声明，建空端口节点由检查器补） */
  io?: { inputs: DcirPort[]; outputs: DcirPort[] } | null;
}

/**
 * 从目录条目建节点：type = 条目自报 engineType（未自报落 unknown，不臆测具体引擎）
 * + 端口采用条目声明 io（声明即执行面）+ 远程意图自动落位（modelKey；
 * 「目录条目 = 全远程节点」）。
 */
export function addModelNode(
  value: EditorValue,
  item: CatalogModelSpec,
  position: XY,
): { value: EditorValue; name: string } {
  const { value: withNode, name } = addNode(
    value,
    {
      type: item.engineType && item.engineType.length > 0 ? item.engineType : "unknown",
      affinity: "Compute",
      inputs: item.io?.inputs ?? [],
      outputs: item.io?.outputs ?? [],
    },
    position,
  );
  return { value: setRemote(withNode, name, item.modelKey), name };
}

export function hasInputBinding(value: EditorValue, nodeName: string, portName: string): boolean {
  return (value.graph.inputBindings ?? []).some((b) => b.nodeName === nodeName && b.portName === portName);
}

export function hasOutputBinding(value: EditorValue, nodeName: string, portName: string): boolean {
  return (value.graph.outputBindings ?? []).some((b) => b.nodeName === nodeName && b.portName === portName);
}

// ── IO 控件元素（图级输入/输出绑定的可视化载体）──

/** 解绑后的元素形态（无 node/port 键） */
function unbound(el: IoElement): IoElement {
  return { id: el.id, kind: el.kind };
}

/** 元素 id 生成：io-in-N / io-out-N（避让节点名与既有元素 id） */
function freshIoId(taken: ReadonlySet<string>, kind: "input" | "output"): string {
  const prefix = kind === "input" ? "io-in-" : "io-out-";
  for (let i = 1; ; i += 1) {
    const id = `${prefix}${i}`;
    if (!taken.has(id)) return id;
  }
}

/**
 * 绑定投影同步（不变量出口）：以 io 元素表为准重算 graph.inputBindings /
 * outputBindings；同 key 的原 binding 对象复用（unknown 字段保留）；
 * 未绑定元素不投影。所有改动 io 的操作均经此出口。
 */
export function syncBindings(value: EditorValue): EditorValue {
  const keyOf = (b: DcirBinding): string => `${b.nodeName}\u0000${b.portName}`;
  const prevIn = new Map((value.graph.inputBindings ?? []).map((b) => [keyOf(b), b]));
  const prevOut = new Map((value.graph.outputBindings ?? []).map((b) => [keyOf(b), b]));
  const inputBindings: DcirBinding[] = [];
  const outputBindings: DcirBinding[] = [];
  const seen = new Set<string>();
  for (const el of value.io) {
    if (!el.node || !el.port) continue;
    const key = `${el.node}\u0000${el.port}`;
    if (seen.has(`${el.kind}\u0000${key}`)) continue; // 单源兜底（同口同向仅首元素投影）
    seen.add(`${el.kind}\u0000${key}`);
    if (el.kind === "input") {
      inputBindings.push(prevIn.get(key) ?? { nodeName: el.node, portName: el.port });
    } else {
      outputBindings.push(prevOut.get(key) ?? { nodeName: el.node, portName: el.port });
    }
  }
  const graph: DcirGraphJson = { ...value.graph, inputBindings, outputBindings };
  return { ...value, graph };
}

/** 从 bindings 派生 IO 元素（入档路径；id 避让节点名） */
export function ioFromBindings(graph: DcirGraphJson): IoElement[] {
  const taken = new Set(graph.nodes.map((n) => n.name));
  const io: IoElement[] = [];
  for (const b of graph.inputBindings ?? []) {
    const id = freshIoId(taken, "input");
    taken.add(id);
    io.push({ id, kind: "input", node: b.nodeName, port: b.portName });
  }
  for (const b of graph.outputBindings ?? []) {
    const id = freshIoId(taken, "output");
    taken.add(id);
    io.push({ id, kind: "output", node: b.nodeName, port: b.portName });
  }
  return io;
}

/** 新建 IO 元素（未绑定；坐标落位） */
export function addIoElement(value: EditorValue, kind: "input" | "output", position: XY): { value: EditorValue; id: string } {
  const taken = new Set<string>([...value.graph.nodes.map((n) => n.name), ...value.io.map((e) => e.id)]);
  const id = freshIoId(taken, kind);
  const io = [...value.io, { id, kind }];
  return { value: { ...value, io, layout: { ...value.layout, [id]: position } }, id };
}

/** 删除 IO 元素（解除绑定 + 坐标清理；bindings 由 syncBindings 重算剔除） */
export function removeIoElement(value: EditorValue, id: string): EditorValue {
  if (!value.io.some((e) => e.id === id)) return value;
  const io = value.io.filter((e) => e.id !== id);
  const layout = { ...value.layout };
  delete layout[id];
  return syncBindings({ ...value, io, layout });
}

/**
 * 绑定元素 ↔ 节点端口：
 * - input 元素 → 节点输入口（单源替换：清该口入边 + 解绑同口其他输入元素）；
 * - output 元素 → 节点输出口（同口解绑替换）。
 */
export function bindIoElement(value: EditorValue, id: string, node: string, port: string): EditorValue {
  const el = value.io.find((e) => e.id === id);
  const target = value.graph.nodes.find((n) => n.name === node);
  if (!el || !target) return value;
  const ports = el.kind === "input" ? target.inputs : target.outputs;
  if (!ports.some((p) => p.name === port)) return value;
  const graph = cloneGraph(value.graph);
  if (el.kind === "input") {
    // 输入口单源（外部供数）：入边替换式清除
    graph.edges = graph.edges.filter((e) => !(e.dstNode === node && e.dstPort === port));
  }
  const io = value.io.map((e) => {
    if (e.id === id) return { ...e, node, port };
    if (e.kind === el.kind && e.node === node && e.port === port) return unbound(e);
    return e;
  });
  return syncBindings({ ...value, graph, io });
}

/** 解除元素绑定（元素保留，回到未连接态） */
export function unbindIoElement(value: EditorValue, id: string): EditorValue {
  const el = value.io.find((e) => e.id === id);
  if (!el || el.node === undefined) return value;
  const io = value.io.map((e) => (e.id === id ? unbound(e) : e));
  return syncBindings({ ...value, io });
}

/** 按 id 取元素（rules / 渲染消费） */
export function ioElementOf(value: EditorValue, id: string): IoElement | undefined {
  return value.io.find((e) => e.id === id);
}

// ── 序列化 / 导入 / 提交派生 ──

/** 提交用图 JSON（layout / remote 自然不在图内；unknown 字段原样保留） */
export function serializeGraph(value: EditorValue): DcirGraphJson {
  return cloneGraph(value.graph);
}

/** JSON 导入（防御式清洗；layout 保留存续节点坐标，remote 保留存续键） */
export function applyImport(value: EditorValue, text: string): { value: EditorValue; issues: EditorIssue[] } {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    return { value, issues: [{ level: "error", code: "import-parse", detail: `JSON 解析失败：${String(e)}` }] };
  }
  const { graph, issues } = sanitizeGraph(parsed);
  const names = new Set(graph.nodes.map((n) => n.name));
  const layout: Record<string, XY> = {};
  for (const n of graph.nodes) {
    const kept = value.layout[n.name];
    if (kept) layout[n.name] = kept;
  }
  const remote: Record<string, string> = {};
  for (const [k, v] of Object.entries(value.remote)) if (names.has(k) && v) remote[k] = v;
  // IO 元素从清洗后的 bindings 重建（坐标由 fillLayout 补种）
  return { value: { graph, layout, remote, io: ioFromBindings(graph) }, issues };
}

/** 提交载荷派生：远程意图（按图内节点顺序，稳定输出；空白键视为未设置） */
export function remoteNodesOf(value: EditorValue): { nodeId: string; modelKey: string }[] {
  const out: { nodeId: string; modelKey: string }[] = [];
  for (const n of value.graph.nodes) {
    const key = (value.remote[n.name] ?? "").trim();
    if (key) out.push({ nodeId: n.name, modelKey: key });
  }
  return out;
}

/** 提交载荷派生：本地图级输入（= inputBindings，两步流上传的寻址面） */
export function localInputsOf(value: EditorValue): { node: string; port: string }[] {
  return (value.graph.inputBindings ?? []).map((b) => ({ node: b.nodeName, port: b.portName }));
}
