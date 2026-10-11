/**
 * 推理图编辑器 — 值模型与 DCIr JSON 窄读取类型（core 层，无运行时依赖）
 *
 * 字段口径与 DCIr C++ 序列化（GraphCompiler）逐字对齐。
 * 分层纪律：
 * - graph 为权威 DCIr 序列化 JSON：编辑器只管理已知字段，未知字段原样保留（无损往返）；
 * - layout / remote 为编辑器侧元数据（side-band）：不进图 JSON、不进提交载荷。
 */

/** 端口类别：tensor = 数据面张量端口；signal = 控制面布尔信号端口。缺省视为 tensor（向后读兼容） */
export type PortKind = "tensor" | "signal";

/**
 * DCIr 序列化端口（与 capability 域 IoPort 同编码；unknown = 未建模字段保留）
 *
 * kind 判别数据/信号两类端口，混居同一 inputs/outputs 数组：
 * - tensor：tensorType/typeSize/shape 描述张量形状；
 * - signal：仅传布尔，tensorType/typeSize/shape 无意义（序列化时不写）。
 */
export interface DcirPort {
  name: string;
  /** 端口类别；缺省 = tensor */
  kind?: PortKind;
  /** 以下三项为 tensor 端口专有；signal 端口忽略 */
  tensorType?: string;
  typeSize?: number;
  shape?: number[];
  required: boolean;
  [key: string]: unknown;
}

/** 信号编排逻辑节点种类（只有信号端子的图内控制流原语） */
export type SignalLogicKind = "source" | "const" | "not" | "and" | "or";

/** DCIr 序列化节点 */
export interface DcirNode {
  name: string;
  type: string;
  affinity?: string;
  /** SignalSource 对外信号名：宿主经 setSignal(name) 驱动的入口；仅信号源节点使用 */
  signalName?: string;
  /** SignalConst 常量布尔值；仅常量信号节点使用 */
  signalValue?: boolean;
  inputs: DcirPort[];
  outputs: DcirPort[];
  [key: string]: unknown;
}

/** DCIr 序列化边（端口到端口四元组） */
export interface DcirEdge {
  srcNode: string;
  srcPort: string;
  dstNode: string;
  dstPort: string;
  [key: string]: unknown;
}

/** 图级输入 / 输出绑定项 */
export interface DcirBinding {
  nodeName: string;
  portName: string;
  [key: string]: unknown;
}

/** DCIr 序列化图（opaque JSON 的窄读取） */
export interface DcirGraphJson {
  version?: string;
  nodes: DcirNode[];
  edges: DcirEdge[];
  inputBindings?: DcirBinding[];
  outputBindings?: DcirBinding[];
  [key: string]: unknown;
}

/** 画布坐标 */
export interface XY {
  x: number;
  y: number;
}

/**
 * IO 控件元素（输入 / 输出）——图级输入/输出绑定的可视化载体
 *
 * 语义（2026-10-10 敲定）：专门的可拖放元素；输入元素仅右侧接线点（连向
 * 某节点的输入端口）、输出元素仅左侧接线点（某节点的输出端口连向它）。
 * 元素是编辑器侧构造：序列化时投影为 graph.inputBindings / outputBindings
 * （DCIr 节点表不含 IO 元素）；坐标经 layout side-band 存取。
 */
export interface IoElement {
  /** 元素稳定 id（不与节点名冲突：生成时避让） */
  id: string;
  kind: "input" | "output";
  /** 绑定目标：input 元素 → 节点输入口；output 元素 → 节点输出口；未连接时缺省 */
  node?: string;
  port?: string;
}

/** 编辑器受控值：graph 为权威，layout / remote / io 为 side-band */
export interface EditorValue {
  graph: DcirGraphJson;
  /** node name / io element id → 画布坐标（不进提交） */
  layout: Record<string, XY>;
  /** node name → modelKey（远程意图；不进图 JSON，提交时映射 remoteNodes） */
  remote: Record<string, string>;
  /** IO 控件元素（图级输入/输出绑定的可视化载体；不进图 JSON） */
  io: IoElement[];
}

/** 校验问题级别：error 阻断提交；warn 仅提示 */
export type IssueLevel = "error" | "warn";

/** 结构化校验问题（断言断 code，不锁文案） */
export interface EditorIssue {
  level: IssueLevel;
  /** 机器可读类别 */
  code: string;
  /** 相关节点（可空） */
  node?: string;
  detail: string;
}
