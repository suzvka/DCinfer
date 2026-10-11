/**
 * ui 层 — React Flow 类型桥（core 结构投影 → RF 运行时类型，单点收口）
 *
 * core/model.toFlowModel 产出框架无关的 FlowNode/FlowEdge 结构（字段为 RF
 * Node/Edge 的子集）；本模块在 ui 边界一次性断言为 RF 类型，core 不 import
 * @xyflow/react，保持 headless。
 */

import type { Edge, Node } from "@xyflow/react";
import type { InferIoNodeData, InferNodeData } from "../core/model";

export type RfInferNode = Node<InferNodeData, "inferNode">;
export type RfIoNode = Node<InferIoNodeData, "inferIoNode">;
export type AnyRfNode = RfInferNode | RfIoNode;
export type RfEdge = Edge;

/** core 投影节点 → RF 节点（结构子集，断言安全） */
export function toRfNodes<T>(nodes: readonly { id: string; type: string; position: { x: number; y: number }; selected?: boolean; data: T }[]): Node[] {
  return nodes as unknown as Node[];
}

/** core 投影边 → RF 边 */
export function toRfEdges(edges: readonly unknown[]): Edge[] {
  return edges as unknown as Edge[];
}
