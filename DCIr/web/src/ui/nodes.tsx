/**
 * 自定义节点渲染：端子 = handle（id = in:<port> / out:<port>）
 *
 * 视觉约定（与 core/layout.ts 尺寸估算一致：宽 236、头 46、端子行 22）：
 * - 亲和色带：Compute 紫蓝 / Operator 蓝青；
 * - required 输入未供数：端子名警示色；图输入/图输出绑定：端子名下划虚线标记。
 *
 * IO 控件元素（InferIoNode）：输入元素仅右侧出点、输出元素仅左侧入点；
 * 已绑定时展示 node.port，未绑定时虚线提示（拖线到节点端子即绑定）。
 */

import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { DcirPort } from "../core/types";
import { IO_HANDLE_IN, IO_HANDLE_OUT, inHandle, isSignalNodeType, isSignalPort, outHandle, sigLabel } from "../core/model";
import type { RfInferNode, RfIoNode } from "./flow-types";

function portTag(p: DcirPort): string {
  const type = p.tensorType === "Float" && p.typeSize === 4 ? "f32" : `${p.tensorType}:${p.typeSize}`;
  return `${type}[${(p.shape ?? []).join(",")}]`;
}

function portTitle(p: DcirPort, extra: string): string {
  return `${p.name}：${p.tensorType}（typeSize ${p.typeSize}）shape [${(p.shape ?? []).join(", ")}]${p.required ? " · required" : ""}${extra}`;
}

export function InferNode({ data, selected }: NodeProps<RfInferNode>) {
  const fed = new Set(data.fedInputs);
  const boundIn = new Set(data.boundInputs);
  const boundOut = new Set(data.boundOutputs);
  const isSigNode = isSignalNodeType(data.type);
  const affinity = isSigNode ? "system" : data.affinity === "Compute" ? "compute" : "operator";
  const tensorInputs = data.inputs.filter((p) => !isSignalPort(p));
  const tensorOutputs = data.outputs.filter((p) => !isSignalPort(p));
  const signalInputs = data.inputs.filter(isSignalPort);
  const signalOutputs = data.outputs.filter(isSignalPort);
  return (
    <div
      className={`ig-node aff-${affinity}${isSigNode ? " is-signal-node" : ""}${selected ? " selected" : ""}${
        data.issueLevel ? ` issue-${data.issueLevel}` : ""
      }`}
    >
      {signalInputs.length > 0 && (
        <div className="ig-sig-band ig-sig-band-top">
          {signalInputs.map((p) => (
            <div className="ig-sig-port" key={p.name}>
              <Handle type="target" position={Position.Top} id={inHandle(p.name)} className="ig-handle ig-handle-signal" />
              <span className="ig-sig-name" title={`信号输入 ${p.name}（布尔；无连线 = 放行）`}>
                {sigLabel(p.name)}
              </span>
            </div>
          ))}
        </div>
      )}
      <div className="ig-head">
        <span className="ig-name" title={data.name}>
          {data.name}
        </span>
        <span className="ig-type" title={`节点类型：${data.type || "（未设置）"}`}>
          {data.type || "?"}
        </span>
      </div>
      {(tensorInputs.length > 0 || tensorOutputs.length > 0) && (
        <div className="ig-body">
          <div className="ig-col ig-col-in">
            {tensorInputs.map((p) => {
              const missing = p.required !== false && !fed.has(p.name);
              const bound = boundIn.has(p.name);
              const extra = `${bound ? " · 图输入（外部供数）" : ""}${missing ? " · 未供数" : ""}`;
              return (
                <div className="ig-port" key={p.name}>
                  <Handle type="target" position={Position.Left} id={inHandle(p.name)} className="ig-handle" />
                  <span
                    className={`ig-port-name${missing ? " missing" : ""}${bound ? " bound-in" : ""}`}
                    title={portTitle(p, extra)}
                  >
                    {p.name}
                    {p.required ? <em>*</em> : null}
                  </span>
                  <span className="ig-port-tag">{portTag(p)}</span>
                </div>
              );
            })}
          </div>
          <div className="ig-col ig-col-out">
            {tensorOutputs.map((p) => {
              const bound = boundOut.has(p.name);
              return (
                <div className="ig-port ig-port-right" key={p.name}>
                  <span className="ig-port-tag">{portTag(p)}</span>
                  <span
                    className={`ig-port-name${bound ? " bound-out" : ""}`}
                    title={portTitle(p, bound ? " · 图输出（结果取回）" : "")}
                  >
                    {p.name}
                    {p.required ? <em>*</em> : null}
                  </span>
                  <Handle type="source" position={Position.Right} id={outHandle(p.name)} className="ig-handle" />
                </div>
              );
            })}
          </div>
        </div>
      )}
      {signalOutputs.length > 0 && (
        <div className="ig-sig-band ig-sig-band-bottom">
          {signalOutputs.map((p) => (
            <div className="ig-sig-port" key={p.name}>
              <span className="ig-sig-name" title={`信号输出 ${p.name}（布尔状态端子，由节点背后算法侧赋值）`}>
                {sigLabel(p.name)}
              </span>
              <Handle type="source" position={Position.Bottom} id={outHandle(p.name)} className="ig-handle ig-handle-signal" />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** IO 控件元素（输入 / 输出）：单接线点 + 绑定目标展示 */
export function InferIoNode({ data, selected }: NodeProps<RfIoNode>) {
  const kind = data.kind;
  const bound = data.node !== undefined && data.port !== undefined;
  return (
    <div className={`ig-io-node io-${kind}${selected ? " selected" : ""}`}>
      {kind === "output" ? (
        <Handle type="target" position={Position.Left} id={IO_HANDLE_IN} className="ig-handle" />
      ) : null}
      <span className="ig-io-kind">{kind === "input" ? "输入" : "输出"}</span>
      <span
        className={`ig-io-target${bound ? "" : " unbound"}`}
        title={bound ? `绑定：${data.node}.${data.port}` : "未绑定（拖线到节点端子即绑定）"}
      >
        {bound ? `${data.node}.${data.port}` : "未绑定"}
      </span>
      {kind === "input" ? (
        <Handle type="source" position={Position.Right} id={IO_HANDLE_OUT} className="ig-handle" />
      ) : null}
    </div>
  );
}
