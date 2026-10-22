/**
 * 自定义节点渲染：端子 = handle（id = in:<port> / out:<port>）
 *
 * 重设计原则（面向用户）：
 * - 节点头优先展示模型短名（modelKey 末段），引擎类型缩成小 badge；
 * - 端口行只显示端口名，数据类型/形状经 tooltip 展示（hover 时可见技术细节）；
 * - 缺必需输入的端口保留警示色，但不再在节点上显示原始 shape 标签；
 * - IO 元素节点：文案改用户语言，保持单接线点语义。
 */

import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { DcirPort } from "../core/types";
import {
  IO_HANDLE_IN,
  IO_HANDLE_OUT,
  inHandle,
  isSignalNodeType,
  isSignalPort,
  outHandle,
  sigLabel,
} from "../core/model";
import type { RfInferNode, RfIoNode } from "./flow-types";
import { friendlyNodeType, portTooltip, shortModelName } from "./i18n";

/** 端口 tooltip（hover 时展示完整技术细节） */
function nodePortTitle(p: DcirPort, extra: string): string {
  return portTooltip(p, extra);
}

export function InferNode({ data, selected }: NodeProps<RfInferNode>) {
  const fed = new Set(data.fedInputs);
  const boundIn = new Set(data.boundInputs);
  const boundOut = new Set(data.boundOutputs);
  const isSigNode = isSignalNodeType(data.type);
  const affinity = isSigNode ? "system" : data.affinity === "Compute" ? "compute" : "operator";

  // 用户可读的节点短名（模型节点用 modelKey 末段；信号节点用友好类型名）
  const displayName = isSigNode
    ? friendlyNodeType(data.type)
    : data.remoteKey
    ? shortModelName(data.remoteKey)
    : data.name;

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
              <span className="ig-sig-name" title={`信号输入 ${sigLabel(p.name)}（布尔）`}>
                {sigLabel(p.name)}
              </span>
            </div>
          ))}
        </div>
      )}
      <div className="ig-head">
        <span className="ig-name" title={displayName !== data.name ? `${data.name}（模型：${data.remoteKey ?? data.type}）` : data.name}>
          {displayName}
        </span>
        {!isSigNode && (
          <span className="ig-type" title={`引擎类型：${data.type || "未设置"}`}>
            {data.type || "?"}
          </span>
        )}
      </div>
      {(tensorInputs.length > 0 || tensorOutputs.length > 0) && (
        <div className="ig-body">
          <div className="ig-col ig-col-in">
            {tensorInputs.map((p) => {
              const missing = p.required !== false && !fed.has(p.name);
              const bound = boundIn.has(p.name);
              const extra = `${bound ? " · 已设为图输入" : ""}${missing ? " · 尚未连接" : ""}`;
              return (
                <div className="ig-port" key={p.name}>
                  <Handle type="target" position={Position.Left} id={inHandle(p.name)} className="ig-handle" />
                  <span
                    className={`ig-port-name${missing ? " missing" : ""}${bound ? " bound-in" : ""}`}
                    title={nodePortTitle(p, extra)}
                  >
                    {p.name}
                    {p.required !== false ? <em>*</em> : null}
                  </span>
                  {missing && <span className="ig-port-dot ig-port-dot-warn" aria-hidden="true" />}
                </div>
              );
            })}
          </div>
          <div className="ig-col ig-col-out">
            {tensorOutputs.map((p) => {
              const bound = boundOut.has(p.name);
              const extra = bound ? " · 已设为图输出" : "";
              return (
                <div className="ig-port ig-port-right" key={p.name}>
                  {bound && <span className="ig-port-dot ig-port-dot-bound" aria-hidden="true" />}
                  <span
                    className={`ig-port-name${bound ? " bound-out" : ""}`}
                    title={nodePortTitle(p, extra)}
                  >
                    {p.name}
                    {p.required !== false ? <em>*</em> : null}
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
              <span className="ig-sig-name" title={`信号输出 ${sigLabel(p.name)}（布尔状态）`}>
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
        title={bound ? `已连接：${data.node}.${data.port}` : "尚未连接"}
      >
        {bound ? `${data.node}.${data.port}` : "待连接"}
      </span>
      {kind === "input" ? (
        <Handle type="source" position={Position.Right} id={IO_HANDLE_OUT} className="ig-handle" />
      ) : null}
    </div>
  );
}
