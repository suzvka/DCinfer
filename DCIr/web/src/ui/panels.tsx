/**
 * 编辑器面板：检查器（右）/ JSON 抽屉 / 问题面板
 *
 * 全部为受控展示组件：变更经回调上抛到领域层纯函数（ui/index.tsx 统一 commit）。
 * 样式全部作用域在 .ig-* 命名空间下（styles.css），不依赖宿主页面样式。
 * 调色板一元化后移交 palette.tsx + sources/*（协议见 palette-protocol.ts）。
 */

import { useState } from "react";
import type { PortDirection } from "../core/model";
import {
  SIGNAL_NODE_TYPES,
  SIG_LOGIC_MAX_INPUTS,
  hasInputBinding,
  hasOutputBinding,
  isSignalNodeType,
  isSignalPort,
  sigLabel,
} from "../core/model";
import type { DcirNode, DcirPort, EditorIssue, EditorValue, IoElement } from "../core/types";
import { detailPortTag } from "./sources/parts";

// ── 通用小件 ──

/** 提交式文本输入（Enter/blur 才提交；错误经回调文案展示） */
function EditableText({
  value,
  commit,
  className,
  placeholder,
  title,
  disabled,
}: {
  value: string;
  commit: (next: string) => string | null;
  className?: string;
  placeholder?: string;
  title?: string;
  disabled?: boolean;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const [err, setErr] = useState("");
  const shown = draft ?? value;
  return (
    <span className={`ig-editable${err ? " err" : ""}`}>
      <input
        className={className ?? "ig-input"}
        value={shown}
        placeholder={placeholder}
        title={err || title || undefined}
        disabled={disabled}
        spellCheck={false}
        onChange={(e) => {
          setDraft(e.target.value);
          if (err) setErr("");
        }}
        onBlur={() => {
          const next = draft;
          setDraft(null);
          if (next === null || next === value) return;
          const msg = commit(next);
          if (msg) setErr(msg);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
          else if (e.key === "Escape") {
            setDraft(null);
            setErr("");
          }
        }}
      />
      {err ? <span className="ig-err">{err}</span> : null}
    </span>
  );
}

// ── 检查器（右）──

export interface InspectorProps {
  value: EditorValue;
  node: DcirNode;
  onClose: () => void;
  /** 返回错误文案（null = 成功提交） */
  onRename: (newName: string) => string | null;
  onPatchNode: (patch: Partial<Pick<DcirNode, "type" | "affinity">>) => void;
  onRemoveNode: () => void;
  onSetRemote: (modelKey: string) => void;
  /** SignalSource 对外信号名 */
  onSetSignalSourceName: (name: string) => void;
  /** SignalConst 常量布尔值 */
  onSetSignalConstValue: (value: boolean) => void;
  /** AND/OR 增一个信号输入 */
  onAddLogicInput: () => void;
  /** AND/OR 减一个信号输入 */
  onRemoveLogicInput: (portName: string) => void;
}

/** 端子行（只读展示；端子 Schema 经检查器/目录声明维护） */
function PortRow({
  direction,
  port,
  bound,
}: {
  direction: PortDirection;
  port: DcirPort;
  bound: boolean;
}) {
  return (
    <div className="ig-portrow">
      <div className="ig-portrow-line">
        <span className="ig-portrow-name ig-mono">{port.name}{port.required !== false ? <em>*</em> : null}</span>
        <span className="ig-portrow-type">{detailPortTag(port)}</span>
        {bound ? (
          <span
            className="ig-portrow-bound"
            title={direction === "inputs" ? "图输入（外部供数）" : "图输出（结果取回）"}
          >
            {direction === "inputs" ? "图输入 ✓" : "图输出 ✓"}
          </span>
        ) : null}
      </div>
    </div>
  );
}

export function Inspector(props: InspectorProps) {
  const { value, node } = props;
  const remote = value.remote[node.name] ?? "";
  const affinity = node.affinity ?? "Operator";
  const affinityOptions = affinity === "Operator" || affinity === "Compute" ? ["Operator", "Compute"] : [affinity, "Operator", "Compute"];
  const isSigNode = isSignalNodeType(node.type);
  const isLogicVar = node.type === SIGNAL_NODE_TYPES.and || node.type === SIGNAL_NODE_TYPES.or;
  const tensorInputs = node.inputs.filter((p) => !isSignalPort(p));
  const tensorOutputs = node.outputs.filter((p) => !isSignalPort(p));
  const signalInputs = node.inputs.filter(isSignalPort);
  const signalOutputs = node.outputs.filter(isSignalPort);

  return (
    <aside className="ig-inspector">
      <div className="ig-panel-title">
        节点
        <span className="ig-spacer" />
        <button type="button" className="ig-iconbtn" title="关闭检查器" onClick={props.onClose}>
          ×
        </button>
      </div>
      <div className="ig-field">
        <label>名称（图内唯一；对象键安全字符）</label>
        <EditableText value={node.name} commit={props.onRename} className="ig-input ig-mono" />
      </div>
      <div className="ig-field">
        <label>节点类型（node.type）</label>
        <EditableText
          value={node.type}
          commit={(v) => {
            const t = v.trim();
            if (!t) return "节点类型不能为空";
            props.onPatchNode({ type: t });
            return null;
          }}
          className="ig-input ig-mono"
        />
      </div>
      {!isSigNode && (
        <div className="ig-field">
          <label>亲和（ResourceClass）</label>
          <select className="ig-input" value={affinity} onChange={(e) => props.onPatchNode({ affinity: e.target.value })}>
            {affinityOptions.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </div>
      )}
      {!isSigNode && (
        <div className="ig-field">
          <label>模型标识 (modelKey)</label>
          <EditableText
            value={remote}
            placeholder="models/…"
            title="modelKey（/storage 产物键）"
            commit={(v) => {
              const t = v.trim();
              props.onSetRemote(t);
              return null;
            }}
            className="ig-input ig-mono"
          />
        </div>
      )}

      {node.type === SIGNAL_NODE_TYPES.source && (
        <div className="ig-field">
          <label>对外信号名（宿主 setSignal 入口）</label>
          <EditableText
            value={typeof node.signalName === "string" ? node.signalName : ""}
            placeholder="如 pause"
            title="宿主经 graph.setSignal(名, bool) 驱动本信号源"
            commit={(v) => {
              props.onSetSignalSourceName(v);
              return null;
            }}
            className="ig-input ig-mono"
          />
        </div>
      )}
      {node.type === SIGNAL_NODE_TYPES.const && (
        <div className="ig-field">
          <label>常量值</label>
          <select
            className="ig-input"
            value={node.signalValue === false ? "false" : "true"}
            onChange={(e) => props.onSetSignalConstValue(e.target.value === "true")}
          >
            <option value="true">true（放行）</option>
            <option value="false">false（阻断）</option>
          </select>
        </div>
      )}

      {(tensorInputs.length > 0 || tensorOutputs.length > 0) && (
        <>
          <div className="ig-field">
            <label>输入端子（{tensorInputs.length}）</label>
            {tensorInputs.map((p) => (
              <PortRow
                key={`in:${p.name}`}
                direction="inputs"
                port={p}
                bound={hasInputBinding(value, node.name, p.name)}
              />
            ))}
          </div>

          <div className="ig-field">
            <label>输出端子（{tensorOutputs.length}）</label>
            {tensorOutputs.map((p) => (
              <PortRow
                key={`out:${p.name}`}
                direction="outputs"
                port={p}
                bound={hasOutputBinding(value, node.name, p.name)}
              />
            ))}
          </div>
        </>
      )}

      <div className="ig-field">
        <label>信号端子（{signalInputs.length + signalOutputs.length}）</label>
        {signalInputs.map((p) => (
          <div className="ig-portrow" key={`sin:${p.name}`}>
            <div className="ig-portrow-line">
              <span className="ig-portrow-name ig-mono">{`▸ ${sigLabel(p.name)}`}</span>
              <span className="ig-portrow-type">signal in</span>
              {isLogicVar && signalInputs.length > 1 ? (
                <button
                  type="button"
                  className="ig-iconbtn ig-portrow-del"
                  title="移除该信号输入"
                  onClick={() => props.onRemoveLogicInput(p.name)}
                >
                  ×
                </button>
              ) : null}
            </div>
          </div>
        ))}
        {signalOutputs.map((p) => (
          <div className="ig-portrow" key={`sout:${p.name}`}>
            <div className="ig-portrow-line">
              <span className="ig-portrow-name ig-mono">{`◂ ${sigLabel(p.name)}`}</span>
              <span className="ig-portrow-type">signal out</span>
            </div>
          </div>
        ))}
        {isLogicVar && signalInputs.length < SIG_LOGIC_MAX_INPUTS ? (
          <button type="button" className="ig-btn" onClick={props.onAddLogicInput}>
            ＋ 增加信号输入
          </button>
        ) : null}
        {!isSigNode ? (
          <p className="ig-hint">gate 无连线 = 放行；状态端子（idle/running/blocked/error/done）由节点背后算法侧在运行期赋值。</p>
        ) : null}
      </div>

      <button type="button" className="ig-btn ig-danger" onClick={props.onRemoveNode}>
        删除节点
      </button>
    </aside>
  );
}

// ── JSON 抽屉 ──

export function JsonDrawer({
  json,
  readOnly,
  onApply,
  onClose,
}: {
  json: string;
  readOnly: boolean;
  /** 返回清洗/错误问题清单（空 = 干净导入） */
  onApply: (text: string) => EditorIssue[];
  onClose: () => void;
}) {
  const [text, setText] = useState(json);
  const [issues, setIssues] = useState<EditorIssue[] | null>(null);
  const [copied, setCopied] = useState(false);
  return (
    <div className="ig-json">
      <div className="ig-json-head">
        <span className="ig-panel-title">graph JSON（DCIr 序列化；不含坐标与远程意图）</span>
        <span className="ig-spacer" />
        <button
          type="button"
          className="ig-btn"
          onClick={() => {
            void navigator.clipboard
              .writeText(json)
              .then(() => {
                setCopied(true);
                window.setTimeout(() => setCopied(false), 1200);
              })
              .catch(() => undefined);
          }}
        >
          {copied ? "已复制 ✓" : "复制当前"}
        </button>
        <button type="button" className="ig-btn" onClick={onClose}>
          关闭
        </button>
      </div>
      <div className="ig-json-body">
        <pre className="ig-json-pre">{json}</pre>
        {!readOnly && (
          <div className="ig-json-edit">
            <label className="ig-json-label">导入 JSON（替换当前图；悬垂引用自动清洗）</label>
            <textarea
              className="ig-input ig-json-textarea"
              value={text}
              spellCheck={false}
              onChange={(e) => setText(e.target.value)}
            />
            <div className="ig-json-actions">
              <button type="button" className="ig-btn ig-primary" onClick={() => setIssues(onApply(text))}>
                应用导入
              </button>
              <button type="button" className="ig-btn" onClick={() => setText(json)}>
                重置为当前
              </button>
            </div>
            {issues && issues.length > 0 && (
              <ul className="ig-json-issues">
                {issues.map((i, idx) => (
                  <li key={idx} className={`ig-issue-${i.level}`}>{`[${i.level}] ${i.detail}`}</li>
                ))}
              </ul>
            )}
            {issues && issues.length === 0 && <p className="ig-hint">导入成功（无清洗项）</p>}
          </div>
        )}
      </div>
    </div>
  );
}

// ── 问题面板（工具栏下方浮层） ──

export interface IssuePanelProps {
  issues: EditorIssue[];
  visible: boolean;
  onClose: () => void;
  onFocusNode: (nodeName: string) => void;
}

export function IssuePanel({ issues, visible, onClose, onFocusNode }: IssuePanelProps) {
  if (!visible) return null;
  const errors = issues.filter((i) => i.level === "error");
  const warns = issues.filter((i) => i.level === "warn");
  return (
    <div className="ig-issues-panel">
      <div className="ig-issues-head">
        <span>问题 ({issues.length})</span>
        <span className="ig-spacer" />
        <button type="button" className="ig-iconbtn" title="关闭" onClick={onClose}>×</button>
      </div>
      <div className="ig-issues-list">
        {errors.length > 0 && (
          <div className="ig-issues-group">
            {errors.map((iss, idx) => (
              <div
                key={`e${idx}`}
                className="ig-issues-item level-error"
                onClick={iss.node ? () => onFocusNode(iss.node!) : undefined}
                style={iss.node ? { cursor: "pointer" } : undefined}
              >
                <span className="ig-issues-dot" />{iss.detail}
              </div>
            ))}
          </div>
        )}
        {warns.length > 0 && (
          <div className="ig-issues-group">
            {warns.map((iss, idx) => (
              <div
                key={`w${idx}`}
                className="ig-issues-item level-warn"
                onClick={iss.node ? () => onFocusNode(iss.node!) : undefined}
                style={iss.node ? { cursor: "pointer" } : undefined}
              >
                <span className="ig-issues-dot" />{iss.detail}
              </div>
            ))}
          </div>
        )}
        {issues.length === 0 && <p className="ig-hint">无问题</p>}
      </div>
    </div>
  );
}

// ── IO 元素检查器（右侧面板） ──

export interface IoInspectorProps {
  element: IoElement;
  onClose: () => void;
  onDelete: () => void;
}

export function IoInspector({ element, onClose, onDelete }: IoInspectorProps) {
  const kindLabel = element.kind === "input" ? "输入元素" : "输出元素";
  const bound = element.node !== undefined && element.port !== undefined;
  return (
    <aside className="ig-inspector">
      <div className="ig-panel-title">
        {kindLabel}
        <span className="ig-spacer" />
        <button type="button" className="ig-iconbtn" title="关闭检查器" onClick={onClose}>×</button>
      </div>
      <div className="ig-field">
        <label>绑定状态</label>
        {bound ? (
          <span className="ig-io-bound-ok">{element.node}.{element.port} ✓</span>
        ) : (
          <span className="ig-io-bound-none">未连接（拖线到节点端子即完成绑定）</span>
        )}
      </div>
      <p className="ig-hint">此元素通过连线定义角色，无需额外配置。</p>
      <button type="button" className="ig-btn ig-danger" onClick={onDelete}>
        删除元素
      </button>
    </aside>
  );
}
