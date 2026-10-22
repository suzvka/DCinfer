/**
 * 编辑器面板：检查器（右）/ JSON 抽屉 / 问题面板
 *
 * 重设计原则（面向用户而非测试人员）：
 * - Inspector：主层展示显示名 / 模型来源 / 端口连接状态；
 *   「高级设置」折叠区才暴露 node.type / affinity / modelKey 原始值；
 * - IssuePanel：通过 i18n.friendlyIssueDetail 将工程术语转为自然语言；
 * - IoInspector：文案改为用户引导语。
 *
 * 全部为受控展示组件：变更经回调上抛到领域层纯函数（ui/index.tsx 统一 commit）。
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
import { friendlyIssueDetail, friendlyNodeType, friendlyPortShape, friendlyPortType, portTooltip, shortModelName } from "./i18n";

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
  onRename: (newName: string) => string | null;
  onPatchNode: (patch: Partial<Pick<DcirNode, "type" | "affinity">>) => void;
  onRemoveNode: () => void;
  onSetRemote: (modelKey: string) => void;
  onSetSignalSourceName: (name: string) => void;
  onSetSignalConstValue: (value: boolean) => void;
  onAddLogicInput: () => void;
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
  const isSignal = port.kind === "signal";
  const typeLabel = friendlyPortType(port);
  const shapeLabel = friendlyPortShape(port);
  const tip = portTooltip(port, bound ? (direction === "inputs" ? " · 已设为图输入" : " · 已设为图输出") : "");
  return (
    <div className="ig-portrow" title={tip}>
      <div className="ig-portrow-line">
        <span className="ig-portrow-name ig-mono">
          {port.name}{port.required !== false && !isSignal ? <em>*</em> : null}
        </span>
        <span className="ig-portrow-type">
          {isSignal ? (
            <span className="ig-badge ig-badge-signal">信号</span>
          ) : (
            <>
              <span className="ig-badge ig-badge-type">{typeLabel}</span>
              {shapeLabel && <span className="ig-badge ig-badge-shape">{shapeLabel}</span>}
            </>
          )}
        </span>
        {bound ? (
          <span
            className="ig-portrow-bound"
            title={direction === "inputs" ? "已设为图输入（外部供数）" : "已设为图输出（结果取回）"}
          >
            {direction === "inputs" ? "图输入" : "图输出"}
          </span>
        ) : null}
      </div>
    </div>
  );
}

/** 折叠面板（「高级设置」专用） */
function Collapsible({ title, children }: { title: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="ig-collapsible">
      <button
        type="button"
        className="ig-collapsible-toggle"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="ig-collapsible-arrow" aria-hidden="true">{open ? "▾" : "▸"}</span>
        {title}
      </button>
      {open && <div className="ig-collapsible-body">{children}</div>}
    </div>
  );
}

export function Inspector(props: InspectorProps) {
  const { value, node } = props;
  const remote = value.remote[node.name] ?? "";
  const affinity = node.affinity ?? "Operator";
  const affinityOptions = affinity === "Operator" || affinity === "Compute"
    ? ["Operator", "Compute"]
    : [affinity, "Operator", "Compute"];

  const isSigNode = isSignalNodeType(node.type);
  const isLogicVar = node.type === SIGNAL_NODE_TYPES.and || node.type === SIGNAL_NODE_TYPES.or;

  const tensorInputs = node.inputs.filter((p) => !isSignalPort(p));
  const tensorOutputs = node.outputs.filter((p) => !isSignalPort(p));
  const signalInputs = node.inputs.filter(isSignalPort);
  const signalOutputs = node.outputs.filter(isSignalPort);

  // 用户可读的节点短名（模型节点用 modelKey 末段；信号节点用友好名；否则用 node.name）
  const displayName = isSigNode
    ? friendlyNodeType(node.type)
    : remote
    ? shortModelName(remote)
    : node.name;

  return (
    <aside className="ig-inspector">
      <div className="ig-panel-title">
        节点属性
        <span className="ig-spacer" />
        <button type="button" className="ig-iconbtn" title="关闭" onClick={props.onClose}>
          ×
        </button>
      </div>

      {/* ── 主层：用户直接关心的字段 ── */}
      <div className="ig-field">
        <label>显示名称</label>
        <EditableText
          value={node.name}
          commit={props.onRename}
          className="ig-input"
          placeholder="节点名称"
          title="修改节点名（仅允许字母、数字、_、. 和 -）"
        />
        {displayName !== node.name && (
          <span className="ig-field-note">来自模型：{displayName}</span>
        )}
      </div>

      {/* 信号源信号名 */}
      {node.type === SIGNAL_NODE_TYPES.source && (
        <div className="ig-field">
          <label>信号名称</label>
          <EditableText
            value={typeof node.signalName === "string" ? node.signalName : ""}
            placeholder="如：pause、ready"
            title="外部通过此名称触发本信号"
            commit={(v) => {
              props.onSetSignalSourceName(v);
              return null;
            }}
            className="ig-input"
          />
        </div>
      )}

      {/* 常量信号值 */}
      {node.type === SIGNAL_NODE_TYPES.const && (
        <div className="ig-field">
          <label>信号值</label>
          <select
            className="ig-input"
            value={node.signalValue === false ? "false" : "true"}
            onChange={(e) => props.onSetSignalConstValue(e.target.value === "true")}
          >
            <option value="true">始终放行（true）</option>
            <option value="false">始终阻断（false）</option>
          </select>
        </div>
      )}

      {/* 端口连接状态 */}
      {(tensorInputs.length > 0 || tensorOutputs.length > 0) && (
        <>
          {tensorInputs.length > 0 && (
            <div className="ig-field">
              <label>输入端口（{tensorInputs.length}）</label>
              {tensorInputs.map((p) => (
                <PortRow
                  key={`in:${p.name}`}
                  direction="inputs"
                  port={p}
                  bound={hasInputBinding(value, node.name, p.name)}
                />
              ))}
            </div>
          )}
          {tensorOutputs.length > 0 && (
            <div className="ig-field">
              <label>输出端口（{tensorOutputs.length}）</label>
              {tensorOutputs.map((p) => (
                <PortRow
                  key={`out:${p.name}`}
                  direction="outputs"
                  port={p}
                  bound={hasOutputBinding(value, node.name, p.name)}
                />
              ))}
            </div>
          )}
        </>
      )}

      {/* 信号控制端口（AND/OR 可增减） */}
      {(signalInputs.length + signalOutputs.length > 0 || isLogicVar) && (
        <div className="ig-field">
          <label>信号端口</label>
          {signalInputs.map((p) => (
            <div className="ig-portrow" key={`sin:${p.name}`}>
              <div className="ig-portrow-line">
                <span className="ig-portrow-name ig-mono">{sigLabel(p.name)}</span>
                <span className="ig-portrow-type"><span className="ig-badge ig-badge-signal">信号 输入</span></span>
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
                <span className="ig-portrow-name ig-mono">{sigLabel(p.name)}</span>
                <span className="ig-portrow-type"><span className="ig-badge ig-badge-signal">信号 输出</span></span>
              </div>
            </div>
          ))}
          {isLogicVar && signalInputs.length < SIG_LOGIC_MAX_INPUTS && (
            <button type="button" className="ig-btn" onClick={props.onAddLogicInput}>
              ＋ 增加信号输入
            </button>
          )}
        </div>
      )}

      {/* ── 高级设置（折叠） ── */}
      {!isSigNode && (
        <Collapsible title="高级设置">
          <div className="ig-field">
            <label>引擎类型（node.type）</label>
            <EditableText
              value={node.type}
              commit={(v) => {
                const t = v.trim();
                if (!t) return "引擎类型不能为空";
                props.onPatchNode({ type: t });
                return null;
              }}
              className="ig-input ig-mono"
            />
          </div>
          <div className="ig-field">
            <label>资源亲和（ResourceClass）</label>
            <select
              className="ig-input"
              value={affinity}
              onChange={(e) => props.onPatchNode({ affinity: e.target.value })}
            >
              {affinityOptions.map((a) => (
                <option key={a} value={a}>{a}</option>
              ))}
            </select>
          </div>
          <div className="ig-field">
            <label>模型标识（modelKey）</label>
            <EditableText
              value={remote}
              placeholder="models/…"
              title="模型在存储中的键值"
              commit={(v) => {
                props.onSetRemote(v.trim());
                return null;
              }}
              className="ig-input ig-mono"
            />
          </div>
          {!isSigNode && (
            <p className="ig-hint">
              gate 端口无连线时节点自动放行；状态端口（idle/running/blocked/error/done）由运行时赋值，编辑期只读。
            </p>
          )}
        </Collapsible>
      )}

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
  onApply: (text: string) => EditorIssue[];
  onClose: () => void;
}) {
  const [text, setText] = useState(json);
  const [issues, setIssues] = useState<EditorIssue[] | null>(null);
  const [copied, setCopied] = useState(false);
  return (
    <div className="ig-json">
      <div className="ig-json-head">
        <span className="ig-panel-title">图结构 JSON（DCIr 标准格式）</span>
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
          {copied ? "已复制 ✓" : "复制"}
        </button>
        <button type="button" className="ig-btn" onClick={onClose}>
          关闭
        </button>
      </div>
      <div className="ig-json-body">
        <pre className="ig-json-pre">{json}</pre>
        {!readOnly && (
          <div className="ig-json-edit">
            <label className="ig-json-label">粘贴 JSON 替换当前图（系统会自动修复可修复的问题）</label>
            <textarea
              className="ig-input ig-json-textarea"
              value={text}
              spellCheck={false}
              onChange={(e) => setText(e.target.value)}
            />
            <div className="ig-json-actions">
              <button type="button" className="ig-btn ig-primary" onClick={() => setIssues(onApply(text))}>
                应用
              </button>
              <button type="button" className="ig-btn" onClick={() => setText(json)}>
                重置
              </button>
            </div>
            {issues && issues.length > 0 && (
              <ul className="ig-json-issues">
                {issues.map((i, idx) => (
                  <li key={idx} className={`ig-issue-${i.level}`}>
                    {`[${i.level === "error" ? "错误" : "提示"}] ${friendlyIssueDetail(i)}`}
                  </li>
                ))}
              </ul>
            )}
            {issues && issues.length === 0 && <p className="ig-hint">导入成功</p>}
          </div>
        )}
      </div>
    </div>
  );
}

// ── 问题面板 ──

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
        <span>{issues.length > 0 ? `发现 ${issues.length} 个问题` : "没有问题"}</span>
        <span className="ig-spacer" />
        <button type="button" className="ig-iconbtn" title="关闭" onClick={onClose}>×</button>
      </div>
      <div className="ig-issues-list">
        {errors.length > 0 && (
          <div className="ig-issues-group">
            <div className="ig-issues-group-label">错误（{errors.length}）——需要修复才能提交</div>
            {errors.map((iss, idx) => (
              <div
                key={`e${idx}`}
                className="ig-issues-item level-error"
                onClick={iss.node ? () => onFocusNode(iss.node!) : undefined}
                style={iss.node ? { cursor: "pointer" } : undefined}
                title={iss.node ? `点击定位节点「${iss.node}」` : undefined}
              >
                <span className="ig-issues-dot" />
                <span>
                  {iss.node && <strong className="ig-issues-node">{iss.node}：</strong>}
                  {friendlyIssueDetail(iss)}
                </span>
              </div>
            ))}
          </div>
        )}
        {warns.length > 0 && (
          <div className="ig-issues-group">
            <div className="ig-issues-group-label">提示（{warns.length}）——不影响提交</div>
            {warns.map((iss, idx) => (
              <div
                key={`w${idx}`}
                className="ig-issues-item level-warn"
                onClick={iss.node ? () => onFocusNode(iss.node!) : undefined}
                style={iss.node ? { cursor: "pointer" } : undefined}
                title={iss.node ? `点击定位节点「${iss.node}」` : undefined}
              >
                <span className="ig-issues-dot" />
                <span>
                  {iss.node && <strong className="ig-issues-node">{iss.node}：</strong>}
                  {friendlyIssueDetail(iss)}
                </span>
              </div>
            ))}
          </div>
        )}
        {issues.length === 0 && <p className="ig-hint" style={{ padding: "10px 12px" }}>所有检查均通过。</p>}
      </div>
    </div>
  );
}

// ── IO 元素检查器 ──

export interface IoInspectorProps {
  element: IoElement;
  onClose: () => void;
  onDelete: () => void;
}

export function IoInspector({ element, onClose, onDelete }: IoInspectorProps) {
  const isInput = element.kind === "input";
  const bound = element.node !== undefined && element.port !== undefined;
  return (
    <aside className="ig-inspector">
      <div className="ig-panel-title">
        {isInput ? "图输入" : "图输出"}
        <span className="ig-spacer" />
        <button type="button" className="ig-iconbtn" title="关闭" onClick={onClose}>×</button>
      </div>
      <div className="ig-field">
        <label>连接状态</label>
        {bound ? (
          <div className="ig-io-bound-ok">
            <span>✓ 已连接到 {element.node}.{element.port}</span>
          </div>
        ) : (
          <div className="ig-io-bound-none">
            {isInput
              ? "尚未连接——将右侧接线点拖到某节点的输入端口即完成绑定"
              : "尚未连接——将某节点的输出端口拖到左侧接线点即完成绑定"}
          </div>
        )}
      </div>
      <p className="ig-hint">
        {isInput
          ? "图输入元素代表从外部送入推理图的数据。连接后，外部调用者可通过该入口为指定节点端口提供数据。"
          : "图输出元素代表从推理图取回结果的出口。连接后，外部调用者可从指定节点端口读取计算结果。"}
      </p>
      <button type="button" className="ig-btn ig-danger" onClick={onDelete}>
        删除
      </button>
    </aside>
  );
}
