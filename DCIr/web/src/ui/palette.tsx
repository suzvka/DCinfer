/**
 * Palette — 调色板容器（导轨 + 浮层舞台）
 *
 * 一元化后的极简职责：导轨按 sources 渲染分组按钮；单开的浮层把舞台交给
 * active 源的 program（经 ActiveProgram 以组件语义调用，key = 组 id，
 * 切组重挂载、组内状态自持）。容器只提供 close（点击外部 / Esc 收起），
 * 对 program 返回的交互面零假设。
 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { PaletteHost, PaletteSource, SourceCtx, SourceProgram } from "./palette-protocol";

export function Palette({ sources, host }: { sources: readonly PaletteSource[]; host: PaletteHost }) {
  const [openId, setOpenId] = useState<string | null>(null);
  const rootRef = useRef<HTMLElement | null>(null);

  // 点击外部 / Esc：收起浮层
  useEffect(() => {
    if (openId === null) return;
    const onPointerDown = (e: PointerEvent) => {
      if (rootRef.current && e.target instanceof Node && rootRef.current.contains(e.target)) return;
      setOpenId(null);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpenId(null);
    };
    document.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [openId]);

  const active = openId === null ? null : (sources.find((s) => s.id === openId) ?? null);
  const ctx: SourceCtx = useMemo(() => ({ ...host, close: () => setOpenId(null) }), [host]);

  return (
    <aside className="ig-palette" ref={rootRef}>
      <div className="ig-rail">
        {sources.map((s) => (
          <button
            key={s.id}
            type="button"
            className={`ig-rail-item${openId === s.id ? " open" : ""}`}
            title={s.label}
            aria-label={s.label}
            aria-expanded={openId === s.id}
            onClick={() => setOpenId((cur) => (cur === s.id ? null : s.id))}
          >
            <span className="ig-rail-icon" aria-hidden="true">{s.icon ?? s.label.slice(0, 1)}</span>
            <span className="ig-rail-label">{s.label}</span>
          </button>
        ))}
      </div>
      {active && (
        <div className="ig-flyout">
          <ActiveProgram key={active.id} program={active.program} ctx={ctx} />
        </div>
      )}
    </aside>
  );
}

/** 以组件语义调用源程序（hooks 归属本实例；key 驱动切组重挂载） */
function ActiveProgram({ program, ctx }: { program: SourceProgram; ctx: SourceCtx }) {
  return <>{program({ ctx })}</>;
}
