/**
 * BrushOptionsPopover — Fase 7 do refactor.
 *
 * Popover exibido inline abaixo do controle de pincel na toolbar.
 * Aparece quando tool=brush.
 *
 * Controles:
 *  - Chip de cor ativa + 8 swatches recentes + input color nativo
 *  - Slider Tamanho (4–160px)
 *  - Slider Opacidade (0–100%)
 *  - Slider Dureza (0–100%) — afeta a visualização do overlay
 */

import { useRef, useEffect, useState } from "react";
import { useEditorStore } from "../../../lib/stores/editorStore";

interface BrushOptionsPopoverProps {
  /** Coordenadas CSS do ponto de ancoragem (bottom-left do trigger). */
  anchorX: number;
  anchorY: number;
  onClose: () => void;
}

export function BrushOptionsPopover({ onClose }: BrushOptionsPopoverProps) {
  const brushColor = useEditorStore((s) => s.brushColor);
  const brushOpacity = useEditorStore((s) => s.brushOpacity);
  const brushHardness = useEditorStore((s) => s.brushHardness);
  const brushSize = useEditorStore((s) => s.brushSize);
  const recentBrushColors = useEditorStore((s) => s.recentBrushColors);
  const setBrushColor = useEditorStore((s) => s.setBrushColor);
  const setBrushOpacity = useEditorStore((s) => s.setBrushOpacity);
  const setBrushHardness = useEditorStore((s) => s.setBrushHardness);
  const setBrushSize = useEditorStore((s) => s.setBrushSize);

  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onMouseDown(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose();
      }
    }
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [onClose]);

  return (
    <div
      ref={ref}
      className="absolute z-50 mt-1 w-[220px] rounded-xl border border-border bg-bg-secondary shadow-lg p-3 space-y-3"
    >
      <ColorSection
        brushColor={brushColor}
        recentBrushColors={recentBrushColors}
        setBrushColor={setBrushColor}
      />

      {/* Tamanho */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted">Tamanho</p>
          <span className="font-mono text-[10px] text-text-secondary">{brushSize}px</span>
        </div>
        <input
          type="range"
          min={4}
          max={160}
          value={brushSize}
          onChange={(e) => setBrushSize(Number(e.target.value))}
          className="w-full"
          title="Tamanho do pincel"
        />
      </div>

      {/* Opacidade */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted">Opacidade</p>
          <span className="font-mono text-[10px] text-text-secondary">{Math.round(brushOpacity * 100)}%</span>
        </div>
        <input
          type="range"
          min={0}
          max={100}
          value={Math.round(brushOpacity * 100)}
          onChange={(e) => setBrushOpacity(Number(e.target.value) / 100)}
          className="w-full"
          title="Opacidade do pincel"
        />
      </div>

      {/* Dureza */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted">Dureza</p>
          <span className="font-mono text-[10px] text-text-secondary">{Math.round(brushHardness * 100)}%</span>
        </div>
        <input
          type="range"
          min={0}
          max={100}
          value={Math.round(brushHardness * 100)}
          onChange={(e) => setBrushHardness(Number(e.target.value) / 100)}
          className="w-full"
          title="Dureza da borda do pincel"
        />

        {/* Preview circular da dureza */}
        <div className="mt-2 flex items-center justify-center">
          <HardnessPreview size={32} hardness={brushHardness} color={brushColor} />
        </div>
      </div>
    </div>
  );
}

/**
 * Seção de cor — usa state local para preview ao vivo, só commita ao
 * store quando o picker fecha (onBlur) ou quando seleciona swatch recente.
 * Evita re-renders no canvas a cada movimento do HSV picker.
 */
function ColorSection({
  brushColor,
  recentBrushColors,
  setBrushColor,
}: {
  brushColor: string;
  recentBrushColors: string[];
  setBrushColor: (c: string) => void;
}) {
  const [draft, setDraft] = useState(brushColor);

  useEffect(() => {
    setDraft(brushColor);
  }, [brushColor]);

  return (
    <div>
      <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted mb-1.5">Cor</p>
      <div className="flex items-center gap-2">
        <div className="relative flex-shrink-0">
          <input
            type="color"
            value={draft}
            // onChange (= mudança final do picker) commita ao store
            onChange={(e) => setBrushColor(e.target.value)}
            // onInput (= drag contínuo) só atualiza preview local
            onInput={(e) => setDraft((e.target as HTMLInputElement).value)}
            onBlur={(e) => setBrushColor(e.target.value)}
            className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
            title="Escolher cor"
          />
          <div
            className="h-7 w-7 rounded-full border-2 border-white/20 shadow-sm"
            style={{ backgroundColor: draft }}
          />
        </div>

        <div className="flex flex-wrap gap-1">
          {recentBrushColors.slice(0, 8).map((c) => (
            <button
              key={c}
              onClick={() => setBrushColor(c)}
              className={`h-5 w-5 rounded-full border-2 transition-smooth ${
                c === brushColor ? "border-white/60 scale-110" : "border-white/15 hover:border-white/30"
              }`}
              style={{ backgroundColor: c }}
              title={c}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

/** Preview circular que ilustra o efeito da dureza. */
function HardnessPreview({
  size,
  hardness,
  color,
}: {
  size: number;
  hardness: number;
  color: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const r = size / 2;
    ctx.clearRect(0, 0, size, size);
    const grad = ctx.createRadialGradient(r, r, 0, r, r, r);

    const hexR = parseInt(color.slice(1, 3), 16);
    const hexG = parseInt(color.slice(3, 5), 16);
    const hexB = parseInt(color.slice(5, 7), 16);

    grad.addColorStop(0, `rgba(${hexR},${hexG},${hexB},1)`);
    // Fade ponto baseado na dureza: dureza=1 → borda dura; dureza=0 → gradiente total
    const fadeStart = hardness * 0.85;
    grad.addColorStop(fadeStart, `rgba(${hexR},${hexG},${hexB},${(1 - hardness * 0.5).toFixed(2)})`);
    grad.addColorStop(1, `rgba(${hexR},${hexG},${hexB},0)`);

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(r, r, r, 0, Math.PI * 2);
    ctx.fill();
  }, [size, hardness, color]);

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      className="rounded-full border border-border/30"
      style={{ background: "rgba(255,255,255,0.04)" }}
    />
  );
}

/**
 * Inline version — usado diretamente na toolbar quando tool=brush.
 * Não precisa de anchorX/Y, renderiza como inline na linha de controles.
 */
export function BrushOptionsInline() {
  const [open, setOpen] = useState(false);
  const brushColor = useEditorStore((s) => s.brushColor);
  const brushOpacity = useEditorStore((s) => s.brushOpacity);
  const brushSize = useEditorStore((s) => s.brushSize);
  const setBrushSize = useEditorStore((s) => s.setBrushSize);
  const triggerRef = useRef<HTMLDivElement>(null);

  return (
    <div className="relative flex items-center gap-1.5 rounded-lg border border-border bg-bg-tertiary/40 px-2 py-1">
      {/* Color swatch trigger */}
      <div ref={triggerRef} className="relative">
        <button
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-1 text-[10px] text-text-muted hover:text-text-primary transition-smooth"
          title="Opções do pincel"
        >
          <div
            className="h-4 w-4 rounded-full border border-white/20 shadow-sm"
            style={{ backgroundColor: brushColor, opacity: brushOpacity }}
          />
        </button>
        {open && (
          <div className="absolute top-full left-0 z-50 mt-1">
            <BrushOptionsPopover anchorX={0} anchorY={0} onClose={() => setOpen(false)} />
          </div>
        )}
      </div>

      <span className="text-[10px] text-text-muted">Pincel</span>
      <input
        type="range"
        min={4}
        max={96}
        value={brushSize}
        title="Tamanho do pincel"
        aria-label="Tamanho do pincel"
        onChange={(event) => setBrushSize(Number(event.target.value))}
        className="w-20"
      />
      <span className="w-7 text-right font-mono text-[10px] text-text-secondary">{brushSize}</span>
    </div>
  );
}
