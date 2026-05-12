/**
 * TypesettingBar — Fase 4 do refactor.
 *
 * Barra horizontal contextual que aparece abaixo da toolbar principal
 * quando uma camada de texto está selecionada. Substitui as seções
 * "Estilo" e "Efeitos" que ficavam no painel direito (PropertyEditor).
 *
 * Controles: Fonte | Tamanho | Cor | Alinhamento | B | I | Contorno ▾ | Sombra ▾ | Brilho ▾
 */

import { useState, useRef, useEffect } from "react";
import { createPortal } from "react-dom";
import {
  AlignLeft,
  AlignCenter,
  AlignRight,
  Bold,
  Italic,
  ChevronDown,
  RotateCcw,
  RotateCw,
} from "lucide-react";
import { useEditorStore } from "../../../lib/stores/editorStore";
import { BUNDLE_FONTS } from "../../../lib/fonts";

// Lista canônica de fontes disponíveis extraída do BUNDLE_FONTS.
// O `value` é o nome do arquivo (ex: "CCDaveGibbonsLower W00 Regular.ttf")
// que corresponde ao campo `estilo.fonte` no project.json.
const AVAILABLE_FONTS = Object.values(BUNDLE_FONTS).map((entry) => {
  // Extrai o filename do path "/fonts/<filename>.ttf"
  const path = entry.files.regular ?? Object.values(entry.files)[0] ?? "";
  const filename = path.split("/").pop() ?? path;
  return { label: entry.cssFamily, value: filename };
});

// Popover reutilizável para efeitos (Contorno / Sombra / Brilho)
function EffectPopover({
  label,
  active,
  children,
}: {
  label: string;
  active: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);

  // Recalcula posição abaixo do botão sempre que abre
  useEffect(() => {
    if (!open || !buttonRef.current) return;
    const rect = buttonRef.current.getBoundingClientRect();
    setPos({
      left: rect.left,
      top: rect.bottom + 4, // 4px gap abaixo do botão
    });
  }, [open]);

  useEffect(() => {
    if (!open) return;
    function onClickOutside(e: MouseEvent) {
      const target = e.target as Node;
      if (
        buttonRef.current?.contains(target) ||
        popoverRef.current?.contains(target)
      )
        return;
      setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, [open]);

  return (
    <>
      <button
        ref={buttonRef}
        onClick={() => setOpen((v) => !v)}
        className={`flex items-center gap-0.5 rounded-md px-2 py-1 text-[11px] font-medium transition-smooth ${
          active
            ? "bg-brand/15 text-brand"
            : "text-text-muted hover:bg-white/[0.04] hover:text-text-primary"
        }`}
        title={label}
      >
        {label}
        <ChevronDown size={10} className={`transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && pos &&
        createPortal(
          <div
            ref={popoverRef}
            style={{ position: "fixed", left: pos.left, top: pos.top, zIndex: 9999 }}
            className="min-w-[220px] rounded-xl border border-border bg-bg-secondary shadow-[0_8px_32px_rgba(0,0,0,0.45)] backdrop-blur-md p-3 space-y-2.5"
          >
            {children}
          </div>,
          document.body,
        )}
    </>
  );
}

function PopoverLabel({ children }: { children: React.ReactNode }) {
  return <label className="block text-[10px] font-medium text-text-muted mb-0.5">{children}</label>;
}

function NumInput({
  value,
  onChange,
  min = 0,
  max = 999,
  className = "",
}: {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  className?: string;
}) {
  return (
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      onChange={(e) => onChange(parseInt(e.target.value) || 0)}
      className={`w-full px-2 py-1 bg-bg-tertiary/60 border border-border rounded-md text-[11px] text-text-primary focus:border-brand/40 focus:outline-none ${className}`}
    />
  );
}

function clampRotation(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(-180, Math.min(180, Math.round(value)));
}

function colorInputValue(value: unknown, fallback: string) {
  const color = typeof value === "string" ? value.trim() : "";
  return /^#[0-9a-fA-F]{6}$/.test(color) ? color : fallback;
}

export function TypesettingBar() {
  const selectedLayerId = useEditorStore((s) => s.selectedLayerId);
  const currentPage = useEditorStore((s) => s.currentPage);
  const pendingEdits = useEditorStore((s) => s.pendingEdits);
  const updateEstilo = useEditorStore((s) => s.updatePendingEstilo);

  const selectedLayer = currentPage?.text_layers.find((t) => t.id === selectedLayerId);
  if (!selectedLayer || !selectedLayerId) return null;

  const edit = pendingEdits[selectedLayerId];
  const estilo = edit?.estilo ? { ...selectedLayer.estilo, ...edit.estilo } : (selectedLayer.estilo ?? {});

  const fonte = estilo.fonte ?? "ComicNeue-Bold.ttf";
  const tamanho = estilo.tamanho ?? 28;
  const cor = colorInputValue(estilo.cor, "#000000");
  const alinhamento = estilo.alinhamento ?? "center";
  const bold = estilo.bold ?? true;
  const italico = estilo.italico ?? false;
  const contornoPx = estilo.contorno_px ?? 0;
  const contornoCor = colorInputValue(estilo.contorno, "#000000");
  const glow = estilo.glow ?? false;
  const glowCor = colorInputValue(estilo.glow_cor, "#FFFFFF");
  const glowPx = estilo.glow_px ?? 0;
  const sombra = estilo.sombra ?? false;
  const sombraCor = colorInputValue(estilo.sombra_cor, "#000000");
  const sombraOffsetX = estilo.sombra_offset?.[0] ?? 2;
  const sombraOffsetY = estilo.sombra_offset?.[1] ?? 2;
  const rotacao = clampRotation(Number(estilo.rotacao ?? 0));

  return (
    <div className="flex items-center gap-1 border-b border-border bg-bg-secondary/80 px-3 py-1 overflow-x-auto overflow-y-visible">
      {/* Fonte */}
      <select
        value={fonte}
        title="Fonte"
        onChange={(e) => updateEstilo(selectedLayerId, { fonte: e.target.value })}
        className="h-7 rounded-md border border-border bg-bg-tertiary/60 px-2 text-[11px] text-text-primary focus:border-brand/40 focus:outline-none max-w-[160px]"
      >
        {AVAILABLE_FONTS.map((f) => (
          <option key={f.value} value={f.value}>
            {f.label}
          </option>
        ))}
      </select>

      <div className="h-4 w-px bg-border mx-0.5 shrink-0" />

      {/* Tamanho */}
      <div className="flex items-center gap-0.5">
        <button
          onClick={() => updateEstilo(selectedLayerId, { tamanho: Math.max(6, tamanho - 1) })}
          className="flex h-7 w-6 items-center justify-center rounded-md text-text-muted hover:bg-white/[0.04] hover:text-text-primary text-[12px] font-bold"
          title="Diminuir tamanho"
        >
          −
        </button>
        <input
          type="number"
          value={tamanho}
          min={6}
          max={200}
          title="Tamanho da fonte"
          onChange={(e) => updateEstilo(selectedLayerId, { tamanho: parseInt(e.target.value) || 12 })}
          className="h-7 w-12 rounded-md border border-border bg-bg-tertiary/60 text-center text-[11px] text-text-primary focus:border-brand/40 focus:outline-none [appearance:textfield]"
        />
        <button
          onClick={() => updateEstilo(selectedLayerId, { tamanho: Math.min(200, tamanho + 1) })}
          className="flex h-7 w-6 items-center justify-center rounded-md text-text-muted hover:bg-white/[0.04] hover:text-text-primary text-[12px] font-bold"
          title="Aumentar tamanho"
        >
          +
        </button>
      </div>

      <div className="h-4 w-px bg-border mx-0.5 shrink-0" />

      {/* Rotacao */}
      <div className="flex items-center gap-0.5 rounded-lg border border-border bg-bg-tertiary/30 p-0.5">
        <button
          onClick={() => updateEstilo(selectedLayerId, { rotacao: clampRotation(rotacao - 15) })}
          className="flex h-6 w-6 items-center justify-center rounded-md text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary"
          title="Girar -15 graus"
        >
          <RotateCcw size={12} />
        </button>
        <input
          type="number"
          value={rotacao}
          min={-180}
          max={180}
          title="Rotacao"
          onChange={(e) => updateEstilo(selectedLayerId, { rotacao: clampRotation(Number(e.target.value)) })}
          className="h-6 w-12 rounded-md border border-border bg-bg-tertiary/60 text-center text-[11px] text-text-primary focus:border-brand/40 focus:outline-none [appearance:textfield]"
        />
        <button
          onClick={() => updateEstilo(selectedLayerId, { rotacao: 0 })}
          className={`h-6 w-6 rounded-md text-[10px] font-semibold transition-smooth ${
            rotacao === 0 ? "bg-brand/15 text-brand" : "text-text-muted hover:bg-white/[0.04] hover:text-text-primary"
          }`}
          title="Zerar rotacao"
        >
          0
        </button>
        <button
          onClick={() => updateEstilo(selectedLayerId, { rotacao: clampRotation(rotacao + 15) })}
          className="flex h-6 w-6 items-center justify-center rounded-md text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary"
          title="Girar +15 graus"
        >
          <RotateCw size={12} />
        </button>
      </div>

      <div className="h-4 w-px bg-border mx-0.5 shrink-0" />

      {/* Cor */}
      <div className="relative flex h-7 w-7 items-center justify-center">
        <input
          type="color"
          value={cor}
          title="Cor do texto"
          onChange={(e) => updateEstilo(selectedLayerId, { cor: e.target.value })}
          className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
        />
        <div
          className="h-5 w-5 rounded-full border-2 border-white/20 shadow-sm"
          style={{ backgroundColor: cor }}
          title={`Cor: ${cor}`}
        />
      </div>

      <div className="h-4 w-px bg-border mx-0.5 shrink-0" />

      {/* Alinhamento + B + I */}
      <div className="flex items-center gap-0.5 rounded-lg border border-border bg-bg-tertiary/30 p-0.5">
        {(["left", "center", "right"] as const).map((align) => {
          const Icon = align === "left" ? AlignLeft : align === "center" ? AlignCenter : AlignRight;
          return (
            <button
              key={align}
              title={`Alinhar ${align}`}
              onClick={() => updateEstilo(selectedLayerId, { alinhamento: align })}
              className={`p-1.5 rounded-md transition-smooth ${
                alinhamento === align
                  ? "bg-brand/15 text-brand"
                  : "text-text-muted hover:text-text-primary"
              }`}
            >
              <Icon size={12} />
            </button>
          );
        })}

        <div className="w-px bg-border mx-0.5 h-4" />

        <button
          onClick={() => updateEstilo(selectedLayerId, { bold: !bold })}
          title="Negrito (B)"
          className={`p-1.5 rounded-md transition-smooth ${
            bold ? "bg-brand/15 text-brand" : "text-text-muted hover:text-text-primary"
          }`}
        >
          <Bold size={12} />
        </button>
        <button
          onClick={() => updateEstilo(selectedLayerId, { italico: !italico })}
          title="Itálico (I)"
          className={`p-1.5 rounded-md transition-smooth ${
            italico ? "bg-brand/15 text-brand" : "text-text-muted hover:text-text-primary"
          }`}
        >
          <Italic size={12} />
        </button>
      </div>

      <div className="h-4 w-px bg-border mx-0.5 shrink-0" />

      {/* Contorno ▾ */}
      <EffectPopover label="Contorno" active={contornoPx > 0}>
        <div>
          <PopoverLabel>Cor</PopoverLabel>
          <input
            type="color"
            value={contornoCor}
            title="Cor do contorno"
            onChange={(e) => updateEstilo(selectedLayerId, { contorno: e.target.value })}
            className="w-full h-7 rounded border border-border cursor-pointer bg-transparent"
          />
        </div>
        <div>
          <PopoverLabel>Espessura (px)</PopoverLabel>
          <NumInput
            value={contornoPx}
            onChange={(v) => updateEstilo(selectedLayerId, { contorno_px: v })}
            max={20}
          />
        </div>
      </EffectPopover>

      {/* Sombra ▾ */}
      <EffectPopover label="Sombra" active={sombra}>
        <div className="flex items-center justify-between">
          <PopoverLabel>Ativar sombra</PopoverLabel>
          <button
            onClick={() => updateEstilo(selectedLayerId, { sombra: !sombra })}
            className={`text-[9px] font-semibold px-2 py-0.5 rounded-md transition-smooth ${
              sombra ? "bg-accent-cyan/15 text-accent-cyan" : "bg-bg-tertiary/50 text-text-muted"
            }`}
          >
            {sombra ? "ON" : "OFF"}
          </button>
        </div>
        {sombra && (
          <>
            <div>
              <PopoverLabel>Cor</PopoverLabel>
              <input
                type="color"
                value={sombraCor}
                title="Cor da sombra"
                onChange={(e) => updateEstilo(selectedLayerId, { sombra_cor: e.target.value })}
                className="w-full h-7 rounded border border-border cursor-pointer bg-transparent"
              />
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              <div>
                <PopoverLabel>Offset X</PopoverLabel>
                <NumInput
                  value={sombraOffsetX}
                  onChange={(v) =>
                    updateEstilo(selectedLayerId, { sombra_offset: [v, sombraOffsetY] })
                  }
                  min={-50}
                  max={50}
                />
              </div>
              <div>
                <PopoverLabel>Offset Y</PopoverLabel>
                <NumInput
                  value={sombraOffsetY}
                  onChange={(v) =>
                    updateEstilo(selectedLayerId, { sombra_offset: [sombraOffsetX, v] })
                  }
                  min={-50}
                  max={50}
                />
              </div>
            </div>
          </>
        )}
      </EffectPopover>

      {/* Brilho ▾ */}
      <EffectPopover label="Brilho" active={glow}>
        <div className="flex items-center justify-between">
          <PopoverLabel>Ativar brilho</PopoverLabel>
          <button
            onClick={() => updateEstilo(selectedLayerId, { glow: !glow })}
            className={`text-[9px] font-semibold px-2 py-0.5 rounded-md transition-smooth ${
              glow ? "bg-accent-cyan/15 text-accent-cyan" : "bg-bg-tertiary/50 text-text-muted"
            }`}
          >
            {glow ? "ON" : "OFF"}
          </button>
        </div>
        {glow && (
          <>
            <div>
              <PopoverLabel>Cor</PopoverLabel>
              <input
                type="color"
                value={glowCor}
                title="Cor do brilho"
                onChange={(e) => updateEstilo(selectedLayerId, { glow_cor: e.target.value })}
                className="w-full h-7 rounded border border-border cursor-pointer bg-transparent"
              />
            </div>
            <div>
              <PopoverLabel>Intensidade (px)</PopoverLabel>
              <NumInput
                value={glowPx}
                onChange={(v) => updateEstilo(selectedLayerId, { glow_px: v })}
                max={30}
              />
            </div>
          </>
        )}
      </EffectPopover>

      {/* Tipo + confiança — info contextual */}
      <div className="ml-auto flex items-center gap-2 shrink-0 pl-2">
        <span className="rounded-md border border-border bg-bg-tertiary/50 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted">
          {selectedLayer.tipo}
        </span>
        {(() => {
          const conf = Math.round(
            ((selectedLayer.confianca_ocr ?? selectedLayer.ocr_confidence ?? 0) || 0) * 100,
          );
          const color =
            conf >= 80
              ? "text-status-success"
              : conf >= 50
                ? "text-status-warning"
                : "text-status-error";
          return (
            <span className={`font-mono text-[10px] font-medium ${color}`} title="Confiança OCR">
              {conf}%
            </span>
          );
        })()}
      </div>
    </div>
  );
}
