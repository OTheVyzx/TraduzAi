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
import { useAppStore, type ProjectFontAssets } from "../../../lib/stores/appStore";
import { ensureEditorFontOptionReady, type EditorFontOption } from "../../../lib/fontCatalog";
import { resolveEditorTextStyle } from "../../../lib/editorTextStyleResolver";
import { EditorFontPicker } from "../EditorFontPicker";
import { TextStylePresetPopover } from "./TextStylePresetPopover";

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
            data-editor-preserve-text-selection="true"
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

function defaultGradientEnd(startColor: string) {
  return startColor.toLowerCase() === "#ffffff" ? "#000000" : "#ffffff";
}

function gradientAngleFromPoint(clientX: number, clientY: number, rect: DOMRect) {
  const angle = Math.atan2(clientY - (rect.top + rect.height / 2), clientX - (rect.left + rect.width / 2)) * 180 / Math.PI;
  return Math.round(angle);
}

function gradientDirectionLine(angle: number) {
  const width = 240;
  const height = 48;
  const radians = angle * Math.PI / 180;
  const dx = Math.cos(radians);
  const dy = Math.sin(radians);
  const distance = Math.min(
    Math.abs(dx) < 0.0001 ? Number.POSITIVE_INFINITY : (width / 2 - 14) / Math.abs(dx),
    Math.abs(dy) < 0.0001 ? Number.POSITIVE_INFINITY : (height / 2 - 8) / Math.abs(dy),
  );
  return {
    x1: width / 2 - dx * distance,
    y1: height / 2 - dy * distance,
    x2: width / 2 + dx * distance,
    y2: height / 2 + dy * distance,
  };
}

function upsertSystemFontAsset(assets: ProjectFontAssets | undefined, option: EditorFontOption): ProjectFontAssets {
  if (option.source !== "system" || !option.localPath) return assets ?? {};
  return {
    ...(assets ?? {}),
    system: {
      ...(assets?.system ?? {}),
      [option.value]: {
        family: option.cssFamily,
        path: option.localPath,
        weight: option.variant ?? "400",
        style: option.style ?? "normal",
      },
    },
  };
}

export function TypesettingBar() {
  const selectedLayerId = useEditorStore((s) => s.selectedLayerId);
  const currentPage = useEditorStore((s) => s.currentPage);
  const pendingEdits = useEditorStore((s) => s.pendingEdits);
  const updateEstilo = useEditorStore((s) => s.updatePendingEstilo);
  const updateProject = useAppStore((s) => s.updateProject);
  const fontAssets = useAppStore((s) => s.project?.font_assets);
  const [loadingFont, setLoadingFont] = useState<string | null>(null);
  const gradientDirectionRef = useRef<HTMLDivElement>(null);

  const selectedLayer = currentPage?.text_layers.find((t) => t.id === selectedLayerId);
  if (!selectedLayer || !selectedLayerId) return null;
  const activeLayerId = selectedLayerId;

  const edit = pendingEdits[selectedLayerId];
  const estilo = edit?.estilo ? { ...selectedLayer.estilo, ...edit.estilo } : (selectedLayer.estilo ?? {});
  const resolvedStyle = resolveEditorTextStyle(estilo);

  const fonte = estilo.fonte ?? "ComicNeue-Bold.ttf";
  const tamanho = estilo.tamanho ?? 28;
  const cor = colorInputValue(estilo.cor, "#000000");
  const gradientColors = Array.isArray(estilo.cor_gradiente) ? estilo.cor_gradiente : [];
  const resolvedGradient = resolvedStyle.fills.find((fill) => fill.type === "linear-gradient");
  const gradientActive = estilo.cor_gradiente_ativo === false
    ? false
    : gradientColors.length >= 2 || Boolean(resolvedGradient);
  const gradientStart = colorInputValue(
    gradientColors[0] ?? (resolvedGradient?.type === "linear-gradient" ? resolvedGradient.stops[0]?.color : undefined),
    cor,
  );
  const gradientEnd = colorInputValue(
    gradientColors[1] ?? (resolvedGradient?.type === "linear-gradient" ? resolvedGradient.stops[resolvedGradient.stops.length - 1]?.color : undefined),
    defaultGradientEnd(gradientStart),
  );
  const rawGradientAngle = Number(estilo.cor_gradiente_angulo ?? (resolvedGradient?.type === "linear-gradient" ? resolvedGradient.angle : 90));
  const gradientAngle = Number.isFinite(rawGradientAngle) ? Math.max(-180, Math.min(180, rawGradientAngle)) : 90;
  const gradientLine = gradientDirectionLine(gradientAngle);
  const alinhamento = estilo.alinhamento ?? "center";
  const bold = estilo.bold ?? true;
  const italico = estilo.italico ?? false;
  const contornoPx = estilo.contorno_px ?? 0;
  const contornoCor = colorInputValue(estilo.contorno, "#000000");
  const contornoAtivo = estilo.contorno_ativo ?? (contornoPx > 0 || resolvedStyle.strokes.length > 0);
  const glow = estilo.glow ?? false;
  const glowCor = colorInputValue(estilo.glow_cor, "#FFFFFF");
  const glowPx = estilo.glow_px ?? 0;
  const sombra = estilo.sombra === true || (estilo.sombra_blur === undefined && resolvedStyle.effects.dropShadows.length > 0);
  const sombraCor = colorInputValue(estilo.sombra_cor ?? resolvedStyle.effects.dropShadows[0]?.color, "#000000");
  const sombraOffsetX = estilo.sombra_offset?.[0] ?? 2;
  const sombraOffsetY = estilo.sombra_offset?.[1] ?? 2;
  const rawShadowBlur = Number(estilo.sombra_blur ?? resolvedStyle.effects.dropShadows[0]?.blur ?? 8);
  const sombraBlur = Number.isFinite(rawShadowBlur) ? Math.max(0, Math.min(30, rawShadowBlur)) : 8;
  const rotacao = clampRotation(Number(estilo.rotacao ?? 0));

  const setGradientAngleFromPointer = (clientX: number, clientY: number) => {
    const bounds = gradientDirectionRef.current?.getBoundingClientRect();
    if (!bounds) return;
    updateEstilo(selectedLayerId, {
      cor: gradientStart,
      cor_gradiente: [gradientStart, gradientEnd],
      cor_gradiente_ativo: true,
      cor_gradiente_angulo: gradientAngleFromPoint(clientX, clientY, bounds),
    });
  };

  async function handleFontChange(value: string, option?: EditorFontOption) {
    setLoadingFont(value);
    try {
      const prepared = await ensureEditorFontOptionReady(option ?? value);
      if (prepared?.source === "system") {
        updateProject({ font_assets: upsertSystemFontAsset(fontAssets, prepared) });
      }
      updateEstilo(activeLayerId, { fonte: value });
    } catch (error) {
      console.warn("[fonts] falha ao preparar fonte do editor:", error);
    } finally {
      setLoadingFont(null);
    }
  }

  return (
    <div
      data-editor-preserve-text-selection="true"
      className="flex items-center gap-1 border-b border-border bg-bg-secondary/80 px-3 py-1 overflow-x-auto overflow-y-visible"
    >
      {/* Fonte */}
      <EditorFontPicker
        value={fonte}
        loadingFont={loadingFont}
        onChange={handleFontChange}
        variant="toolbar"
        selectTestId="text-font-select"
      />

      <TextStylePresetPopover
        currentStyle={estilo}
        onApply={(stylePatch) => updateEstilo(selectedLayerId, stylePatch)}
      />

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
          onChange={(e) => updateEstilo(selectedLayerId, { cor: e.target.value, cor_gradiente: [], cor_gradiente_ativo: false })}
          className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
        />
        <div
          className="h-5 w-5 rounded-full border-2 border-white/20 shadow-sm"
          style={{ backgroundColor: cor }}
          title={`Cor: ${cor}`}
        />
      </div>

      {/* Gradiente */}
      <EffectPopover label="Gradiente" active={gradientActive}>
        <div
          data-testid="text-gradient-preview"
          data-gradient-active={String(gradientActive)}
          className="h-8 rounded-md border border-border"
          style={{ background: `linear-gradient(${gradientAngle + 90}deg, ${gradientStart}, ${gradientEnd})` }}
          title="Preview do gradiente"
        />
        <div>
          <PopoverLabel>Direção do gradiente</PopoverLabel>
          <div
            ref={gradientDirectionRef}
            data-testid="text-gradient-direction"
            role="slider"
            aria-label="Direção do gradiente"
            aria-valuemin={-180}
            aria-valuemax={180}
            aria-valuenow={gradientAngle}
            tabIndex={0}
            onPointerDown={(event) => {
              event.currentTarget.setPointerCapture(event.pointerId);
              setGradientAngleFromPointer(event.clientX, event.clientY);
            }}
            onPointerMove={(event) => {
              if (event.buttons === 1) setGradientAngleFromPointer(event.clientX, event.clientY);
            }}
            onKeyDown={(event) => {
              if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
              event.preventDefault();
              updateEstilo(selectedLayerId, {
                cor_gradiente_angulo: Math.max(-180, Math.min(180, gradientAngle + (event.key === "ArrowRight" ? 5 : -5))),
              });
            }}
            className="relative h-12 touch-none cursor-crosshair overflow-hidden rounded-md border border-border bg-bg-tertiary/70 outline-none focus:border-brand/50"
            title="Arraste a linha para mudar a direção"
          >
            <svg viewBox="0 0 240 48" className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
              <line x1={gradientLine.x1} y1={gradientLine.y1} x2={gradientLine.x2} y2={gradientLine.y2} stroke="white" strokeOpacity="0.9" strokeWidth="2" />
              <circle cx={gradientLine.x1} cy={gradientLine.y1} r="5" fill={gradientStart} stroke="white" strokeWidth="2" />
              <circle cx={gradientLine.x2} cy={gradientLine.y2} r="5" fill={gradientEnd} stroke="white" strokeWidth="2" />
            </svg>
            <span className="pointer-events-none absolute bottom-1 right-1 rounded bg-bg-primary/75 px-1 text-[9px] text-text-muted">{gradientAngle}°</span>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-1.5">
          <div>
            <PopoverLabel>Cor inicial</PopoverLabel>
            <input
              data-testid="text-gradient-start"
              type="color"
              value={gradientStart}
              title="Cor inicial do gradiente"
              onChange={(e) =>
                updateEstilo(selectedLayerId, {
                  cor: e.target.value,
                  cor_gradiente: [e.target.value, gradientEnd],
                  cor_gradiente_ativo: true,
                  cor_gradiente_angulo: gradientAngle,
                })
              }
              className="w-full h-7 rounded border border-border cursor-pointer bg-transparent"
            />
          </div>
          <div>
            <PopoverLabel>Cor final</PopoverLabel>
            <input
              data-testid="text-gradient-end"
              type="color"
              value={gradientEnd}
              title="Cor final do gradiente"
              onChange={(e) =>
                updateEstilo(selectedLayerId, {
                  cor: gradientStart,
                  cor_gradiente: [gradientStart, e.target.value],
                  cor_gradiente_ativo: true,
                  cor_gradiente_angulo: gradientAngle,
                })
              }
              className="w-full h-7 rounded border border-border cursor-pointer bg-transparent"
            />
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            data-testid="text-gradient-enable"
            onClick={() =>
              updateEstilo(selectedLayerId, {
                cor: gradientStart,
                cor_gradiente: [gradientStart, gradientEnd],
                cor_gradiente_ativo: true,
                cor_gradiente_angulo: gradientAngle,
              })
            }
            className="flex-1 rounded-md bg-brand px-2 py-1 text-[10px] font-semibold text-white"
          >
            Aplicar
          </button>
          <button
            type="button"
            data-testid="text-gradient-clear"
            onClick={() => updateEstilo(selectedLayerId, { cor: gradientStart, cor_gradiente: [], cor_gradiente_ativo: false })}
            className="rounded-md border border-border px-2 py-1 text-[10px] font-semibold text-text-muted hover:text-text-primary"
          >
            Solido
          </button>
        </div>
      </EffectPopover>

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
      <EffectPopover label="Contorno" active={contornoAtivo}>
        <div className="flex items-center justify-between">
          <PopoverLabel>Ativar contorno</PopoverLabel>
          <button
            type="button"
            data-testid="text-outline-toggle"
            onClick={() => updateEstilo(selectedLayerId, {
              contorno_ativo: !contornoAtivo,
              ...(!contornoAtivo && contornoPx <= 0 ? { contorno_px: 2 } : {}),
            })}
            className={`text-[9px] font-semibold px-2 py-0.5 rounded-md transition-smooth ${
              contornoAtivo ? "bg-accent-cyan/15 text-accent-cyan" : "bg-bg-tertiary/50 text-text-muted"
            }`}
          >
            {contornoAtivo ? "ON" : "OFF"}
          </button>
        </div>
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
            onClick={() => updateEstilo(selectedLayerId, { sombra: !sombra, sombra_blur: sombraBlur })}
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
            <label className="block">
              <div className="mb-1 flex items-center justify-between">
                <PopoverLabel>Suavidade (smooth)</PopoverLabel>
                <span className="text-[9px] text-text-muted">{sombraBlur}px</span>
              </div>
              <input
                data-testid="text-shadow-smoothness"
                type="range"
                min={0}
                max={30}
                value={sombraBlur}
                onChange={(event) => updateEstilo(selectedLayerId, { sombra_blur: Number(event.target.value) })}
                className="w-full accent-[rgb(var(--color-brand))]"
              />
            </label>
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
