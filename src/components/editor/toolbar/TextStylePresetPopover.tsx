import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import { Check, ChevronDown, Palette, Save, Trash2 } from "lucide-react";
import type { TextLayerStyle } from "../../../lib/stores/appStore";
import { resolveLegacyFontFamily } from "../../../lib/fonts";
import {
  cloneTextStylePresetPatch,
  createCustomTextStylePreset,
  mergeTextStylePresetLists,
  type EditorTextStylePreset,
  type EditorTextStylePresetPatch,
} from "../../../lib/editorTextStylePresets";
import {
  loadCustomTextStylePresets,
  saveCustomTextStylePresets,
} from "../../../lib/editorTextStylePresetStorage";

interface TextStylePresetPopoverProps {
  currentStyle: Partial<TextLayerStyle>;
  onApply: (patch: EditorTextStylePresetPatch) => void;
}

function previewTextStyle(preset: EditorTextStylePreset): CSSProperties {
  const style = preset.stylePatch;
  const gradient = style.cor_gradiente?.length
    ? `linear-gradient(180deg, ${style.cor_gradiente.join(", ")})`
    : undefined;
  const shadows: string[] = [];
  if (style.sombra) {
    const [x, y] = style.sombra_offset ?? [2, 2];
    shadows.push(`${x}px ${y}px 0 ${style.sombra_cor || "#000000"}`);
  }
  if (style.glow) {
    shadows.push(`0 0 ${Math.max(1, style.glow_px ?? 4)}px ${style.glow_cor || "#ffffff"}`);
  }

  return {
    fontFamily: resolveLegacyFontFamily(style.fonte ?? "ComicNeue-Bold.ttf"),
    fontWeight: style.bold === false ? 500 : 800,
    fontStyle: style.italico ? "italic" : "normal",
    color: gradient ? "transparent" : (style.cor || "#ffffff"),
    backgroundImage: gradient,
    WebkitBackgroundClip: gradient ? "text" : undefined,
    WebkitTextFillColor: gradient ? "transparent" : undefined,
    WebkitTextStroke: `${Math.max(0, style.contorno_px ?? 0) / 2}px ${style.contorno || "transparent"}`,
    textShadow: shadows.join(", ") || undefined,
  };
}

export function TextStylePresetPopover({ currentStyle, onApply }: TextStylePresetPopoverProps) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const [customPresets, setCustomPresets] = useState<EditorTextStylePreset[]>([]);
  const [customName, setCustomName] = useState("");
  const [saving, setSaving] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);

  const presets = useMemo(() => mergeTextStylePresetLists(customPresets), [customPresets]);

  useEffect(() => {
    if (!open || !buttonRef.current) return;
    const rect = buttonRef.current.getBoundingClientRect();
    const width = 340;
    setPos({
      left: Math.max(8, Math.min(rect.left, window.innerWidth - width - 8)),
      top: rect.bottom + 4,
    });
  }, [open]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    void loadCustomTextStylePresets().then((loaded) => {
      if (!cancelled) setCustomPresets(loaded);
    });
    return () => {
      cancelled = true;
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    function onClickOutside(event: MouseEvent) {
      const target = event.target as Node;
      if (buttonRef.current?.contains(target) || popoverRef.current?.contains(target)) return;
      setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, [open]);

  function applyPreset(preset: EditorTextStylePreset) {
    onApply(cloneTextStylePresetPatch(preset.stylePatch));
    setOpen(false);
  }

  async function createPreset() {
    setSaving(true);
    try {
      const preset = createCustomTextStylePreset(currentStyle, customName);
      const next = [...customPresets, preset];
      setCustomPresets(next);
      setCustomName("");
      await saveCustomTextStylePresets(next);
    } finally {
      setSaving(false);
    }
  }

  async function deletePreset(id: string) {
    const next = customPresets.filter((preset) => preset.id !== id);
    setCustomPresets(next);
    await saveCustomTextStylePresets(next);
  }

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        onClick={() => setOpen((value) => !value)}
        data-testid="text-style-preset-button"
        className={`flex h-7 items-center gap-1 rounded-md px-2 text-[11px] font-medium transition-smooth ${
          open
            ? "bg-brand/15 text-brand"
            : "text-text-muted hover:bg-white/[0.04] hover:text-text-primary"
        }`}
        title="Presets de estilo"
      >
        <Palette size={12} />
        Preset
        <ChevronDown size={10} className={`transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {open && pos &&
        createPortal(
          <div
            ref={popoverRef}
            data-testid="text-style-preset-popover"
            style={{ position: "fixed", left: pos.left, top: pos.top, zIndex: 9999 }}
            className="w-[340px] rounded-xl border border-border bg-bg-secondary p-3 shadow-[0_8px_32px_rgba(0,0,0,0.45)] backdrop-blur-md"
          >
            <div className="grid grid-cols-2 gap-2">
              {presets.map((preset) => (
                <div
                  key={preset.id}
                  role="button"
                  tabIndex={0}
                  data-testid={`text-style-preset-option-${preset.id}`}
                  onClick={() => applyPreset(preset)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      applyPreset(preset);
                    }
                  }}
                  className="group rounded-lg border border-border bg-bg-tertiary/50 p-2 text-left transition-smooth hover:border-brand/50 hover:bg-bg-tertiary"
                  title={`Aplicar ${preset.name}`}
                >
                  <div className="mb-1 flex h-10 items-center justify-center overflow-hidden rounded-md bg-black/30">
                    <span className="max-w-full truncate text-xl leading-none" style={previewTextStyle(preset)}>
                      Aa!
                    </span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="min-w-0 flex-1 truncate text-[11px] font-semibold text-text-primary">
                      {preset.name}
                    </span>
                    {preset.kind === "custom" && (
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          void deletePreset(preset.id);
                        }}
                        className="rounded p-0.5 text-text-muted opacity-70 hover:bg-white/[0.06] hover:text-status-error group-hover:opacity-100"
                        title="Remover preset"
                      >
                        <Trash2 size={11} />
                      </button>
                    )}
                  </div>
                  {preset.description && (
                    <p className="mt-0.5 text-[9px] leading-tight text-text-muted">
                      {preset.description}
                    </p>
                  )}
                </div>
              ))}
            </div>

            <div className="mt-3 border-t border-border pt-3">
              <label className="mb-1 block text-[10px] font-medium text-text-muted" htmlFor="text-style-preset-name">
                Criar preset do texto selecionado
              </label>
              <div className="flex gap-1.5">
                <input
                  id="text-style-preset-name"
                  data-testid="text-style-preset-name"
                  value={customName}
                  onChange={(event) => setCustomName(event.target.value)}
                  placeholder="Nome do preset"
                  className="h-7 min-w-0 flex-1 rounded-md border border-border bg-bg-tertiary/60 px-2 text-[11px] text-text-primary outline-none focus:border-brand/40"
                />
                <button
                  type="button"
                  data-testid="text-style-preset-create"
                  onClick={() => void createPreset()}
                  disabled={saving}
                  className="flex h-7 items-center gap-1 rounded-md bg-brand px-2 text-[11px] font-medium text-white disabled:cursor-wait disabled:opacity-70"
                  title="Criar preset"
                >
                  {saving ? <Check size={12} /> : <Save size={12} />}
                  Criar
                </button>
              </div>
            </div>
          </div>,
          document.body,
        )}
    </>
  );
}
