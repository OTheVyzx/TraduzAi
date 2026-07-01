import { useEditorStore } from "../../lib/stores/editorStore";
import { useAppStore, type ProjectFontAssets } from "../../lib/stores/appStore";
import { buildEditorScene } from "../../lib/editorScene";
import { ensureEditorFontOptionReady, type EditorFontOption } from "../../lib/fontCatalog";
import { useTextEditSession } from "../../lib/useTextEditSession";
import { EditorFontPicker } from "./EditorFontPicker";
import {
  AlignLeft,
  AlignCenter,
  AlignRight,
  Bold,
  Italic,
  ChevronDown,
  ChevronRight,
  Search,
  RefreshCw,
  Link2Off,
} from "lucide-react";
import { useMemo, useState } from "react";

function Section({
  title,
  defaultOpen = true,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-b border-border">
      <button
        className="flex items-center gap-1.5 w-full px-4 py-2 text-[11px] font-semibold text-text-muted hover:text-text-primary transition-smooth"
        onClick={() => setOpen(!open)}
      >
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        {title}
      </button>
      {open && <div className="px-4 pb-3 space-y-2">{children}</div>}
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <label className="block text-[10px] font-medium text-text-muted mb-0.5">{children}</label>;
}

function InputField({
  value,
  onChange,
  type = "text",
  readOnly = false,
  title = "Campo de entrada",
}: {
  value: string | number;
  onChange: (val: string) => void;
  type?: "text" | "number";
  readOnly?: boolean;
  title?: string;
}) {
  return (
    <input
      type={type}
      value={value}
      title={title}
      onChange={(e) => onChange(e.target.value)}
      readOnly={readOnly}
      className={`w-full px-2 py-1.5 bg-bg-tertiary/60 border border-border rounded-md text-[11px] text-text-primary
        focus:border-brand/40 focus:outline-none transition-smooth
        ${readOnly ? "opacity-50 cursor-default" : ""}`}
    />
  );
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

export function PropertyEditor() {
  const {
    selectedLayerId,
    pendingEdits,
    currentPage,
    isRetypesetting
  } = useEditorStore();

  const commitTextBbox = useEditorStore((s) => s.commitTextBbox);
  const updateEstilo = useEditorStore((s) => s.updatePendingEstilo);
  const updateProject = useAppStore((s) => s.updateProject);
  const fontAssets = useAppStore((s) => s.project?.font_assets);
  const textSession = useTextEditSession(selectedLayerId ?? "");
  const [loadingFont, setLoadingFont] = useState<string | null>(null);

  const selectedLayer = useMemo(
    () => buildEditorScene({ page: currentPage, pendingEdits, selectedLayerId }).selectedTextLayer,
    [currentPage, pendingEdits, selectedLayerId],
  );
  const entry = selectedLayer;

  if (!entry || !selectedLayerId) {
    return (
      <div className="flex-1 flex items-center justify-center p-4">
        <p className="text-[11px] text-text-muted text-center leading-relaxed">
          Selecione uma camada<br />para editar
        </p>
      </div>
    );
  }
  const activeLayerId = selectedLayerId;

  const traduzido = textSession.value;
  const original = entry.displayOriginal;
  const estilo = entry.estilo;
  const [x1, y1, x2, y2] = entry.effectiveBbox;

  const confidencePercent = Math.round(((entry.confianca_ocr ?? entry.ocr_confidence ?? 0) || 0) * 100);
  const confidenceColor =
    confidencePercent >= 80
      ? "text-status-success"
      : confidencePercent >= 50
        ? "text-status-warning"
        : "text-status-error";

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
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto">
      {/* Header info */}
      <div className="border-b border-border px-4 py-2.5">
        <div className="flex items-center justify-between">
          <span className="rounded-md border border-border bg-bg-tertiary/50 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted">
            {entry.tipo}
          </span>
          <span className={`font-mono text-[10px] font-medium ${confidenceColor}`}>{confidencePercent}%</span>
        </div>
        <p className="mt-1.5 text-[10px] text-text-muted font-mono">
          {x1}, {y1} → {x2}, {y2}
        </p>
      </div>

      {/* Texto */}
      <Section title="Texto" defaultOpen={true}>
        <div>
          <div className="flex items-center justify-between mb-0.5">
            <Label>Original</Label>
            <span className={`text-[9px] font-mono font-medium ${confidenceColor}`}>
              {confidencePercent}%
            </span>
          </div>
          <textarea
            value={original}
            title="Texto original (somente leitura)"
            readOnly
            rows={2}
            className="w-full px-2 py-1.5 bg-bg-tertiary/40 border border-border rounded-md text-[11px]
              text-text-muted resize-none opacity-60 cursor-default"
          />
        </div>
        <div>
          <Label>Traducao</Label>
          <textarea
            value={traduzido}
            title="Texto traduzido"
            onFocus={textSession.onFocus}
            onBlur={textSession.onBlur}
            onCompositionStart={textSession.onCompositionStart}
            onCompositionEnd={textSession.onCompositionEnd}
            onChange={(e) => textSession.onChange(e.target.value)}
            onKeyDown={textSession.onKeyDown}
            rows={4}
            className="w-full px-2 py-1.5 bg-bg-tertiary/60 border border-border rounded-md text-[11px]
              text-text-primary resize-y focus:border-brand/40 focus:outline-none transition-smooth"
          />
        </div>
      </Section>

      {/* Posição */}
      <Section title="Posicao" defaultOpen={false}>
        <div className="grid grid-cols-2 gap-1.5">
          <div>
            <Label>X</Label>
            <InputField
              type="number"
              value={x1}
              onChange={(v) => {
                const nx = parseInt(v) || 0;
                const w = x2 - x1;
                commitTextBbox(selectedLayerId, [nx, y1, nx + w, y2]);
              }}
            />
          </div>
          <div>
            <Label>Y</Label>
            <InputField
              type="number"
              value={y1}
              onChange={(v) => {
                const ny = parseInt(v) || 0;
                const h = y2 - y1;
                commitTextBbox(selectedLayerId, [x1, ny, x2, ny + h]);
              }}
            />
          </div>
          <div>
            <Label>Largura</Label>
            <InputField
              type="number"
              value={x2 - x1}
              onChange={(v) => {
                const w = parseInt(v) || 1;
                commitTextBbox(selectedLayerId, [x1, y1, x1 + w, y2]);
              }}
            />
          </div>
          <div>
            <Label>Altura</Label>
            <InputField
              type="number"
              value={y2 - y1}
              onChange={(v) => {
                const h = parseInt(v) || 1;
                commitTextBbox(selectedLayerId, [x1, y1, x2, y1 + h]);
              }}
            />
          </div>
        </div>
      </Section>

      {/* Estilo */}
      <Section title="Estilo" defaultOpen={false}>
        <div>
          <Label>Fonte</Label>
          <EditorFontPicker
            value={estilo.fonte}
            loadingFont={loadingFont}
            onChange={handleFontChange}
            variant="panel"
          />
        </div>

        <div className="grid grid-cols-2 gap-1.5">
          <div>
            <Label>Tamanho</Label>
            <InputField
              type="number"
              value={estilo.tamanho}
              onChange={(v) => updateEstilo(selectedLayerId, { tamanho: parseInt(v) || 12 })}
            />
          </div>
          <div>
            <Label>Cor</Label>
            <div className="flex items-center gap-1">
              <input
                type="color"
                value={estilo.cor}
                title="Cor do texto"
                onChange={(e) => updateEstilo(selectedLayerId, { cor: e.target.value, cor_gradiente: [] })}
                className="w-6 h-6 rounded border border-border cursor-pointer bg-transparent"
              />
              <InputField
                value={estilo.cor}
                onChange={(v) => updateEstilo(selectedLayerId, { cor: v, cor_gradiente: [] })}
              />
            </div>
          </div>
        </div>

        {/* Alignment + Bold/Italic */}
        <div>
          <Label>Alinhamento</Label>
          <div className="flex gap-0.5 rounded-lg border border-border bg-bg-tertiary/30 p-0.5 w-fit">
            {(["left", "center", "right"] as const).map((align) => {
              const Icon = align === "left" ? AlignLeft : align === "center" ? AlignCenter : AlignRight;
              return (
                <button
                  key={align}
                  title={`Alinhamento ${align}`}
                  onClick={() => updateEstilo(selectedLayerId, { alinhamento: align })}
                  className={`p-1.5 rounded-md transition-smooth ${
                    estilo.alinhamento === align
                      ? "bg-brand/15 text-brand"
                      : "text-text-muted hover:text-text-primary"
                  }`}
                >
                  <Icon size={12} />
                </button>
              );
            })}

            <div className="w-px bg-border mx-0.5" />

            <button
              onClick={() => updateEstilo(selectedLayerId, { bold: !estilo.bold })}
              title="Negrito"
              className={`p-1.5 rounded-md transition-smooth ${
                estilo.bold
                  ? "bg-brand/15 text-brand"
                  : "text-text-muted hover:text-text-primary"
              }`}
            >
              <Bold size={12} />
            </button>
            <button
              onClick={() => updateEstilo(selectedLayerId, { italico: !estilo.italico })}
              title="Italico"
              className={`p-1.5 rounded-md transition-smooth ${
                estilo.italico
                  ? "bg-brand/15 text-brand"
                  : "text-text-muted hover:text-text-primary"
              }`}
            >
              <Italic size={12} />
            </button>
          </div>
        </div>
      </Section>

      {/* Efeitos */}
      <Section title="Efeitos" defaultOpen={false}>
        {/* Contorno */}
        <div>
          <Label>Contorno</Label>
          <div className="flex items-center gap-1.5">
            <input
              type="color"
              value={estilo.contorno || "#000000"}
              title="Cor do contorno"
              onChange={(e) => updateEstilo(selectedLayerId, { contorno: e.target.value })}
              className="w-6 h-6 rounded border border-border cursor-pointer bg-transparent"
            />
            <div className="flex-1">
              <InputField
                type="number"
                value={estilo.contorno_px}
                onChange={(v) => updateEstilo(selectedLayerId, { contorno_px: parseInt(v) || 0 })}
              />
            </div>
            <span className="text-[10px] text-text-muted">px</span>
          </div>
        </div>

        {/* Glow */}
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Label>Brilho</Label>
            <button
              onClick={() => updateEstilo(selectedLayerId, { glow: !estilo.glow })}
              className={`text-[9px] font-semibold px-1.5 py-0.5 rounded-md transition-smooth ${
                estilo.glow
                  ? "bg-accent-cyan/15 text-accent-cyan"
                  : "bg-bg-tertiary/50 text-text-muted"
              }`}
            >
              {estilo.glow ? "ON" : "OFF"}
            </button>
          </div>
          {estilo.glow && (
            <div className="flex items-center gap-1.5">
              <input
                type="color"
                value={estilo.glow_cor || "#FFFFFF"}
                title="Cor do brilho"
                onChange={(e) => updateEstilo(selectedLayerId, { glow_cor: e.target.value })}
                className="w-6 h-6 rounded border border-border cursor-pointer bg-transparent"
              />
              <div className="flex-1">
                <InputField
                  type="number"
                  value={estilo.glow_px}
                  onChange={(v) => updateEstilo(selectedLayerId, { glow_px: parseInt(v) || 0 })}
                />
              </div>
              <span className="text-[10px] text-text-muted">px</span>
            </div>
          )}
        </div>

        {/* Sombra */}
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Label>Sombra</Label>
            <button
              onClick={() => updateEstilo(selectedLayerId, { sombra: !estilo.sombra })}
              className={`text-[9px] font-semibold px-1.5 py-0.5 rounded-md transition-smooth ${
                estilo.sombra
                  ? "bg-accent-cyan/15 text-accent-cyan"
                  : "bg-bg-tertiary/50 text-text-muted"
              }`}
            >
              {estilo.sombra ? "ON" : "OFF"}
            </button>
          </div>
          {estilo.sombra && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5">
                <input
                  type="color"
                  value={estilo.sombra_cor || "#000000"}
                  title="Cor da sombra"
                  onChange={(e) => updateEstilo(selectedLayerId, { sombra_cor: e.target.value })}
                  className="w-6 h-6 rounded border border-border cursor-pointer bg-transparent"
                />
                <span className="text-[10px] text-text-muted">cor</span>
              </div>
              <div className="grid grid-cols-2 gap-1.5">
                <div>
                  <Label>Offset X</Label>
                  <InputField
                    type="number"
                    value={estilo.sombra_offset?.[0] ?? 0}
                    onChange={(v) =>
                      updateEstilo(selectedLayerId, {
                        sombra_offset: [parseInt(v) || 0, estilo.sombra_offset?.[1] ?? 0],
                      })
                    }
                  />
                </div>
                <div>
                  <Label>Offset Y</Label>
                  <InputField
                    type="number"
                    value={estilo.sombra_offset?.[1] ?? 0}
                    onChange={(v) =>
                      updateEstilo(selectedLayerId, {
                        sombra_offset: [estilo.sombra_offset?.[0] ?? 0, parseInt(v) || 0],
                      })
                    }
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </Section>
    </div>

    {/* Manual Actions */}
    <div className="p-3 border-t border-border space-y-2 bg-bg-secondary/30">
      <h3 className="text-[9px] font-bold text-text-muted/60 uppercase tracking-[0.14em]">
        Acoes manuais
      </h3>
      <div className="grid grid-cols-2 gap-1.5">
        <button
          onClick={() => useEditorStore.getState().reProcessBlock("ocr")}
          disabled={isRetypesetting}
          className="flex items-center justify-center gap-1.5 px-2.5 py-2 bg-indigo-500/8 hover:bg-indigo-500/15 disabled:opacity-40 text-[10px] font-medium text-indigo-300 rounded-lg border border-indigo-500/15 transition-smooth active:scale-95"
          title="Detecta o texto original dentro da area selecionada"
        >
          <Search className="w-3 h-3" />
          Ler Texto
        </button>
        <button
          onClick={() => useEditorStore.getState().reProcessBlock("inpaint")}
          disabled={isRetypesetting}
          className="flex items-center justify-center gap-1.5 px-2.5 py-2 bg-rose-500/8 hover:bg-rose-500/15 disabled:opacity-40 text-[10px] font-medium text-rose-300 rounded-lg border border-rose-500/15 transition-smooth active:scale-95"
          title="Remove o texto original e limpa o balao (Inpainting)"
        >
          <div className="w-3 h-3 border-[1.5px] border-rose-300/50 rounded-full" />
          Limpar
        </button>
        <button
          onClick={() => useEditorStore.getState().reProcessBlock("translate")}
          disabled={isRetypesetting}
          className="flex items-center justify-center gap-1.5 px-2.5 py-2 bg-emerald-500/8 hover:bg-emerald-500/15 disabled:opacity-40 text-[10px] font-medium text-emerald-300 rounded-lg border border-emerald-500/15 transition-smooth active:scale-95"
          title="Traduz o texto original para o idioma de destino"
        >
          <RefreshCw className={`w-3 h-3 ${isRetypesetting ? "animate-spin" : ""}`} />
          Traduzir
        </button>
      </div>

      {/* Disconnect group */}
      {selectedLayer?.layout_group_size && selectedLayer.layout_group_size > 1 && (
        <button
          onClick={() => useEditorStore.getState().disconnectBlock()}
          className="w-full flex items-center justify-center gap-1.5 px-2.5 py-2 bg-rose-500/8 hover:bg-rose-500/15 text-[10px] font-medium text-rose-300 rounded-lg border border-rose-500/15 transition-smooth active:scale-95"
        >
          <Link2Off className="w-3 h-3" />
          Desconectar de Grupo
        </button>
      )}
    </div>
  </div>
);
}
