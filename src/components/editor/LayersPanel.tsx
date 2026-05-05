import { useMemo, useState } from "react";
import { Check, Eye, EyeOff, Image as ImageIcon, Layers, Search, Trash2, Wand2 } from "lucide-react";
import { useEditorStore } from "../../lib/stores/editorStore";
import type { ImageLayerKey } from "../../lib/stores/appStore";
import { LayerItem } from "./LayerItem";

const IMAGE_LAYER_LABELS: Record<ImageLayerKey, string> = {
  base: "Base",
  mask: "Mascara",
  inpaint: "Inpaint",
  brush: "Pintura",  // Fase 7: brush = camada de pintura livre
  rendered: "Render",
};

export function LayersPanel() {
  const {
    currentPage,
    selectedLayerId,
    selectedImageLayerKey,
    pendingEdits,
    toggleImageLayerVisibility,
    selectImageLayer,
    deleteSelectedLayer,
    commitEdits,
  } = useEditorStore();
  const [query, setQuery] = useState("");

  const textLayers = currentPage?.text_layers ?? [];
  const imageLayers = currentPage?.image_layers ?? {};
  const hasPendingEdits = Object.keys(pendingEdits).length > 0;

  const filteredTextLayers = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return textLayers;
    return textLayers.filter((entry) => {
      const haystack = `${entry.traduzido ?? entry.translated ?? ""} ${entry.original} ${entry.tipo}`.toLowerCase();
      return haystack.includes(normalized);
    });
  }, [query, textLayers]);

  return (
    <div className="flex h-full w-[340px] flex-col border-l border-border bg-bg-primary">
      {/* Header */}
      <div className="border-b border-border px-4 py-2.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers size={13} className="text-brand" />
            <span className="text-[13px] font-semibold tracking-tight">Camadas</span>
            <span className="rounded-full bg-white/[0.04] px-1.5 py-0.5 text-[10px] font-mono text-text-muted">
              {filteredTextLayers.length}/{textLayers.length}
            </span>
          </div>

          <div className="flex items-center gap-0.5">
            <button
              onClick={() => void commitEdits()}
              disabled={!hasPendingEdits}
              className="rounded-md p-1.5 text-status-success transition-smooth hover:bg-status-success/8 disabled:opacity-25"
              title="Salvar alteracoes"
            >
              <Check size={13} />
            </button>
            <button
              onClick={() => void deleteSelectedLayer()}
              disabled={!selectedLayerId}
              className="rounded-md p-1.5 text-status-error transition-smooth hover:bg-status-error/8 disabled:opacity-25"
              title="Excluir camada"
            >
              <Trash2 size={13} />
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="relative mt-2">
          <Search size={12} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Buscar camada..."
            className="w-full rounded-lg border border-border bg-bg-tertiary/50 py-1.5 pl-7 pr-3 text-[11px] text-text-primary outline-none transition-smooth placeholder:text-text-muted focus:border-brand/30 focus:bg-bg-tertiary"
          />
        </div>
      </div>

      {/* Image layers */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="px-3 py-2.5">
          <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-text-muted">
            <ImageIcon size={10} />
            Bitmap
          </div>
          <div className="space-y-0.5">
            {(Object.keys(IMAGE_LAYER_LABELS) as ImageLayerKey[]).map((key) => {
              const layer = imageLayers[key];
              const visible = layer?.visible ?? (key === "base");
              const selected = selectedImageLayerKey === key;
              return (
                <div
                  key={key}
                  className={`flex w-full items-center justify-between rounded-lg border px-2.5 py-1.5 text-left transition-smooth ${
                    selected
                      ? "border-accent-cyan/25 bg-accent-cyan/8"
                      : "border-transparent hover:bg-white/[0.03]"
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => selectImageLayer(key)}
                    className="min-w-0 flex-1 text-left"
                  >
                    <p className="text-[11px] font-medium text-text-primary">{IMAGE_LAYER_LABELS[key]}</p>
                  </button>

                  <button
                    type="button"
                    className="rounded-md p-1 text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary"
                    onClick={(event) => {
                      event.stopPropagation();
                      void toggleImageLayerVisibility(key);
                    }}
                    title={visible ? "Ocultar" : "Mostrar"}
                  >
                    {visible ? <Eye size={12} /> : <EyeOff size={12} />}
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {/* Text layers */}
        <div className="border-t border-border px-3 py-2.5">
          <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-text-muted">
            <Wand2 size={10} />
            Texto
          </div>
          {filteredTextLayers.length === 0 ? (
            <div className="rounded-lg border border-dashed border-border bg-bg-tertiary/30 px-4 py-5 text-center text-[11px] text-text-muted">
              {textLayers.length === 0
                ? "Nenhuma camada de texto"
                : "Sem resultados"}
            </div>
          ) : (
            <div className="space-y-0.5">
              {filteredTextLayers.map((entry, index) => (
                <LayerItem key={entry.id} entry={entry} index={index + 1} />
              ))}
            </div>
          )}
        </div>
      </div>

    </div>
  );
}
