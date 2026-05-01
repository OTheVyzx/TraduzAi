import { useMemo, useState } from "react";
import { Check, Eye, EyeOff, Image as ImageIcon, Layers, Search, Trash2, Wand2 } from "lucide-react";
import { useEditorStore } from "../../lib/stores/editorStore";
import type { ImageLayerKey } from "../../lib/stores/appStore";
import { LayerItem } from "./LayerItem";
import { PropertyEditor } from "./PropertyEditor";

const IMAGE_LAYER_LABELS: Record<ImageLayerKey, string> = {
  base: "Base",
  mask: "Máscara",
  inpaint: "Inpaint",
  brush: "Brush",
  rendered: "Render final",
};

export function LayersPanel() {
  const {
    currentPage,
    selectedLayerId,
    selectedImageLayerKey,
    pendingEdits,
    pendingStructuralEdits,
    toggleImageLayerVisibility,
    selectImageLayer,
    deleteSelectedLayer,
    commitEdits,
  } = useEditorStore();
  const [query, setQuery] = useState("");

  const textLayers = currentPage?.text_layers ?? [];
  const imageLayers = currentPage?.image_layers ?? {};
  const structuralPendingCount =
    pendingStructuralEdits.created.length +
    Object.keys(pendingStructuralEdits.deleted).length +
    (pendingStructuralEdits.order ? 1 : 0);
  const pendingCount = Object.keys(pendingEdits).length + structuralPendingCount;
  const hasPendingEdits = pendingCount > 0;

  const filteredTextLayers = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return textLayers;
    return textLayers.filter((entry) => {
      const haystack = `${entry.traduzido ?? entry.translated ?? ""} ${entry.original} ${entry.tipo}`.toLowerCase();
      return haystack.includes(normalized);
    });
  }, [query, textLayers]);

  return (
    <div className="flex h-full w-[360px] flex-col border-l border-border bg-[linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0.01))] backdrop-blur">
      <div className="border-b border-border px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers size={14} className="text-brand" />
            <span className="text-sm font-medium">Camadas</span>
            <span className="text-xs text-text-muted">
              {filteredTextLayers.length}/{textLayers.length}
            </span>
          </div>

          <div className="flex items-center gap-1">
            <button
              onClick={() => void commitEdits()}
              disabled={!hasPendingEdits}
              className="rounded p-1 text-status-success transition-smooth hover:bg-status-success/10 disabled:opacity-35"
              title="Salvar alterações"
            >
              <Check size={14} />
            </button>
            <button
              onClick={() => void deleteSelectedLayer()}
              disabled={!selectedLayerId}
              className="rounded p-1 text-status-error transition-smooth hover:bg-status-error/10 disabled:opacity-35"
              title="Excluir camada selecionada"
            >
              <Trash2 size={14} />
            </button>
          </div>
        </div>

        <div className="relative mt-3">
          <Search size={14} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Buscar texto, tipo ou camada..."
            className="w-full rounded-xl border border-border bg-bg-tertiary py-2 pl-8 pr-3 text-sm text-text-primary outline-none transition-smooth focus:border-brand/40"
          />
        </div>

        <div className="mt-2 flex items-center justify-between text-[11px] text-text-muted">
          <span>{pendingCount} alteração(ões) pendentes</span>
          <span>
            {selectedLayerId
              ? "Texto selecionado"
              : selectedImageLayerKey
                ? `Imagem: ${IMAGE_LAYER_LABELS[selectedImageLayerKey]}`
                : "Nenhuma seleção"}
          </span>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto border-b border-border">
        <div className="px-3 py-3">
          <div className="mb-2 flex items-center gap-2 text-[11px] uppercase tracking-[0.16em] text-text-muted">
            <ImageIcon size={12} />
            Layers bitmap
          </div>
          <div className="space-y-1">
            {(Object.keys(IMAGE_LAYER_LABELS) as ImageLayerKey[]).map((key) => {
              const layer = imageLayers[key];
              const visible = layer?.visible ?? (key === "base");
              const selected = selectedImageLayerKey === key;
              return (
                <div
                  key={key}
                  className={`flex w-full items-center justify-between rounded-xl border px-3 py-2 text-left transition-smooth ${
                    selected
                      ? "border-accent-cyan/35 bg-accent-cyan/10"
                      : "border-border bg-bg-tertiary/45 hover:border-white/12"
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => selectImageLayer(key)}
                    className="min-w-0 flex-1 text-left"
                  >
                    <p className="text-sm text-text-primary">{IMAGE_LAYER_LABELS[key]}</p>
                    <p className="truncate text-[11px] text-text-muted">
                      {layer?.path?.split(/[\\/]/).pop() ?? "Sem arquivo"}
                    </p>
                  </button>

                  <button
                    type="button"
                    className="rounded p-1 text-text-secondary transition-smooth hover:bg-white/[0.03] hover:text-text-primary"
                    onClick={(event) => {
                      event.stopPropagation();
                      void toggleImageLayerVisibility(key);
                    }}
                    title={visible ? "Ocultar camada" : "Mostrar camada"}
                  >
                    {visible ? <Eye size={14} /> : <EyeOff size={14} />}
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        <div className="border-t border-border px-3 py-3">
          <div className="mb-2 flex items-center gap-2 text-[11px] uppercase tracking-[0.16em] text-text-muted">
            <Wand2 size={12} />
            Blocos de texto
          </div>
          {filteredTextLayers.length === 0 ? (
            <div className="rounded-xl border border-dashed border-white/8 bg-bg-tertiary/35 px-4 py-6 text-center text-xs text-text-muted">
              {textLayers.length === 0
                ? "Nenhuma camada de texto nesta página"
                : "Nenhum bloco encontrado para esse filtro"}
            </div>
          ) : (
            <div className="space-y-1">
              {filteredTextLayers.map((entry, index) => (
                <LayerItem key={entry.id} entry={entry} index={index + 1} />
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="flex min-h-0 flex-[0_0_44%] flex-col">
        <div className="border-b border-border px-4 py-2">
          <span className="text-xs font-medium text-text-secondary">Propriedades</span>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto">
          <PropertyEditor />
        </div>
      </div>
    </div>
  );
}
