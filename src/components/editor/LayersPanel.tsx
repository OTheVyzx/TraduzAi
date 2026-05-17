import { useMemo, useState } from "react";
import { Check, Image as ImageIcon, Lock, LockOpen, Search, Trash2, Wand2 } from "lucide-react";
import { useEditorStore } from "../../lib/stores/editorStore";
import { buildEditorScene, searchTextLayers } from "../../lib/editorScene";
import { LayerItem } from "./LayerItem";

export function LayersPanel() {
  const currentPage = useEditorStore((s) => s.currentPage);
  const selectedLayerId = useEditorStore((s) => s.selectedLayerId);
  const pendingEdits = useEditorStore((s) => s.pendingEdits);
  const deleteSelectedLayer = useEditorStore((s) => s.deleteSelectedLayer);
  const commitEdits = useEditorStore((s) => s.commitEdits);
  const toggleImageLayerVisibility = useEditorStore((s) => s.toggleImageLayerVisibility);
  const setImageLayerLocked = useEditorStore((s) => s.setImageLayerLocked);

  const [query, setQuery] = useState("");

  const scene = useMemo(
    () => buildEditorScene({ page: currentPage, pendingEdits, selectedLayerId }),
    [currentPage, pendingEdits, selectedLayerId],
  );
  const hasPendingEdits = Object.keys(pendingEdits).length > 0;

  const filteredTextLayers = useMemo(() => {
    return searchTextLayers(scene.textLayers, query);
  }, [query, scene.textLayers]);

  return (
    <div className="flex h-full w-[340px] flex-col border-l border-border bg-bg-primary">
      <div className="border-b border-border px-4 py-2.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Wand2 size={13} className="text-brand" />
            <span className="text-[13px] font-semibold tracking-tight">Textos</span>
            <span className="rounded-full bg-white/[0.04] px-1.5 py-0.5 text-[10px] font-mono text-text-muted">
              {filteredTextLayers.length}/{scene.textLayers.length}
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
              title="Excluir texto"
            >
              <Trash2 size={13} />
            </button>
          </div>
        </div>
        <div className="relative mt-2">
          <Search size={12} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Buscar texto..."
            className="w-full rounded-lg border border-border bg-bg-tertiary/50 py-1.5 pl-7 pr-3 text-[11px] text-text-primary outline-none transition-smooth placeholder:text-text-muted focus:border-brand/30 focus:bg-bg-tertiary"
          />
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-2.5">
        <div className="mb-3 rounded-lg border border-border bg-bg-secondary/40 p-2.5">
          <div className="mb-2 flex items-center gap-2">
            <ImageIcon size={12} className="text-text-muted" />
            <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-text-muted">
              Camadas da pagina
            </span>
          </div>
          <div className="space-y-1">
            {scene.imageLayers.map((layer) => {
              const status = layer.visible ? "visivel" : "oculta";
              return (
                <div
                  key={layer.key}
                  className="flex items-center justify-between rounded-md border border-border/70 bg-bg-tertiary/35 px-2 py-1.5"
                  title={layer.hasContent ? `Camada ${layer.key}` : `Camada ${layer.key} sem conteudo`}
                >
                  <div className="min-w-0">
                    <p className="truncate text-[11px] font-medium text-text-secondary">
                      Camada {layer.key}
                    </p>
                    <p className="text-[10px] text-text-muted">{layer.hasContent ? status : "sem conteudo"}</p>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      disabled={!layer.hasContent}
                      onClick={() => {
                        void toggleImageLayerVisibility(layer.key).catch((error) => {
                          console.error("Erro ao alternar visibilidade da camada:", error);
                        });
                      }}
                      className="rounded p-1 text-text-muted transition-smooth hover:bg-white/[0.06] hover:text-text-primary disabled:opacity-25"
                      title={status}
                    >
                      <span className="sr-only">{status}</span>
                      <span className={`block h-1.5 w-1.5 rounded-full ${layer.visible ? "bg-status-success" : "bg-text-muted"}`} />
                    </button>
                    <button
                      disabled={!layer.hasContent}
                      onClick={() => {
                        void setImageLayerLocked(layer.key, !layer.locked).catch((error) => {
                          console.error("Erro ao alternar bloqueio da camada:", error);
                        });
                      }}
                      className="rounded p-1 text-text-muted transition-smooth hover:bg-white/[0.06] hover:text-text-primary disabled:opacity-25"
                      title={layer.locked ? "bloqueada" : "desbloqueada"}
                    >
                      {layer.locked ? <Lock size={12} /> : <LockOpen size={12} />}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {filteredTextLayers.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border bg-bg-tertiary/30 px-4 py-5 text-center text-[11px] text-text-muted">
            {scene.textLayers.length === 0 ? "Nenhum texto" : "Sem resultados"}
          </div>
        ) : (
          <div className="space-y-0.5">
            {filteredTextLayers.map((entry, index) => (
              <LayerItem key={entry.id} entry={entry} index={index + 1} hasEdits={entry.id in pendingEdits} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
