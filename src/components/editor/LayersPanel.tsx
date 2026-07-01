import { useEffect, useMemo, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  FileText,
  Folder,
  Image as ImageIcon,
  Lock,
  LockOpen,
  Search,
  Wand2,
} from "lucide-react";
import { useAppStore, type PageData } from "../../lib/stores/appStore";
import { useEditorStore } from "../../lib/stores/editorStore";
import { buildEditorScene, searchTextLayers, type NormalizedTextLayer } from "../../lib/editorScene";
import { LayerItem } from "./LayerItem";

function pageTextCount(page: PageData | undefined) {
  return (page?.text_layers ?? page?.textos ?? []).length;
}

export function LayersPanel() {
  const projectPages = useAppStore((s) => s.project?.paginas ?? []);
  const currentPage = useEditorStore((s) => s.currentPage);
  const currentPageIndex = useEditorStore((s) => s.currentPageIndex);
  const setCurrentPage = useEditorStore((s) => s.setCurrentPage);
  const isLoadingPage = useEditorStore((s) => s.isLoadingPage);
  const selectedLayerId = useEditorStore((s) => s.selectedLayerId);
  const pendingEdits = useEditorStore((s) => s.pendingEdits);
  const toggleImageLayerVisibility = useEditorStore((s) => s.toggleImageLayerVisibility);
  const setImageLayerLocked = useEditorStore((s) => s.setImageLayerLocked);

  const [query, setQuery] = useState("");
  const [expandedPages, setExpandedPages] = useState<Set<number>>(() => new Set([0]));
  const totalPages = projectPages.length;

  const scene = useMemo(
    () => buildEditorScene({ page: currentPage, pendingEdits, selectedLayerId }),
    [currentPage, pendingEdits, selectedLayerId],
  );

  const filteredTextLayers = useMemo(() => {
    return searchTextLayers(scene.textLayers, query);
  }, [query, scene.textLayers]);

  useEffect(() => {
    if (!selectedLayerId) return;
    setExpandedPages((current) => {
      if (current.has(currentPageIndex)) return current;
      const next = new Set(current);
      next.add(currentPageIndex);
      return next;
    });
  }, [currentPageIndex, selectedLayerId]);

  function goToPage(pageIndex: number) {
    if (pageIndex < 0 || pageIndex >= totalPages) return;
    setExpandedPages((current) => {
      const next = new Set(current);
      next.add(pageIndex);
      return next;
    });
    if (pageIndex === currentPageIndex) return;
    void setCurrentPage(pageIndex);
  }

  function togglePage(pageIndex: number) {
    if (pageIndex !== currentPageIndex) {
      goToPage(pageIndex);
      return;
    }
    setExpandedPages((current) => {
      const next = new Set(current);
      if (next.has(pageIndex)) next.delete(pageIndex);
      else next.add(pageIndex);
      return next;
    });
  }

  async function goToPageAndSelect(pageIndex: number, layerId: string) {
    if (pageIndex !== currentPageIndex) {
      await setCurrentPage(pageIndex);
    }
    useEditorStore.getState().selectLayer(layerId);
  }

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
            {scene.imageLayers.filter((layer) => layer.hasContent || layer.key === "base" || layer.key === "rendered").map((layer) => {
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
                      {layer.visible ? <Eye size={12} /> : <EyeOff size={12} />}
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

        <div className="space-y-1">
          {projectPages.map((page, pageIndex) => {
            const isActivePage = pageIndex === currentPageIndex;
            const rawScene = isActivePage
              ? scene
              : buildEditorScene({ page, pendingEdits: {}, selectedLayerId: null });
            const pageLayers = isActivePage ? filteredTextLayers : searchTextLayers(rawScene.textLayers, query);
            const isExpanded = expandedPages.has(pageIndex);
            const count = query.trim() ? pageLayers.length : pageTextCount(page);

            if (query.trim() && pageLayers.length === 0) return null;

            return (
              <div key={`text-page-group-${page.numero ?? pageIndex + 1}-${pageIndex}`} className="overflow-hidden rounded-md border border-border/70 bg-bg-secondary/35">
                <button
                  type="button"
                  disabled={isLoadingPage}
                  onClick={() => togglePage(pageIndex)}
                  className={`flex w-full items-center gap-2 px-2 py-1.5 text-left transition-smooth ${
                    isActivePage ? "bg-brand/10 text-text-primary" : "text-text-secondary hover:bg-white/[0.04]"
                  } disabled:cursor-wait disabled:opacity-50`}
                  title={`Abrir textos da pagina ${pageIndex + 1}`}
                >
                  {isExpanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                  <Folder size={14} className={isActivePage ? "text-brand" : "text-text-muted"} />
                  <span className="min-w-0 flex-1 truncate text-[11px] font-semibold">
                    Pagina {pageIndex + 1}
                  </span>
                  <span className="rounded bg-bg-tertiary px-1.5 py-0.5 font-mono text-[10px] text-text-muted">
                    {count}
                  </span>
                </button>

                <div
                  className={`grid border-t border-border/60 bg-bg-primary/35 transition-[grid-template-rows,opacity] duration-150 ease-out ${
                    isExpanded ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
                  }`}
                >
                  <div className="min-h-0 overflow-hidden py-1">
                    {pageLayers.length === 0 ? (
                      <div className="px-7 py-2 text-[11px] text-text-muted">
                        {query.trim() ? "Sem resultados" : "Nenhum texto"}
                      </div>
                    ) : isActivePage ? (
                      <div className="space-y-0.5">
                        {pageLayers.map((entry, index) => (
                          <LayerItem key={entry.id} entry={entry} index={index + 1} hasEdits={entry.id in pendingEdits} />
                        ))}
                      </div>
                    ) : (
                      <div className="space-y-0.5">
                        {pageLayers.map((entry, index) => (
                          <InactiveTextLayerRow
                            key={entry.id}
                            entry={entry}
                            index={index + 1}
                            onClick={() => void goToPageAndSelect(pageIndex, entry.id)}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })}

          {projectPages.length === 0 && (
            <div className="rounded-lg border border-dashed border-border bg-bg-tertiary/30 px-4 py-5 text-center text-[11px] text-text-muted">
              Nenhuma pagina
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function InactiveTextLayerRow({
  entry,
  index,
  onClick,
}: {
  entry: NormalizedTextLayer;
  index: number;
  onClick: () => void;
}) {
  const displayText = entry.displayText || entry.displayOriginal || "(vazio)";
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex w-full items-center gap-2 px-7 py-1.5 text-left text-[11px] text-text-secondary transition-smooth hover:bg-white/[0.04] hover:text-text-primary"
      title="Abrir pagina e selecionar texto"
    >
      <span className="rounded bg-bg-tertiary px-1.5 py-0.5 font-mono text-[10px] text-text-muted">
        {index}
      </span>
      <FileText size={13} className="shrink-0 text-text-muted" />
      <span className="min-w-0 flex-1 truncate">{displayText.trim() || "(vazio)"}</span>
      <span className="shrink-0 rounded-full border border-border px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em] text-text-muted">
        {entry.tipo}
      </span>
    </button>
  );
}
