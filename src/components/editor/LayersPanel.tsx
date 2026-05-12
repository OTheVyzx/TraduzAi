import { useMemo, useState } from "react";
import { Check, Search, Trash2, Wand2 } from "lucide-react";
import { useEditorStore } from "../../lib/stores/editorStore";
import { LayerItem } from "./LayerItem";

export function LayersPanel() {
  const currentPage = useEditorStore((s) => s.currentPage);
  const selectedLayerId = useEditorStore((s) => s.selectedLayerId);
  const pendingEdits = useEditorStore((s) => s.pendingEdits);
  const deleteSelectedLayer = useEditorStore((s) => s.deleteSelectedLayer);
  const commitEdits = useEditorStore((s) => s.commitEdits);

  const [query, setQuery] = useState("");

  const textLayers = currentPage?.text_layers ?? [];
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
      <div className="border-b border-border px-4 py-2.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Wand2 size={13} className="text-brand" />
            <span className="text-[13px] font-semibold tracking-tight">Textos</span>
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
        {filteredTextLayers.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border bg-bg-tertiary/30 px-4 py-5 text-center text-[11px] text-text-muted">
            {textLayers.length === 0 ? "Nenhum texto" : "Sem resultados"}
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
  );
}
