import { useMemo, useState } from "react";
import { Layers, Check, X, Search } from "lucide-react";
import { useAppStore } from "../../lib/stores/appStore";
import { useEditorStore } from "../../lib/stores/editorStore";
import { LayerItem } from "./LayerItem";
import { PropertyEditor } from "./PropertyEditor";

export function LayersPanel() {
  const project = useAppStore((s) => s.project);
  const { currentPageIndex, selectedLayerId, pendingEdits } = useEditorStore();
  const commitEdits = useEditorStore((s) => s.commitEdits);
  const discardEdits = useEditorStore((s) => s.discardEdits);
  const [query, setQuery] = useState("");

  const page = project?.paginas[currentPageIndex];
  const textos = page?.textos ?? [];
  const hasPendingEdits = Object.keys(pendingEdits).length > 0;

  const filteredTextos = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return textos;
    return textos.filter((entry) => {
      const preview = `${entry.traduzido} ${entry.original}`.toLowerCase();
      return preview.includes(normalized) || entry.tipo.toLowerCase().includes(normalized);
    });
  }, [query, textos]);

  return (
    <div className="flex h-full w-[340px] flex-col bg-bg-secondary">
      <div className="border-b border-white/5 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers size={14} className="text-accent-purple" />
            <span className="text-sm font-medium">Blocos</span>
            <span className="text-xs text-text-muted">
              {filteredTextos.length}/{textos.length}
            </span>
          </div>

          {hasPendingEdits && (
            <div className="flex items-center gap-1">
              <button
                onClick={() => void commitEdits()}
                className="rounded p-1 text-status-success transition-smooth hover:bg-status-success/10"
                title="Aplicar edicoes"
              >
                <Check size={14} />
              </button>
              <button
                onClick={discardEdits}
                className="rounded p-1 text-status-error transition-smooth hover:bg-status-error/10"
                title="Descartar edicoes"
              >
                <X size={14} />
              </button>
            </div>
          )}
        </div>

        <div className="relative mt-3">
          <Search size={14} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Buscar texto, tipo ou revisao..."
            className="w-full rounded-lg border border-white/5 bg-bg-tertiary py-2 pl-8 pr-3 text-sm text-text-primary outline-none transition-smooth focus:border-accent-purple/40"
          />
        </div>

        <div className="mt-2 flex items-center justify-between text-[11px] text-text-muted">
          <span>{Object.keys(pendingEdits).length} alteracao(oes) pendentes</span>
          <span>{selectedLayerId ? "Bloco selecionado" : "Nenhum bloco selecionado"}</span>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto border-b border-white/5">
        {filteredTextos.length === 0 ? (
          <div className="flex h-full items-center justify-center p-4">
            <p className="text-center text-xs text-text-muted">
              {textos.length === 0 ? "Nenhum texto detectado nesta pagina" : "Nenhum bloco encontrado para esse filtro"}
            </p>
          </div>
        ) : (
          filteredTextos.map((entry, index) => (
            <LayerItem
              key={entry.id}
              entry={entry}
              index={index + 1}
            />
          ))
        )}
      </div>

      <div className="flex min-h-0 flex-[0_0_46%] flex-col">
        <div className="border-b border-white/5 px-4 py-2">
          <span className="text-xs font-medium text-text-secondary">Propriedades</span>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto">
          <PropertyEditor />
        </div>
      </div>
    </div>
  );
}
