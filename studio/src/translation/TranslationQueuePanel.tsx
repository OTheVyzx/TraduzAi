import { useMemo, useState } from "react";
import { Check, Circle, Clock3, ListFilter, Search } from "lucide-react";
import type { StudioProject, TranslationStatus } from "../project/studioProject";
import {
  buildTranslationQueue,
  calculatePageTranslationProgress,
  calculateTranslationProgress,
  type TranslationQueueFilter,
  type TranslationQueueItem,
} from "./translationQueue";

export interface TranslationTarget {
  pageIndex: number;
  layerId: string;
}

const FILTERS: Array<[TranslationQueueFilter, string]> = [
  ["all", "Todos"],
  ["pending", "Pendentes"],
  ["review", "Revisão"],
  ["approved", "Aprovados"],
];

const STATUS_LABELS: Record<TranslationStatus, string> = {
  pending: "Pendente",
  translated: "Traduzido",
  review: "Revisão",
  approved: "Aprovado",
};

function StatusIcon({ status }: { status: TranslationStatus }) {
  if (status === "approved") return <Check size={11} />;
  if (status === "review") return <Clock3 size={11} />;
  return <Circle size={9} fill={status === "translated" ? "currentColor" : "none"} />;
}

export function TranslationQueuePanel({
  project,
  filter,
  currentPageIndex,
  selectedLayerId,
  onFilterChange,
  onSelectTarget,
}: {
  project: StudioProject;
  filter: TranslationQueueFilter;
  currentPageIndex: number;
  selectedLayerId: string | null;
  onFilterChange: (filter: TranslationQueueFilter) => void;
  onSelectTarget: (target: TranslationTarget) => void | Promise<void>;
}) {
  const [query, setQuery] = useState("");
  const [textClass, setTextClass] = useState("");
  const progress = calculateTranslationProgress(project);
  const textClasses = useMemo(() => [...new Set(project.paginas.flatMap((page) => (
    page.text_layers.map((layer) => String(layer.content_class ?? layer.tipo ?? "sem classe").trim() || "sem classe")
  )))].sort((left, right) => left.localeCompare(right, "pt-BR")), [project]);
  const queue = buildTranslationQueue(project, filter, { query, textClass });
  const pages = project.paginas.flatMap((page, pageIndex) => {
    const items = queue.filter((item) => item.pageIndex === pageIndex);
    return items.length > 0 ? [{ page, pageIndex, items }] : [];
  });

  return (
    <aside className="flex w-[224px] shrink-0 flex-col border-r border-border bg-bg-primary" aria-label="Fila de tradução">
      <header className="border-b border-border px-3 py-3">
        <div className="flex items-center gap-2">
          <ListFilter size={13} className="text-accent-cyan" />
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-secondary">Fila de tradução</h2>
          <span className="ml-auto font-mono text-[10px] text-accent-cyan">{progress.percentage}%</span>
        </div>
        <div className="mt-2 h-1 overflow-hidden rounded-full bg-white/[0.05]">
          <div className="h-full rounded-full bg-accent-cyan transition-all" style={{ width: `${progress.percentage}%` }} />
        </div>
        <p className="mt-1.5 text-[9px] text-text-muted">{progress.completed} de {progress.total} blocos encaminhados</p>
      </header>

      <div className="grid grid-cols-2 gap-1 border-b border-border p-2">
        {FILTERS.map(([value, label]) => (
          <button
            key={value}
            type="button"
            aria-pressed={filter === value}
            onClick={() => onFilterChange(value)}
            className={`rounded-md px-2 py-1.5 text-[9px] font-medium transition ${
              filter === value ? "bg-accent-cyan/12 text-accent-cyan" : "text-text-muted hover:bg-white/[0.04] hover:text-text-primary"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="space-y-1.5 border-b border-border p-2">
        <label className="flex items-center gap-1.5 rounded-md border border-border bg-bg-secondary px-2 py-1.5 focus-within:border-accent-cyan/50">
          <Search size={11} className="shrink-0 text-text-muted" />
          <input
            aria-label="Buscar texto na fila"
            value={query}
            onChange={(event) => setQuery(event.currentTarget.value)}
            placeholder="Buscar origem, tradução ou motivo"
            className="min-w-0 flex-1 bg-transparent text-[9px] text-text-primary outline-none placeholder:text-text-muted"
          />
        </label>
        <select
          aria-label="Filtrar por classe"
          value={textClass}
          onChange={(event) => setTextClass(event.currentTarget.value)}
          className="w-full rounded-md border border-border bg-bg-secondary px-2 py-1.5 text-[9px] text-text-secondary outline-none focus:border-accent-cyan/50"
        >
          <option value="">Todas as classes</option>
          {textClasses.map((value) => <option key={value} value={value}>{value}</option>)}
        </select>
        {(query || textClass) && <p className="px-1 text-[9px] text-text-muted">{queue.length} resultado(s) neste filtro</p>}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {pages.length === 0 ? (
          <p className="rounded-lg border border-dashed border-border px-3 py-6 text-center text-[10px] leading-4 text-text-muted">
            Nenhum bloco neste filtro.
          </p>
        ) : pages.map(({ page, pageIndex, items }) => {
          const pageProgress = calculatePageTranslationProgress(page);
          return (
            <section key={`${page.numero}-${pageIndex}`} className="mb-3" aria-label={`Página ${page.numero}`}>
              <div className="mb-1.5 flex items-center gap-2 px-1">
                <h3 className={`text-[10px] font-semibold ${currentPageIndex === pageIndex ? "text-accent-cyan" : "text-text-secondary"}`}>
                  Página {page.numero}
                </h3>
                <span className="ml-auto font-mono text-[9px] text-text-muted">{pageProgress.completed}/{pageProgress.total}</span>
              </div>
              <div className="space-y-1">
                {items.map((item) => (
                  <QueueItem
                    key={`${item.pageIndex}:${item.layerId}`}
                    item={item}
                    selected={currentPageIndex === item.pageIndex && selectedLayerId === item.layerId}
                    onSelect={() => onSelectTarget({ pageIndex: item.pageIndex, layerId: item.layerId })}
                  />
                ))}
              </div>
            </section>
          );
        })}
      </div>
    </aside>
  );
}

function QueueItem({ item, selected, onSelect }: { item: TranslationQueueItem; selected: boolean; onSelect: () => void | Promise<void> }) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      onClick={() => void onSelect()}
      className={`w-full rounded-lg border px-2 py-2 text-left transition ${
        selected
          ? "border-accent-cyan/35 bg-accent-cyan/10 shadow-[inset_2px_0_0_rgba(34,211,238,0.8)]"
          : "border-transparent bg-white/[0.025] hover:border-white/[0.08] hover:bg-white/[0.04]"
      }`}
    >
      <div className="flex items-center gap-1.5">
        <span className={`inline-flex h-4 w-4 items-center justify-center rounded-full ${
          item.status === "approved" ? "bg-status-success/15 text-status-success" :
          item.status === "review" ? "bg-status-warning/15 text-status-warning" :
          item.status === "translated" ? "bg-brand/15 text-brand" : "bg-white/[0.04] text-text-muted"
        }`}>
          <StatusIcon status={item.status} />
        </span>
        <span className="truncate text-[10px] font-medium text-text-primary">{item.original || `Bloco ${item.blockIndex + 1}`}</span>
      </div>
      <div className="mt-1 flex items-center gap-2 pl-[22px]">
        <span className="truncate text-[9px] text-text-muted">{item.translated || "Sem tradução"}</span>
        <span className="ml-auto shrink-0 text-[8px] text-text-muted/70">{item.textClass}</span>
        <span className="shrink-0 text-[8px] uppercase tracking-wide text-text-muted/70">{STATUS_LABELS[item.status]}</span>
      </div>
      {(item.notes || item.qaFlags.length > 0) && <p className="mt-1 truncate pl-[22px] text-[8px] text-status-warning/80">
        {[item.notes, ...item.qaFlags].filter(Boolean).join(" · ")}
      </p>}
    </button>
  );
}
