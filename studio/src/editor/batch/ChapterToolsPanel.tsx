import { useMemo, useState } from "react";
import {
  BookOpenText,
  Check,
  ClipboardCopy,
  ClipboardPaste,
  Redo2,
  Search,
  Undo2,
  X,
} from "lucide-react";
import type { StudioProject } from "../../project/studioProject";
import { useStudioProjectStore } from "../../store/projectStore";
import {
  buildChapterReviewQueue,
  copyStyleFromLayer,
  createApplyStyleCommand,
  createReplaceTextCommand,
  createResolveReviewCommand,
  previewChapterReplacements,
  type ChapterStyleClipboard,
} from "./chapterCommands";

export function ChapterToolsPanel({
  project,
  currentPageIndex,
  selectedLayerId,
  onPrepareProject,
  onNavigateToLayer,
  openByDefault = false,
}: {
  project: StudioProject;
  currentPageIndex: number;
  selectedLayerId: string | null;
  onPrepareProject: () => Promise<StudioProject>;
  onNavigateToLayer: (pageIndex: number, layerId: string) => Promise<void>;
  openByDefault?: boolean;
}) {
  const [open, setOpen] = useState(openByDefault);
  const [clipboard, setClipboard] = useState<ChapterStyleClipboard | null>(null);
  const [query, setQuery] = useState("");
  const [replacement, setReplacement] = useState("");
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [wholeWord, setWholeWord] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const executeCommand = useStudioProjectStore((state) => state.executeChapterCommand);
  const undoChapterCommand = useStudioProjectStore((state) => state.undoChapterCommand);
  const redoChapterCommand = useStudioProjectStore((state) => state.redoChapterCommand);
  const historyIndex = useStudioProjectStore((state) => state.chapterHistoryIndex);
  const historyLength = useStudioProjectStore((state) => state.chapterHistory.length);

  const replacementPreview = useMemo(() => {
    if (!query) return [];
    return previewChapterReplacements(project, {
      query,
      replacement,
      caseSensitive,
      wholeWord,
    });
  }, [caseSensitive, project, query, replacement, wholeWord]);
  const reviewQueue = useMemo(() => buildChapterReviewQueue(project), [project]);
  const occurrenceCount = replacementPreview.reduce((total, item) => total + item.occurrences, 0);

  const run = async (operation: () => Promise<string | null>) => {
    if (busy) return;
    setBusy(true);
    setMessage(null);
    try {
      setMessage(await operation());
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };

  const copySelectedStyle = () => run(async () => {
    if (!selectedLayerId) throw new Error("Selecione uma camada de texto");
    const latest = await onPrepareProject();
    setClipboard(copyStyleFromLayer(latest, { pageIndex: currentPageIndex, layerId: selectedLayerId }));
    return "Estilo copiado";
  });

  const applyStyle = (scope: "selection" | "chapter") => run(async () => {
    if (!clipboard) throw new Error("Copie um estilo primeiro");
    const latest = await onPrepareProject();
    const targets = scope === "chapter"
      ? latest.paginas.flatMap((page, pageIndex) => page.text_layers.map((layer) => ({ pageIndex, layerId: layer.id })))
      : selectedLayerId
        ? [{ pageIndex: currentPageIndex, layerId: selectedLayerId }]
        : [];
    if (targets.length === 0) throw new Error("Selecione uma camada de texto");
    const changed = await executeCommand(createApplyStyleCommand(latest, clipboard, targets));
    return changed ? `Estilo aplicado em ${targets.length} camada(s)` : "Nenhuma camada alterada";
  });

  const replaceAll = () => run(async () => {
    const latest = await onPrepareProject();
    const matches = previewChapterReplacements(latest, {
      query,
      replacement,
      caseSensitive,
      wholeWord,
    });
    if (matches.length === 0) return "Nenhuma ocorrência encontrada";
    const changed = await executeCommand(createReplaceTextCommand(latest, matches));
    const count = matches.reduce((total, item) => total + item.occurrences, 0);
    return changed ? `${count} ocorrência(s) substituída(s)` : "Nenhum texto alterado";
  });

  const resolveReview = (itemId: string) => run(async () => {
    const latest = await onPrepareProject();
    const item = buildChapterReviewQueue(latest).find((candidate) => candidate.id === itemId);
    if (!item) return "Item já resolvido";
    const changed = await executeCommand(createResolveReviewCommand(latest, [item]));
    return changed ? "Item marcado como resolvido" : "Item não alterado";
  });

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex items-center gap-1 rounded-lg border border-border bg-bg-tertiary/50 px-2.5 py-1 text-[11px] font-medium text-text-secondary transition-smooth hover:text-text-primary"
        title="Ferramentas de produtividade do capítulo"
      >
        <BookOpenText size={12} />
        Capítulo
        {reviewQueue.length > 0 && (
          <span className="rounded-full bg-status-warning/20 px-1.5 text-[9px] text-status-warning">{reviewQueue.length}</span>
        )}
      </button>

      {open && (
        <section data-testid="chapter-tools-panel" className="absolute right-0 top-8 z-[80] w-[360px] max-h-[calc(100vh-90px)] overflow-y-auto rounded-xl border border-border bg-bg-secondary p-3 shadow-2xl">
          <header className="mb-3 flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold text-text-primary">Ferramentas do capítulo</p>
              <p className="text-[10px] text-text-muted">Estilo, texto e revisão em lote</p>
            </div>
            <button type="button" onClick={() => setOpen(false)} className="rounded p-1 text-text-muted hover:bg-white/5">
              <X size={13} />
            </button>
          </header>

          <div className="mb-3 flex items-center gap-1 rounded-lg border border-border bg-bg-primary/60 p-1">
            <button
              type="button"
              disabled={busy || historyIndex <= 0}
              onClick={() => void run(async () => (await undoChapterCommand()) ? "Lote desfeito" : null)}
              className="flex flex-1 items-center justify-center gap-1 rounded px-2 py-1 text-[10px] text-text-secondary hover:bg-white/5 disabled:opacity-30"
            >
              <Undo2 size={11} /> Desfazer lote
            </button>
            <button
              type="button"
              disabled={busy || historyIndex >= historyLength}
              onClick={() => void run(async () => (await redoChapterCommand()) ? "Lote refeito" : null)}
              className="flex flex-1 items-center justify-center gap-1 rounded px-2 py-1 text-[10px] text-text-secondary hover:bg-white/5 disabled:opacity-30"
            >
              <Redo2 size={11} /> Refazer lote
            </button>
          </div>

          <section className="mb-3 rounded-lg border border-border bg-bg-primary/40 p-2.5">
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-text-muted">Estilo</p>
            <button
              type="button"
              disabled={busy || !selectedLayerId}
              onClick={() => void copySelectedStyle()}
              className="mb-2 flex w-full items-center justify-center gap-1 rounded-md border border-border px-2 py-1.5 text-[10px] text-text-secondary hover:bg-white/5 disabled:opacity-30"
            >
              <ClipboardCopy size={11} /> Copiar estilo selecionado
            </button>
            <div className="grid grid-cols-2 gap-1.5">
              <button
                type="button"
                disabled={busy || !clipboard || !selectedLayerId}
                onClick={() => void applyStyle("selection")}
                className="flex items-center justify-center gap-1 rounded-md bg-accent/15 px-2 py-1.5 text-[10px] text-accent disabled:opacity-30"
              >
                <ClipboardPaste size={11} /> Na seleção
              </button>
              <button
                type="button"
                disabled={busy || !clipboard}
                onClick={() => void applyStyle("chapter")}
                className="flex items-center justify-center gap-1 rounded-md bg-accent/15 px-2 py-1.5 text-[10px] text-accent disabled:opacity-30"
              >
                <ClipboardPaste size={11} /> No capítulo
              </button>
            </div>
          </section>

          <section className="mb-3 rounded-lg border border-border bg-bg-primary/40 p-2.5">
            <p className="mb-2 flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wide text-text-muted">
              <Search size={11} /> Buscar e substituir
            </p>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Buscar no texto traduzido"
              className="mb-1.5 w-full rounded-md border border-border bg-bg-secondary px-2 py-1.5 text-[11px] text-text-primary outline-none focus:border-accent/60"
            />
            <input
              value={replacement}
              onChange={(event) => setReplacement(event.target.value)}
              placeholder="Substituir por"
              className="mb-2 w-full rounded-md border border-border bg-bg-secondary px-2 py-1.5 text-[11px] text-text-primary outline-none focus:border-accent/60"
            />
            <div className="mb-2 flex items-center gap-3 text-[10px] text-text-muted">
              <label className="flex items-center gap-1"><input type="checkbox" checked={caseSensitive} onChange={(event) => setCaseSensitive(event.target.checked)} /> Diferenciar maiúsculas</label>
              <label className="flex items-center gap-1"><input type="checkbox" checked={wholeWord} onChange={(event) => setWholeWord(event.target.checked)} /> Palavra inteira</label>
            </div>
            <button
              type="button"
              disabled={busy || !query || occurrenceCount === 0}
              onClick={() => void replaceAll()}
              className="flex w-full items-center justify-center gap-1 rounded-md bg-accent px-2 py-1.5 text-[10px] font-medium text-white disabled:opacity-30"
            >
              Substituir {occurrenceCount > 0 ? `${occurrenceCount} ocorrência(s)` : "todas"}
            </button>
          </section>

          <section className="rounded-lg border border-border bg-bg-primary/40 p-2.5">
            <div className="mb-2 flex items-center justify-between">
              <p className="text-[10px] font-semibold uppercase tracking-wide text-text-muted">Fila de revisão</p>
              <span className="text-[10px] text-status-warning">{reviewQueue.length} pendente(s)</span>
            </div>
            <div className="max-h-48 space-y-1.5 overflow-y-auto">
              {reviewQueue.length === 0 && <p className="py-3 text-center text-[10px] text-text-muted">Capítulo sem pendências.</p>}
              {reviewQueue.map((item) => (
                <div key={item.id} className="rounded-md border border-border bg-bg-secondary/70 p-2">
                  <button
                    type="button"
                    onClick={() => void onNavigateToLayer(item.pageIndex, item.layerId)}
                    className="block w-full text-left"
                  >
                    <span className="text-[9px] font-medium text-accent">Página {item.pageNumber}</span>
                    <span className="block truncate text-[10px] text-text-primary">{item.translated || item.original}</span>
                    <span className="block truncate text-[9px] text-text-muted">{item.reasons.join(" · ")}</span>
                  </button>
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void resolveReview(item.id)}
                    className="mt-1.5 flex items-center gap-1 text-[9px] text-status-success disabled:opacity-30"
                  >
                    <Check size={10} /> Marcar resolvido
                  </button>
                </div>
              ))}
            </div>
          </section>

          {message && <p className="mt-2 rounded-md bg-bg-primary px-2 py-1.5 text-[10px] text-text-secondary">{message}</p>}
        </section>
      )}
    </div>
  );
}
