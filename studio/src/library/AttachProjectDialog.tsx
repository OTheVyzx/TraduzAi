import { useEffect, useMemo, useState, type FormEvent } from "react";
import { AlertTriangle, FileJson, FolderOpen, X } from "lucide-react";
import type { LibraryWork } from "./libraryModel";

export interface ProjectAttachmentDraft {
  projectPath: string;
  workTitle: string;
  chapterLabel: string;
  pageCount: number;
  coverPath?: string | null;
}

function comparablePath(path: string): string {
  return path.trim().replace(/\\/g, "/").replace(/\/+$/, "").toLocaleLowerCase("en-US");
}

export function AttachProjectDialog({
  open,
  work,
  draft: providedDraft = null,
  onChooseProject,
  onClose,
  onConfirm,
}: {
  open: boolean;
  work: LibraryWork;
  draft?: ProjectAttachmentDraft | null;
  onChooseProject: () => Promise<ProjectAttachmentDraft | null>;
  onClose: () => void;
  onConfirm: (draft: ProjectAttachmentDraft) => void | Promise<void>;
}) {
  const [draft, setDraft] = useState<ProjectAttachmentDraft | null>(providedDraft);
  const [chapterLabel, setChapterLabel] = useState(providedDraft?.chapterLabel ?? "");
  const [duplicateConfirmed, setDuplicateConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setDraft(providedDraft);
    setChapterLabel(providedDraft?.chapterLabel ?? "");
    setDuplicateConfirmed(false);
    setError(null);
  }, [open, providedDraft]);

  const duplicate = useMemo(() => (
    draft ? work.chapters.find((chapter) => comparablePath(chapter.projectPath) === comparablePath(draft.projectPath)) ?? null : null
  ), [draft, work.chapters]);

  if (!open) return null;

  const chooseProject = async () => {
    setBusy(true);
    setError(null);
    try {
      const selected = await onChooseProject();
      if (!selected) return;
      setDraft(selected);
      setChapterLabel(selected.chapterLabel);
      setDuplicateConfirmed(false);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!draft) {
      setError("Escolha um project.json.");
      return;
    }
    if (!chapterLabel.trim()) {
      setError("Informe o capítulo.");
      return;
    }
    await onConfirm({ ...draft, chapterLabel: chapterLabel.trim() });
    onClose();
  };

  return (
    <div className="studio-dialog-backdrop" role="presentation">
      <section className="studio-dialog studio-attach-dialog" role="dialog" aria-modal="true" aria-labelledby="studio-attach-title">
        <header>
          <div><small>{work.title}</small><h2 id="studio-attach-title">Anexar projeto existente</h2></div>
          <button type="button" className="studio-dialog-icon" aria-label="Fechar" onClick={onClose}><X size={17} /></button>
        </header>

        <form onSubmit={(event) => void submit(event)}>
          <div className="studio-attach-picker">
            <FileJson size={28} />
            <div>
              <strong>{draft ? "project.json selecionado" : "Escolha um projeto TraduzAI"}</strong>
              <small>{draft?.projectPath ?? "O arquivo será apenas referenciado pela biblioteca."}</small>
            </div>
            <button type="button" onClick={() => void chooseProject()} disabled={busy}>
              <FolderOpen size={14} /> {busy ? "Lendo…" : draft ? "Trocar" : "Escolher"}
            </button>
          </div>

          {draft && (
            <div className="studio-attach-metadata">
              <label><span>Obra detectada</span><input value={draft.workTitle} readOnly /></label>
              <label><span>Capítulo</span><input value={chapterLabel} onChange={(event) => setChapterLabel(event.currentTarget.value)} /></label>
              <div><span>Páginas</span><strong>{draft.pageCount} páginas</strong></div>
            </div>
          )}

          {duplicate && (
            <label className="studio-dialog-warning">
              <AlertTriangle size={16} />
              <span><strong>Este project.json já está anexado.</strong> Confirmar atualização da referência.</span>
              <input type="checkbox" checked={duplicateConfirmed} onChange={(event) => setDuplicateConfirmed(event.currentTarget.checked)} />
            </label>
          )}
          {error && <p className="studio-dialog-error">{error}</p>}

          <footer>
            <p>O Studio não move nem altera o arquivo ao anexá-lo.</p>
            <div className="studio-dialog-actions">
              <button type="button" onClick={onClose}>Cancelar</button>
              <button type="submit" className="studio-dialog-primary" disabled={!draft || Boolean(duplicate && !duplicateConfirmed)}>Anexar capítulo</button>
            </div>
          </footer>
        </form>
      </section>
    </div>
  );
}
