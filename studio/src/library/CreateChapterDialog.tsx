import { useEffect, useState, type FormEvent } from "react";
import { Archive, FileJson, FolderOpen, Save, X } from "lucide-react";
import {
  ManualChapterCreationError,
  type ManualChapterCreationInput,
  type PreparedManualPage,
} from "../backend/projectDialog";
import type { LibraryWork } from "./libraryModel";

export interface CreateChapterDraft {
  chapterLabel: string;
  sourceLanguage: string;
  targetLanguage: string;
  sourcePath: string | null;
  projectJsonPath: string | null;
}

export function validateCreateChapterDraft(draft: CreateChapterDraft): string | null {
  if (!draft.chapterLabel.trim()) return "Informe o capítulo.";
  if (!draft.sourcePath) return "Escolha uma pasta de imagens, ZIP ou CBZ.";
  if (!draft.projectJsonPath) return "Escolha onde salvar o project.json.";
  if (!draft.sourceLanguage.trim() || !draft.targetLanguage.trim()) return "Informe os idiomas do capítulo.";
  return null;
}

export function CreateChapterDialog({
  open,
  work,
  onClose,
  onChooseFolder,
  onChooseArchive,
  onChooseDestination,
  onAttachExisting,
  onCreate,
}: {
  open: boolean;
  work: LibraryWork;
  onClose: () => void;
  onChooseFolder: () => Promise<string | null>;
  onChooseArchive: () => Promise<string | null>;
  onChooseDestination: (suggestedName: string) => Promise<string | null>;
  onAttachExisting: () => void;
  onCreate: (input: ManualChapterCreationInput, preparedPages?: PreparedManualPage[] | null) => Promise<void>;
}) {
  const [chapterLabel, setChapterLabel] = useState("");
  const [chapterTitle, setChapterTitle] = useState("");
  const [sourceLanguage, setSourceLanguage] = useState("ko");
  const [targetLanguage, setTargetLanguage] = useState("pt-BR");
  const [sourcePath, setSourcePath] = useState<string | null>(null);
  const [projectJsonPath, setProjectJsonPath] = useState<string | null>(null);
  const [preparedPages, setPreparedPages] = useState<PreparedManualPage[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setChapterLabel("");
    setChapterTitle("");
    setSourceLanguage("ko");
    setTargetLanguage("pt-BR");
    setSourcePath(null);
    setProjectJsonPath(null);
    setPreparedPages(null);
    setBusy(false);
    setError(null);
  }, [open, work.id]);

  if (!open) return null;

  const draft = { chapterLabel, sourceLanguage, targetLanguage, sourcePath, projectJsonPath };
  const validationError = validateCreateChapterDraft(draft);

  const chooseSource = async (chooser: () => Promise<string | null>) => {
    setError(null);
    try {
      const selected = await chooser();
      if (selected) {
        setSourcePath(selected);
        setPreparedPages(null);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  };

  const chooseDestination = async () => {
    setError(null);
    try {
      const safeLabel = chapterLabel.trim().replace(/[<>:"/\\|?*]+/g, "-") || "capitulo";
      const selected = await onChooseDestination(`${work.title} - ${safeLabel}/project.json`);
      if (selected) {
        setProjectJsonPath(selected);
        setPreparedPages(null);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (validationError || !sourcePath || !projectJsonPath) {
      setError(validationError);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await onCreate({
        workTitle: work.title,
        chapterLabel: chapterLabel.trim(),
        chapterTitle: chapterTitle.trim() || undefined,
        sourceLanguage: sourceLanguage.trim(),
        targetLanguage: targetLanguage.trim(),
        sourcePath,
        projectJsonPath,
      }, preparedPages);
      onClose();
    } catch (cause) {
      if (cause instanceof ManualChapterCreationError) setPreparedPages(cause.preparedPages);
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="studio-dialog-backdrop" role="presentation">
      <section className="studio-dialog studio-create-chapter-dialog" role="dialog" aria-modal="true" aria-labelledby="studio-create-chapter-title">
        <header>
          <div><small>{work.title}</small><h2 id="studio-create-chapter-title">Criar capítulo manual</h2></div>
          <button type="button" className="studio-dialog-icon" aria-label="Fechar" onClick={onClose}><X size={17} /></button>
        </header>

        <form onSubmit={(event) => void submit(event)}>
          <div className="studio-create-chapter-body">
            <div className="studio-create-source-grid">
              <button type="button" onClick={() => void chooseSource(onChooseFolder)}>
                <FolderOpen size={20} /><strong>Pasta de imagens</strong><small>PNG, JPEG ou WebP</small>
              </button>
              <button type="button" onClick={() => void chooseSource(onChooseArchive)}>
                <Archive size={20} /><strong>ZIP ou CBZ</strong><small>Extração local protegida</small>
              </button>
            </div>

            <div className="studio-create-selected-path">
              <span>Origem</span>
              <strong title={sourcePath ?? undefined}>{sourcePath ?? "Nenhuma origem selecionada"}</strong>
            </div>

            <div className="studio-dialog-fields">
              <div className="studio-dialog-field-pair">
                <label><span>Capítulo *</span><input autoFocus value={chapterLabel} onChange={(event) => setChapterLabel(event.currentTarget.value)} /></label>
                <label><span>Título opcional</span><input value={chapterTitle} onChange={(event) => setChapterTitle(event.currentTarget.value)} /></label>
              </div>
              <div className="studio-dialog-field-pair">
                <label><span>Idioma de origem</span><input value={sourceLanguage} onChange={(event) => setSourceLanguage(event.currentTarget.value)} /></label>
                <label><span>Idioma de destino</span><input value={targetLanguage} onChange={(event) => setTargetLanguage(event.currentTarget.value)} /></label>
              </div>
            </div>

            <div className="studio-create-destination">
              <div><span>Projeto</span><strong title={projectJsonPath ?? undefined}>{projectJsonPath ?? "Escolha onde salvar o project.json"}</strong></div>
              <button type="button" onClick={() => void chooseDestination()}><Save size={14} /> Escolher destino</button>
            </div>

            <button type="button" className="studio-create-attach-existing" onClick={onAttachExisting}>
              <FileJson size={15} /> Anexar project.json existente
            </button>

            {preparedPages && <p className="studio-create-retry-note">{preparedPages.length} páginas já preparadas; tentar novamente não extrairá os arquivos outra vez.</p>}
            {error && <p className="studio-dialog-error">{error}</p>}
          </div>

          <footer>
            <p>Os arquivos serão copiados para a pasta do novo projeto. A origem não será alterada.</p>
            <div className="studio-dialog-actions">
              <button type="button" onClick={onClose}>Cancelar</button>
              <button type="submit" className="studio-dialog-primary" disabled={Boolean(validationError) || busy}>
                {busy ? "Criando…" : preparedPages ? "Tentar novamente" : "Criar e abrir"}
              </button>
            </div>
          </footer>
        </form>
      </section>
    </div>
  );
}
