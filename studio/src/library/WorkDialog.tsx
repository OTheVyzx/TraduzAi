import { useEffect, useState, type FormEvent } from "react";
import { BookOpen, ImagePlus, Link2, Trash2, X } from "lucide-react";
import type { AddLibraryWorkInput } from "../store/libraryStore";
import type { LibraryWork, PublicationStatus } from "./libraryModel";

export interface WorkDraft {
  title: string;
  aliases: string;
}

export function validateWorkDraft(draft: WorkDraft): string | null {
  return draft.title.trim() ? null : "Informe o título da obra.";
}

function aliasesFromText(value: string): string[] {
  return value
    .split(/[\n,]/)
    .map((alias) => alias.trim())
    .filter((alias, index, aliases) => Boolean(alias) && aliases.indexOf(alias) === index);
}

export function WorkDialog({
  open,
  work = null,
  onClose,
  onSave,
  onRemove,
  onChooseCover,
  onLinkTracking,
}: {
  open: boolean;
  work?: LibraryWork | null;
  onClose: () => void;
  onSave: (input: AddLibraryWorkInput) => void | Promise<void>;
  onRemove?: (workId: string) => void | Promise<void>;
  onChooseCover?: () => Promise<string | null>;
  onLinkTracking?: () => void;
}) {
  const [title, setTitle] = useState(work?.title ?? "");
  const [aliases, setAliases] = useState(work?.aliases.join(", ") ?? "");
  const [coverPath, setCoverPath] = useState<string | null>(work?.coverPath ?? null);
  const [publicationStatus, setPublicationStatus] = useState<PublicationStatus>(work?.publicationStatus ?? "unknown");
  const [anilistId, setAnilistId] = useState(work?.external.anilistId?.toString() ?? "");
  const [mangaDexId, setMangaDexId] = useState(work?.external.mangaDexId ?? "");
  const [canonicalUrl, setCanonicalUrl] = useState(work?.external.canonicalUrl ?? "");
  const [manualStatus, setManualStatus] = useState(work?.external.manualStatusOverride != null);
  const [error, setError] = useState<string | null>(null);
  const [removalArmed, setRemovalArmed] = useState(false);

  useEffect(() => {
    if (!open) return;
    setTitle(work?.title ?? "");
    setAliases(work?.aliases.join(", ") ?? "");
    setCoverPath(work?.coverPath ?? null);
    setPublicationStatus(work?.publicationStatus ?? "unknown");
    setAnilistId(work?.external.anilistId?.toString() ?? "");
    setMangaDexId(work?.external.mangaDexId ?? "");
    setCanonicalUrl(work?.external.canonicalUrl ?? "");
    setManualStatus(work?.external.manualStatusOverride != null);
    setError(null);
    setRemovalArmed(false);
  }, [open, work]);

  if (!open) return null;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const validationError = validateWorkDraft({ title, aliases });
    if (validationError) {
      setError(validationError);
      return;
    }
    await onSave({
      id: work?.id ?? `work-${Date.now().toString(36)}`,
      title: title.trim(),
      aliases: aliasesFromText(aliases),
      coverPath,
      publicationStatus,
      external: {
        ...(anilistId.trim() && Number.isFinite(Number(anilistId)) ? { anilistId: Number(anilistId) } : {}),
        ...(mangaDexId.trim() ? { mangaDexId: mangaDexId.trim() } : {}),
        ...(canonicalUrl.trim() ? { canonicalUrl: canonicalUrl.trim() } : {}),
        ...(work?.external.tracking ? { tracking: work.external.tracking } : {}),
        ...(work && (work.external.anilistId || work.external.mangaDexId)
          ? { manualStatusOverride: manualStatus ? publicationStatus : null }
          : {}),
      },
    });
    onClose();
  };

  return (
    <div className="studio-dialog-backdrop" role="presentation">
      <section className="studio-dialog" role="dialog" aria-modal="true" aria-labelledby="studio-work-dialog-title">
        <header>
          <div>
            <small>Biblioteca local</small>
            <h2 id="studio-work-dialog-title">{work ? "Editar obra" : "Adicionar obra"}</h2>
          </div>
          <button type="button" className="studio-dialog-icon" aria-label="Fechar" onClick={onClose}><X size={17} /></button>
        </header>

        <form onSubmit={(event) => void submit(event)}>
          <div className="studio-work-dialog-grid">
            <div className="studio-work-dialog-cover">
              <span>{coverPath ? <img src={coverPath} alt="Capa escolhida" /> : <BookOpen size={32} />}</span>
              <button type="button" onClick={() => void onChooseCover?.().then((path) => path && setCoverPath(path))}>
                <ImagePlus size={14} /> Escolher capa
              </button>
              {coverPath && <small title={coverPath}>{coverPath}</small>}
            </div>

            <div className="studio-dialog-fields">
              <label>
                <span>Título *</span>
                <input autoFocus value={title} onChange={(event) => setTitle(event.currentTarget.value)} />
              </label>
              <label>
                <span>Aliases</span>
                <input value={aliases} placeholder="Separados por vírgula" onChange={(event) => setAliases(event.currentTarget.value)} />
              </label>
              <label>
                <span>Status da publicação</span>
                <select value={publicationStatus} onChange={(event) => {
                  setPublicationStatus(event.currentTarget.value as PublicationStatus);
                  if (work?.external.anilistId || work?.external.mangaDexId) setManualStatus(true);
                }}>
                  <option value="unknown">Sem status</option>
                  <option value="releasing">Em publicação</option>
                  <option value="hiatus">Hiato</option>
                  <option value="completed">Completa</option>
                  <option value="cancelled">Cancelada</option>
                  <option value="not_yet_released">Não iniciada</option>
                </select>
              </label>
              {(work?.external.anilistId || work?.external.mangaDexId) && (
                <label className="flex-row">
                  <input type="checkbox" checked={manualStatus} onChange={(event) => setManualStatus(event.currentTarget.checked)} />
                  <span>Manter este status manual mesmo se as fontes divergirem</span>
                </label>
              )}
              <div className="studio-dialog-field-pair">
                <label><span>AniList ID</span><input inputMode="numeric" value={anilistId} onChange={(event) => setAnilistId(event.currentTarget.value)} /></label>
                <label><span>MangaDex ID</span><input value={mangaDexId} onChange={(event) => setMangaDexId(event.currentTarget.value)} /></label>
              </div>
              <label><span>Fonte canônica</span><input type="url" value={canonicalUrl} placeholder="https://" onChange={(event) => setCanonicalUrl(event.currentTarget.value)} /></label>
              {work && onLinkTracking && (
                <button type="button" className="studio-dialog-secondary" onClick={onLinkTracking}>
                  <Link2 size={14} /> Vincular ou trocar fonte
                </button>
              )}
            </div>
          </div>

          {error && <p className="studio-dialog-error">{error}</p>}

          <footer>
            {work && onRemove && (
              <div className="studio-dialog-remove">
                <button
                  type="button"
                  onClick={() => removalArmed ? void Promise.resolve(onRemove(work.id)).then(onClose) : setRemovalArmed(true)}
                >
                  <Trash2 size={14} /> {removalArmed ? "Confirmar remoção" : "Remover da biblioteca"}
                </button>
                <small>Remove só a referência: não apaga capítulos nem arquivos do disco.</small>
              </div>
            )}
            <div className="studio-dialog-actions">
              <button type="button" onClick={onClose}>Cancelar</button>
              <button type="submit" className="studio-dialog-primary">{work ? "Salvar alterações" : "Adicionar obra"}</button>
            </div>
          </footer>
        </form>
      </section>
    </div>
  );
}
