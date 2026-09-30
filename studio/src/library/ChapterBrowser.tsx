import { useEffect, useState, type CSSProperties, type KeyboardEvent } from "react";
import { AlertTriangle, BookOpen, CheckCircle2, FileInput, FolderOpen, Image, Link2, Plus, Trash2 } from "lucide-react";
import { loadImageSource } from "../../../src/lib/imageSource";
import { chapterProgress, type LibraryChapter, type LibraryWork } from "./libraryModel";

const WORKFLOW_LABELS: Record<NonNullable<LibraryChapter["workflowStatus"]>, string> = {
  pending: "Não iniciado",
  translating: "Tradução",
  editing: "Edição",
  review: "Revisão",
  completed: "Concluído",
};

const CHAPTER_ARROW_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]);
const CHAPTER_BATCH_SIZE = 60;

type ChapterThumbnailState = "loading" | "loaded" | "failed";

function chapterCoverMime(path: string) {
  const cleanPath = path.split(/[?#]/, 1)[0].toLocaleLowerCase("en-US");
  if (cleanPath.endsWith(".png")) return "image/png";
  if (cleanPath.endsWith(".webp")) return "image/webp";
  if (cleanPath.endsWith(".gif")) return "image/gif";
  if (cleanPath.endsWith(".avif")) return "image/avif";
  return "image/jpeg";
}

function ChapterThumbnail({ chapter, progress }: { chapter: LibraryChapter; progress: number }) {
  const [state, setState] = useState<ChapterThumbnailState>(chapter.coverPath ? "loading" : "failed");
  const [source, setSource] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let revoke: (() => void) | undefined;
    setSource(null);
    setState(chapter.coverPath ? "loading" : "failed");
    if (!chapter.coverPath) return () => { active = false; };

    void loadImageSource(chapter.coverPath, chapterCoverMime(chapter.coverPath))
      .then((loaded) => {
        revoke = loaded.revoke;
        if (!active) {
          revoke?.();
          return;
        }
        setSource(loaded.src);
      })
      .catch(() => {
        if (active) setState("failed");
      });

    return () => {
      active = false;
      revoke?.();
    };
  }, [chapter.coverPath]);

  const workflowStatus = chapter.workflowStatus ?? "pending";
  return (
    <>
      <span className={`studio-chapter-placeholder studio-chapter-placeholder-${state}`} aria-hidden="true">
        <span className="studio-chapter-preview-icon"><BookOpen size={18} /></span>
        <span className="studio-chapter-preview-kicker">Capítulo</span>
        <span className="studio-chapter-preview-number">{chapter.label}</span>
        <span className="studio-chapter-preview-status">{WORKFLOW_LABELS[workflowStatus]}</span>
        <span className="studio-chapter-preview-percent">{progress}%</span>
      </span>
      {source && (
        <img
          className={`studio-chapter-image studio-chapter-image-${state}`}
          src={source}
          alt=""
          onLoad={() => setState("loaded")}
          onError={() => setState("failed")}
        />
      )}
    </>
  );
}

export function shouldHandleChapterArrowKey(
  key: string,
  target: { tagName?: string; isContentEditable?: boolean },
): boolean {
  if (!CHAPTER_ARROW_KEYS.has(key) || target.isContentEditable) return false;
  return !["INPUT", "TEXTAREA", "SELECT"].includes((target.tagName ?? "").toUpperCase());
}

export function nextChapterSelection<T extends { id: string }>(
  chapters: readonly T[],
  selectedChapterId: string | null,
  key: string,
  columns: number,
): string | null {
  if (chapters.length === 0) return null;
  const currentIndex = Math.max(0, chapters.findIndex((chapter) => chapter.id === selectedChapterId));
  const safeColumns = Math.max(1, Math.trunc(columns));
  const delta = key === "ArrowLeft"
    ? -1
    : key === "ArrowRight"
      ? 1
      : key === "ArrowUp"
        ? -safeColumns
        : key === "ArrowDown"
          ? safeColumns
          : 0;
  const nextIndex = Math.min(chapters.length - 1, Math.max(0, currentIndex + delta));
  return chapters[nextIndex].id;
}

export function ChapterBrowser({
  work,
  query = "",
  view,
  thumbnailSize,
  selectedChapterId,
  onSelectChapter,
  onOpenChapter,
  onImportProject,
  onAddChapter,
  missingProjectPaths = new Set<string>(),
  onRelinkChapter,
  onRemoveChapter,
}: {
  work: LibraryWork | null;
  query?: string;
  view: "grid" | "list";
  thumbnailSize: number;
  selectedChapterId: string | null;
  onSelectChapter: (chapterId: string) => void;
  onOpenChapter: (projectPath: string) => void;
  onImportProject?: () => void;
  onAddChapter?: () => void;
  missingProjectPaths?: ReadonlySet<string>;
  onRelinkChapter?: (chapterId: string) => void;
  onRemoveChapter?: (chapterId: string) => void;
}) {
  const [visibleCount, setVisibleCount] = useState(CHAPTER_BATCH_SIZE);
  const normalizedQuery = query.trim().toLocaleLowerCase("pt-BR");
  const chapters = (work?.chapters ?? []).filter((chapter) => {
    if (!normalizedQuery) return true;
    const missing = missingProjectPaths.has(chapter.projectPath);
    return [chapter.label, chapter.title ?? "", chapter.projectPath, missing ? "caminho ausente relocalizar" : ""]
      .join(" ")
      .toLocaleLowerCase("pt-BR")
      .includes(normalizedQuery);
  });
  const selectedChapter = chapters.find((chapter) => chapter.id === selectedChapterId) ?? null;
  const selectedChapterIndex = selectedChapterId
    ? chapters.findIndex((chapter) => chapter.id === selectedChapterId)
    : -1;
  const visibleLimit = Math.max(visibleCount, selectedChapterIndex + 1);
  const visibleChapters = chapters.slice(0, visibleLimit);

  useEffect(() => {
    setVisibleCount(CHAPTER_BATCH_SIZE);
  }, [normalizedQuery, work?.id]);

  const handleChapterKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement;
    if (!shouldHandleChapterArrowKey(event.key, target)) return;
    const columns = view === "list"
      ? 1
      : Math.max(1, Math.floor(event.currentTarget.clientWidth / Math.max(1, thumbnailSize + 22)));
    const focusedId = target.dataset.chapterId ?? selectedChapterId;
    const nextId = nextChapterSelection(chapters, focusedId, event.key, columns);
    if (!nextId || nextId === focusedId) return;
    event.preventDefault();
    const nextIndex = chapters.findIndex((chapter) => chapter.id === nextId);
    if (nextIndex >= visibleLimit) {
      setVisibleCount(Math.ceil((nextIndex + 1) / CHAPTER_BATCH_SIZE) * CHAPTER_BATCH_SIZE);
    }
    onSelectChapter(nextId);
    const collection = event.currentTarget;
    window.requestAnimationFrame(() => {
      [...collection.querySelectorAll<HTMLButtonElement>("[data-chapter-id]")]
        .find((button) => button.dataset.chapterId === nextId)
        ?.focus();
    });
  };

  return (
    <>
      {view === "list" && work && chapters.length > 0 && (
        <div className="studio-chapter-table-heading" aria-hidden="true">
          <span>Capítulo</span><span>Status</span><span>Páginas</span><span>Progresso</span><span>Última atualização</span><span>Ações</span>
        </div>
      )}
      <section
        className={`studio-chapter-browser studio-chapter-browser-${view}`}
        style={{ "--chapter-card-size": `${thumbnailSize}px` } as CSSProperties}
        aria-label="Capítulos da obra"
      >
        {!work ? (
          <div className="studio-chapter-empty">
            <BookOpen size={34} aria-hidden="true" />
            <strong>Selecione ou adicione uma obra</strong>
            <span>Os capítulos da obra escolhida aparecerão aqui.</span>
          </div>
        ) : chapters.length === 0 ? (
          <div className="studio-chapter-empty">
            <Image size={34} aria-hidden="true" />
            <strong>{query ? "Nenhum capítulo encontrado" : "Ainda não há capítulos"}</strong>
            <span>{query ? "Tente outro termo de busca." : "Anexe um projeto TraduzAI ou crie um capítulo manual."}</span>
          </div>
        ) : (
          <div className="studio-chapter-collection" onKeyDown={handleChapterKeyDown}>
            {visibleChapters.map((chapter, chapterIndex) => {
              const progress = chapterProgress(chapter);
              const selected = chapter.id === selectedChapterId;
              const missing = missingProjectPaths.has(chapter.projectPath);
              return (
                <article key={chapter.id} className={`studio-chapter-entry${missing ? " studio-chapter-entry-missing" : ""}`}>
                  <button
                    type="button"
                    className="studio-chapter-card"
                    aria-label={`Selecionar capítulo ${chapter.label}`}
                    aria-pressed={selected}
                    data-chapter-id={chapter.id}
                    tabIndex={selected || (!selectedChapter && chapterIndex === 0) ? 0 : -1}
                    onClick={() => onSelectChapter(chapter.id)}
                    onDoubleClick={() => !missing && onOpenChapter(chapter.projectPath)}
                  >
                    <span className="studio-chapter-thumbnail">
                      <ChapterThumbnail chapter={chapter} progress={progress} />
                      {selected && <span className="studio-chapter-selected"><CheckCircle2 size={17} /></span>}
                      {missing && <span className="studio-chapter-missing-mark"><AlertTriangle size={16} /></span>}
                      <span className="studio-chapter-progress" style={{ "--chapter-progress": `${progress}%` } as CSSProperties} />
                    </span>
                    <span className="studio-chapter-details">
                      <strong>Capítulo {chapter.label}</strong>
                      {chapter.title && <span>{chapter.title}</span>}
                      <small>
                        {missing ? "Caminho ausente" : `${chapter.completedPages ?? 0} de ${chapter.pageCount ?? 0} páginas${chapter.workflowStatus ? ` · ${WORKFLOW_LABELS[chapter.workflowStatus]}` : ""}`}
                      </small>
                    </span>
                    {view === "list" && (
                      <>
                        <span className={`studio-chapter-row-status studio-chapter-row-status-${chapter.workflowStatus ?? "pending"}`}>{WORKFLOW_LABELS[chapter.workflowStatus ?? "pending"]}</span>
                        <span className="studio-chapter-row-pages">{chapter.pageCount ?? 0}</span>
                        <span className="studio-chapter-row-progress"><b>{progress}%</b><i style={{ width: `${progress}%` }} /></span>
                        <time className="studio-chapter-row-updated">{chapter.lastOpenedAt ? new Intl.DateTimeFormat("pt-BR", { dateStyle: "short" }).format(new Date(chapter.lastOpenedAt)) : "—"}</time>
                      </>
                    )}
                  </button>
                  {(missing || selected) && (
                    <div className="studio-chapter-reference-actions">
                      {missing && <button type="button" onClick={() => onRelinkChapter?.(chapter.id)}><Link2 size={11} /> Relocalizar</button>}
                      {selected && <button type="button" onClick={() => onRemoveChapter?.(chapter.id)}><Trash2 size={11} /> Remover referência</button>}
                    </div>
                  )}
                </article>
              );
            })}
            {visibleLimit < chapters.length && (
              <button
                type="button"
                className="studio-chapter-load-more"
                onClick={() => setVisibleCount((current) => current + CHAPTER_BATCH_SIZE)}
              >
                Carregar mais {Math.min(CHAPTER_BATCH_SIZE, chapters.length - visibleLimit)} capítulos
              </button>
            )}
          </div>
        )}
      </section>

      <footer className="studio-library-footer">
        <button type="button" onClick={onImportProject}>
          <FileInput size={15} />
          Importar projeto
        </button>
        <div className="studio-library-footer-primary">
          <button type="button" disabled={!work} onClick={onAddChapter}>
            <Plus size={15} />
            Adicionar capítulo
          </button>
          <button
            type="button"
            className="studio-library-open"
            disabled={!selectedChapter || missingProjectPaths.has(selectedChapter.projectPath)}
            onClick={() => selectedChapter && !missingProjectPaths.has(selectedChapter.projectPath) && onOpenChapter(selectedChapter.projectPath)}
          >
            <FolderOpen size={15} />
            Abrir
          </button>
        </div>
      </footer>
    </>
  );
}
