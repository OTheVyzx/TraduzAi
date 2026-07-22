import type { CSSProperties } from "react";
import { BookOpen, CheckCircle2, FileInput, FolderOpen, Image, Plus } from "lucide-react";
import { chapterProgress, type LibraryChapter, type LibraryWork } from "./libraryModel";

const WORKFLOW_LABELS: Record<NonNullable<LibraryChapter["workflowStatus"]>, string> = {
  pending: "Não iniciado",
  translating: "Tradução",
  editing: "Edição",
  review: "Revisão",
  completed: "Concluído",
};

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
}) {
  const normalizedQuery = query.trim().toLocaleLowerCase("pt-BR");
  const chapters = (work?.chapters ?? []).filter((chapter) => {
    if (!normalizedQuery) return true;
    return [chapter.label, chapter.title ?? ""]
      .join(" ")
      .toLocaleLowerCase("pt-BR")
      .includes(normalizedQuery);
  });
  const selectedChapter = chapters.find((chapter) => chapter.id === selectedChapterId) ?? null;

  return (
    <>
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
          <div className="studio-chapter-collection">
            {chapters.map((chapter) => {
              const progress = chapterProgress(chapter);
              const selected = chapter.id === selectedChapterId;
              return (
                <button
                  key={chapter.id}
                  type="button"
                  className="studio-chapter-card"
                  aria-label={`Selecionar capítulo ${chapter.label}`}
                  aria-pressed={selected}
                  onClick={() => onSelectChapter(chapter.id)}
                  onDoubleClick={() => onOpenChapter(chapter.projectPath)}
                >
                  <span className="studio-chapter-thumbnail">
                    {chapter.coverPath ? (
                      <img src={chapter.coverPath} alt="" />
                    ) : (
                      <span className="studio-chapter-placeholder"><Image size={28} /></span>
                    )}
                    {selected && <span className="studio-chapter-selected"><CheckCircle2 size={17} /></span>}
                    <span className="studio-chapter-progress" style={{ "--chapter-progress": `${progress}%` } as CSSProperties} />
                  </span>
                  <span className="studio-chapter-details">
                    <strong>Capítulo {chapter.label}</strong>
                    {chapter.title && <span>{chapter.title}</span>}
                    <small>
                      {chapter.completedPages ?? 0} de {chapter.pageCount ?? 0} páginas
                      {chapter.workflowStatus ? ` · ${WORKFLOW_LABELS[chapter.workflowStatus]}` : ""}
                    </small>
                  </span>
                </button>
              );
            })}
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
            disabled={!selectedChapter}
            onClick={() => selectedChapter && onOpenChapter(selectedChapter.projectPath)}
          >
            <FolderOpen size={15} />
            Abrir
          </button>
        </div>
      </footer>
    </>
  );
}
