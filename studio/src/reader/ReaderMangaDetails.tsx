import { ArrowLeft, BookOpenCheck, Download, MoreVertical, RefreshCw } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { normalizeGenres } from "./readerModel";
import type { RuntimeChapter, RuntimeManga } from "./sourceRuntimeClient";
import { ReaderCover } from "./ReaderCover";

const STATUS_LABELS: Record<number, string> = { 0: "Desconhecido", 1: "Em andamento", 2: "Concluído", 3: "Licenciado", 4: "Publicação finalizada", 5: "Cancelado", 6: "Em hiato" };
const CHAPTER_BATCH_SIZE = 80;

function chapterDate(value?: number): string | null {
  return value ? new Date(value).toLocaleDateString("pt-BR", { timeZone: "UTC" }) : null;
}

export function ReaderMangaDetails({ manga, sourceName, chapters, inLibrary, busy, chaptersLoading, chaptersError, onBack, onToggleLibrary, onRetryChapters, onChapter, renderChapterAction }: {
  manga: RuntimeManga;
  sourceName: string;
  chapters: RuntimeChapter[];
  inLibrary: boolean;
  busy: boolean;
  chaptersLoading: boolean;
  chaptersError: string | null;
  onBack(): void;
  onToggleLibrary(): void;
  onRetryChapters(): void;
  onChapter?(chapter: RuntimeChapter): void;
  renderChapterAction?(chapter: RuntimeChapter): ReactNode;
}) {
  const genres = normalizeGenres(manga.genre);
  const countLabel = `${chapters.length} ${chapters.length === 1 ? "capítulo" : "capítulos"}`;
  const [visibleChapterCount, setVisibleChapterCount] = useState(CHAPTER_BATCH_SIZE);
  useEffect(() => setVisibleChapterCount(CHAPTER_BATCH_SIZE), [manga.url]);
  const visibleChapters = chapters.slice(0, visibleChapterCount);
  const remainingChapters = Math.max(0, chapters.length - visibleChapters.length);
  return <section className="reader-manga-details">
    <header className="reader-manga-details-toolbar">
      <button type="button" aria-label="Voltar" onClick={onBack}><ArrowLeft size={22} /></button>
      <h1>{manga.title}</h1>
      <button type="button" aria-label="Atualizar detalhes" disabled={busy} onClick={onRetryChapters}><RefreshCw size={19} /></button>
    </header>
    <div className="reader-manga-details-layout">
      <aside className="reader-manga-summary">
        <div className="reader-manga-backdrop" aria-hidden="true"><ReaderCover src={manga.thumbnailUrl} /></div>
        <ReaderCover className="reader-manga-primary-cover" src={manga.thumbnailUrl} alt={`Capa de ${manga.title}`} />
        <div className="reader-manga-meta">
          <h2>{manga.title}</h2>
          {manga.author && <p><span>Autor</span><strong>{manga.author}</strong></p>}
          {manga.artist && <p><span>Artista</span><strong>{manga.artist}</strong></p>}
          <p><span>Status</span><strong>{STATUS_LABELS[manga.status ?? 0] ?? "Desconhecido"}</strong></p>
          <p><span>Fonte</span><strong>{sourceName}</strong></p>
        </div>
        <button className="reader-manga-library-action" type="button" disabled={busy} onClick={onToggleLibrary}><BookOpenCheck size={18} />{inLibrary ? "Remover da biblioteca" : "Adicionar à biblioteca"}</button>
        {manga.description && <p className="reader-manga-description">{manga.description}</p>}
        {genres.length > 0 && <div className="reader-manga-genres">{genres.map((genre) => <span key={genre}>{genre}</span>)}</div>}
      </aside>
      <section className="reader-manga-chapter-pane" aria-label="Capítulos">
        <header><h2>{countLabel}</h2><div><BookOpenCheck size={18} /><Download size={18} /></div></header>
        {chaptersLoading && <p className="reader-manga-chapter-state">Carregando capítulos…</p>}
        {chaptersError && <div className="reader-manga-chapter-error" role="alert"><p>{chaptersError}</p><button type="button" onClick={onRetryChapters}>Tentar novamente</button></div>}
        {!chaptersLoading && !chaptersError && <div className="reader-manga-chapter-list">{visibleChapters.map((chapter) => {
          const date = chapterDate(chapter.dateUpload);
          return <article key={chapter.url}>
            <button type="button" className="reader-manga-chapter-open" onClick={() => onChapter?.(chapter)}><strong>{chapter.name}</strong>{chapter.scanlator && <small>{chapter.scanlator}</small>}{date && <small>{date}</small>}</button>
            {renderChapterAction?.(chapter) ?? <button type="button" aria-label={`Ações de ${chapter.name}`}><MoreVertical size={18} /></button>}
          </article>;
        })}{remainingChapters > 0 && <button
          className="reader-manga-load-more"
          type="button"
          onClick={() => setVisibleChapterCount((count) => Math.min(chapters.length, count + CHAPTER_BATCH_SIZE))}
        >Carregar mais {remainingChapters} {remainingChapters === 1 ? "capítulo" : "capítulos"}</button>}</div>}
      </section>
    </div>
  </section>;
}
