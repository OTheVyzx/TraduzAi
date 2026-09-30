import { BookOpen, CheckCircle2, FileDown, PencilLine } from "lucide-react";
import { chapterProgress, type LibraryWork } from "./libraryModel";

function workStatus(work: LibraryWork): string {
  if (work.chapters.some((chapter) => chapter.workflowStatus === "review")) return "Em revisão";
  if (work.chapters.some((chapter) => chapter.workflowStatus === "editing")) return "Em edição";
  if (work.chapters.length > 0 && work.chapters.every((chapter) => chapter.workflowStatus === "completed")) return "Concluída";
  return work.publicationStatus === "releasing" ? "Em andamento" : "Sem status";
}

export function WorkInspector({ work, onEditWork }: { work: LibraryWork | null; onEditWork?: () => void }) {
  if (!work) return <aside className="studio-work-inspector studio-work-inspector-empty"><BookOpen size={28} /><strong>Selecione uma obra</strong><p>Os detalhes da obra aparecerão aqui.</p></aside>;

  const completedChapters = work.chapters.filter((chapter) => chapter.workflowStatus === "completed").length;
  const totalProgress = work.chapters.length
    ? Math.round(work.chapters.reduce((total, chapter) => total + chapterProgress(chapter), 0) / work.chapters.length)
    : 0;

  return (
    <aside className="studio-work-inspector" aria-label={`Informações de ${work.title}`}>
      <header>
        <span className="studio-work-inspector-cover-wrap">
          {work.coverPath ? <img className="studio-work-inspector-cover-ambient" src={work.coverPath} alt="" aria-hidden="true" /> : null}
          <span className="studio-work-inspector-cover">{work.coverPath ? <img src={work.coverPath} alt="" /> : <BookOpen size={32} />}</span>
        </span>
        <div><h2>{work.title}</h2><p>{workStatus(work)}</p><small>{work.chapters.length} capítulo{work.chapters.length === 1 ? "" : "s"}</small></div>
      </header>
      {work.genres?.length ? <section><h3>Gêneros</h3><div className="studio-work-inspector-tags">{work.genres.map((genre) => <span key={genre}>{genre}</span>)}</div></section> : null}
      {work.description ? <section><h3>Resumo da obra</h3><p>{work.description}</p></section> : null}
      {(work.sourceLanguage || work.targetLanguage) && <section><h3>Idiomas</h3><p className="studio-work-inspector-languages"><b>{work.sourceLanguage?.toUpperCase() ?? "—"}</b><span>→</span><b>{work.targetLanguage?.toUpperCase() ?? "—"}</b></p></section>}
      <section className="studio-work-inspector-progress"><div><h3>Progresso geral</h3><strong>{totalProgress}% concluído</strong></div><span><i style={{ width: `${totalProgress}%` }} /></span><p>{completedChapters} capítulo{completedChapters === 1 ? "" : "s"} concluído{completedChapters === 1 ? "" : "s"}<em>{Math.max(0, work.chapters.length - completedChapters)} restante{work.chapters.length - completedChapters === 1 ? "" : "s"}</em></p></section>
      <footer><h3>Ações</h3><button type="button" onClick={onEditWork}><PencilLine size={16} /> Editar obra</button><button type="button" disabled title="A exportação é habilitada por capítulo quando houver um formato disponível."><FileDown size={16} /> Exportar projeto</button></footer>
    </aside>
  );
}
