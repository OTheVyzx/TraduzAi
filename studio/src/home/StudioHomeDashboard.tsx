import { Archive, BookOpen, CheckCircle2, ChevronRight, Clock3, FolderInput, PencilLine, Plus, Settings2 } from "lucide-react";
import type { LibraryWork, StudioLibrary } from "../library/libraryModel";
import { selectHomeWorkMetrics, selectRecentWorks } from "../library/librarySelectors";

function workStatus(work: LibraryWork): string {
  if (work.chapters.some((chapter) => chapter.workflowStatus === "review")) return "Em revisão";
  if (work.chapters.some((chapter) => chapter.workflowStatus === "editing")) return "Em edição";
  if (work.chapters.length > 0 && work.chapters.every((chapter) => chapter.workflowStatus === "completed")) return "Concluída";
  return work.publicationStatus === "releasing" ? "Em andamento" : "Sem status";
}

function latestTimestamp(work: LibraryWork): number {
  return Math.max(0, ...work.chapters.map((chapter) => Date.parse(chapter.lastOpenedAt ?? "") || 0));
}

function relativeDate(work: LibraryWork): string {
  const timestamp = latestTimestamp(work);
  if (!timestamp) return "Sem edição registrada";
  const minutes = Math.max(0, Math.round((Date.now() - timestamp) / 60_000));
  if (minutes < 60) return `Atualizado há ${minutes || 1} min`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `Atualizado há ${hours} h`;
  return `Atualizado há ${Math.round(hours / 24)} dia${hours >= 48 ? "s" : ""}`;
}

function WorkCover({ work }: { work: LibraryWork }) {
  return (
    <span className="studio-home-work-cover" aria-hidden="true">
      {work.coverPath ? <img src={work.coverPath} alt="" /> : <BookOpen size={22} />}
    </span>
  );
}

export function StudioHomeDashboard({
  document,
  onOpenWork,
  onCreateWork,
  onImportProject,
  onOpenSettings,
}: {
  document: StudioLibrary;
  onOpenWork: (workId: string) => void;
  onCreateWork: () => void;
  onImportProject: () => void;
  onOpenSettings: () => void;
}) {
  const metrics = selectHomeWorkMetrics(document);
  const visibleWorks = selectRecentWorks(document);
  const metricCards = [
    { label: "Obras totais", value: metrics.totalWorks, icon: BookOpen, tone: "blue" },
    { label: "Obras traduzidas", value: metrics.translatedWorks, icon: CheckCircle2, tone: "green" },
    { label: "Em edição", value: metrics.editingWorks, icon: PencilLine, tone: "violet" },
    { label: "Em revisão", value: metrics.reviewWorks, icon: Clock3, tone: "amber" },
  ];

  return (
    <section className="studio-home-dashboard">
      <div className="studio-home-main">
        <div className="studio-home-metrics">
          {metricCards.map(({ label, value, icon: Icon, tone }) => <article className={`studio-home-metric studio-home-metric-${tone}`} key={label}><Icon size={27} /><div className="studio-home-metric-line"><strong>{value}</strong><h2>{label}</h2></div></article>)}
        </div>
        <div className="studio-home-heading"><h1>Visão geral</h1><p>Acompanhe suas obras e continue de onde parou.</p></div>
        <section className="studio-home-recent" aria-labelledby="studio-home-recent-title"><h2 id="studio-home-recent-title">Obras recentemente editadas</h2><div>{visibleWorks.length ? visibleWorks.slice(0, 8).map((work) => <button type="button" className="studio-home-work-row" key={work.id} onClick={() => onOpenWork(work.id)}><WorkCover work={work} /><span className="studio-home-work-copy"><strong>{work.title}</strong><small className={`studio-status-${workStatus(work).replaceAll(" ", "-").toLocaleLowerCase("pt-BR")}`}>{workStatus(work)}</small><em>{work.chapters.length} capítulo{work.chapters.length === 1 ? "" : "s"}</em></span><time>{relativeDate(work)}</time><ChevronRight size={19} /></button>) : <p className="studio-home-empty">Nenhuma obra adicionada ainda.</p>}</div></section>
      </div>

      <aside className="studio-home-quick-actions"><h2>Ações rápidas</h2><button type="button" onClick={onCreateWork}><Plus /> <span><strong>Nova obra</strong><small>Crie uma nova obra</small></span><ChevronRight /></button><button type="button" onClick={onImportProject}><FolderInput /> <span><strong>Importar obra/projeto</strong><small>Importe um projeto existente</small></span><ChevronRight /></button><button type="button" onClick={onOpenSettings}><Settings2 /> <span><strong>Configurações</strong><small>Ajustes do aplicativo</small></span><ChevronRight /></button><p><Archive size={15} /> A exportação fica disponível dentro de uma obra ou capítulo.</p></aside>
    </section>
  );
}
