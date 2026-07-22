import { BookOpen, Library, Plus, Search } from "lucide-react";
import type { LibraryWork, PublicationStatus } from "./libraryModel";

const STATUS_LABELS: Record<PublicationStatus, string> = {
  releasing: "Em publicação",
  hiatus: "Hiato",
  completed: "Completa",
  cancelled: "Cancelada",
  not_yet_released: "Não iniciada",
  unknown: "Sem status",
};

export function WorkLibrarySidebar({
  works,
  selectedWorkId,
  query,
  onQueryChange,
  onSelectWork,
  onAddWork,
}: {
  works: LibraryWork[];
  selectedWorkId: string | null;
  query: string;
  onQueryChange: (query: string) => void;
  onSelectWork: (workId: string) => void;
  onAddWork: () => void;
}) {
  const normalizedQuery = query.trim().toLocaleLowerCase("pt-BR");
  const visibleWorks = works.filter((work) => {
    if (!normalizedQuery) return true;
    return [work.title, ...work.aliases]
      .join(" ")
      .toLocaleLowerCase("pt-BR")
      .includes(normalizedQuery);
  });

  return (
    <aside className="studio-library-sidebar" aria-label="Biblioteca de obras">
      <div className="studio-library-sidebar-heading">
        <Library size={16} aria-hidden="true" />
        <h1>Obras</h1>
        <span>{works.length}</span>
      </div>

      <label className="studio-library-search">
        <Search size={14} aria-hidden="true" />
        <span className="studio-sr-only">Buscar obras</span>
        <input
          type="search"
          value={query}
          placeholder="Buscar obras"
          onChange={(event) => onQueryChange(event.currentTarget.value)}
        />
      </label>

      <div className="studio-work-list">
        {visibleWorks.length > 0 ? visibleWorks.map((work) => (
          <button
            key={work.id}
            type="button"
            className="studio-work-item"
            aria-current={work.id === selectedWorkId ? "true" : undefined}
            onClick={() => onSelectWork(work.id)}
          >
            <span className="studio-work-cover" aria-hidden="true">
              {work.coverPath ? <img src={work.coverPath} alt="" /> : <BookOpen size={18} />}
            </span>
            <span className="studio-work-copy">
              <strong>{work.title}</strong>
              <small>{work.chapters.length} capítulos · {STATUS_LABELS[work.publicationStatus]}</small>
            </span>
          </button>
        )) : (
          <div className="studio-library-sidebar-empty">
            {works.length === 0 ? "Sua biblioteca está vazia." : "Nenhuma obra encontrada."}
          </div>
        )}
      </div>

      <button type="button" className="studio-library-add-work" onClick={onAddWork}>
        <Plus size={15} aria-hidden="true" />
        Adicionar obra
      </button>
    </aside>
  );
}
