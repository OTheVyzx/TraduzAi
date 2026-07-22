import { Grid3X3, List, Pencil, Search } from "lucide-react";

export function LibraryToolbar({
  title,
  chapterCount,
  query,
  view,
  thumbnailSize,
  onQueryChange,
  onSetView,
  onSetThumbnailSize,
  onEditWork,
}: {
  title: string;
  chapterCount: number;
  query: string;
  view: "grid" | "list";
  thumbnailSize: number;
  onQueryChange: (query: string) => void;
  onSetView: (view: "grid" | "list") => void;
  onSetThumbnailSize: (size: number) => void;
  onEditWork?: () => void;
}) {
  return (
    <header className="studio-library-toolbar">
      <div className="studio-library-title">
        <h2>Capítulos</h2>
        <span>{title}</span>
        <small>{chapterCount}</small>
        {onEditWork && (
          <button type="button" className="studio-library-edit-work" aria-label="Editar obra" onClick={onEditWork}>
            <Pencil size={12} />
          </button>
        )}
      </div>

      <label className="studio-library-chapter-search">
        <Search size={14} aria-hidden="true" />
        <span className="studio-sr-only">Buscar capítulos</span>
        <input
          type="search"
          value={query}
          placeholder="Buscar capítulos"
          onChange={(event) => onQueryChange(event.currentTarget.value)}
        />
      </label>

      <div className="studio-library-size-control" aria-label="Tamanho das miniaturas">
        <input
          type="range"
          min="112"
          max="240"
          step="8"
          value={thumbnailSize}
          disabled={view === "list"}
          onChange={(event) => onSetThumbnailSize(Number(event.currentTarget.value))}
        />
      </div>

      <div className="studio-library-view-switcher" aria-label="Visualização dos capítulos">
        <button
          type="button"
          aria-label="Visualização em grade"
          aria-pressed={view === "grid"}
          onClick={() => onSetView("grid")}
        >
          <Grid3X3 size={16} />
        </button>
        <button
          type="button"
          aria-label="Visualização em lista"
          aria-pressed={view === "list"}
          onClick={() => onSetView("list")}
        >
          <List size={17} />
        </button>
      </div>
    </header>
  );
}
