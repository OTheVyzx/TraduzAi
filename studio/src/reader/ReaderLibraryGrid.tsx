import { MoreVertical, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import type { ReaderDocument } from "./readerModel";
import { ReaderCover } from "./ReaderCover";

export function ReaderLibraryGrid({ document, newCategoryName, onNewCategoryName, onAddCategory, onToggleCategory, onOpen, onSetAutoUpdate, onRemove }: {
  document: ReaderDocument;
  newCategoryName: string;
  onNewCategoryName(value: string): void;
  onAddCategory(): void;
  onToggleCategory(mangaId: string, categoryId: string): void;
  onOpen(mangaId: string): void;
  onSetAutoUpdate(mangaId: string, enabled: boolean): void;
  onRemove(mangaId: string): void;
}) {
  const [menuMangaId, setMenuMangaId] = useState<string | null>(null);
  const [categoryFormOpen, setCategoryFormOpen] = useState(false);
  return <section className="reader-library-surface">
    <header className="reader-library-toolbar"><div><p className="eyebrow">Obras salvas</p><h1>Biblioteca do leitor</h1></div><button type="button" aria-label="Nova categoria" onClick={() => setCategoryFormOpen((open) => !open)}><Plus size={18} /></button></header>
    {categoryFormOpen && <form className="reader-library-category-form" onSubmit={(event) => { event.preventDefault(); onAddCategory(); }}><input autoFocus aria-label="Nova categoria" value={newCategoryName} onChange={(event) => onNewCategoryName(event.currentTarget.value)} placeholder="Nome da categoria…" /><button type="submit" disabled={!newCategoryName.trim()}>Adicionar</button></form>}
    <div className="reader-library-cover-grid">{document.manga.map((manga) => {
      const unread = manga.chapters.filter((chapter) => !chapter.read).length;
      return <article className="reader-library-card" key={manga.id}>
        <ReaderCover src={manga.thumbnailUrl} className="reader-library-ambient" />
        <button className="reader-library-open" type="button" aria-label={`Abrir ${manga.title}`} onClick={() => onOpen(manga.id)}><ReaderCover src={manga.thumbnailUrl} alt={`Capa de ${manga.title}`} /><span className="reader-library-card-copy"><strong>{manga.title}</strong><small>{unread} {unread === 1 ? "não lido" : "não lidos"}</small></span></button>
        <button className="reader-library-menu-trigger" type="button" aria-label={`Ações de ${manga.title}`} aria-expanded={menuMangaId === manga.id} onClick={() => setMenuMangaId((current) => current === manga.id ? null : manga.id)}><MoreVertical size={19} /></button>
        {menuMangaId === manga.id && <div className="reader-library-menu" role="menu">
          <label><input type="checkbox" checked={manga.autoUpdate} onChange={(event) => onSetAutoUpdate(manga.id, event.currentTarget.checked)} /> Atualizar automaticamente</label>
          {document.categories.map((category) => <label key={category.id}><input type="checkbox" checked={manga.categoryIds.includes(category.id)} onChange={() => onToggleCategory(manga.id, category.id)} /> {category.name}</label>)}
          <button type="button" role="menuitem" onClick={() => onRemove(manga.id)}><Trash2 size={16} /> Remover da biblioteca</button>
        </div>}
      </article>;
    })}</div>
  </section>;
}
