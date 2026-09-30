import { useState } from "react";
import { ArrowLeft, Boxes, Check, Filter, Grid3X3, Heart, LibraryBig, Pin, Plus, RefreshCw, Search, X } from "lucide-react";
import { extensionSections, groupSources, safeRemoteImageUrl, type ExtensionSectionItem } from "./navigateModel";
import { languageLabel, type InstalledSource } from "./sourceCatalogModel";
import type { InstalledExtension, RepositoryExtension, RuntimeManga } from "./sourceRuntimeClient";
import type { SourceFilter, SourceFilterChange } from "./sourceRuntimeClient";
import { ReaderSourceFilters } from "./ReaderSourceFilters";
import "./readerNavigate.css";

export type NavigateTab = "sources" | "extensions";
export type NavigateSourceMode = "popular" | "latest";
export type NavigateManga = RuntimeManga & { extensionPackage: string; sourceName: string };

export function NavigateTabs({ active, busy, searchOpen, query, onChange, onRefresh, onToggleSearch, onQueryChange, onOpenRepositories, onOpenLanguages }: { active: NavigateTab; busy: boolean; searchOpen: boolean; query: string; onChange(tab: NavigateTab): void; onRefresh(): void; onToggleSearch(): void; onQueryChange(value: string): void; onOpenRepositories(): void; onOpenLanguages(): void }) {
  return <div className="reader-navigate-toolbar">
    <div className="reader-navigate-tabs" role="tablist" aria-label="Navegar"><button type="button" role="tab" aria-selected={active === "sources"} onClick={() => onChange("sources")}>Fontes</button><button type="button" role="tab" aria-selected={active === "extensions"} onClick={() => onChange("extensions")}>Extensões</button></div>
    <div className="reader-navigate-actions">
      {searchOpen && <label><Search size={17} /><input autoFocus aria-label="Filtrar catálogo" value={query} onChange={(event) => onQueryChange(event.currentTarget.value)} placeholder={active === "sources" ? "Filtrar fontes…" : "Filtrar extensões…"} onKeyDown={(event) => { if (event.key === "Escape") onToggleSearch(); }} /></label>}
      <button type="button" aria-label={searchOpen ? "Fechar pesquisa" : "Pesquisar catálogo"} title="Pesquisar" onClick={onToggleSearch}>{searchOpen ? <X size={19} /> : <Search size={19} />}</button>
      <button type="button" aria-label="Gerenciar repositórios" title="Gerenciar repositórios" onClick={onOpenRepositories}><Plus size={20} /></button>
      <button type="button" aria-label="Filtrar idiomas" title="Idiomas permitidos" onClick={onOpenLanguages}><Filter size={19} /></button>
      <button type="button" aria-label="Atualizar catálogo" title="Atualizar catálogo" disabled={busy} onClick={onRefresh}><RefreshCw size={18} /></button>
    </div>
  </div>;
}

function SourceIcon({ source }: { source: InstalledSource }) {
  const iconUrl = safeRemoteImageUrl(source.iconUrl);
  return <span className="reader-source-icon">{iconUrl ? <img src={iconUrl} alt="" loading="lazy" /> : <LibraryBig size={25} aria-hidden="true" />}</span>;
}

export function SourcesTab({ catalogLoaded, sources, recentSourceIdentity, busy, onOpen, onShowExtensions }: { catalogLoaded: boolean; sources: InstalledSource[]; recentSourceIdentity?: string | null; busy: boolean; onOpen(source: InstalledSource, mode: NavigateSourceMode): void; onShowExtensions(): void }) {
  if (!catalogLoaded) return <section className="reader-navigate-empty"><RefreshCw size={28} /><h2>Catálogo ainda não carregado</h2><p>Atualize o catálogo para mostrar suas fontes.</p></section>;
  if (sources.length === 0) return <section className="reader-navigate-empty"><Boxes size={28} /><h2>Nenhuma fonte instalada</h2><p>Instale e ative uma extensão para começar a navegar.</p><button type="button" onClick={onShowExtensions}>Ver extensões</button></section>;
  return <div className="reader-source-groups">{groupSources(sources, recentSourceIdentity).map((group) => <section key={group.id} className="reader-source-group" aria-labelledby={`source-group-${group.id}`}>
    <h2 id={`source-group-${group.id}`}>{group.label}</h2>
    <div>{group.sources.map((source) => <article className="reader-source-row" key={`${source.extensionPackage}:${source.id}`}>
      <button className="reader-source-open" type="button" disabled={busy} onClick={() => onOpen(source, "popular")}>
        <SourceIcon source={source} />
        <span><strong>{source.name}</strong><small>{languageLabel(source.lang)}{(source.nsfw ?? 0) > 0 && <b>18+</b>}</small></span>
      </button>
      <button className="reader-source-latest" type="button" disabled={busy} onClick={() => onOpen(source, "latest")}>MAIS RECENTES</button>
      <Pin size={19} aria-hidden="true" />
    </article>)}</div>
  </section>)}</div>;
}

function ExtensionIcon({ extension }: { extension: RepositoryExtension }) {
  const iconUrl = safeRemoteImageUrl(extension.iconUrl);
  return <span className="reader-extension-icon">{iconUrl ? <img src={iconUrl} alt="" loading="lazy" /> : <Boxes size={24} aria-hidden="true" />}</span>;
}

function ExtensionActions({ item, busy, onInstall, onSetEnabled, onRollback, onUninstall }: { item: ExtensionSectionItem; busy: boolean; onInstall(extension: RepositoryExtension): void; onSetEnabled(extension: InstalledExtension, enabled: boolean): void; onRollback(extension: InstalledExtension): void; onUninstall(extension: InstalledExtension): void }) {
  const { extension, installed } = item;
  const incompatible = !extension.jarUrl;
  const update = Boolean(installed && installed.versionCode < extension.versionCode);
  if (incompatible) return <button type="button" disabled>INCOMPATÍVEL</button>;
  if (update) return <button type="button" disabled={busy} onClick={() => onInstall(extension)}>ATUALIZAR</button>;
  if (!installed) return <button type="button" disabled={busy} onClick={() => onInstall(extension)}>INSTALAR</button>;
  return <div className="reader-extension-controls">
    <button type="button" disabled={busy} onClick={() => onSetEnabled(installed, !installed.enabled)}>{installed.enabled ? "DESATIVAR" : "ATIVAR"}</button>
    {installed.rollbackAvailable && <button type="button" disabled={busy} onClick={() => onRollback(installed)}>VERSÃO ANTERIOR</button>}
    <button className="danger" type="button" disabled={busy} onClick={() => onUninstall(installed)}>DESINSTALAR</button>
  </div>;
}

export function ExtensionsTab({ busy, catalogLoaded, extensions, installed, visibleCount, onInstall, onSetEnabled, onRollback, onUninstall, onLoadMore }: { busy: boolean; catalogLoaded: boolean; extensions: RepositoryExtension[]; installed: InstalledExtension[]; visibleCount: number; onInstall(extension: RepositoryExtension): void; onSetEnabled(extension: InstalledExtension, enabled: boolean): void; onRollback(extension: InstalledExtension): void; onUninstall(extension: InstalledExtension): void; onLoadMore(): void }) {
  if (!catalogLoaded) return <section className="reader-navigate-empty"><RefreshCw size={28} /><h2>Catálogo ainda não carregado</h2><p>Atualize o catálogo para ver as extensões.</p></section>;
  if (extensions.length === 0) return <section className="reader-navigate-empty"><Search size={28} /><h2>Nenhuma extensão encontrada</h2><p>Ajuste o idioma ou o termo pesquisado.</p></section>;
  const shownPackages = new Set(extensions.slice(0, visibleCount).map((extension) => extension.packageName));
  return <div className="reader-extension-sections">{extensionSections(extensions, installed).map((section) => {
    const items = section.id === "updates" || section.id === "installed"
      ? section.items
      : section.items.filter((item) => shownPackages.has(item.extension.packageName));
    if (items.length === 0) return null;
    return <section key={section.id} className={`reader-extension-section ${section.id}`}><h2>{section.label}</h2><div>{items.map((item) => <article key={`${item.extension.repositoryId}:${item.extension.packageName}:${item.extension.versionCode}`}>
      <ExtensionIcon extension={item.extension} />
      <span className="reader-extension-copy"><strong>{item.extension.name}</strong><small>{languageLabel(item.extension.lang)} · {item.extension.versionName}{(item.extension.nsfw ?? 0) > 0 && <b>18+</b>}</small></span>
      {item.installed?.enabled && !item.extension.jarUrl ? <Check size={18} aria-label="Instalada" /> : null}
      <ExtensionActions item={item} busy={busy} onInstall={onInstall} onSetEnabled={onSetEnabled} onRollback={onRollback} onUninstall={onUninstall} />
    </article>)}</div></section>;
  })}{visibleCount < extensions.length && <button className="reader-load-more" type="button" onClick={onLoadMore}>MOSTRAR MAIS</button>}</div>;
}

export function SourceCatalogView({ source, mode, query, heading, busy, items, filters, filtersBusy, filtersError, onLoadFilters, onApplyFilters, onBack, onModeChange, onQueryChange, onSearch, onOpen, onAdd }: { source: InstalledSource; mode: NavigateSourceMode; query: string; heading: string; busy: boolean; items: NavigateManga[]; filters: SourceFilter[]; filtersBusy: boolean; filtersError?: string | null; onLoadFilters(): void; onApplyFilters(changes: SourceFilterChange[]): void; onBack(): void; onModeChange(mode: NavigateSourceMode): void; onQueryChange(value: string): void; onSearch(): void; onOpen(item: NavigateManga): void; onAdd(item: NavigateManga): void }) {
  const [searchOpen, setSearchOpen] = useState(Boolean(query));
  const [filtersOpen, setFiltersOpen] = useState(false);
  return <section className="reader-source-catalog">
    <header className="reader-source-toolbar"><button type="button" aria-label="Voltar para fontes" onClick={onBack}><ArrowLeft size={22} /></button><div><h1>{source.name}</h1><small>{languageLabel(source.lang)}</small></div>{searchOpen && <label><Search size={19} /><span className="sr-only">Pesquisar nesta fonte</span><input autoFocus aria-label="Pesquisar nesta fonte" value={query} onChange={(event) => onQueryChange(event.currentTarget.value)} onKeyDown={(event) => { if (event.key === "Enter") onSearch(); if (event.key === "Escape") { onQueryChange(""); setSearchOpen(false); } }} placeholder="Pesquisar…" /></label>}<button type="button" aria-label={searchOpen ? "Fechar pesquisa na fonte" : "Pesquisar na fonte"} onClick={() => { if (searchOpen) onQueryChange(""); setSearchOpen(!searchOpen); }}>{searchOpen ? <X size={19} /> : <Search size={19} />}</button><Grid3X3 size={19} aria-label="Visualização em grade" /></header>
    <div className="reader-source-modebar">
      <button type="button" aria-pressed={mode === "popular"} disabled={busy} onClick={() => onModeChange("popular")}><Heart size={16} /> POPULARES</button>
      <button type="button" aria-pressed={mode === "latest"} disabled={busy} onClick={() => onModeChange("latest")}><RefreshCw size={16} /> MAIS RECENTES</button>
      <button type="button" aria-expanded={filtersOpen} title="Filtros fornecidos pela fonte" onClick={() => { setFiltersOpen(true); onLoadFilters(); }}><Filter size={16} /> FILTRO</button>
      {query.trim() && <button className="reader-source-search-submit" type="button" disabled={busy} onClick={onSearch}>{busy ? "PESQUISANDO…" : "PESQUISAR"}</button>}
    </div>
    {heading && <p className="reader-source-result-heading">{heading}</p>}
    {!busy && heading && items.length === 0 && <section className="reader-navigate-empty"><Search size={28} /><h2>Nenhuma obra encontrada</h2></section>}
    <div className="reader-cover-grid">{items.map((item) => {
      const cover = safeRemoteImageUrl(item.thumbnailUrl);
      return <article key={`${item.extensionPackage}:${item.sourceId}:${item.url}`}>
        <div className="reader-cover-media">
          <button className="reader-cover-open" type="button" aria-label={`Abrir ${item.title}`} disabled={busy} onClick={() => onOpen(item)}>{cover ? <img src={cover} alt={`Capa de ${item.title}`} loading="lazy" /> : <span className="reader-cover-placeholder"><LibraryBig size={36} /></span>}</button>
          <button className="reader-cover-quick-add" type="button" aria-label={`Adicionar ${item.title} à biblioteca`} title="Adicionar à biblioteca" disabled={busy} onClick={() => onAdd(item)}><Plus size={17} /></button>
        </div>
        <strong title={item.title}>{item.title}</strong>
      </article>;
    })}</div>
    <ReaderSourceFilters open={filtersOpen} filters={filters} busy={filtersBusy} error={filtersError} onClose={() => setFiltersOpen(false)} onApply={(changes) => { onApplyFilters(changes); setFiltersOpen(false); }} />
  </section>;
}
