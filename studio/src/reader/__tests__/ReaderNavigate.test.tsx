import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ExtensionsTab, NavigateTabs, SourceCatalogView, SourcesTab } from "../ReaderNavigate";
import { ReaderSourceFilters } from "../ReaderSourceFilters";
import type { InstalledExtension, RepositoryExtension } from "../sourceRuntimeClient";
import type { InstalledSource } from "../sourceCatalogModel";

const source: InstalledSource = {
  extensionPackage: "ext.flix",
  extensionName: "MangaFlix",
  id: "20",
  name: "MangaFlix (PT-BR)",
  lang: "pt-BR",
  iconUrl: "https://repo.invalid/flix.png",
  nsfw: 2,
};

const extension: RepositoryExtension = {
  repositoryId: "repo",
  name: "MangaFlix",
  packageName: "ext.flix",
  versionCode: 2,
  versionName: "1.4.2",
  lang: "pt-BR",
  nsfw: 2,
  iconUrl: "https://repo.invalid/flix.png",
  jarUrl: "https://repo.invalid/flix.jar",
  sources: [],
};

const installed: InstalledExtension = {
  packageName: "ext.flix",
  name: "MangaFlix",
  versionCode: 1,
  versionName: "1.4.1",
  repositoryId: "repo",
  sha256: "hash",
  enabled: true,
  rollbackAvailable: false,
};

describe("ReaderNavigate", () => {
  it("renders source-provided filter controls in a floating panel", () => {
    const html = renderToStaticMarkup(createElement(ReaderSourceFilters, {
      open: true,
      busy: false,
      filters: [{ path: [0], type: "select", name: "Categoria", values: ["Todas", "Ação"], value: 0 }],
      onClose: () => undefined,
      onApply: () => undefined,
    }));
    expect(html).toContain("Refinar catálogo");
    expect(html).toContain("Categoria");
    expect(html).toContain("Ação");
    expect(html).toContain("studio-floating-layer");
  });
  it("renders only the approved Fontes and Extensões tabs", () => {
    const html = renderToStaticMarkup(createElement(NavigateTabs, { active: "sources", busy: false, searchOpen: false, query: "", onChange: () => undefined, onRefresh: () => undefined, onToggleSearch: () => undefined, onQueryChange: () => undefined, onOpenRepositories: () => undefined, onOpenLanguages: () => undefined }));
    expect(html).toContain("Fontes");
    expect(html).toContain("Extensões");
    expect(html).not.toContain("Migrar");
    expect(html).toContain('aria-label="Pesquisar catálogo"');
    expect(html).toContain('aria-label="Gerenciar repositórios"');
    expect(html).toContain('aria-label="Filtrar idiomas"');
  });

  it("renders Suwayomi-style source groups with icon, language, warning and latest action", () => {
    const html = renderToStaticMarkup(createElement(SourcesTab, {
      catalogLoaded: true,
      sources: [source, { ...source, id: "21", name: "English Source", lang: "en", nsfw: 0 }],
      recentSourceIdentity: "ext.flix::20",
      busy: false,
      onOpen: () => undefined,
      onShowExtensions: () => undefined,
    }));
    expect(html).toContain("Usado por último");
    expect(html).toContain("Português (Brasil)");
    expect(html).toContain("Inglês");
    expect(html).toContain("MAIS RECENTES");
    expect(html).toContain("18+");
    expect(html).toContain("src=\"https://repo.invalid/flix.png\"");
  });

  it("groups extensions by lifecycle and exposes update controls", () => {
    const html = renderToStaticMarkup(createElement(ExtensionsTab, {
      busy: false,
      catalogLoaded: true,
      extensions: [extension],
      installed: [installed],
      visibleCount: 60,
      onInstall: () => undefined,
      onSetEnabled: () => undefined,
      onRollback: () => undefined,
      onUninstall: () => undefined,
      onLoadMore: () => undefined,
    }));
    expect(html).toContain("Atualização pendente");
    expect(html).toContain("ATUALIZAR");
    expect(html).toContain("MangaFlix");
  });

  it("renders the internal source catalog as a real cover grid", () => {
    const html = renderToStaticMarkup(createElement(SourceCatalogView, {
      source,
      mode: "popular",
      query: "",
      heading: "Obras populares",
      busy: false,
      filters: [],
      filtersBusy: false,
      items: [{ sourceId: "20", url: "/martial-peak", title: "Martial Peak", thumbnailUrl: "https://covers.invalid/martial.webp", status: 1, extensionPackage: "ext.flix", sourceName: "MangaFlix (PT-BR)" }],
      onLoadFilters: () => undefined,
      onApplyFilters: () => undefined,
      onBack: () => undefined,
      onModeChange: () => undefined,
      onQueryChange: () => undefined,
      onSearch: () => undefined,
      onOpen: () => undefined,
      onAdd: () => undefined,
    }));
    expect(html).toContain("POPULARES");
    expect(html).toContain("MAIS RECENTES");
    expect(html).toContain("FILTRO");
    expect(html).toContain("src=\"https://covers.invalid/martial.webp\"");
    expect(html).toContain("Martial Peak");
    expect(html).toContain('aria-label="Abrir Martial Peak"');
    expect(html).toContain('aria-label="Adicionar Martial Peak à biblioteca"');
  });
});
