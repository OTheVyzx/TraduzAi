import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { StudioHomeDashboard } from "../StudioHomeDashboard";
import { createEmptyLibrary, type StudioLibrary } from "../../library/libraryModel";

const library: StudioLibrary = {
  ...createEmptyLibrary(),
  works: [{
    id: "work-1",
    title: "Gosu",
    aliases: ["The Master"],
    genres: ["Ação", "Artes Marciais"],
    publicationStatus: "releasing",
    external: {},
    chapters: [{
      id: "chapter-1", label: "1", title: "Despertar", projectPath: "memory://gosu/1", pageCount: 24,
      completedPages: 24, workflowStatus: "completed", lastOpenedAt: "2026-09-02T10:00:00Z",
    }],
  }],
};

describe("StudioHomeDashboard", () => {
  it("presents work-level metrics and routes a recently edited work into Biblioteca", () => {
    const html = renderToStaticMarkup(createElement(StudioHomeDashboard, {
      document: library,
      onOpenWork: () => undefined,
      onCreateWork: () => undefined,
      onImportProject: () => undefined,
      onOpenSettings: () => undefined,
    }));

    expect(html).toContain("Obras totais");
    expect(html).toContain("Obras traduzidas");
    expect(html).toContain("Em edição");
    expect(html).toContain("Em revisão");
    expect(html).toContain("Obras recentemente editadas");
    expect(html).toContain("Gosu");
    expect(html).not.toContain('aria-label="Filtros da biblioteca"');
    expect(html).not.toContain("Navegação");
    expect(html).not.toContain("Etiquetas");
    expect(html).not.toContain("Ação");
    expect(html).toContain("Nova obra");
    expect(html).toContain("Importar obra/projeto");
    expect(html).not.toContain("Capítulos totais");
    expect(html).toContain('class="studio-home-metric-line"><strong>1</strong><h2>Obras totais</h2>');
    expect(html).not.toContain("Obras cadastradas");
    expect(html).not.toContain("% do total de obras");
    expect(html).not.toContain("Obras em processo de edição");
    expect(html).not.toContain("Aguardando revisão final");
  });
});
