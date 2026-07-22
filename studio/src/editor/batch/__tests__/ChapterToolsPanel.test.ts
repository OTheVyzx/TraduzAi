import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { importStudioProject } from "../../../project/adapters";
import { ChapterToolsPanel } from "../ChapterToolsPanel";

describe("ChapterToolsPanel", () => {
  it("exposes style, find/replace, review and chapter history controls", () => {
    const project = importStudioProject({
      versao: "1.0",
      paginas: [{ numero: 1, textos: [{ id: "a", bbox: [0, 0, 10, 10], traduzido: "Ola", review_required: true }] }],
    }).project;
    const html = renderToStaticMarkup(createElement(ChapterToolsPanel, {
      project,
      currentPageIndex: 0,
      selectedLayerId: "a",
      openByDefault: true,
      onPrepareProject: async () => project,
      onNavigateToLayer: async () => undefined,
    }));

    expect(html).toContain("Ferramentas do capítulo");
    expect(html).toContain("Copiar estilo selecionado");
    expect(html).toContain("Buscar e substituir");
    expect(html).toContain("Fila de revisão");
    expect(html).toContain("Desfazer lote");
  });
});
