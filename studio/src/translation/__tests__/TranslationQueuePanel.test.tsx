import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { importStudioProject } from "../../project/adapters";
import { TranslationQueuePanel } from "../TranslationQueuePanel";

describe("TranslationQueuePanel", () => {
  it("exposes text, class and status filters without collapsing their meanings", () => {
    const project = importStudioProject({
      versao: "2.0",
      paginas: [{
        numero: 1,
        textos: [
          { id: "speech", bbox: [0, 0, 10, 10], texto: "Origem", traduzido: "Destino", tipo: "fala" },
          { id: "sfx", bbox: [0, 10, 10, 20], texto: "BOOM", traduzido: "BUM", tipo: "sfx", translation_status: "review", qa_flags: ["mask_review_required"] },
        ],
      }],
    }).project;
    const html = renderToStaticMarkup(createElement(TranslationQueuePanel, {
      project,
      filter: "all",
      currentPageIndex: 0,
      selectedLayerId: null,
      onFilterChange: () => undefined,
      onSelectTarget: () => undefined,
    }));

    expect(html).toContain('aria-label="Buscar texto na fila"');
    expect(html).toContain('aria-label="Filtrar por classe"');
    expect(html).toContain("Todas as classes");
    expect(html).toContain(">fala<");
    expect(html).toContain(">sfx<");
    expect(html).toContain("Revisão");
    expect(html).toContain("mask_review_required");
  });
});
