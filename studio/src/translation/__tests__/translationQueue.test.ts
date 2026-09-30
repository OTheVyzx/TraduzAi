import { describe, expect, it } from "vitest";
import { importStudioProject } from "../../project/adapters";
import {
  buildTranslationQueue,
  calculatePageTranslationProgress,
  calculateTranslationProgress,
} from "../translationQueue";

function queueProject() {
  return importStudioProject({
    versao: "2.0",
    paginas: [
      {
        numero: 1,
        textos: [
          { id: "pending", bbox: [0, 0, 10, 10], texto: "A", traduzido: "", tipo: "fala" },
          { id: "translated", bbox: [0, 10, 10, 20], texto: "B", traduzido: "Pronto", tipo: "sfx" },
        ],
      },
      {
        numero: 2,
        textos: [
          {
            id: "review",
            bbox: [0, 0, 10, 10],
            texto: "C",
            traduzido: "Revisar",
            translation_status: "review",
            translation_notes: "Checar contexto",
            qa_flags: ["ocr_low_confidence"],
          },
          {
            id: "approved",
            bbox: [0, 10, 10, 20],
            texto: "D",
            traduzido: "Aprovado",
            translation_status: "approved",
          },
        ],
      },
      { numero: 3, textos: [] },
    ],
  }).project;
}

describe("translationQueue", () => {
  it("derives an ordered queue and filters it without mutating the project", () => {
    const project = queueProject();
    const before = JSON.stringify(project);

    expect(buildTranslationQueue(project).map((item) => [item.pageNumber, item.layerId, item.status])).toEqual([
      [1, "pending", "pending"],
      [1, "translated", "translated"],
      [2, "review", "review"],
      [2, "approved", "approved"],
    ]);
    expect(buildTranslationQueue(project, "pending").map((item) => item.layerId)).toEqual(["pending"]);
    expect(buildTranslationQueue(project, "review")[0]).toMatchObject({
      pageIndex: 1,
      blockIndex: 0,
      notes: "Checar contexto",
    });
    expect(JSON.stringify(project)).toBe(before);
  });

  it("filters the queue by text and content class without merging review states", () => {
    const project = queueProject();

    expect(buildTranslationQueue(project, "all", { query: "pronto" }).map((item) => item.layerId))
      .toEqual(["translated"]);
    expect(buildTranslationQueue(project, "all", { textClass: "sfx" }).map((item) => item.layerId))
      .toEqual(["translated"]);
    expect(buildTranslationQueue(project, "review", { query: "contexto" }).map((item) => item.layerId))
      .toEqual(["review"]);
    expect(buildTranslationQueue(project, "all", { query: "ocr_low" })[0]).toMatchObject({
      layerId: "review",
      qaFlags: ["ocr_low_confidence"],
    });
    expect(buildTranslationQueue(project, "pending", { textClass: "sfx" })).toEqual([]);
  });

  it("calculates block progress and keeps empty pages/projects at zero", () => {
    const project = queueProject();

    expect(calculateTranslationProgress(project)).toEqual({
      total: 4,
      completed: 3,
      pending: 1,
      translated: 1,
      review: 1,
      approved: 1,
      percentage: 75,
    });
    expect(calculatePageTranslationProgress(project.paginas[2])).toEqual({
      total: 0,
      completed: 0,
      pending: 0,
      translated: 0,
      review: 0,
      approved: 0,
      percentage: 0,
    });
    expect(calculateTranslationProgress({ ...project, paginas: [] })).toMatchObject({
      total: 0,
      completed: 0,
      percentage: 0,
    });
  });
});
