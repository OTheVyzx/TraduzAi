import { describe, expect, it } from "vitest";
import { importStudioProject } from "../../project/adapters";
import { toAppProject } from "../StudioSharedEditor";

describe("projeção do projeto Studio para o editor", () => {
  it("does not invent full OCR confidence when provenance is absent", () => {
    const project = importStudioProject({
      versao: "2.0",
      paginas: [{
        numero: 1,
        textos: [
          { id: "unknown", bbox: [0, 0, 10, 10], texto: "Sem proveniência" },
          { id: "measured", bbox: [0, 10, 10, 20], texto: "Medido", ocr_confidence: 0.72 },
        ],
      }],
    }).project;

    const projected = toAppProject(project, "memory://projection");

    expect(projected.paginas[0].text_layers[0]).toMatchObject({
      confianca_ocr: 0,
      ocr_confidence: 0,
    });
    expect(projected.paginas[0].text_layers[1]).toMatchObject({
      confianca_ocr: 0.72,
      ocr_confidence: 0.72,
    });
    expect(projected.status).toBe("idle");
  });

  it("preserves status but binds editor writes to the currently opened project", () => {
    const project = importStudioProject({
      versao: "2.0",
      status: "needs_review",
      mode: "auto",
      qualidade: "rapida",
      output_path: "N:/outputs/chapter-1",
      paginas: [{ numero: 1, textos: [] }],
    }).project;

    expect(toAppProject(project, "memory://status")).toMatchObject({
      status: "needs_review",
      mode: "auto",
      qualidade: "rapida",
      source_path: "memory://status",
      output_path: "memory://status",
    });
  });

  it("does not save a promoted project back into its seed output path", () => {
    const project = importStudioProject({
      versao: "2.0",
      source_path: "N:/seed",
      output_path: "N:/seed/project.json",
      paginas: [{ numero: 1, textos: [] }],
    }).project;

    expect(toAppProject(project, "N:/promoted/r47/project.json")).toMatchObject({
      source_path: "N:/promoted/r47/project.json",
      output_path: "N:/promoted/r47/project.json",
    });
  });

  it("derives a blocked editor state from a persisted export gate", () => {
    const project = importStudioProject({
      versao: "2.0",
      qa: {
        export_gate: {
          status: "BLOCK",
          critical_issue_count: 1,
          review_issue_count: 0,
          needs_review: false,
        },
      },
      paginas: [{ numero: 1, textos: [] }],
    }).project;

    expect(toAppProject(project, "memory://blocked").status).toBe("done_blocked");
  });
});
