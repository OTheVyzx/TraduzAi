import { describe, expect, it } from "vitest";

import {
  isTraduzAiProjectSourceError,
  shouldLeaveProcessingForCompletedProject,
  shouldOpenExistingProjectFromImport,
} from "../projectSourceGuards.ts";
import type { Project } from "../stores/appStore";

describe("projectSourceGuards", () => {
  it("abre pasta de projeto em vez de nova traducao", () => {
    expect(shouldOpenExistingProjectFromImport("N:/TraduzAI/saida", {
      valid: false,
      pages: 0,
      has_project_json: true,
      error: "Esta pasta ja e um projeto TraduzAi.",
    })).toBe(true);
  });

  it("nao tenta abrir ZIP exportado direto", () => {
    expect(shouldOpenExistingProjectFromImport("N:/TraduzAI/traduzido.zip", {
      valid: false,
      pages: 0,
      has_project_json: true,
      error: "Este arquivo ja e um projeto/exportacao do TraduzAi.",
    })).toBe(false);
  });

  it("reconhece erro defensivo do pipeline", () => {
    expect(isTraduzAiProjectSourceError(
      "Esta pasta ja e um projeto TraduzAi. Use Abrir projeto para continuar, nao Nova traducao.",
    )).toBe(true);
  });

  it("evita reprocessar projeto pronto", () => {
    const project = {
      status: "done",
      paginas: [{ numero: 1 }],
    } as Project;

    expect(shouldLeaveProcessingForCompletedProject(project)).toBe(true);
  });
});
