import { describe, expect, it } from "vitest";
import {
  EDITOR_PROFESSIONAL_SHORTCUTS,
  EDITOR_PROFESSIONAL_TOOLS,
  editorProfessionalReadiness,
} from "../editorProfessionalTools";

describe("editorProfessionalTools", () => {
  it("lista ferramentas visuais esperadas para revisao manual", () => {
    expect(EDITOR_PROFESSIONAL_TOOLS.map((tool) => tool.label)).toEqual([
      "Selecionar",
      "Mover texto",
      "Editar texto",
      "Brush de mascara",
      "Borracha",
      "Reprocessar regiao",
      "Comparar original/final",
    ]);
  });

  it("expoe atalhos essenciais do editor", () => {
    expect(EDITOR_PROFESSIONAL_SHORTCUTS).toContain("Ctrl+Z desfaz");
    expect(EDITOR_PROFESSIONAL_SHORTCUTS).toContain("Ctrl+S salva");
    expect(EDITOR_PROFESSIONAL_SHORTCUTS).toContain("1/2/3 troca visualizacao");
  });

  it("calcula prontidao profissional a partir dos recursos disponiveis", () => {
    expect(
      editorProfessionalReadiness({
        hasLayersPanel: true,
        hasTextProperties: true,
        hasMaskTools: true,
        hasBeforeAfter: true,
        hasUndoRedo: true,
      }),
    ).toEqual({ passed: 5, total: 5, ready: true });
  });
});
