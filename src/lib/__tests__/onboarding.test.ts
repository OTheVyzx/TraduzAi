import { describe, expect, it } from "vitest";
import { buildOnboardingChecklist, ONBOARDING_FLOW } from "../onboarding";
import type { Project } from "../stores/appStore";

describe("onboarding", () => {
  it("describes the product flow", () => {
    expect(ONBOARDING_FLOW).toEqual([
      "Importe um capitulo",
      "Selecione a obra",
      "Busque contexto online",
      "Revise o glossario",
      "Traduza",
      "Corrija alertas",
      "Exporte",
    ]);
  });

  it("builds setup checklist from project state", () => {
    const checklist = buildOnboardingChecklist({
      obra: "Fixture",
      work_context: { selected: true, context_loaded: false, glossary_loaded: false },
      contexto: { glossario: {} },
    } as unknown as Project);

    expect(checklist.map((step) => [step.id, step.status])).toContainEqual(["import", "done"]);
    expect(checklist.map((step) => [step.id, step.status])).toContainEqual(["work", "done"]);
    expect(checklist.map((step) => [step.id, step.status])).toContainEqual(["context", "warning"]);
    expect(checklist.map((step) => [step.id, step.status])).toContainEqual(["glossary", "warning"]);
  });
});
