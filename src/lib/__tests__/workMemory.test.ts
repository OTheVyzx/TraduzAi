import { describe, expect, it } from "vitest";
import { buildWorkMemorySummary, mergeMemorySuggestions } from "../workMemory";
import type { Project } from "../stores/appStore";

const project = {
  capitulo: 4,
  contexto: {
    glossario: { Hero: "Heroi" },
    personagens: ["Hero"],
    memoria_lexical: { "Thank you": "Obrigado" },
  },
  paginas: [
    {
      text_layers: [
        {
          qa_actions: [
            { flag_id: "ocr_suspect", status: "ignored", ignored_reason: "corrigido", ignored_at: "" },
            { flag_id: "visual_text_leak", status: "ignored", ignored_reason: "SFX preservado", ignored_at: "" },
          ],
        },
      ],
    },
  ],
} as unknown as Project;

describe("workMemory", () => {
  it("summarizes loaded work memory", () => {
    expect(buildWorkMemorySummary(project)).toMatchObject({
      reviewed_terms: 1,
      characters: 1,
      ocr_corrections: 1,
      previous_chapters: 3,
      translation_memory: 1,
      sfx_decisions: 1,
    });
  });

  it("does not overwrite reviewed glossary with memory suggestions", () => {
    expect(mergeMemorySuggestions({ Hero: "Heroi revisado" }, { Hero: "Heroi automatico", Villain: "Vilao" })).toEqual({
      Hero: "Heroi revisado",
      Villain: "Vilao",
    });
  });
});
