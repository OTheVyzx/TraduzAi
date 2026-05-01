import type { Project } from "./stores/appStore";

export interface WorkMemorySummary {
  reviewed_terms: number;
  characters: number;
  ocr_corrections: number;
  previous_chapters: number;
  translation_memory: number;
  sfx_decisions: number;
}

export function buildWorkMemorySummary(project: Project | null): WorkMemorySummary {
  if (!project) {
    return {
      reviewed_terms: 0,
      characters: 0,
      ocr_corrections: 0,
      previous_chapters: 0,
      translation_memory: 0,
      sfx_decisions: 0,
    };
  }
  const memory = project.contexto.memoria_lexical ?? {};
  const qaActions = project.paginas.flatMap((page) =>
    (page.text_layers ?? []).flatMap((layer) => layer.qa_actions ?? []),
  );

  return {
    reviewed_terms: Object.keys(project.contexto.glossario ?? {}).length,
    characters: project.contexto.personagens.length,
    ocr_corrections: qaActions.filter((action) => action.flag_id.includes("ocr")).length,
    previous_chapters: project.capitulo > 1 ? project.capitulo - 1 : 0,
    translation_memory: Object.keys(memory).length,
    sfx_decisions: qaActions.filter((action) => action.ignored_reason?.toLocaleLowerCase("pt-BR").includes("sfx")).length,
  };
}

export function mergeMemorySuggestions(
  reviewedGlossary: Record<string, string>,
  suggestions: Record<string, string>,
) {
  return {
    ...suggestions,
    ...reviewedGlossary,
  };
}
