import { importStudioProject } from "./adapters";
import type { StudioProject } from "./studioProject";

export interface ManualProjectPage {
  number: number;
  relativePath: string;
  width: number;
  height: number;
}

export interface CreateManualProjectInput {
  workTitle: string;
  chapterLabel: string;
  chapterTitle?: string;
  sourceLanguage: string;
  targetLanguage: string;
  pages: ManualProjectPage[];
}

export function createManualProject(input: CreateManualProjectInput): StudioProject {
  if (input.pages.length === 0) {
    throw new Error("O capítulo manual precisa de pelo menos uma página válida.");
  }

  return importStudioProject({
    app: "traduzai",
    versao: "2.0",
    studio_schema_version: "1.0",
    obra: input.workTitle.trim() || "Obra sem título",
    capitulo: input.chapterLabel.trim() || "1",
    ...(input.chapterTitle?.trim() ? { chapter_title: input.chapterTitle.trim() } : {}),
    idioma_origem: input.sourceLanguage.trim() || "en",
    idioma_destino: input.targetLanguage.trim() || "pt-BR",
    paginas: input.pages.map((page, index) => ({
      numero: page.number || index + 1,
      width: page.width,
      height: page.height,
      arquivo_original: page.relativePath,
      arquivo_traduzido: null,
      image_layers: {
        base: {
          key: "base",
          path: page.relativePath,
          visible: true,
          locked: true,
          opacity: 1,
          order: 0,
        },
      },
      text_layers: [],
      textos: [],
    })),
    work_context: {
      manual_chapter: true,
      created_by: "traduzai-studio",
      ...(input.chapterTitle?.trim() ? { chapter_title: input.chapterTitle.trim() } : {}),
    },
  }).project;
}
