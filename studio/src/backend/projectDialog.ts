import { invoke as tauriInvoke } from "@tauri-apps/api/core";
import { createManualProject } from "../project/createManualProject";
import type { StudioProject } from "../project/studioProject";
import { getStudioEditorBackend } from "./editorBackend";

function isTauriRuntime() {
  return typeof window !== "undefined" && ("__TAURI_INTERNALS__" in window || "__TAURI__" in window);
}

export function canUseProjectDialogs() {
  return isTauriRuntime();
}

export async function openProjectDialog() {
  if (!isTauriRuntime()) {
    throw new Error("Abrir projeto pelo sistema de arquivos está disponível apenas no app desktop.");
  }
  const { open } = await import("@tauri-apps/plugin-dialog");
  const selected = await open({
    multiple: false,
    directory: false,
    filters: [{ name: "Projeto TraduzAI", extensions: ["json"] }],
    title: "Abrir project.json",
  });
  return typeof selected === "string" ? selected : null;
}

export async function openProjectForAttachment() {
  const projectPath = await openProjectDialog();
  if (!projectPath) return null;
  const project = await getStudioEditorBackend().loadProject({ project_path: projectPath });
  return { projectPath, project };
}

export async function openCoverImageDialog() {
  if (!isTauriRuntime()) {
    throw new Error("Escolher capa pelo sistema de arquivos está disponível apenas no app desktop.");
  }
  const { open } = await import("@tauri-apps/plugin-dialog");
  const selected = await open({
    multiple: false,
    directory: false,
    filters: [{ name: "Imagem", extensions: ["png", "jpg", "jpeg", "webp"] }],
    title: "Escolher capa da obra",
  });
  return typeof selected === "string" ? selected : null;
}

export async function openManualChapterFolderDialog() {
  if (!isTauriRuntime()) {
    throw new Error("Escolher uma pasta está disponível apenas no app desktop.");
  }
  const { open } = await import("@tauri-apps/plugin-dialog");
  const selected = await open({
    multiple: false,
    directory: true,
    title: "Escolher pasta de imagens",
  });
  return typeof selected === "string" ? selected : null;
}

export async function openManualChapterArchiveDialog() {
  if (!isTauriRuntime()) {
    throw new Error("Escolher ZIP ou CBZ está disponível apenas no app desktop.");
  }
  const { open } = await import("@tauri-apps/plugin-dialog");
  const selected = await open({
    multiple: false,
    directory: false,
    filters: [{ name: "Capítulo compactado", extensions: ["zip", "cbz"] }],
    title: "Escolher ZIP ou CBZ",
  });
  return typeof selected === "string" ? selected : null;
}

export interface PreparedManualPage {
  number: number;
  relativePath: string;
  width: number;
  height: number;
}

export interface ManualChapterCreationInput {
  workTitle: string;
  chapterLabel: string;
  chapterTitle?: string;
  sourceLanguage: string;
  targetLanguage: string;
  sourcePath: string;
  projectJsonPath: string;
}

export interface ManualChapterCreationRuntime {
  prepare(sourcePath: string, projectJsonPath: string): Promise<PreparedManualPage[]>;
  save(projectJsonPath: string, project: StudioProject): Promise<void>;
}

export class ManualChapterCreationError extends Error {
  readonly preparedPages: PreparedManualPage[] | null;

  constructor(message: string, preparedPages: PreparedManualPage[] | null) {
    super(message);
    this.name = "ManualChapterCreationError";
    this.preparedPages = preparedPages;
  }
}

function defaultManualChapterRuntime(): ManualChapterCreationRuntime {
  return {
    prepare: (sourcePath, projectJsonPath) => tauriInvoke<PreparedManualPage[]>("studio_prepare_manual_chapter", {
      sourcePath,
      projectJsonPath,
    }),
    save: (projectJsonPath, project) => getStudioEditorBackend().saveProjectJson({
      project_path: projectJsonPath,
      project_json: project,
    }),
  };
}

export async function createManualChapterFromImages(
  input: ManualChapterCreationInput,
  runtime: ManualChapterCreationRuntime = defaultManualChapterRuntime(),
  preparedPages: PreparedManualPage[] | null = null,
) {
  let pages = preparedPages;
  try {
    pages ??= await runtime.prepare(input.sourcePath, input.projectJsonPath);
    const project = {
      ...createManualProject({
        workTitle: input.workTitle,
        chapterLabel: input.chapterLabel,
        chapterTitle: input.chapterTitle,
        sourceLanguage: input.sourceLanguage,
        targetLanguage: input.targetLanguage,
        pages,
      }),
      source_path: input.projectJsonPath,
      output_path: input.projectJsonPath,
    };
    await runtime.save(input.projectJsonPath, project);
    return { project, preparedPages: pages };
  } catch (cause) {
    const message = cause instanceof Error ? cause.message : String(cause);
    throw new ManualChapterCreationError(message, pages);
  }
}

export async function projectPathExists(projectPath: string) {
  if (!isTauriRuntime()) return true;
  const { exists } = await import("@tauri-apps/plugin-fs");
  return exists(projectPath);
}

export async function saveProjectDialog(defaultPath = "project.json") {
  if (!isTauriRuntime()) {
    throw new Error("Salvar projeto pelo sistema de arquivos está disponível apenas no app desktop.");
  }
  const { save } = await import("@tauri-apps/plugin-dialog");
  const selected = await save({
    defaultPath,
    filters: [{ name: "Projeto TraduzAI", extensions: ["json"] }],
    title: "Salvar project.json",
  });
  return typeof selected === "string" ? selected : null;
}
