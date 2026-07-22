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
import { getStudioEditorBackend } from "./editorBackend";
