import type { Project } from "./stores/appStore";

export interface ImportValidationResult {
  valid: boolean;
  pages: number;
  has_project_json: boolean;
  error?: string;
}

export function isProjectArchivePath(path: string): boolean {
  return /\.(zip|cbz)$/i.test(path.trim().replace(/\\/g, "/"));
}

export function shouldOpenExistingProjectFromImport(
  path: string,
  validation: ImportValidationResult,
): boolean {
  return !validation.valid && validation.has_project_json && !isProjectArchivePath(path);
}

export function isTraduzAiProjectSourceError(error: unknown): boolean {
  const message = String(error ?? "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLocaleLowerCase("pt-BR");
  return (
    message.includes("projeto") &&
    message.includes("traduzai") &&
    message.includes("nova traducao")
  );
}

export function shouldLeaveProcessingForCompletedProject(project: Project | null): boolean {
  return Boolean(project && project.status === "done" && project.paginas.length > 0);
}
