import type { StudioProject } from "../project/studioProject";

export type StudioWorkspace = "translation" | "editing";

const WORKSPACE_STORAGE_PREFIX = "traduzai-studio-workspace:";

export function defaultStudioWorkspace(project: StudioProject): StudioWorkspace {
  const hasTranslatedText = project.paginas.some((page) => (
    page.text_layers.some((layer) => layer.translated.trim().length > 0)
  ));
  if (hasTranslatedText) return "editing";

  const context = project.work_context;
  return context?.manual_chapter === true ? "translation" : "editing";
}

export function readStudioWorkspace(
  project: StudioProject,
  projectPath: string,
  storage?: Pick<Storage, "getItem"> | null,
): StudioWorkspace {
  const saved = storage?.getItem(`${WORKSPACE_STORAGE_PREFIX}${normalizeProjectPath(projectPath)}`);
  return saved === "translation" || saved === "editing" ? saved : defaultStudioWorkspace(project);
}

export function writeStudioWorkspace(
  projectPath: string,
  workspace: StudioWorkspace,
  storage?: Pick<Storage, "setItem"> | null,
) {
  storage?.setItem(`${WORKSPACE_STORAGE_PREFIX}${normalizeProjectPath(projectPath)}`, workspace);
}

export function requestWorkspaceClose(dirty: boolean, confirmDiscard: () => boolean): boolean {
  return !dirty || confirmDiscard();
}

function normalizeProjectPath(projectPath: string) {
  const normalized = projectPath.replace(/\\/g, "/");
  return /^[A-Za-z]:\//.test(normalized) ? normalized.toLocaleLowerCase("en-US") : normalized;
}
