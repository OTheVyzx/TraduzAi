import type { StudioEditorBackend } from "../backend/editorBackend";
import type { StudioProject } from "../project/studioProject";

export const STUDIO_RECOVERY_VERSION = "1.0" as const;

export interface StudioRecoverySnapshot {
  version: typeof STUDIO_RECOVERY_VERSION;
  projectPath: string;
  savedAt: number;
  project: StudioProject;
}

export function parseStudioRecoverySnapshot(
  value: unknown,
  expectedProjectPath?: string,
): StudioRecoverySnapshot | null {
  if (typeof value !== "object" || value === null) return null;
  const record = value as Record<string, unknown>;
  const project = record.project;
  if (
    record.version !== STUDIO_RECOVERY_VERSION
    || typeof record.projectPath !== "string"
    || typeof record.savedAt !== "number"
    || !Number.isFinite(record.savedAt)
    || record.savedAt < 0
    || typeof project !== "object"
    || project === null
    || (project as Record<string, unknown>).app !== "traduzai"
    || !Array.isArray((project as Record<string, unknown>).paginas)
  ) {
    return null;
  }
  if (expectedProjectPath && normalizeProjectPath(record.projectPath) !== normalizeProjectPath(expectedProjectPath)) {
    return null;
  }
  return clone(value as StudioRecoverySnapshot);
}

function normalizeProjectPath(projectPath: string) {
  const normalized = projectPath.replace(/\\/g, "/").replace(/\/$/, "");
  return /^[A-Za-z]:\//.test(normalized) ? normalized.toLocaleLowerCase() : normalized;
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export function createRecoverySnapshot(
  projectPath: string,
  project: StudioProject,
  savedAt = Date.now(),
): StudioRecoverySnapshot {
  return {
    version: STUDIO_RECOVERY_VERSION,
    projectPath,
    savedAt,
    project: clone(project),
  };
}

export function isRecoveryCandidate(
  snapshot: StudioRecoverySnapshot | null,
  currentProject: StudioProject | null,
  expectedProjectPath?: string,
) {
  if (snapshot && expectedProjectPath && normalizeProjectPath(snapshot.projectPath) !== normalizeProjectPath(expectedProjectPath)) {
    return false;
  }
  if (!snapshot || !currentProject) return Boolean(snapshot);
  return JSON.stringify(snapshot.project) !== JSON.stringify(currentProject);
}

export async function runStudioAutosaveCycle({
  backend,
  projectPath,
  dirty,
  runAutoSave,
  now = Date.now,
}: {
  backend: StudioEditorBackend;
  projectPath: string;
  dirty: boolean;
  runAutoSave: () => Promise<void>;
  now?: () => number;
}) {
  if (dirty) await runAutoSave();
  const project = await backend.loadProject({ project_path: projectPath });
  const snapshot = createRecoverySnapshot(projectPath, project, now());
  await backend.saveRecoverySnapshot({ project_path: projectPath, snapshot });
  return snapshot;
}

export async function recoverStudioProject(
  backend: StudioEditorBackend,
  projectPath: string,
) {
  const snapshot = await backend.loadRecoverySnapshot({ project_path: projectPath });
  const verified = parseStudioRecoverySnapshot(snapshot, projectPath);
  if (!verified) return null;
  await backend.saveProjectJson({ project_path: projectPath, project_json: verified.project });
  const restored = await backend.loadProject({ project_path: projectPath });
  await backend.clearRecoverySnapshot({ project_path: projectPath });
  return restored;
}
