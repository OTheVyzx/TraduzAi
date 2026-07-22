import { create } from "zustand";
import { configureStudioEditorBackend, getStudioEditorBackend } from "../backend/editorBackend";
import { createLegacyEditorBackendAdapter } from "../backend/editorBackendCompat";
import { openProjectDialog, saveProjectDialog } from "../backend/projectDialog";
import { createDefaultStudioBackend } from "../backend/tauriBackend";
import {
  createRecoverySnapshot,
  isRecoveryCandidate,
  recoverStudioProject,
  type StudioRecoverySnapshot,
} from "../autosave/recovery";
import {
  applyChapterHistoryEntry,
  createChapterHistoryEntry,
  type ChapterCommand,
  type ChapterHistoryEntry,
} from "../editor/batch/chapterCommands";
import { importStudioProject, toTraduzAiV2Compat } from "../project/adapters";
import type { ImageLayerKey, ProjectImportResult, StudioProject, StudioTextLayer } from "../project/studioProject";

const DEFAULT_PROJECT_PATH = "memory://current";
const MAX_CHAPTER_HISTORY = 30;

async function refreshRecoverySnapshot(projectPath: string, project: StudioProject) {
  try {
    await getStudioEditorBackend().saveRecoverySnapshot({
      project_path: projectPath,
      snapshot: createRecoverySnapshot(projectPath, project),
    });
  } catch (error) {
    console.error("Falha ao atualizar snapshot de recuperacao do Studio:", error);
  }
}

export interface StudioProjectState {
  project: StudioProject | null;
  projectPath: string | null;
  currentPageIndex: number;
  lastImport: ProjectImportResult | null;
  error: string | null;
  chapterHistory: ChapterHistoryEntry[];
  chapterHistoryIndex: number;
  isProjectSaving: boolean;
  hasUnsavedChanges: boolean;
  recoverySnapshot: StudioRecoverySnapshot | null;
  importProjectJson: (jsonText: string, projectPath?: string) => Promise<void>;
  loadProject: (projectPath: string) => Promise<void>;
  openProjectFromDialog: () => Promise<void>;
  closeProject: (force?: boolean) => boolean;
  saveProject: () => Promise<void>;
  saveProjectAsFromDialog: () => Promise<void>;
  setCurrentPageIndex: (pageIndex: number) => void;
  patchCurrentTextLayer: (layerId: string, patch: Partial<StudioTextLayer>) => Promise<void>;
  setCurrentTextLayerVisibility: (layerId: string, visible: boolean) => Promise<void>;
  setCurrentImageLayerVisibility: (layerKey: ImageLayerKey, visible: boolean) => Promise<void>;
  executeChapterCommand: (command: ChapterCommand) => Promise<boolean>;
  undoChapterCommand: () => Promise<boolean>;
  redoChapterCommand: () => Promise<boolean>;
  restoreRecovery: () => Promise<boolean>;
  dismissRecovery: () => Promise<void>;
  clearError: () => void;
}

const studioBackend = createDefaultStudioBackend();
configureStudioEditorBackend(studioBackend);

export const useStudioProjectStore = create<StudioProjectState>((set, get) => ({
  project: null,
  projectPath: null,
  currentPageIndex: 0,
  lastImport: null,
  error: null,
  chapterHistory: [],
  chapterHistoryIndex: 0,
  isProjectSaving: false,
  hasUnsavedChanges: false,
  recoverySnapshot: null,

  importProjectJson: async (jsonText, projectPath = DEFAULT_PROJECT_PATH) => {
    try {
      const payload = JSON.parse(jsonText) as unknown;
      const result = importStudioProject(payload);
      studioBackend.putProject(projectPath, result.project);
      set({
        project: result.project,
        projectPath,
        currentPageIndex: 0,
        lastImport: result,
        error: null,
        chapterHistory: [],
        chapterHistoryIndex: 0,
        isProjectSaving: false,
        hasUnsavedChanges: true,
        recoverySnapshot: null,
      });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error) });
    }
  },

  loadProject: async (projectPath) => {
    try {
      const backend = getStudioEditorBackend();
      const project = await backend.loadProject({ project_path: projectPath });
      let snapshot: StudioRecoverySnapshot | null = null;
      try {
        snapshot = await backend.loadRecoverySnapshot({ project_path: projectPath });
      } catch (error) {
        console.warn("Projeto aberto sem acesso aos snapshots de recuperacao:", error);
      }
      set({
        project,
        projectPath,
        currentPageIndex: 0,
        error: null,
        chapterHistory: [],
        chapterHistoryIndex: 0,
        isProjectSaving: false,
        hasUnsavedChanges: false,
        recoverySnapshot: isRecoveryCandidate(snapshot, project, projectPath) ? snapshot : null,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      try {
        const snapshot = await getStudioEditorBackend().loadRecoverySnapshot({ project_path: projectPath });
        set({ projectPath, recoverySnapshot: snapshot, error: message });
      } catch {
        set({ projectPath, recoverySnapshot: null, error: message });
      }
    }
  },

  openProjectFromDialog: async () => {
    try {
      const selected = await openProjectDialog();
      if (!selected) return;
      await get().loadProject(selected);
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error) });
    }
  },

  closeProject: (force = false) => {
    if (get().hasUnsavedChanges && !force) {
      set({ error: "Há alterações não salvas. Salve o capítulo ou confirme o descarte antes de fechar." });
      return false;
    }
    set({
      project: null,
      projectPath: null,
      currentPageIndex: 0,
      lastImport: null,
      error: null,
      chapterHistory: [],
      chapterHistoryIndex: 0,
      isProjectSaving: false,
      hasUnsavedChanges: false,
      recoverySnapshot: null,
    });
    return true;
  },

  saveProject: async () => {
    const { project, projectPath } = get();
    if (!project || !projectPath) return;
    try {
      const compatProject = importStudioProject(toTraduzAiV2Compat(project)).project;
      await getStudioEditorBackend().saveProjectJson({ project_path: projectPath, project_json: compatProject });
      const savedProject = await getStudioEditorBackend().loadProject({ project_path: projectPath });
      await refreshRecoverySnapshot(projectPath, savedProject);
      set({ project: savedProject, error: null, hasUnsavedChanges: false });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error), hasUnsavedChanges: true });
    }
  },

  saveProjectAsFromDialog: async () => {
    const { project, projectPath } = get();
    if (!project) return;
    try {
      const selected = await saveProjectDialog(projectPath && !projectPath.startsWith("memory://") ? projectPath : "project.json");
      if (!selected) return;
      const compatProject = importStudioProject(toTraduzAiV2Compat({
        ...project,
        source_path: selected,
        output_path: selected,
      })).project;
      await getStudioEditorBackend().saveProjectJson({ project_path: selected, project_json: compatProject });
      const savedProject = await getStudioEditorBackend().loadProject({ project_path: selected });
      await refreshRecoverySnapshot(selected, savedProject);
      set({ project: savedProject, projectPath: selected, error: null, hasUnsavedChanges: false });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error), hasUnsavedChanges: true });
    }
  },

  setCurrentPageIndex: (pageIndex) => {
    const total = get().project?.paginas.length ?? 0;
    set({ currentPageIndex: Math.max(0, Math.min(Math.max(0, total - 1), pageIndex)) });
  },

  patchCurrentTextLayer: async (layerId, patch) => {
    const { projectPath, currentPageIndex } = get();
    if (!projectPath) return;
    try {
      const backend = getStudioEditorBackend();
      const compat = createLegacyEditorBackendAdapter(backend);
      await compat.patchEditorTextLayer({
        project_path: projectPath,
        page_index: currentPageIndex,
        layer_id: layerId,
        patch,
      });
      const project = await backend.loadProject({ project_path: projectPath });
      await refreshRecoverySnapshot(projectPath, project);
      set({ project, error: null, hasUnsavedChanges: false });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error), hasUnsavedChanges: true });
    }
  },

  setCurrentTextLayerVisibility: async (layerId, visible) => {
    const { projectPath, currentPageIndex } = get();
    if (!projectPath) return;
    try {
      const backend = getStudioEditorBackend();
      const compat = createLegacyEditorBackendAdapter(backend);
      await compat.setEditorLayerVisibility({
        project_path: projectPath,
        page_index: currentPageIndex,
        layer_kind: "text",
        layer_id: layerId,
        visible,
      });
      const project = await backend.loadProject({ project_path: projectPath });
      await refreshRecoverySnapshot(projectPath, project);
      set({ project, error: null, hasUnsavedChanges: false });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error), hasUnsavedChanges: true });
    }
  },

  setCurrentImageLayerVisibility: async (layerKey, visible) => {
    const { projectPath, currentPageIndex } = get();
    if (!projectPath) return;
    try {
      const backend = getStudioEditorBackend();
      const compat = createLegacyEditorBackendAdapter(backend);
      await compat.setEditorLayerVisibility({
        project_path: projectPath,
        page_index: currentPageIndex,
        layer_kind: "image",
        layer_key: layerKey,
        visible,
      });
      const project = await backend.loadProject({ project_path: projectPath });
      await refreshRecoverySnapshot(projectPath, project);
      set({ project, error: null, hasUnsavedChanges: false });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error), hasUnsavedChanges: true });
    }
  },

  executeChapterCommand: async (command) => {
    const { projectPath, chapterHistory, chapterHistoryIndex, isProjectSaving } = get();
    if (!projectPath || isProjectSaving) return false;
    set({ isProjectSaving: true, error: null });
    try {
      const backend = getStudioEditorBackend();
      const historyEntry = createChapterHistoryEntry(command);
      if (historyEntry.patches.length === 0) {
        set({ isProjectSaving: false });
        return false;
      }
      const { project: savedProject } = await backend.mutateProject({
        project_path: projectPath,
        mutate: (latest) => replaceProjectDraft(
          latest,
          applyChapterHistoryEntry(latest, historyEntry, "redo"),
        ),
      });
      await refreshRecoverySnapshot(projectPath, savedProject);
      let history = [
        ...chapterHistory.slice(0, chapterHistoryIndex),
        historyEntry,
      ];
      if (history.length > MAX_CHAPTER_HISTORY) history = history.slice(-MAX_CHAPTER_HISTORY);
      set({
        project: savedProject,
        chapterHistory: history,
        chapterHistoryIndex: history.length,
        isProjectSaving: false,
        hasUnsavedChanges: false,
        error: null,
      });
      return true;
    } catch (error) {
      set({
        isProjectSaving: false,
        hasUnsavedChanges: true,
        error: error instanceof Error ? error.message : String(error),
      });
      return false;
    }
  },

  undoChapterCommand: async () => {
    const { projectPath, chapterHistory, chapterHistoryIndex, isProjectSaving } = get();
    if (!projectPath || isProjectSaving || chapterHistoryIndex <= 0) return false;
    const command = chapterHistory[chapterHistoryIndex - 1];
    set({ isProjectSaving: true, error: null });
    try {
      const backend = getStudioEditorBackend();
      const { project: savedProject } = await backend.mutateProject({
        project_path: projectPath,
        mutate: (latest) => replaceProjectDraft(
          latest,
          applyChapterHistoryEntry(latest, command, "undo"),
        ),
      });
      await refreshRecoverySnapshot(projectPath, savedProject);
      set({
        project: savedProject,
        chapterHistoryIndex: chapterHistoryIndex - 1,
        isProjectSaving: false,
        hasUnsavedChanges: false,
        error: null,
      });
      return true;
    } catch (error) {
      set({
        isProjectSaving: false,
        hasUnsavedChanges: true,
        error: error instanceof Error ? error.message : String(error),
      });
      return false;
    }
  },

  redoChapterCommand: async () => {
    const { projectPath, chapterHistory, chapterHistoryIndex, isProjectSaving } = get();
    if (!projectPath || isProjectSaving || chapterHistoryIndex >= chapterHistory.length) return false;
    const command = chapterHistory[chapterHistoryIndex];
    set({ isProjectSaving: true, error: null });
    try {
      const backend = getStudioEditorBackend();
      const { project: savedProject } = await backend.mutateProject({
        project_path: projectPath,
        mutate: (latest) => replaceProjectDraft(
          latest,
          applyChapterHistoryEntry(latest, command, "redo"),
        ),
      });
      await refreshRecoverySnapshot(projectPath, savedProject);
      set({
        project: savedProject,
        chapterHistoryIndex: chapterHistoryIndex + 1,
        isProjectSaving: false,
        hasUnsavedChanges: false,
        error: null,
      });
      return true;
    } catch (error) {
      set({
        isProjectSaving: false,
        hasUnsavedChanges: true,
        error: error instanceof Error ? error.message : String(error),
      });
      return false;
    }
  },

  restoreRecovery: async () => {
    const { projectPath, recoverySnapshot, isProjectSaving } = get();
    if (!projectPath || !recoverySnapshot || isProjectSaving) return false;
    set({ isProjectSaving: true, error: null });
    try {
      const project = await recoverStudioProject(getStudioEditorBackend(), projectPath);
      if (!project) {
        set({ isProjectSaving: false, recoverySnapshot: null });
        return false;
      }
      set({
        project,
        recoverySnapshot: null,
        isProjectSaving: false,
        hasUnsavedChanges: false,
        chapterHistory: [],
        chapterHistoryIndex: 0,
        error: null,
      });
      return true;
    } catch (error) {
      set({
        isProjectSaving: false,
        hasUnsavedChanges: true,
        error: error instanceof Error ? error.message : String(error),
      });
      return false;
    }
  },

  dismissRecovery: async () => {
    const { projectPath } = get();
    if (!projectPath) return;
    try {
      await getStudioEditorBackend().clearRecoverySnapshot({ project_path: projectPath });
      set({ recoverySnapshot: null, error: null });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error) });
    }
  },

  clearError: () => set({ error: null }),
}));

function replaceProjectDraft(target: StudioProject, source: StudioProject) {
  const targetRecord = target as unknown as Record<string, unknown>;
  for (const key of Object.keys(targetRecord)) delete targetRecord[key];
  Object.assign(targetRecord, source as unknown as Record<string, unknown>);
}
