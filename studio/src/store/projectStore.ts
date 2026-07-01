import { create } from "zustand";
import { configureStudioEditorBackend, getStudioEditorBackend } from "../backend/editorBackend";
import { createLegacyEditorBackendAdapter } from "../backend/editorBackendCompat";
import { openProjectDialog, saveProjectDialog } from "../backend/projectDialog";
import { createDefaultStudioBackend } from "../backend/tauriBackend";
import { importStudioProject, toTraduzAiV2Compat } from "../project/adapters";
import type { ImageLayerKey, ProjectImportResult, StudioProject, StudioTextLayer } from "../project/studioProject";

const DEFAULT_PROJECT_PATH = "memory://current";

export interface StudioProjectState {
  project: StudioProject | null;
  projectPath: string | null;
  currentPageIndex: number;
  lastImport: ProjectImportResult | null;
  error: string | null;
  importProjectJson: (jsonText: string, projectPath?: string) => Promise<void>;
  loadProject: (projectPath: string) => Promise<void>;
  openProjectFromDialog: () => Promise<void>;
  saveProject: () => Promise<void>;
  saveProjectAsFromDialog: () => Promise<void>;
  setCurrentPageIndex: (pageIndex: number) => void;
  patchCurrentTextLayer: (layerId: string, patch: Partial<StudioTextLayer>) => Promise<void>;
  setCurrentTextLayerVisibility: (layerId: string, visible: boolean) => Promise<void>;
  setCurrentImageLayerVisibility: (layerKey: ImageLayerKey, visible: boolean) => Promise<void>;
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
      });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error) });
    }
  },

  loadProject: async (projectPath) => {
    try {
      const project = await getStudioEditorBackend().loadProject({ project_path: projectPath });
      set({ project, projectPath, currentPageIndex: 0, error: null });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error) });
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

  saveProject: async () => {
    const { project, projectPath } = get();
    if (!project || !projectPath) return;
    try {
      const compatProject = importStudioProject(toTraduzAiV2Compat(project)).project;
      await getStudioEditorBackend().saveProjectJson({ project_path: projectPath, project_json: compatProject });
      const savedProject = await getStudioEditorBackend().loadProject({ project_path: projectPath });
      set({ project: savedProject, error: null });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error) });
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
      set({ project: savedProject, projectPath: selected, error: null });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error) });
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
      set({ project, error: null });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error) });
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
      set({ project, error: null });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error) });
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
      set({ project, error: null });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error) });
    }
  },

  clearError: () => set({ error: null }),
}));
