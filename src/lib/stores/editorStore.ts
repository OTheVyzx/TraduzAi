import { create } from "zustand";
import { useAppStore, type TextEntry } from "./appStore";

export interface EditorState {
  currentPageIndex: number;
  selectedLayerId: string | null;
  hoveredLayerId: string | null;
  hiddenLayers: Record<number, string[]>;
  viewMode: "translated" | "inpainted" | "original";
  showOverlays: boolean;
  zoom: number;
  panOffset: { x: number; y: number };
  pendingEdits: Record<string, Partial<TextEntry>>;

  isRetypesetting: boolean;
  isReinpainting: boolean;
  lastRetypesetTime: number;

  setCurrentPage: (index: number) => void;
  selectLayer: (id: string | null) => void;
  hoverLayer: (id: string | null) => void;
  toggleLayerVisibility: (layerId: string) => void;
  setViewMode: (mode: EditorState["viewMode"]) => void;
  toggleOverlays: () => void;
  setZoom: (zoom: number) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  setPan: (offset: { x: number; y: number }) => void;
  resetViewport: () => void;
  updatePendingEdit: (layerId: string, changes: Partial<TextEntry>) => void;
  updatePendingEstilo: (layerId: string, estiloChanges: Partial<TextEntry["estilo"]>) => void;
  commitEdits: () => Promise<void>;
  discardEdits: () => void;
  retypesetCurrentPage: () => Promise<void>;
  reinpaintCurrentPage: () => Promise<void>;
  resetEditor: () => void;
}

export const useEditorStore = create<EditorState>((set, get) => ({
  currentPageIndex: 0,
  selectedLayerId: null,
  hoveredLayerId: null,
  hiddenLayers: {},
  viewMode: "inpainted",
  showOverlays: true,
  zoom: 1,
  panOffset: { x: 0, y: 0 },
  pendingEdits: {},
  isRetypesetting: false,
  isReinpainting: false,
  lastRetypesetTime: 0,

  setCurrentPage: (index) =>
    set({
      currentPageIndex: index,
      selectedLayerId: null,
      hoveredLayerId: null,
      zoom: 1,
      panOffset: { x: 0, y: 0 },
    }),

  selectLayer: (id) => set({ selectedLayerId: id }),

  hoverLayer: (id) => set({ hoveredLayerId: id }),

  toggleLayerVisibility: (layerId) =>
    set((state) => {
      const pageIdx = state.currentPageIndex;
      const current = state.hiddenLayers[pageIdx] ?? [];
      const isHidden = current.includes(layerId);
      return {
        hiddenLayers: {
          ...state.hiddenLayers,
          [pageIdx]: isHidden
            ? current.filter((id) => id !== layerId)
            : [...current, layerId],
        },
      };
    }),

  setViewMode: (mode) => set({ viewMode: mode }),

  toggleOverlays: () => set((state) => ({ showOverlays: !state.showOverlays })),

  setZoom: (zoom) => set({ zoom: Math.max(0.25, Math.min(4, zoom)) }),

  zoomIn: () => set((state) => ({ zoom: Math.max(0.25, Math.min(4, state.zoom + 0.2)) })),

  zoomOut: () => set((state) => ({ zoom: Math.max(0.25, Math.min(4, state.zoom - 0.2)) })),

  setPan: (offset) => set({ panOffset: offset }),

  resetViewport: () => set({ zoom: 1, panOffset: { x: 0, y: 0 } }),

  updatePendingEdit: (layerId, changes) =>
    set((state) => ({
      pendingEdits: {
        ...state.pendingEdits,
        [layerId]: { ...state.pendingEdits[layerId], ...changes },
      },
    })),

  updatePendingEstilo: (layerId, estiloChanges) =>
    set((state) => {
      const existing = state.pendingEdits[layerId] ?? {};
      const existingEstilo = existing.estilo ?? {};
      return {
        pendingEdits: {
          ...state.pendingEdits,
          [layerId]: {
            ...existing,
            estilo: { ...existingEstilo, ...estiloChanges } as TextEntry["estilo"],
          },
        },
      };
    }),

  commitEdits: async () => {
    const { pendingEdits, currentPageIndex } = get();
    if (Object.keys(pendingEdits).length === 0) return;

    const appStore = useAppStore.getState();
    const project = appStore.project;
    if (!project) return;

    const page = project.paginas[currentPageIndex];
    if (!page) return;

    const updatedTextos = page.textos.map((texto) => {
      const edit = pendingEdits[texto.id];
      if (!edit) return texto;
      return {
        ...texto,
        ...edit,
        estilo: edit.estilo
          ? { ...texto.estilo, ...edit.estilo }
          : texto.estilo,
      };
    });

    const updatedPaginas = [...project.paginas];
    updatedPaginas[currentPageIndex] = { ...page, textos: updatedTextos };

    appStore.updateProject({ paginas: updatedPaginas });
    
    // Save to disk
    const updatedProject = { ...project, paginas: updatedPaginas };
    try {
      const { saveProjectJson } = await import("../tauri");
      await saveProjectJson({
        project_path: updatedProject.output_path ?? updatedProject.source_path,
        project_json: updatedProject,
      });
    } catch (e) {
      console.error("Falha ao salvar edições no disco", e);
    }

    set({ pendingEdits: {} });
  },

  discardEdits: () => set({ pendingEdits: {} }),

  retypesetCurrentPage: async () => {
    const state = get();
    await state.commitEdits();
    
    const appStore = useAppStore.getState();
    const project = appStore.project;
    if (!project) return;
    
    set({ isRetypesetting: true });
    try {
      const { retypesetPage } = await import("../tauri");
      const projectPath = project.output_path ?? project.source_path;
      
      await retypesetPage({
        project_path: projectPath,
        page_index: state.currentPageIndex,
      });
      
      set({ lastRetypesetTime: Date.now() });
    } catch (e) {
      console.error("Falha ao re-renderizar a página:", e);
    } finally {
      set({ isRetypesetting: false });
    }
  },

  reinpaintCurrentPage: async () => {
    const state = get();
    await state.commitEdits();
    const appStore = useAppStore.getState();
    const project = appStore.project;
    if (!project) return;

    set({ isReinpainting: true });
    try {
      const { reinpaintPage } = await import("../tauri");
      const projectPath = project.output_path ?? project.source_path;

      await reinpaintPage({
        project_path: projectPath,
        page_index: state.currentPageIndex,
      });

      set({ lastRetypesetTime: Date.now() });
    } catch (e) {
      console.error("Falha ao refazer a pagina limpa:", e);
    } finally {
      set({ isReinpainting: false });
    }
  },

  resetEditor: () =>
    set({
      currentPageIndex: 0,
      selectedLayerId: null,
      hoveredLayerId: null,
      hiddenLayers: {},
      viewMode: "inpainted",
      showOverlays: true,
      zoom: 1,
      panOffset: { x: 0, y: 0 },
      pendingEdits: {},
      isRetypesetting: false,
      isReinpainting: false,
      lastRetypesetTime: 0,
    }),
}));
