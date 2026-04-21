import { create } from "zustand";
import { useAppStore, type ImageLayerKey, type PageData, type TextEntry } from "./appStore";
import {
  createEditorTextLayer,
  deleteEditorTextLayer,
  loadEditorPage,
  patchEditorTextLayer,
  retypesetPage,
  reinpaintPage,
  processBlock,
  setEditorLayerVisibility,
  updateBrushRegion,
  updateMaskRegion,
} from "../tauri";

export type EditorToolMode = "select" | "block" | "brush" | "repairBrush" | "eraser";
export type EditorViewMode = "translated" | "inpainted" | "original";

function projectPath() {
  const project = useAppStore.getState().project;
  return project ? project.output_path ?? project.source_path : null;
}

function sortTextLayers(layers: TextEntry[]) {
  return [...layers].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
}

function syncCurrentPageIntoProject(page: PageData, pageIndex: number) {
  const appStore = useAppStore.getState();
  const project = appStore.project;
  if (!project) return;
  const paginas = [...project.paginas];
  paginas[pageIndex] = page;
  appStore.updateProject({ paginas });
}

function updateLayerInPage(
  page: PageData,
  layerId: string,
  updater: (layer: TextEntry) => TextEntry,
): PageData {
  const updatedLayers = sortTextLayers(
    page.text_layers.map((layer) => (layer.id === layerId ? updater(layer) : layer)),
  );
  return {
    ...page,
    text_layers: updatedLayers,
    textos: updatedLayers,
  };
}

interface EditorState {
  currentPageIndex: number;
  currentPage: PageData | null;
  selectedLayerId: string | null;
  selectedImageLayerKey: ImageLayerKey | null;
  hoveredLayerId: string | null;
  viewMode: EditorViewMode;
  toolMode: EditorToolMode;
  showOverlays: boolean;
  zoom: number;
  panOffset: { x: number; y: number };
  pendingEdits: Record<string, Partial<TextEntry>>;
  isRetypesetting: boolean;
  isReinpainting: boolean;
  isLoadingPage: boolean;
  lastRetypesetTime: number;
  brushSize: number;

  loadCurrentPage: () => Promise<void>;
  setCurrentPage: (index: number) => Promise<void>;
  selectLayer: (id: string | null) => void;
  selectImageLayer: (key: ImageLayerKey | null) => void;
  hoverLayer: (id: string | null) => void;
  setViewMode: (mode: EditorViewMode) => void;
  setToolMode: (mode: EditorToolMode) => void;
  toggleOverlays: () => void;
  setZoom: (zoom: number) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  setPan: (offset: { x: number; y: number }) => void;
  resetViewport: () => void;
  setBrushSize: (size: number) => void;
  updatePendingEdit: (layerId: string, changes: Partial<TextEntry>) => void;
  updatePendingEstilo: (layerId: string, estiloChanges: Partial<TextEntry["estilo"]>) => void;
  commitEdits: () => Promise<void>;
  discardEdits: () => void;
  toggleTextLayerVisibility: (layerId: string) => Promise<void>;
  toggleImageLayerVisibility: (layerKey: ImageLayerKey) => Promise<void>;
  createTextLayer: (bbox: [number, number, number, number]) => Promise<void>;
  deleteSelectedLayer: () => Promise<void>;
  applyBitmapStroke: (payload: {
    width: number;
    height: number;
    strokes: [number, number][][];
    clear?: boolean;
  }) => Promise<void>;
  retypesetCurrentPage: () => Promise<void>;
  reinpaintCurrentPage: () => Promise<void>;
  reProcessBlock: (mode: "ocr" | "translate" | "inpaint") => Promise<void>;
  disconnectBlock: () => Promise<void>;
  resetEditor: () => void;
}

export const useEditorStore = create<EditorState>((set, get) => ({
  currentPageIndex: 0,
  currentPage: null,
  selectedLayerId: null,
  selectedImageLayerKey: null,
  hoveredLayerId: null,
  viewMode: "translated",
  toolMode: "select",
  showOverlays: true,
  zoom: 1,
  panOffset: { x: 0, y: 0 },
  pendingEdits: {},
  isRetypesetting: false,
  isReinpainting: false,
  isLoadingPage: false,
  lastRetypesetTime: 0,
  brushSize: 18,

  loadCurrentPage: async () => {
    const path = projectPath();
    const project = useAppStore.getState().project;
    if (!path || !project) {
      set({ currentPage: null, isLoadingPage: false });
      return;
    }

    set({ isLoadingPage: true });
    try {
      const payload = await loadEditorPage({
        project_path: path,
        page_index: get().currentPageIndex,
      });
      syncCurrentPageIntoProject(payload.page, payload.page_index);
      set({
        currentPage: payload.page,
        pendingEdits: {},
        selectedLayerId: null,
        selectedImageLayerKey: null,
        hoveredLayerId: null,
      });
    } finally {
      set({ isLoadingPage: false });
    }
  },

  setCurrentPage: async (index) => {
    set({
      currentPageIndex: index,
      currentPage: null,
      pendingEdits: {},
      selectedLayerId: null,
      selectedImageLayerKey: null,
      hoveredLayerId: null,
      zoom: 1,
      panOffset: { x: 0, y: 0 },
    });
    await get().loadCurrentPage();
  },

  selectLayer: (id) => set({ selectedLayerId: id, selectedImageLayerKey: null }),
  selectImageLayer: (key) => set({ selectedImageLayerKey: key, selectedLayerId: null }),
  hoverLayer: (id) => set({ hoveredLayerId: id }),
  setViewMode: (mode) => set({ viewMode: mode }),
  setToolMode: (mode) => set({ toolMode: mode }),
  toggleOverlays: () => set((state) => ({ showOverlays: !state.showOverlays })),
  setZoom: (zoom) => set({ zoom: Math.max(0.2, Math.min(5, zoom)) }),
  zoomIn: () => set((state) => ({ zoom: Math.max(0.2, Math.min(5, state.zoom + 0.15)) })),
  zoomOut: () => set((state) => ({ zoom: Math.max(0.2, Math.min(5, state.zoom - 0.15)) })),
  setPan: (offset) => set({ panOffset: offset }),
  resetViewport: () => set({ zoom: 1, panOffset: { x: 0, y: 0 } }),
  setBrushSize: (brushSize) => set({ brushSize: Math.max(4, Math.min(160, brushSize)) }),

  updatePendingEdit: (layerId, changes) =>
    set((state) => ({
      pendingEdits: {
        ...state.pendingEdits,
        [layerId]: { ...state.pendingEdits[layerId], ...changes },
      },
      selectedLayerId: layerId,
      selectedImageLayerKey: null,
    })),

  updatePendingEstilo: (layerId, estiloChanges) =>
    set((state) => {
      const current = state.pendingEdits[layerId] ?? {};
      const estiloBase = current.estilo ?? {};
      return {
        pendingEdits: {
          ...state.pendingEdits,
          [layerId]: {
            ...current,
            estilo: { ...estiloBase, ...estiloChanges } as TextEntry["estilo"],
          },
        },
      };
    }),

  commitEdits: async () => {
    const path = projectPath();
    const { pendingEdits, currentPageIndex, currentPage } = get();
    if (!path || !currentPage || Object.keys(pendingEdits).length === 0) return;

    for (const [layerId, edit] of Object.entries(pendingEdits)) {
      const patch: Record<string, unknown> = {};
      if (edit.traduzido !== undefined || edit.translated !== undefined) {
        patch.translated = edit.traduzido ?? edit.translated ?? "";
      }
      if (edit.tipo) patch.tipo = edit.tipo;
      if (edit.bbox) {
        patch.layout_bbox = edit.bbox;
        patch.balloon_bbox = edit.bbox;
        patch.bbox = edit.bbox;
      }
      if (edit.estilo) patch.style = edit.estilo;

      await patchEditorTextLayer({
        project_path: path,
        page_index: currentPageIndex,
        layer_id: layerId,
        patch,
      });
    }

    set({ pendingEdits: {} });
    await get().loadCurrentPage();
  },

  discardEdits: () => set({ pendingEdits: {} }),

  toggleTextLayerVisibility: async (layerId) => {
    const path = projectPath();
    const page = get().currentPage;
    if (!path || !page) return;
    const layer = page.text_layers.find((item) => item.id === layerId);
    if (!layer) return;
    await setEditorLayerVisibility({
      project_path: path,
      page_index: get().currentPageIndex,
      layer_kind: "text",
      layer_id: layerId,
      visible: !(layer.visible ?? true),
    });
    const updatedPage = updateLayerInPage(page, layerId, (item) => ({
      ...item,
      visible: !(item.visible ?? true),
    }));
    syncCurrentPageIntoProject(updatedPage, get().currentPageIndex);
    set({ currentPage: updatedPage });
  },

  toggleImageLayerVisibility: async (layerKey) => {
    const path = projectPath();
    const page = get().currentPage;
    if (!path || !page) return;
    const layer = page.image_layers?.[layerKey];
    await setEditorLayerVisibility({
      project_path: path,
      page_index: get().currentPageIndex,
      layer_kind: "image",
      layer_key: layerKey,
      visible: !(layer?.visible ?? false),
    });
    const updatedPage: PageData = {
      ...page,
      image_layers: {
        ...page.image_layers,
        [layerKey]: {
          key: layerKey,
          path: layer?.path ?? null,
          visible: !(layer?.visible ?? false),
          locked: layer?.locked ?? false,
        },
      },
    };
    syncCurrentPageIntoProject(updatedPage, get().currentPageIndex);
    set({ currentPage: updatedPage, selectedImageLayerKey: layerKey });
  },

  createTextLayer: async (bbox) => {
    const path = projectPath();
    const page = get().currentPage;
    if (!path || !page) return;
    const created = await createEditorTextLayer({
      project_path: path,
      page_index: get().currentPageIndex,
      layout_bbox: bbox,
    });
    
    // Auto-run OCR on the newly created block
    set({ isRetypesetting: true, selectedLayerId: created.id });
    try {
      await processBlock({
        project_path: path,
        page_index: get().currentPageIndex,
        block_id: created.id,
        mode: "ocr",
      });
      await get().loadCurrentPage();
    } finally {
      set({ 
        isRetypesetting: false,
        selectedLayerId: created.id,
        selectedImageLayerKey: null,
        toolMode: "select",
        lastRetypesetTime: Date.now(),
      });
    }
  },

  deleteSelectedLayer: async () => {
    const path = projectPath();
    const page = get().currentPage;
    const layerId = get().selectedLayerId;
    if (!path || !page || !layerId) return;
    await deleteEditorTextLayer({
      project_path: path,
      page_index: get().currentPageIndex,
      layer_id: layerId,
    });
    const updatedLayers = page.text_layers.filter((layer) => layer.id !== layerId);
    const updatedPage = { ...page, text_layers: updatedLayers, textos: updatedLayers };
    syncCurrentPageIntoProject(updatedPage, get().currentPageIndex);
    set({
      currentPage: updatedPage,
      selectedLayerId: null,
      pendingEdits: Object.fromEntries(
        Object.entries(get().pendingEdits).filter(([id]) => id !== layerId),
      ),
    });
  },

  applyBitmapStroke: async ({ width, height, strokes, clear = false }) => {
    const path = projectPath();
    const page = get().currentPage;
    if (!path || !page || strokes.length === 0) return;

    const layerKey = get().toolMode === "brush" ? "brush" : "mask";
    const erase = get().toolMode === "eraser";
    const fn = layerKey === "brush" ? updateBrushRegion : updateMaskRegion;
    const absolutePath = await fn({
      project_path: path,
      page_index: get().currentPageIndex,
      width,
      height,
      brush_size: get().brushSize,
      clear,
      erase,
      strokes,
    });

    const updatedPage: PageData = {
      ...page,
      image_layers: {
        ...page.image_layers,
        [layerKey]: {
          key: layerKey,
          path: absolutePath,
          visible: true,
          locked: false,
        },
      },
    };
    syncCurrentPageIntoProject(updatedPage, get().currentPageIndex);
    set({
      currentPage: updatedPage,
      selectedImageLayerKey: layerKey,
      selectedLayerId: null,
      lastRetypesetTime: Date.now(),
    });
  },

  retypesetCurrentPage: async () => {
    const path = projectPath();
    if (!path) return;
    await get().commitEdits();
    set({ isRetypesetting: true });
    try {
      await retypesetPage({
        project_path: path,
        page_index: get().currentPageIndex,
      });
      await get().loadCurrentPage();
      set({ lastRetypesetTime: Date.now() });
    } finally {
      set({ isRetypesetting: false });
    }
  },

  reProcessBlock: async (mode) => {
    const { selectedLayerId, currentPageIndex } = get();
    const path = projectPath();
    if (!path || !selectedLayerId) return;

    await get().commitEdits();
    set({ isRetypesetting: true });
    try {
      await processBlock({
        project_path: path,
        page_index: currentPageIndex,
        block_id: selectedLayerId,
        mode,
      });
      await get().loadCurrentPage();
      set({ lastRetypesetTime: Date.now() });
    } finally {
      set({ isRetypesetting: false });
    }
  },

  disconnectBlock: async () => {
    const { selectedLayerId, currentPageIndex, currentPage } = get();
    const path = projectPath();
    if (!path || !selectedLayerId || !currentPage) return;

    const layer = currentPage.text_layers.find((l: TextEntry) => l.id === selectedLayerId);
    if (!layer) return;

    // Reset layout fields to force independent rendering
    await patchEditorTextLayer({
      project_path: path,
      page_index: currentPageIndex,
      layer_id: selectedLayerId,
      patch: {
        layout_bbox: layer.source_bbox,
        balloon_bbox: layer.source_bbox,
        layout_group_size: 1,
        connected_children: null,
        connected_text_groups: [],
        connected_lobe_bboxes: [],
        _connected_slot_index: null,
      },
    });

    await get().loadCurrentPage();
  },

  reinpaintCurrentPage: async () => {
    const path = projectPath();
    if (!path) return;
    await get().commitEdits();
    set({ isReinpainting: true });
    try {
      await reinpaintPage({
        project_path: path,
        page_index: get().currentPageIndex,
      });
      await get().loadCurrentPage();
      set({ lastRetypesetTime: Date.now() });
    } finally {
      set({ isReinpainting: false });
    }
  },

  resetEditor: () =>
    set({
      currentPageIndex: 0,
      currentPage: null,
      selectedLayerId: null,
      selectedImageLayerKey: null,
      hoveredLayerId: null,
      viewMode: "translated",
      toolMode: "select",
      showOverlays: true,
      zoom: 1,
      panOffset: { x: 0, y: 0 },
      pendingEdits: {},
      isRetypesetting: false,
      isReinpainting: false,
      isLoadingPage: false,
      lastRetypesetTime: 0,
      brushSize: 18,
    }),
}));
