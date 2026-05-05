import { create } from "zustand";
import { useAppStore, type ImageLayerKey, type PageData, type Project, type TextEntry } from "./appStore";
import type { TextLayerStyle } from "./appStore";
import {
  loadEditorPage,
  patchEditorTextLayer,
  renderPreviewPage as renderPreviewPageCommand,
  retypesetPage,
  reinpaintPage,
  detectPage,
  ocrPage,
  translatePage,
  processBlock,
  saveProjectJson,
  setEditorLayerVisibility,
  updateBrushRegion,
  updateMaskRegion,
  runPageActionWithOptionalMask,
} from "../tauri";
import {
  createHistoryStack,
  disposeAllForPage,
  executeCommand,
  getPageKey,
  pruneHistoryStacksByGlobalCap,
  recordCommand,
  redo as redoHistory,
  undo as undoHistory,
  updateHistoryBaseFingerprint,
  type Bbox,
  type EditorCommand,
  type HistoryStack,
  type ValidationResult,
  type WorkingStateDraft,
} from "../editorHistory";
import {
  defaultWorkContext,
  type WorkCharacter,
  type WorkGlossaryEntry,
} from "../workContext";

export type EditorToolMode = "select" | "block" | "brush" | "repairBrush" | "eraser" | "mask";
export type EditorViewMode = "translated" | "inpainted" | "original";
export type RenderPreviewStatus = "fresh" | "stale" | "rendering" | "error";

// Chaves de camadas bitmap — "base" é imutável (nunca reescrita pelo brush/inpaint)
export type BitmapLayerKey = "base" | "mask" | "inpaint" | "brush" | "rendered" | "preview";
export type MutableBitmapLayerKey = Exclude<BitmapLayerKey, "base">;

export interface RenderPreviewCacheEntry {
  fingerprint: string;
  status: RenderPreviewStatus;
  path: string | null;
  previewPath: string | null;
  generatedAt: number | null;
  error: string | null;
}

export type RenderPreviewCacheByPageKey = Record<string, RenderPreviewCacheEntry>;

function projectPath() {
  const project = useAppStore.getState().project;
  return project ? project.output_path ?? project.source_path : null;
}

function sortTextLayers(layers: TextEntry[]) {
  return [...layers].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
}

function normalizeTextLayerOrder(layers: TextEntry[]) {
  return layers.map((layer, index) => ({ ...layer, order: index }));
}

function syncCurrentPageIntoProject(page: PageData, pageIndex: number) {
  const appStore = useAppStore.getState();
  const project = appStore.project;
  if (!project) return;
  const paginas = [...project.paginas];
  paginas[pageIndex] = page;
  appStore.updateProject({ paginas });
}

function pageFingerprint(page: PageData | null) {
  if (!page) return "";
  return JSON.stringify(
    page.text_layers.map((layer) => ({
      id: layer.id,
      traduzido: layer.traduzido,
      translated: layer.translated,
      bbox: layer.bbox,
      layout_bbox: layer.layout_bbox,
      estilo: layer.estilo,
      visible: layer.visible,
      locked: layer.locked,
      order: layer.order,
    })),
  );
}

function renderedPathForPage(page: PageData | null) {
  return page?.image_layers?.rendered?.path ?? page?.arquivo_traduzido ?? null;
}

function previewCacheKey(fingerprint: string) {
  let hash = 0x811c9dc5;
  for (let index = 0; index < fingerprint.length; index += 1) {
    hash ^= fingerprint.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

function sameBbox(a: Bbox | undefined, b: Bbox | undefined) {
  if (!a || !b) return false;
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) < 0.01);
}

function removeEmptyPendingEdit(pendingEdits: Record<string, Partial<TextEntry>>, layerId: string) {
  const current = pendingEdits[layerId];
  if (!current) return pendingEdits;
  const next = { ...current };
  if (next.estilo && Object.keys(next.estilo).length === 0) {
    delete next.estilo;
  }
  if (Object.keys(next).length === 0) {
    const { [layerId]: _removed, ...rest } = pendingEdits;
    return rest;
  }
  return { ...pendingEdits, [layerId]: next };
}

function mergePendingEdit(layer: TextEntry, edit: Partial<TextEntry> | undefined): TextEntry {
  if (!edit) return layer;
  const estilo = edit.estilo ? { ...layer.estilo, ...edit.estilo } : layer.estilo;
  return {
    ...layer,
    ...edit,
    traduzido: edit.traduzido ?? edit.translated ?? layer.traduzido,
    translated: edit.translated ?? edit.traduzido ?? layer.translated,
    bbox: edit.bbox ?? layer.bbox,
    layout_bbox: edit.bbox ?? layer.layout_bbox,
    balloon_bbox: edit.bbox ?? layer.balloon_bbox,
    estilo,
    style: estilo,
  };
}

function defaultTextStyle(): TextLayerStyle {
  return {
    fonte: "CCDaveGibbonsLower W00 Regular.ttf",
    tamanho: 28,
    cor: "#000000",
    cor_gradiente: [],
    contorno: "",
    contorno_px: 0,
    glow: false,
    glow_cor: "",
    glow_px: 0,
    sombra: false,
    sombra_cor: "",
    sombra_offset: [0, 0],
    bold: false,
    italico: false,
    rotacao: 0,
    alinhamento: "center",
    force_upper: false,
  };
}

function createLocalTextLayer(bbox: [number, number, number, number], order: number): TextEntry {
  const estilo = defaultTextStyle();
  return {
    id: `tl_local_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`,
    kind: "text",
    source_bbox: bbox,
    layout_bbox: bbox,
    render_bbox: null,
    bbox,
    tipo: "fala",
    original: "",
    traduzido: "",
    translated: "",
    confianca_ocr: 1,
    ocr_confidence: 1,
    estilo,
    style: estilo,
    visible: true,
    locked: false,
    order,
    render_preview_path: null,
    detector: null,
    line_polygons: null,
    source_direction: null,
    rendered_direction: null,
    source_language: null,
    rotation_deg: 0,
    detected_font_size_px: null,
    balloon_bbox: bbox,
    balloon_subregions: [],
    layout_group_size: 1,
  };
}

function emptyStructuralEdits(): PendingStructuralEdits {
  return { created: [], deleted: {}, order: undefined };
}

function hasStructuralEdits(edits: PendingStructuralEdits) {
  return edits.created.length > 0 || Object.keys(edits.deleted).length > 0 || !!edits.order;
}

function getBasePage(pageIndex: number) {
  return useAppStore.getState().project?.paginas[pageIndex] ?? null;
}

function getBaseLayer(pageIndex: number, layerId: string) {
  return getBasePage(pageIndex)?.text_layers.find((layer) => layer.id === layerId) ?? null;
}

export type PendingStructuralEdits = {
  created: TextEntry[];
  deleted: Record<string, TextEntry>;
  order?: string[];
};

function materializeWorkingPage(
  page: PageData,
  pendingEdits: Record<string, Partial<TextEntry>>,
  pendingStructuralEdits: PendingStructuralEdits,
) {
  let layers = normalizeTextLayerOrder(
    page.text_layers
      .filter((layer) => !pendingStructuralEdits.deleted[layer.id])
      .map((layer) => mergePendingEdit(layer, pendingEdits[layer.id])),
  );

  if (pendingStructuralEdits.order) {
    const byId = new Map(layers.map((layer) => [layer.id, layer]));
    const ordered = pendingStructuralEdits.order.flatMap((id) => {
      const layer = byId.get(id);
      return layer ? [layer] : [];
    });
    const missing = layers.filter((layer) => !pendingStructuralEdits.order?.includes(layer.id));
    layers = normalizeTextLayerOrder([...ordered, ...missing]);
  }

  return { ...page, text_layers: layers, textos: layers };
}

function renderPreviewFingerprint(
  page: PageData | null,
  pendingEdits: Record<string, Partial<TextEntry>> = {},
  pendingStructuralEdits: PendingStructuralEdits = emptyStructuralEdits(),
) {
  if (!page) return "";
  const materializedPage = materializeWorkingPage(page, pendingEdits, pendingStructuralEdits);

  return JSON.stringify({
    rendered: page.image_layers?.rendered?.path ?? page.arquivo_traduzido,
    inpaint: page.image_layers?.inpaint,
    mask: page.image_layers?.mask,
    brush: page.image_layers?.brush,
    text_layers: materializedPage.text_layers.map((layer) => ({
      id: layer.id,
      traduzido: layer.traduzido,
      translated: layer.translated,
      bbox: layer.bbox,
      layout_bbox: layer.layout_bbox,
      balloon_bbox: layer.balloon_bbox,
      style: layer.estilo ?? layer.style,
      order: layer.order,
      visible: layer.visible ?? true,
    })),
  });
}

export function getRenderPreviewStateForPage(
  pageKey: string,
  page: PageData | null,
  cache: RenderPreviewCacheByPageKey,
): RenderPreviewCacheEntry {
  return (
    cache[pageKey] ?? {
      fingerprint: renderPreviewFingerprint(page),
      status: "fresh",
      path: renderedPathForPage(page),
      previewPath: null,
      generatedAt: null,
      error: null,
    }
  );
}

export function getStaleRenderPreviewPages(project: Project | null, cache: RenderPreviewCacheByPageKey) {
  if (!project) return [];
  return project.paginas.flatMap((page, index) => {
    const pageKey = getPageKey(project, index);
    const entry = getRenderPreviewStateForPage(pageKey, page, cache);
    return entry.status === "fresh" && !entry.previewPath ? [] : [index + 1];
  });
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
  renderPreviewCacheByPageKey: RenderPreviewCacheByPageKey;
  historyByPageKey: Record<string, HistoryStack>;
  isRetypesetting: boolean;
  isReinpainting: boolean;
  isLoadingPage: boolean;
  lastRetypesetTime: number;
  brushSize: number;
  // ── Brush (Fase 7) ───────────────────────────────────────────────────────
  /** Cor atual do brush em hex (#rrggbb). */
  brushColor: string;
  /** Opacidade do traço (0..1). */
  brushOpacity: number;
  /** Dureza da borda (0..1). Afeta a visualização no EditorBitmapOverlay. */
  brushHardness: number;
  /** Até 8 cores recentes. */
  recentBrushColors: string[];
  setBrushColor: (color: string) => void;
  setBrushOpacity: (opacity: number) => void;
  setBrushHardness: (hardness: number) => void;
  // ── Máscara Lasso (Fase 8) ───────────────────────────────────────────────
  /** Modo do lasso: freehand (arrastar) ou polygonal (cliques). */
  maskShape: "freehand" | "polygonal";
  /** Operação: substituir, adicionar ou subtrair da máscara existente. */
  maskOp: "replace" | "add" | "subtract";
  /** Pontos do lasso em construção (espaço da imagem). null = nenhuma seleção em progresso. */
  maskInProgress: { points: Array<[number, number]> } | null;
  setMaskShape: (shape: "freehand" | "polygonal") => void;
  setMaskOp: (op: "replace" | "add" | "subtract") => void;
  setMaskInProgress: (progress: { points: Array<[number, number]> } | null) => void;
  clearMask: () => Promise<void>;
  // ── Borracha Inteligente (Fase 9) ────────────────────────────────────────
  /** Última camada pintada — usada pelo eraser para inferir alvo. */
  lastPaintedLayer: "brush" | "mask";
  /** Alvo explícito do eraser (nulo = inferência automática). */
  eraserTarget: "brush" | "mask" | null;
  setEraserTarget: (target: "brush" | "mask" | null) => void;

  // ── Layers MVP (Fase 10) ────────────────────────────────────────────────
  /** Thumbnails por camada bitmap (dataURL, cache runtime). */
  layerThumbnails: Partial<Record<ImageLayerKey, string>>;
  setImageLayerOpacity: (key: ImageLayerKey, opacity: number) => Promise<void>;
  setImageLayerLocked: (key: ImageLayerKey, locked: boolean) => Promise<void>;
  reorderImageLayers: (orderedKeys: ImageLayerKey[]) => Promise<void>;
  generateLayerThumbnail: (key: ImageLayerKey) => Promise<void>;

  // Cache-bust por camada — nunca persiste no project.json (runtime-only)
  bitmapLayerVersions: Partial<Record<BitmapLayerKey, number>>;
  bumpBitmapLayerVersion: (layerKey: MutableBitmapLayerKey) => void;

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
  currentPageKey: () => string;
  getRenderPreviewState: (pageKey: string) => RenderPreviewCacheEntry;
  markRenderPreviewStale: (pageKey: string) => void;
  markRenderPreviewRendering: (pageKey: string) => void;
  markRenderPreviewFresh: (pageKey: string, path?: string | null, previewPath?: string | null) => void;
  markRenderPreviewError: (pageKey: string, error: string) => void;
  getStaleRenderPreviewPages: () => number[];
  renderPreviewPage: (pageKey: string) => Promise<void>;
  pendingStructuralEdits: PendingStructuralEdits;
  setWorkingTraduzido: (pageKey: string, layerId: string, value: string) => void;
  setWorkingOriginal: (pageKey: string, layerId: string, value: string) => void;
  activePageAction: null | "detect" | "ocr" | "translate" | "inpaint";
  pageActionError: { action: "detect" | "ocr" | "translate" | "inpaint"; message: string } | null;
  clearPageActionError: () => void;
  runMaskedAction: (action: "detect" | "ocr" | "translate" | "inpaint") => Promise<void>;

  // ── Auto-save (Fase 3) ───────────────────────────────────────────────
  /** True quando há edição não persistida (incrementa em todo mutador). */
  dirty: boolean;
  /** Timestamp ms do último save bem-sucedido. */
  lastSavedAt: number | null;
  /** Estado visível pelo AutoSaveIndicator. */
  autoSaveStatus: "idle" | "pending" | "saving" | "saved" | "error";
  /** Versão monotônica das edições pendentes — incrementa em markDirty. */
  saveVersion: number;
  /** Versão sendo salva no momento (para descarte de race antiga). */
  saveInFlightVersion: number | null;
  /** Última mensagem de erro de save (exibida no indicator + tooltip). */
  lastSaveError: string | null;
  /** Pausa o auto-save (usado durante pipeline action). */
  autoSavePaused: boolean;
  markDirty: () => void;
  /** Salva apenas patches incrementais sem chamar loadCurrentPage (no flicker). */
  commitEditsPatchOnly: () => Promise<void>;
  /** Loop a cada 3s — chama commitEditsPatchOnly se dirty + não pausado. */
  runAutoSave: () => Promise<void>;
  /** Salva síncrono. Chame antes de troca de página, unmount, pipeline action. */
  flushAutoSave: () => Promise<void>;
  pauseAutoSave: () => void;
  resumeAutoSave: () => void;

  // ── Auto Fidelity Render (Fase 6) ───────────────────────────────────────
  /** Versão monotônica para detectar edições durante render em curso. */
  renderVersion: number;
  /** Versão do render em curso (null = nenhum rodando). */
  renderInFlightVersion: number | null;
  /** Estado do render fiel. */
  renderStatus: "idle" | "stale" | "rendering" | "updated" | "error";
  /** Mensagem do último erro de render. */
  renderError: string | null;
  /** Marca render como desatualizado + incrementa renderVersion. */
  markRenderStale: () => void;
  /** Debounce 1500ms + chama runAutoFidelityRender se ainda stale. */
  scheduleAutoFidelityRender: () => void;
  /** Executa retypesetCurrentPage com proteção anti-race. */
  runAutoFidelityRender: () => Promise<void>;
  /** Força render imediato (sem debounce). Ctrl+Shift+R. */
  forceFidelityRender: () => Promise<void>;

  setWorkingEstiloPatch: (
    pageKey: string,
    layerId: string,
    patch: Partial<TextLayerStyle>,
    touchedKeys: (keyof TextLayerStyle)[],
  ) => void;
  setWorkingBbox: (pageKey: string, layerId: string, bbox: Bbox) => void;
  insertWorkingLayer: (pageKey: string, layer: TextEntry, insertIndex: number) => void;
  deleteWorkingLayer: (pageKey: string, layerId: string) => void;
  reorderWorkingLayers: (pageKey: string, orderedIds: string[]) => void;
  applyWorkingBitmapRegion: (pageKey: string, bbox: Bbox, bytes: Uint8Array) => void;
  setWorkingVisibility: (pageKey: string, layerId: string, visible: boolean) => void;
  setWorkingLocked: (pageKey: string, layerId: string, locked: boolean) => void;
  hasLayer: (pageKey: string, layerId: string) => boolean;
  getLayer: (pageKey: string, layerId: string) => TextEntry | null;
  getOrderedLayerIds: (pageKey: string) => string[];
  sanitizeSelection: () => void;
  recordEditorCommand: (cmd: EditorCommand) => ValidationResult;
  executeEditorCommand: (cmd: EditorCommand) => ValidationResult;
  undoEditor: () => ValidationResult;
  redoEditor: () => ValidationResult;
  clearHistoryForPage: (pageKey: string) => void;
  commitEdits: () => Promise<void>;
  discardEdits: () => void;
  toggleTextLayerVisibility: (layerId: string) => Promise<void>;
  toggleTextLayerLock: (layerId: string) => void;
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
  detectInPage: () => Promise<void>;
  ocrInPage: () => Promise<void>;
  translateInPage: () => Promise<void>;
  reProcessBlock: (mode: "ocr" | "translate" | "inpaint") => Promise<void>;
  disconnectBlock: () => Promise<void>;
  resetEditor: () => void;
  // ─── Contexto da Obra ────────────────────────────────────────────────────
  setTranslationContextPatch: (patch: Partial<import("../workContext").WorkContext>) => void;
  addCharacter: (char: WorkCharacter) => void;
  updateCharacter: (id: string, patch: Partial<WorkCharacter>) => void;
  removeCharacter: (id: string) => void;
  addGlossaryEntry: (entry: WorkGlossaryEntry) => void;
  updateGlossaryEntry: (id: string, patch: Partial<WorkGlossaryEntry>) => void;
  removeGlossaryEntry: (id: string) => void;
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
  renderPreviewCacheByPageKey: {},
  pendingStructuralEdits: emptyStructuralEdits(),
  historyByPageKey: {},
  isRetypesetting: false,
  isReinpainting: false,
  isLoadingPage: false,
  lastRetypesetTime: 0,
  brushSize: 18,
  brushColor: "#000000",
  brushOpacity: 1,
  brushHardness: 0.8,
  recentBrushColors: ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"],
  // Fase 8 — Lasso
  maskShape: "freehand",
  maskOp: "replace",
  maskInProgress: null,
  // Fase 9 — Eraser
  lastPaintedLayer: "brush",
  eraserTarget: null,
  // Fase 10 — Layers MVP
  layerThumbnails: {},
  bitmapLayerVersions: {},
  activePageAction: null,
  pageActionError: null,
  dirty: false,
  lastSavedAt: null,
  autoSaveStatus: "idle",
  saveVersion: 0,
  saveInFlightVersion: null,
  lastSaveError: null,
  autoSavePaused: false,
  renderVersion: 0,
  renderInFlightVersion: null,
  renderStatus: "idle",
  renderError: null,

  clearPageActionError: () => set({ pageActionError: null }),

  markDirty: () =>
    set((state) => ({
      dirty: true,
      saveVersion: state.saveVersion + 1,
      autoSaveStatus: state.autoSaveStatus === "saving" ? "saving" : "pending",
    })),

  pauseAutoSave: () => set({ autoSavePaused: true }),
  resumeAutoSave: () => set({ autoSavePaused: false }),

  // ── Auto Fidelity Render (Fase 6) ───────────────────────────────────────
  markRenderStale: () =>
    set((state) => ({
      renderStatus: "stale",
      renderVersion: state.renderVersion + 1,
    })),

  // Debounce timer — armazenado fora do store para não causar re-renders.
  // Cada chamada a scheduleAutoFidelityRender cancela o timer anterior.

  scheduleAutoFidelityRender: (() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    return () => {
      if (timer !== null) clearTimeout(timer);
      timer = setTimeout(() => {
        timer = null;
        const state = useEditorStore.getState();
        if (state.renderStatus === "stale" && !state.isRetypesetting && !state.isReinpainting) {
          void state.runAutoFidelityRender();
        }
      }, 1500);
    };
  })(),

  runAutoFidelityRender: async () => {
    const state = get();
    if (state.renderStatus === "rendering") return;
    if (state.isRetypesetting || state.isReinpainting || state.activePageAction !== null) return;
    const version = state.renderVersion;
    set({ renderStatus: "rendering", renderInFlightVersion: version });
    try {
      await get().retypesetCurrentPage();
      if (get().renderVersion !== version) {
        // Edição aconteceu durante o render — fica stale para próximo ciclo
        set({ renderStatus: "stale", renderInFlightVersion: null });
      } else {
        set({ renderStatus: "updated", renderInFlightVersion: null });
      }
    } catch (err) {
      set({ renderStatus: "error", renderError: String(err), renderInFlightVersion: null });
    }
  },

  forceFidelityRender: async () => {
    // Ignora debounce — marca stale e roda imediatamente
    set((state) => ({ renderStatus: "stale", renderVersion: state.renderVersion + 1 }));
    await get().runAutoFidelityRender();
  },

  // Variante de commitEdits sem loadCurrentPage no fim, usada pelo auto-save
  // a cada 3s. Evita o flicker de re-render que aconteceria ao recarregar a
  // página completa do disco. (commitEdits original ainda é chamada via Ctrl+S
  // e nas pipeline actions.)
  commitEditsPatchOnly: async () => {
    const path = projectPath();
    const { pendingEdits, pendingStructuralEdits, currentPageIndex, currentPage } = get();
    const hasPendingUpdates = Object.keys(pendingEdits).length > 0;
    const hasPendingStructural = hasStructuralEdits(pendingStructuralEdits);
    if (!path || !currentPage || (!hasPendingUpdates && !hasPendingStructural)) return;

    if (hasPendingStructural) {
      // Para mudanças estruturais (criar/deletar/reordenar layers) ainda
      // precisamos do save full (saveProjectJson). Sem reload depois.
      const project = useAppStore.getState().project;
      if (!project) return;
      const materializedLayers = normalizeTextLayerOrder(
        currentPage.text_layers
          .filter((layer) => !pendingStructuralEdits.deleted[layer.id])
          .map((layer) => mergePendingEdit(layer, pendingEdits[layer.id])),
      );
      const materializedPage: PageData = {
        ...currentPage,
        text_layers: materializedLayers,
        textos: materializedLayers,
      };
      const paginas = [...project.paginas];
      paginas[currentPageIndex] = materializedPage;
      const nextProject = { ...project, paginas };
      await saveProjectJson({ project_path: path, project_json: nextProject });
      useAppStore.getState().updateProject({ paginas });
      set({ pendingEdits: {}, pendingStructuralEdits: emptyStructuralEdits() });
      return;
    }

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
      if (edit.visible !== undefined) patch.visible = edit.visible;
      if (edit.locked !== undefined) patch.locked = edit.locked;

      await patchEditorTextLayer({
        project_path: path,
        page_index: currentPageIndex,
        layer_id: layerId,
        patch,
      });
    }

    set({ pendingEdits: {}, pendingStructuralEdits: emptyStructuralEdits() });
  },

  runAutoSave: async () => {
    const state = get();
    if (
      !state.dirty ||
      state.autoSavePaused ||
      state.activePageAction !== null ||
      state.isRetypesetting ||
      state.isReinpainting ||
      state.saveInFlightVersion !== null
    ) {
      return;
    }
    const versionAtStart = state.saveVersion;
    set({ autoSaveStatus: "saving", saveInFlightVersion: versionAtStart, lastSaveError: null });
    try {
      await get().commitEditsPatchOnly();
      // Race-safe: se nova edição entrou enquanto salvávamos, mantém dirty pra
      // próxima janela do interval. Senão, marca limpo.
      const nowVersion = get().saveVersion;
      if (nowVersion === versionAtStart) {
        set({
          dirty: false,
          lastSavedAt: Date.now(),
          autoSaveStatus: "saved",
          saveInFlightVersion: null,
        });
      } else {
        set({ saveInFlightVersion: null, autoSaveStatus: "pending" });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error("[AutoSave] error:", message);
      set({
        autoSaveStatus: "error",
        lastSaveError: message,
        saveInFlightVersion: null,
      });
    }
  },

  // flushAutoSave: salvamento síncrono "obrigatório" antes de troca de página,
  // unmount, beforeunload, ou pipeline action. Bloqueia até concluir.
  flushAutoSave: async () => {
    const state = get();
    if (!state.dirty || state.activePageAction !== null) return;
    const versionAtStart = state.saveVersion;
    set({ autoSaveStatus: "saving", saveInFlightVersion: versionAtStart, lastSaveError: null });
    try {
      await get().commitEditsPatchOnly();
      const nowVersion = get().saveVersion;
      set({
        dirty: nowVersion !== versionAtStart,
        lastSavedAt: Date.now(),
        autoSaveStatus: nowVersion === versionAtStart ? "saved" : "pending",
        saveInFlightVersion: null,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error("[AutoSave] flush error:", message);
      set({
        autoSaveStatus: "error",
        lastSaveError: message,
        saveInFlightVersion: null,
      });
      throw error;
    }
  },

  bumpBitmapLayerVersion: (layerKey) =>
    set((state) => ({
      bitmapLayerVersions: {
        ...state.bitmapLayerVersions,
        [layerKey]: (state.bitmapLayerVersions[layerKey] ?? 0) + 1,
      },
    })),

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
      const liveState = get();
      const retainedTextSelection =
        liveState.selectedLayerId && payload.page.text_layers.some((layer) => layer.id === liveState.selectedLayerId)
          ? liveState.selectedLayerId
          : null;
      const retainedImageSelection =
        liveState.selectedImageLayerKey && payload.page.image_layers?.[liveState.selectedImageLayerKey]
          ? liveState.selectedImageLayerKey
          : null;
      syncCurrentPageIntoProject(payload.page, payload.page_index);
      set({
        currentPage: payload.page,
        pendingEdits: {},
        pendingStructuralEdits: emptyStructuralEdits(),
        selectedLayerId: retainedTextSelection,
        selectedImageLayerKey: retainedImageSelection,
        hoveredLayerId: null,
      });
    } finally {
      set({ isLoadingPage: false });
    }
  },

  setCurrentPage: async (index) => {
    // Auto-save desligado — usuário é responsável por salvar antes de trocar.
    // Se houver pendingEdits, o indicador "Não salvo" alerta o usuário.
    set({
      currentPageIndex: index,
      currentPage: null,
      pendingEdits: {},
      pendingStructuralEdits: emptyStructuralEdits(),
      selectedLayerId: null,
      selectedImageLayerKey: null,
      hoveredLayerId: null,
      zoom: 1,
      panOffset: { x: 0, y: 0 },
      // Estado de auto-save resetado para a nova página.
      dirty: false,
      autoSaveStatus: "idle",
      // Estado de render fiel resetado para a nova página (Fase 6).
      renderStatus: "idle",
      renderVersion: 0,
      renderInFlightVersion: null,
      renderError: null,
      maskInProgress: null,
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
  setBrushColor: (color) =>
    set((state) => ({
      brushColor: color,
      recentBrushColors: [
        color,
        ...state.recentBrushColors.filter((c) => c !== color),
      ].slice(0, 8),
    })),
  setBrushOpacity: (opacity) => set({ brushOpacity: Math.max(0, Math.min(1, opacity)) }),
  setBrushHardness: (hardness) => set({ brushHardness: Math.max(0, Math.min(1, hardness)) }),

  // Fase 8 — Lasso setters
  setMaskShape: (shape) => set({ maskShape: shape }),
  setMaskOp: (op) => set({ maskOp: op }),
  setMaskInProgress: (progress) => set({ maskInProgress: progress }),

  // Fase 9 — Eraser setter
  setEraserTarget: (target) => set({ eraserTarget: target }),
  clearMask: async () => {
    const path = projectPath();
    const { currentPageIndex, bumpBitmapLayerVersion } = get();
    if (!path) return;
    const { writeMaskFromPng } = await import("../tauri");
    const blank = document.createElement("canvas");
    blank.width = 1;
    blank.height = 1;
    const pngData = blank.toDataURL("image/png");
    try {
      await writeMaskFromPng({
        project_path: path,
        page_index: currentPageIndex,
        png_data: pngData,
        layer_key: "mask",
        op: "replace",
      });
      bumpBitmapLayerVersion("mask");
    } catch (_e) {
      // silencia falha de limpeza
    }
  },

  // ── Fase 10 — Layers MVP ─────────────────────────────────────────────────
  setImageLayerOpacity: async (key, opacity) => {
    const page = get().currentPage;
    if (!page) return;
    const clamped = Math.max(0, Math.min(1, opacity));
    const existingLayer = page.image_layers?.[key];
    if (!existingLayer) return;
    const updatedPage: PageData = {
      ...page,
      image_layers: {
        ...page.image_layers,
        [key]: { ...existingLayer, opacity: clamped },
      },
    };
    syncCurrentPageIntoProject(updatedPage, get().currentPageIndex);
    set({ currentPage: updatedPage });
    get().markDirty();
    get().markRenderStale();
    get().scheduleAutoFidelityRender();
    // Auto-save será responsável por persistir opacity via commitEdits
    void get().runAutoSave();
  },

  setImageLayerLocked: async (key, locked) => {
    const page = get().currentPage;
    if (!page) return;
    const existingLayer = page.image_layers?.[key];
    if (!existingLayer) return;
    const updatedPage: PageData = {
      ...page,
      image_layers: {
        ...page.image_layers,
        [key]: { ...existingLayer, locked },
      },
    };
    syncCurrentPageIntoProject(updatedPage, get().currentPageIndex);
    set({ currentPage: updatedPage });
    get().markDirty();
  },

  reorderImageLayers: async (orderedKeys) => {
    const page = get().currentPage;
    if (!page) return;
    const imageLayers = { ...page.image_layers };
    orderedKeys.forEach((key, index) => {
      if (imageLayers[key]) {
        imageLayers[key] = { ...imageLayers[key]!, order: index };
      }
    });
    const updatedPage: PageData = { ...page, image_layers: imageLayers };
    syncCurrentPageIntoProject(updatedPage, get().currentPageIndex);
    set({ currentPage: updatedPage });
    get().markDirty();
    get().markRenderStale();
    get().scheduleAutoFidelityRender();
  },

  generateLayerThumbnail: async (key) => {
    const page = get().currentPage;
    const layer = page?.image_layers?.[key];
    if (!layer?.path) return;
    try {
      const { readFile } = await import("@tauri-apps/plugin-fs");
      const bytes = await readFile(layer.path);
      const blob = new Blob([bytes], { type: "image/png" });
      const srcUrl = URL.createObjectURL(blob);
      await new Promise<void>((resolve) => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement("canvas");
          canvas.width = 32;
          canvas.height = 32;
          const ctx = canvas.getContext("2d");
          ctx?.drawImage(img, 0, 0, 32, 32);
          const dataUrl = canvas.toDataURL("image/png");
          URL.revokeObjectURL(srcUrl);
          set((state) => ({ layerThumbnails: { ...state.layerThumbnails, [key]: dataUrl } }));
          resolve();
        };
        img.onerror = () => { URL.revokeObjectURL(srcUrl); resolve(); };
        img.src = srcUrl;
      });
    } catch {
      // silencia falha de thumbnail
    }
  },

  updatePendingEdit: (layerId, changes) => {
    const pageKey = get().currentPageKey();
    set((state) => ({
      pendingEdits: {
        ...state.pendingEdits,
        [layerId]: { ...state.pendingEdits[layerId], ...changes },
      },
      selectedLayerId: layerId,
      selectedImageLayerKey: null,
    }));
    get().markRenderPreviewStale(pageKey);
    // Fase 6: agenda re-render fiel (debounce 1.5s)
    get().markRenderStale();
    get().scheduleAutoFidelityRender();
  },

  updatePendingEstilo: (layerId, estiloChanges) => {
    const pageKey = get().currentPageKey();
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
    });
    get().markRenderPreviewStale(pageKey);
    // Fase 6: agenda re-render fiel (debounce 1.5s)
    get().markRenderStale();
    get().scheduleAutoFidelityRender();
  },

  currentPageKey: () => {
    const project = useAppStore.getState().project;
    if (!project) return "";
    return getPageKey(project, get().currentPageIndex);
  },

  getRenderPreviewState: (pageKey) =>
    getRenderPreviewStateForPage(pageKey, get().currentPage, get().renderPreviewCacheByPageKey),

  markRenderPreviewStale: (pageKey) => {
    if (!pageKey) return;
    set((state) => {
      const current = getRenderPreviewStateForPage(pageKey, state.currentPage, state.renderPreviewCacheByPageKey);
      return {
        renderPreviewCacheByPageKey: {
          ...state.renderPreviewCacheByPageKey,
          [pageKey]: {
            ...current,
            fingerprint: renderPreviewFingerprint(state.currentPage, state.pendingEdits, state.pendingStructuralEdits),
            status: "stale",
            error: null,
          },
        },
      };
    });
  },

  markRenderPreviewRendering: (pageKey) => {
    if (!pageKey) return;
    set((state) => {
      const current = getRenderPreviewStateForPage(pageKey, state.currentPage, state.renderPreviewCacheByPageKey);
      return {
        renderPreviewCacheByPageKey: {
          ...state.renderPreviewCacheByPageKey,
          [pageKey]: {
            ...current,
            fingerprint: renderPreviewFingerprint(state.currentPage, state.pendingEdits, state.pendingStructuralEdits),
            status: "rendering",
            error: null,
          },
        },
      };
    });
  },

  markRenderPreviewFresh: (pageKey, path, previewPath) => {
    if (!pageKey) return;
    set((state) => {
      const current = getRenderPreviewStateForPage(pageKey, state.currentPage, state.renderPreviewCacheByPageKey);
      return {
        renderPreviewCacheByPageKey: {
          ...state.renderPreviewCacheByPageKey,
          [pageKey]: {
            ...current,
            fingerprint: renderPreviewFingerprint(state.currentPage),
            status: "fresh",
            path: path ?? renderedPathForPage(state.currentPage),
            previewPath: previewPath ?? null,
            generatedAt: Date.now(),
            error: null,
          },
        },
      };
    });
  },

  markRenderPreviewError: (pageKey, error) => {
    if (!pageKey) return;
    set((state) => {
      const current = getRenderPreviewStateForPage(pageKey, state.currentPage, state.renderPreviewCacheByPageKey);
      return {
        renderPreviewCacheByPageKey: {
          ...state.renderPreviewCacheByPageKey,
          [pageKey]: {
            ...current,
            fingerprint: renderPreviewFingerprint(state.currentPage, state.pendingEdits, state.pendingStructuralEdits),
            status: "error",
            error,
          },
        },
      };
    });
  },

  getStaleRenderPreviewPages: () =>
    getStaleRenderPreviewPages(useAppStore.getState().project, get().renderPreviewCacheByPageKey),

  renderPreviewPage: async (pageKey) => {
    const path = projectPath();
    const { currentPage, currentPageIndex, pendingEdits, pendingStructuralEdits } = get();
    if (!path || !currentPage || pageKey !== get().currentPageKey()) return;
    const materializedPage = materializeWorkingPage(currentPage, pendingEdits, pendingStructuralEdits);
    const fingerprint = renderPreviewFingerprint(currentPage, pendingEdits, pendingStructuralEdits);
    get().markRenderPreviewRendering(pageKey);
    try {
      const previewPath = await renderPreviewPageCommand({
        project_path: path,
        page_index: currentPageIndex,
        page: materializedPage,
        fingerprint: previewCacheKey(fingerprint),
      });
      get().markRenderPreviewFresh(pageKey, renderedPathForPage(currentPage), previewPath);
      set({ lastRetypesetTime: Date.now(), viewMode: "translated" });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      get().markRenderPreviewError(pageKey, message);
      throw err;
    }
  },

  setWorkingTraduzido: (pageKey, layerId, value) => {
    const shouldMarkStale =
      pageKey === get().currentPageKey() &&
      !!get().currentPage?.text_layers.some((item) => item.id === layerId);
    set((state) => {
      if (pageKey !== get().currentPageKey()) return {};
      const layer = state.currentPage?.text_layers.find((item) => item.id === layerId);
      if (!layer) return {};
      const pending = { ...state.pendingEdits };
      const current = { ...(pending[layerId] ?? {}) };
      const baseText = layer.traduzido ?? layer.translated ?? "";
      if (value === baseText) {
        delete current.traduzido;
        delete current.translated;
      } else {
        current.traduzido = value;
        current.translated = value;
      }
      pending[layerId] = current;
      return {
        pendingEdits: removeEmptyPendingEdit(pending, layerId),
        selectedLayerId: layerId,
        selectedImageLayerKey: null,
      };
    });
    if (shouldMarkStale) get().markRenderPreviewStale(pageKey);
  },

  setWorkingOriginal: (pageKey, layerId, value) => {
    const shouldMarkStale =
      pageKey === get().currentPageKey() &&
      !!get().currentPage?.text_layers.some((item) => item.id === layerId);
    set((state) => {
      if (pageKey !== get().currentPageKey()) return {};
      const layer = state.currentPage?.text_layers.find((item) => item.id === layerId);
      if (!layer) return {};
      const pending = { ...state.pendingEdits };
      const current = { ...(pending[layerId] ?? {}) };
      if (value === layer.original) {
        delete current.original;
      } else {
        current.original = value;
      }
      pending[layerId] = current;
      return {
        pendingEdits: removeEmptyPendingEdit(pending, layerId),
        selectedLayerId: layerId,
        selectedImageLayerKey: null,
      };
    });
    if (shouldMarkStale) get().markRenderPreviewStale(pageKey);
  },

  runMaskedAction: async (action) => {
    if (get().activePageAction) return; // previne duplo clique
    const path = projectPath();
    if (!path) return;
    const pageIndex = get().currentPageIndex;
    // Fase 3: flush auto-save antes de qualquer ação pipeline para garantir
    // que pendingEdits estão persistidos no project.json que o sidecar lê.
    try {
      await get().flushAutoSave();
    } catch {
      /* erro já registrado em lastSaveError; segue mesmo assim */
    }
    set({ activePageAction: action, pageActionError: null });
    console.log(`[EditorAction] start  ${action} page=${pageIndex}`);
    try {
      const result = await runPageActionWithOptionalMask({
        project_path: path,
        page_index: pageIndex,
        action,
      });
      await get().loadCurrentPage();
      // Cache-bust para assets que foram modificados
      for (const asset of result.changed_assets) {
        if (asset === "inpaint") get().bumpBitmapLayerVersion("inpaint");
        if (asset === "rendered") get().bumpBitmapLayerVersion("rendered");
        if (asset === "mask") get().bumpBitmapLayerVersion("mask");
      }
      get().markRenderPreviewStale(get().currentPageKey());
      // Fase 8: limpar lasso após inpaint concluído
      if (action === "inpaint") {
        set({ maskInProgress: null });
      }
      console.log(
        `[EditorAction] success ${action} page=${pageIndex} changed=${result.changed_assets.join(",")}`,
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`[EditorAction] error  ${action} page=${pageIndex}: ${message}`);
      set({ pageActionError: { action, message } });
    } finally {
      set({ activePageAction: null });
    }
  },

  setWorkingEstiloPatch: (pageKey, layerId, patch, touchedKeys) => {
    const shouldMarkStale =
      pageKey === get().currentPageKey() &&
      !!get().currentPage?.text_layers.some((item) => item.id === layerId);
    set((state) => {
      if (pageKey !== get().currentPageKey()) return {};
      const layer = state.currentPage?.text_layers.find((item) => item.id === layerId);
      if (!layer) return {};
      const pending = { ...state.pendingEdits };
      const current = { ...(pending[layerId] ?? {}) };
      const pendingStyle: Partial<TextLayerStyle> = { ...(current.estilo ?? {}) };
      for (const key of touchedKeys) {
        const nextValue = patch[key];
        if (nextValue === layer.estilo[key]) {
          delete pendingStyle[key];
        } else {
          pendingStyle[key] = nextValue as never;
        }
      }
      if (Object.keys(pendingStyle).length > 0) {
        current.estilo = pendingStyle as TextEntry["estilo"];
      } else {
        delete current.estilo;
      }
      pending[layerId] = current;
      return {
        pendingEdits: removeEmptyPendingEdit(pending, layerId),
        selectedLayerId: layerId,
        selectedImageLayerKey: null,
      };
    });
    if (shouldMarkStale) get().markRenderPreviewStale(pageKey);
  },

  setWorkingBbox: (pageKey, layerId, bbox) => {
    const shouldMarkStale =
      pageKey === get().currentPageKey() &&
      !!get().currentPage?.text_layers.some((item) => item.id === layerId);
    set((state) => {
      if (pageKey !== get().currentPageKey()) return {};
      const layer = state.currentPage?.text_layers.find((item) => item.id === layerId);
      if (!layer) return {};
      const pending = { ...state.pendingEdits };
      const current = { ...(pending[layerId] ?? {}) };
      const baseBbox = layer.layout_bbox ?? layer.bbox;
      if (sameBbox(baseBbox, bbox)) {
        delete current.bbox;
      } else {
        current.bbox = bbox;
      }
      pending[layerId] = current;
      return {
        pendingEdits: removeEmptyPendingEdit(pending, layerId),
        selectedLayerId: layerId,
        selectedImageLayerKey: null,
      };
    });
    if (shouldMarkStale) get().markRenderPreviewStale(pageKey);
  },

  insertWorkingLayer: (pageKey, layer, insertIndex) => {
    const shouldMarkStale = pageKey === get().currentPageKey() && !!get().currentPage;
    set((state) => {
      if (pageKey !== get().currentPageKey() || !state.currentPage) return {};
      const withoutExisting = state.currentPage.text_layers.filter((item) => item.id !== layer.id);
      const boundedIndex = Math.max(0, Math.min(insertIndex, withoutExisting.length));
      const nextLayers = normalizeTextLayerOrder([
        ...withoutExisting.slice(0, boundedIndex),
        layer,
        ...withoutExisting.slice(boundedIndex),
      ]);
      const baseLayer = getBaseLayer(state.currentPageIndex, layer.id);
      const structural = {
        ...state.pendingStructuralEdits,
        created: baseLayer
          ? state.pendingStructuralEdits.created.filter((item) => item.id !== layer.id)
          : [
              ...state.pendingStructuralEdits.created.filter((item) => item.id !== layer.id),
              nextLayers.find((item) => item.id === layer.id) ?? layer,
            ],
        deleted: { ...state.pendingStructuralEdits.deleted },
      };
      delete structural.deleted[layer.id];
      const updatedPage = { ...state.currentPage, text_layers: nextLayers, textos: nextLayers };
      return {
        currentPage: updatedPage,
        pendingStructuralEdits: structural,
        selectedLayerId: layer.id,
        selectedImageLayerKey: null,
      };
    });
    if (shouldMarkStale) get().markRenderPreviewStale(pageKey);
  },

  deleteWorkingLayer: (pageKey, layerId) => {
    const shouldMarkStale =
      pageKey === get().currentPageKey() &&
      !!get().currentPage?.text_layers.some((item) => item.id === layerId);
    set((state) => {
      if (pageKey !== get().currentPageKey() || !state.currentPage) return {};
      const existing = state.currentPage.text_layers.find((layer) => layer.id === layerId);
      if (!existing) return {};
      const nextLayers = normalizeTextLayerOrder(state.currentPage.text_layers.filter((layer) => layer.id !== layerId));
      const baseLayer = getBaseLayer(state.currentPageIndex, layerId);
      const structural = {
        ...state.pendingStructuralEdits,
        created: state.pendingStructuralEdits.created.filter((layer) => layer.id !== layerId),
        deleted: { ...state.pendingStructuralEdits.deleted },
      };
      if (baseLayer) {
        structural.deleted[layerId] = baseLayer;
      } else {
        delete structural.deleted[layerId];
      }
      const { [layerId]: _removed, ...pendingEdits } = state.pendingEdits;
      const updatedPage = { ...state.currentPage, text_layers: nextLayers, textos: nextLayers };
      return {
        currentPage: updatedPage,
        pendingEdits,
        pendingStructuralEdits: structural,
        selectedLayerId: state.selectedLayerId === layerId ? null : state.selectedLayerId,
        hoveredLayerId: state.hoveredLayerId === layerId ? null : state.hoveredLayerId,
      };
    });
    if (shouldMarkStale) get().markRenderPreviewStale(pageKey);
  },

  reorderWorkingLayers: (pageKey, orderedIds) => {
    const shouldMarkStale = pageKey === get().currentPageKey() && !!get().currentPage;
    set((state) => {
      if (pageKey !== get().currentPageKey() || !state.currentPage) return {};
      const byId = new Map(state.currentPage.text_layers.map((layer) => [layer.id, layer]));
      const ordered = orderedIds.flatMap((id) => {
        const layer = byId.get(id);
        return layer ? [layer] : [];
      });
      const missing = state.currentPage.text_layers.filter((layer) => !orderedIds.includes(layer.id));
      const nextLayers = normalizeTextLayerOrder([...ordered, ...missing]);
      const baseOrder = sortTextLayers(getBasePage(state.currentPageIndex)?.text_layers ?? []).map((layer) => layer.id);
      const nextOrder = nextLayers.map((layer) => layer.id);
      const updatedPage = { ...state.currentPage, text_layers: nextLayers, textos: nextLayers };
      return {
        currentPage: updatedPage,
        pendingStructuralEdits: {
          ...state.pendingStructuralEdits,
          created: state.pendingStructuralEdits.created.map((created) => nextLayers.find((layer) => layer.id === created.id) ?? created),
          order: nextOrder.length === baseOrder.length && nextOrder.every((id, index) => id === baseOrder[index])
            ? undefined
            : nextOrder,
          },
      };
    });
    if (shouldMarkStale) get().markRenderPreviewStale(pageKey);
  },
  applyWorkingBitmapRegion: () => undefined,

  setWorkingVisibility: (pageKey, layerId, visible) => {
    const shouldMarkStale =
      pageKey === get().currentPageKey() &&
      !!get().currentPage?.text_layers.some((item) => item.id === layerId);
    set((state) => {
      if (pageKey !== get().currentPageKey() || !state.currentPage) return {};
      const updatedLayers = state.currentPage.text_layers.map((layer) =>
        layer.id === layerId ? { ...layer, visible } : layer,
      );
      const baseVisible = getBaseLayer(state.currentPageIndex, layerId)?.visible ?? true;
      const pending = { ...state.pendingEdits };
      const current = { ...(pending[layerId] ?? {}) };
      if (visible === baseVisible) delete current.visible;
      else current.visible = visible;
      pending[layerId] = current;
      const updatedPage = { ...state.currentPage, text_layers: updatedLayers, textos: updatedLayers };
      return {
        currentPage: updatedPage,
        pendingEdits: removeEmptyPendingEdit(pending, layerId),
      };
    });
    if (shouldMarkStale) get().markRenderPreviewStale(pageKey);
  },

  setWorkingLocked: (pageKey, layerId, locked) =>
    set((state) => {
      if (pageKey !== get().currentPageKey() || !state.currentPage) return {};
      const updatedLayers = state.currentPage.text_layers.map((layer) =>
        layer.id === layerId ? { ...layer, locked } : layer,
      );
      const baseLocked = getBaseLayer(state.currentPageIndex, layerId)?.locked ?? false;
      const pending = { ...state.pendingEdits };
      const current = { ...(pending[layerId] ?? {}) };
      if (locked === baseLocked) delete current.locked;
      else current.locked = locked;
      pending[layerId] = current;
      const updatedPage = { ...state.currentPage, text_layers: updatedLayers, textos: updatedLayers };
      return {
        currentPage: updatedPage,
        pendingEdits: removeEmptyPendingEdit(pending, layerId),
      };
    }),

  hasLayer: (pageKey, layerId) =>
    pageKey === get().currentPageKey() &&
    !!get().currentPage?.text_layers.some((layer) => layer.id === layerId),

  getLayer: (pageKey, layerId) => {
    if (pageKey !== get().currentPageKey()) return null;
    const layer = get().currentPage?.text_layers.find((item) => item.id === layerId);
    if (!layer) return null;
    const edit = get().pendingEdits[layerId];
    if (!edit) return layer;
    return {
      ...layer,
      ...edit,
      estilo: edit.estilo ? { ...layer.estilo, ...edit.estilo } : layer.estilo,
      style: edit.estilo ? { ...layer.estilo, ...edit.estilo } : layer.style,
    };
  },

  getOrderedLayerIds: (pageKey) => {
    if (pageKey !== get().currentPageKey()) return [];
    return sortTextLayers(get().currentPage?.text_layers ?? []).map((layer) => layer.id);
  },

  sanitizeSelection: () =>
    set((state) => {
      const selectedLayerStillExists =
        !state.selectedLayerId ||
        !!state.currentPage?.text_layers.some((layer) => layer.id === state.selectedLayerId);
      return selectedLayerStillExists ? {} : { selectedLayerId: null };
    }),

  recordEditorCommand: (cmd) => {
    const pageKey = cmd.pageKey;
    const stack =
      get().historyByPageKey[pageKey] ?? createHistoryStack(pageKey, pageFingerprint(get().currentPage));
    const draft = get() as WorkingStateDraft;
    const result = recordCommand(cmd, draft, stack);
    if (result.ok) {
      const historyByPageKey = { ...get().historyByPageKey, [pageKey]: stack };
      pruneHistoryStacksByGlobalCap(Object.values(historyByPageKey));
      set({ historyByPageKey: { ...historyByPageKey } });
    }
    return result;
  },

  executeEditorCommand: (cmd) => {
    const pageKey = cmd.pageKey;
    const stack =
      get().historyByPageKey[pageKey] ?? createHistoryStack(pageKey, pageFingerprint(get().currentPage));
    const draft = get() as WorkingStateDraft;
    const result = executeCommand(cmd, draft, stack);
    if (result.ok) {
      const historyByPageKey = { ...get().historyByPageKey, [pageKey]: stack };
      pruneHistoryStacksByGlobalCap(Object.values(historyByPageKey));
      set({ historyByPageKey: { ...historyByPageKey } });
    }
    return result;
  },

  undoEditor: () => {
    const pageKey = get().currentPageKey();
    const stack = get().historyByPageKey[pageKey];
    if (!stack) return { ok: false, reason: "nada para desfazer" };
    const result = undoHistory(stack, get() as WorkingStateDraft);
    if (result.ok) set({ historyByPageKey: { ...get().historyByPageKey, [pageKey]: stack } });
    return result;
  },

  redoEditor: () => {
    const pageKey = get().currentPageKey();
    const stack = get().historyByPageKey[pageKey];
    if (!stack) return { ok: false, reason: "nada para refazer" };
    const result = redoHistory(stack, get() as WorkingStateDraft);
    if (result.ok) set({ historyByPageKey: { ...get().historyByPageKey, [pageKey]: stack } });
    return result;
  },

  clearHistoryForPage: (pageKey) => {
    const { [pageKey]: _removed, ...rest } = get().historyByPageKey;
    disposeAllForPage(pageKey);
    set({ historyByPageKey: rest });
  },

  commitEdits: async () => {
    const path = projectPath();
    const { pendingEdits, pendingStructuralEdits, currentPageIndex, currentPage } = get();
    const hasPendingUpdates = Object.keys(pendingEdits).length > 0;
    const hasPendingStructural = hasStructuralEdits(pendingStructuralEdits);
    if (!path || !currentPage || (!hasPendingUpdates && !hasPendingStructural)) return;

    if (hasPendingStructural) {
      const project = useAppStore.getState().project;
      if (!project) return;
      const materializedLayers = normalizeTextLayerOrder(
        currentPage.text_layers
          .filter((layer) => !pendingStructuralEdits.deleted[layer.id])
          .map((layer) => mergePendingEdit(layer, pendingEdits[layer.id])),
      );
      const materializedPage: PageData = {
        ...currentPage,
        text_layers: materializedLayers,
        textos: materializedLayers,
      };
      const paginas = [...project.paginas];
      paginas[currentPageIndex] = materializedPage;
      const nextProject = { ...project, paginas };
      await saveProjectJson({
        project_path: path,
        project_json: nextProject,
      });
      useAppStore.getState().updateProject({ paginas });
      set({ pendingEdits: {}, pendingStructuralEdits: emptyStructuralEdits() });
      await get().loadCurrentPage();
      const pageKey = get().currentPageKey();
      const stack = get().historyByPageKey[pageKey];
      if (stack) {
        updateHistoryBaseFingerprint(stack, pageFingerprint(get().currentPage));
        set({ historyByPageKey: { ...get().historyByPageKey, [pageKey]: stack } });
      }
      return;
    }

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
      if (edit.visible !== undefined) patch.visible = edit.visible;
      if (edit.locked !== undefined) patch.locked = edit.locked;

      await patchEditorTextLayer({
        project_path: path,
        page_index: currentPageIndex,
        layer_id: layerId,
        patch,
      });
    }

    set({ pendingEdits: {}, pendingStructuralEdits: emptyStructuralEdits() });
    await get().loadCurrentPage();
    const pageKey = get().currentPageKey();
    const stack = get().historyByPageKey[pageKey];
    if (stack) {
      updateHistoryBaseFingerprint(stack, pageFingerprint(get().currentPage));
      set({ historyByPageKey: { ...get().historyByPageKey, [pageKey]: stack } });
    }
  },

  discardEdits: () => {
    const basePage = getBasePage(get().currentPageIndex);
    const pageKey = get().currentPageKey();
    set({
      currentPage: basePage,
      pendingEdits: {},
      pendingStructuralEdits: emptyStructuralEdits(),
      selectedLayerId: null,
      selectedImageLayerKey: null,
    });
    get().markRenderPreviewFresh(pageKey, renderedPathForPage(basePage));
  },

  toggleTextLayerVisibility: async (layerId) => {
    const page = get().currentPage;
    if (!page) return;
    const layer = page.text_layers.find((item) => item.id === layerId);
    if (!layer) return;
    get().executeEditorCommand({
      commandId: crypto.randomUUID(),
      pageKey: get().currentPageKey(),
      createdAt: Date.now(),
      type: "toggle-visibility",
      layerId,
      before: layer.visible ?? true,
      after: !(layer.visible ?? true),
    });
  },

  toggleTextLayerLock: (layerId) => {
    const page = get().currentPage;
    if (!page) return;
    const layer = page.text_layers.find((item) => item.id === layerId);
    if (!layer) return;
    get().executeEditorCommand({
      commandId: crypto.randomUUID(),
      pageKey: get().currentPageKey(),
      createdAt: Date.now(),
      type: "toggle-lock",
      layerId,
      before: layer.locked ?? false,
      after: !(layer.locked ?? false),
    });
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
    if (layerKey === "inpaint" || layerKey === "mask" || layerKey === "brush") {
      get().markRenderPreviewStale(get().currentPageKey());
    }
  },

  createTextLayer: async (bbox) => {
    const page = get().currentPage;
    if (!page) return;
    const created = createLocalTextLayer(bbox, page.text_layers.length);
    const previousSelection = get().selectedLayerId;
    get().executeEditorCommand({
      commandId: crypto.randomUUID(),
      pageKey: get().currentPageKey(),
      createdAt: Date.now(),
      type: "create-layer",
      layerId: created.id,
      layer: created,
      insertIndex: page.text_layers.length,
      selectionBefore: { ids: previousSelection ? [previousSelection] : [], primary: previousSelection },
      selectionAfter: { ids: [created.id], primary: created.id },
    });
    set({
      selectedLayerId: created.id,
      selectedImageLayerKey: null,
      toolMode: "select",
      lastRetypesetTime: Date.now(),
    });
  },

  deleteSelectedLayer: async () => {
    const page = get().currentPage;
    const layerId = get().selectedLayerId;
    if (!page || !layerId) return;
    const layer = page.text_layers.find((item) => item.id === layerId);
    if (!layer) return;
    const index = sortTextLayers(page.text_layers).findIndex((item) => item.id === layerId);
    get().executeEditorCommand({
      commandId: crypto.randomUUID(),
      pageKey: get().currentPageKey(),
      createdAt: Date.now(),
      type: "delete-layer",
      layerId,
      layer: mergePendingEdit(layer, get().pendingEdits[layerId]),
      index: Math.max(0, index),
      selectionBefore: { ids: [layerId], primary: layerId },
      selectionAfter: { ids: [], primary: null },
    });
  },

  applyBitmapStroke: async ({ width, height, strokes, clear = false }) => {
    const path = projectPath();
    const page = get().currentPage;
    if (!path || !page || strokes.length === 0) return;

    const mode = get().toolMode;
    const erase = mode === "eraser";

    // Fase 9: Borracha inteligente — inferir alvo
    let layerKey: "brush" | "mask";
    if (mode === "eraser") {
      // Alvo explícito > camada selecionada > última camada pintada > brush (default)
      const explicit = get().eraserTarget;
      const sel = get().selectedImageLayerKey;
      if (explicit) {
        layerKey = explicit;
      } else if (sel === "brush" || sel === "mask") {
        layerKey = sel;
      } else {
        layerKey = get().lastPaintedLayer;
      }
    } else if (mode === "brush") {
      layerKey = "brush";
    } else {
      layerKey = "mask";
    }

    // Guardar última camada pintada (não apagada)
    if (!erase) {
      set({ lastPaintedLayer: layerKey });
    }

    // Bloquear stroke em camadas travadas
    const layerMeta = page.image_layers?.[layerKey];
    if (layerMeta?.locked) {
      console.warn(`[Eraser/Brush] Camada "${layerKey}" está travada — stroke ignorado.`);
      return;
    }

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
    // Cache-bust da camada editada (runtime-only, não persiste no project.json)
    get().bumpBitmapLayerVersion(layerKey as MutableBitmapLayerKey);
    get().markRenderPreviewStale(get().currentPageKey());
  },

  retypesetCurrentPage: async () => {
    const path = projectPath();
    if (!path) return;
    const pageKey = get().currentPageKey();
    await get().commitEdits();
    set({ isRetypesetting: true });
    try {
      await retypesetPage({
        project_path: path,
        page_index: get().currentPageIndex,
      });
      await get().loadCurrentPage();
      get().markRenderPreviewFresh(pageKey, renderedPathForPage(get().currentPage));
      get().bumpBitmapLayerVersion("rendered");
      set({ lastRetypesetTime: Date.now(), viewMode: "translated", showOverlays: true });
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
      get().markRenderPreviewFresh(get().currentPageKey(), renderedPathForPage(get().currentPage));
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
    const pageKey = get().currentPageKey();
    await get().commitEdits();
    set({ isReinpainting: true });
    try {
      await reinpaintPage({
        project_path: path,
        page_index: get().currentPageIndex,
      });
      await get().loadCurrentPage();
      get().markRenderPreviewStale(pageKey);
      get().bumpBitmapLayerVersion("inpaint");
      set({ lastRetypesetTime: Date.now(), viewMode: "inpainted" });
    } finally {
      set({ isReinpainting: false });
    }
  },

  detectInPage: async () => {
    const path = projectPath();
    if (!path) return;
    const pageKey = get().currentPageKey();
    set({ isRetypesetting: true });
    try {
      await detectPage({
        project_path: path,
        page_index: get().currentPageIndex,
      });
      await get().loadCurrentPage();
      get().markRenderPreviewFresh(pageKey, renderedPathForPage(get().currentPage));
      set({
        lastRetypesetTime: Date.now(),
        showOverlays: true,
        selectedLayerId: null,
        selectedImageLayerKey: null,
      });
    } finally {
      set({ isRetypesetting: false });
    }
  },

  ocrInPage: async () => {
    const path = projectPath();
    if (!path) return;
    const pageKey = get().currentPageKey();
    set({ isRetypesetting: true });
    try {
      await ocrPage({
        project_path: path,
        page_index: get().currentPageIndex,
      });
      await get().loadCurrentPage();
      get().markRenderPreviewFresh(pageKey, renderedPathForPage(get().currentPage));
      set({ lastRetypesetTime: Date.now(), viewMode: "translated", showOverlays: true });
    } finally {
      set({ isRetypesetting: false });
    }
  },

  translateInPage: async () => {
    const path = projectPath();
    if (!path) return;
    const pageKey = get().currentPageKey();
    set({ isRetypesetting: true });
    try {
      await translatePage({
        project_path: path,
        page_index: get().currentPageIndex,
      });
      await get().loadCurrentPage();
      get().markRenderPreviewFresh(pageKey, renderedPathForPage(get().currentPage));
      set({ lastRetypesetTime: Date.now(), viewMode: "translated", showOverlays: true });
    } finally {
      set({ isRetypesetting: false });
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
      renderPreviewCacheByPageKey: {},
      pendingStructuralEdits: emptyStructuralEdits(),
      historyByPageKey: {},
      isRetypesetting: false,
      isReinpainting: false,
      isLoadingPage: false,
      lastRetypesetTime: 0,
      brushSize: 18,
      brushColor: "#000000",
      brushOpacity: 1,
      brushHardness: 0.8,
      maskShape: "freehand",
      maskOp: "replace",
      maskInProgress: null,
      lastPaintedLayer: "brush",
      eraserTarget: null,
      layerThumbnails: {},
      dirty: false,
      autoSaveStatus: "idle",
      renderStatus: "idle",
      renderVersion: 0,
      renderInFlightVersion: null,
      renderError: null,
    }),

  // ─── Contexto da Obra ────────────────────────────────────────────────────

  setTranslationContextPatch: (patch) => {
    useAppStore.setState((appState) => {
      if (!appState.project) return {};
      const current = appState.project.translation_context ?? defaultWorkContext();
      return {
        project: {
          ...appState.project,
          translation_context: { ...current, ...patch, updatedAt: new Date().toISOString() },
        },
      };
    });
  },

  addCharacter: (char) => {
    useAppStore.setState((appState) => {
      if (!appState.project) return {};
      const ctx = appState.project.translation_context ?? defaultWorkContext();
      return {
        project: {
          ...appState.project,
          translation_context: {
            ...ctx,
            characters: [...(ctx.characters ?? []), char],
            updatedAt: new Date().toISOString(),
          },
        },
      };
    });
  },

  updateCharacter: (id, patch) => {
    useAppStore.setState((appState) => {
      if (!appState.project) return {};
      const ctx = appState.project.translation_context ?? defaultWorkContext();
      return {
        project: {
          ...appState.project,
          translation_context: {
            ...ctx,
            characters: (ctx.characters ?? []).map((c) => (c.id === id ? { ...c, ...patch } : c)),
            updatedAt: new Date().toISOString(),
          },
        },
      };
    });
  },

  removeCharacter: (id) => {
    useAppStore.setState((appState) => {
      if (!appState.project) return {};
      const ctx = appState.project.translation_context ?? defaultWorkContext();
      return {
        project: {
          ...appState.project,
          translation_context: {
            ...ctx,
            characters: (ctx.characters ?? []).filter((c) => c.id !== id),
            updatedAt: new Date().toISOString(),
          },
        },
      };
    });
  },

  addGlossaryEntry: (entry) => {
    useAppStore.setState((appState) => {
      if (!appState.project) return {};
      const ctx = appState.project.translation_context ?? defaultWorkContext();
      return {
        project: {
          ...appState.project,
          translation_context: {
            ...ctx,
            glossary: [...(ctx.glossary ?? []), entry],
            updatedAt: new Date().toISOString(),
          },
        },
      };
    });
  },

  updateGlossaryEntry: (id, patch) => {
    useAppStore.setState((appState) => {
      if (!appState.project) return {};
      const ctx = appState.project.translation_context ?? defaultWorkContext();
      return {
        project: {
          ...appState.project,
          translation_context: {
            ...ctx,
            glossary: (ctx.glossary ?? []).map((e) => (e.id === id ? { ...e, ...patch } : e)),
            updatedAt: new Date().toISOString(),
          },
        },
      };
    });
  },

  removeGlossaryEntry: (id) => {
    useAppStore.setState((appState) => {
      if (!appState.project) return {};
      const ctx = appState.project.translation_context ?? defaultWorkContext();
      return {
        project: {
          ...appState.project,
          translation_context: {
            ...ctx,
            glossary: (ctx.glossary ?? []).filter((e) => e.id !== id),
            updatedAt: new Date().toISOString(),
          },
        },
      };
    });
  },
}));

// ── Auto-save dirty tracking (Fase 3) ─────────────────────────────────
// Em vez de chamar markDirty() em ~20 mutadores, usamos um subscriber que
// detecta qualquer alteração em pendingEdits ou pendingStructuralEdits.
// Quando os dois ficam vazios (após save bem-sucedido), o subscriber não
// faz nada — markDirty só dispara em transições não-vazio→não-vazio ou
// vazio→não-vazio com conteúdo novo.
let lastTrackedPendingHash = "";
function hashPendingEdits(s: EditorState): string {
  const peKeys = Object.keys(s.pendingEdits).sort().join(",");
  const struct =
    s.pendingStructuralEdits.created.length +
    "_" +
    Object.keys(s.pendingStructuralEdits.deleted).length +
    "_" +
    (s.pendingStructuralEdits.order ? "1" : "0");
  return `${peKeys}|${struct}|${JSON.stringify(s.pendingEdits)}`;
}
useEditorStore.subscribe((state) => {
  const hash = hashPendingEdits(state);
  if (hash !== lastTrackedPendingHash) {
    lastTrackedPendingHash = hash;
    // Só dispara markDirty se há de fato algo pendente. Evita marcar dirty
    // logo após um save bem-sucedido (que limpou pendingEdits).
    const hasAny =
      Object.keys(state.pendingEdits).length > 0 ||
      hasStructuralEdits(state.pendingStructuralEdits);
    if (hasAny && !state.dirty) {
      state.markDirty();
    }
    // Também incrementa saveVersion enquanto continua dirty (cada nova edição
    // invalida saves em curso pelo race-check em runAutoSave).
    if (hasAny && state.dirty) {
      useEditorStore.setState((s) => ({ saveVersion: s.saveVersion + 1 }));
    }
  }
});
