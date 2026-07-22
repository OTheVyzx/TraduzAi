import type { StudioPage, StudioProject, StudioTextLayer } from "../project/studioProject";
import type { BitmapRegionConfig, StudioEditorBackend } from "./editorBackend";

export type PageActionName = "detect" | "detect_boxes" | "ocr" | "translate" | "inpaint";
export type PageActionChangedAsset = "brush" | "mask" | "inpaint" | "rendered" | "preview" | "project_json";

export interface EditorPagePayload {
  project_file?: string;
  project_dir?: string;
  page_index: number;
  total_pages?: number;
  page: StudioPage;
  project?: StudioProject;
}

export interface LegacyBitmapRegionConfig {
  project_path: string;
  page_index: number;
  width: number;
  height: number;
  brush_size: number;
  clear?: boolean;
  erase?: boolean;
  strokes: [number, number][][];
  color?: string;
  opacity?: number;
  hardness?: number;
  png_data?: string;
  dirty_bbox?: [number, number, number, number];
  clip_mask_png?: string;
}

export interface LegacyWriteMaskFromPngConfig {
  project_path: string;
  page_index: number;
  png_data: string;
  layer_key: string;
  op: "replace" | "add" | "subtract";
}

export interface RegionalInpaintResult {
  page_index: number;
  inpaint_path: string;
  before_inpaint_path?: string | null;
  bbox: [number, number, number, number];
}

export interface LegacyEditorBackendApi {
  saveProjectJson(config: { project_path: string; project_json: unknown }): Promise<void>;
  loadEditorPage(config: { project_path: string; page_index: number }): Promise<EditorPagePayload>;
  createEditorTextLayer(config: {
    project_path: string;
    page_index: number;
    layout_bbox: [number, number, number, number];
  }): Promise<StudioTextLayer>;
  patchEditorTextLayer(config: {
    project_path: string;
    page_index: number;
    layer_id: string;
    patch: Record<string, unknown>;
  }): Promise<StudioTextLayer>;
  deleteEditorTextLayer(config: { project_path: string; page_index: number; layer_id: string }): Promise<void>;
  setEditorLayerVisibility(config: {
    project_path: string;
    page_index: number;
    layer_kind: "image" | "text";
    layer_key?: string | null;
    layer_id?: string | null;
    visible: boolean;
  }): Promise<void>;
  snapshotImageLayer?(config: {
    project_path: string;
    page_index: number;
    layer_key: string;
    source_path?: string | null;
  }): Promise<string | null>;
  updateMaskRegion(config: LegacyBitmapRegionConfig): Promise<string>;
  updateBrushRegion(config: LegacyBitmapRegionConfig): Promise<string>;
  updateRecoveryRegion(config: LegacyBitmapRegionConfig): Promise<string>;
  updateReinpaintRegion(config: LegacyBitmapRegionConfig): Promise<string>;
  writeMaskFromPng(config: LegacyWriteMaskFromPngConfig): Promise<string>;
  writeHealingMask(config: {
    project_path: string;
    page_index: number;
    png_data: string;
    bbox?: [number, number, number, number];
  }): Promise<string>;
  healInpaintRegion(config: {
    project_path: string;
    page_index: number;
    bbox: [number, number, number, number];
    mask_path: string;
  }): Promise<RegionalInpaintResult>;
    renderPreviewPage(args: {
    project_path: string;
    page_index: number;
    page: StudioPage;
    fingerprint: string;
  }): Promise<{ output_path: string; renderer_backend: string | null; path?: string; preview_path?: string | null }>;
  runPageActionWithOptionalMask(config: {
    project_path: string;
    page_index: number;
    action: PageActionName;
    bbox?: [number, number, number, number] | null;
    mask_path?: string | null;
  }): Promise<{ action: PageActionName; changed_assets: PageActionChangedAsset[]; message?: string }>;
  runProcessRegion(config: {
    project_path: string;
    page_index: number;
    bbox: [number, number, number, number];
    mask_path?: string | null;
  }): Promise<{
    page_index: number;
    overlay: {
      id: string;
      page_index: number;
      bbox: [number, number, number, number];
      crop_path: string;
      text_layer_ids: string[];
      visible: boolean;
      locked: boolean;
      order: number;
    };
    changed_assets: PageActionChangedAsset[];
    changed_layers: string[];
    message: string;
  }>;
  retypesetPage(config: { project_path: string; page_index: number }): Promise<string>;
  detectPage(config: { project_path: string; page_index: number }): Promise<string>;
  detectBoxesPage(config: { project_path: string; page_index: number }): Promise<string>;
  preloadEditorVisionPage(config: { project_path: string; page_index: number }): Promise<string>;
  ocrPage(config: { project_path: string; page_index: number }): Promise<string>;
  translatePage(config: { project_path: string; page_index: number }): Promise<string>;
  reinpaintPage(config: {
    project_path: string;
    page_index: number;
    bbox?: [number, number, number, number];
    mask_path?: string | null;
  }): Promise<string>;
  processBlock(config: {
    project_path: string;
    page_index: number;
    block_id: string;
    mode: "ocr" | "translate" | "inpaint";
  }): Promise<string>;
}

function requirePngData(config: LegacyBitmapRegionConfig, layer: string) {
  if (!config.png_data) {
    throw new Error(`png_data e obrigatorio para atualizar camada ${layer} no Studio`);
  }
  return config.png_data;
}

const editorBaseProjects = new WeakMap<StudioEditorBackend, Map<string, StudioProject>>();

function cloneValue<T>(value: T): T {
  if (value === undefined) return value;
  return JSON.parse(JSON.stringify(value)) as T;
}

function valuesEqual(left: unknown, right: unknown) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function projectBasesFor(backend: StudioEditorBackend) {
  let projects = editorBaseProjects.get(backend);
  if (!projects) {
    projects = new Map();
    editorBaseProjects.set(backend, projects);
  }
  return projects;
}

function applyChangedRecordFields(
  base: Record<string, unknown>,
  incoming: Record<string, unknown>,
  latest: Record<string, unknown>,
  context: string,
) {
  for (const key of new Set([...Object.keys(base), ...Object.keys(incoming)])) {
    const baseHas = Object.prototype.hasOwnProperty.call(base, key);
    const incomingHas = Object.prototype.hasOwnProperty.call(incoming, key);
    const latestHas = Object.prototype.hasOwnProperty.call(latest, key);
    const baseValue = base[key];
    const incomingValue = incoming[key];
    const latestValue = latest[key];
    if (baseHas === incomingHas && valuesEqual(baseValue, incomingValue)) continue;
    if (latestHas === incomingHas && valuesEqual(latestValue, incomingValue)) continue;
    if (latestHas !== baseHas || !valuesEqual(latestValue, baseValue)) {
      throw new Error(`${context} mudou no campo ${key}; salve ou recarregue antes de repetir a alteracao estrutural`);
    }
    if (incomingHas) latest[key] = cloneValue(incomingValue);
    else delete latest[key];
  }
}

function applyLegacyEditorProjectDelta(base: StudioProject, incoming: StudioProject, latest: StudioProject) {
  if (base.paginas.length !== incoming.paginas.length || latest.paginas.length !== base.paginas.length) {
    throw new Error("A quantidade de paginas mudou durante a edicao estrutural");
  }
  incoming.paginas.forEach((incomingPage, pageIndex) => {
    const basePage = base.paginas[pageIndex];
    const latestPage = latest.paginas[pageIndex];
    if (!basePage || !latestPage) throw new Error(`Pagina ${pageIndex + 1} indisponivel para salvar estrutura`);
    const baseById = new Map(basePage.text_layers.map((layer) => [layer.id, layer]));
    const incomingById = new Map(incomingPage.text_layers.map((layer) => [layer.id, layer]));
    const latestById = new Map(latestPage.text_layers.map((layer) => [layer.id, layer]));
    const baseOrder = basePage.text_layers.map((layer) => layer.id);
    const incomingOrder = incomingPage.text_layers.map((layer) => layer.id);
    const latestOrder = latestPage.text_layers.map((layer) => layer.id);
    const structureChanged = !valuesEqual(baseOrder, incomingOrder);
    if (structureChanged && !valuesEqual(latestOrder, baseOrder) && !valuesEqual(latestOrder, incomingOrder)) {
      throw new Error(`A estrutura da pagina ${pageIndex + 1} mudou em outra operacao`);
    }

    for (const [layerId, baseLayer] of baseById) {
      const incomingLayer = incomingById.get(layerId);
      const latestLayer = latestById.get(layerId);
      if (!incomingLayer) {
        if (latestLayer && !valuesEqual(latestLayer, baseLayer)) {
          throw new Error(`A camada ${layerId} mudou antes de ser removida`);
        }
        continue;
      }
      if (!latestLayer) throw new Error(`A camada ${layerId} foi removida em outra operacao`);
      applyChangedRecordFields(
        baseLayer as unknown as Record<string, unknown>,
        incomingLayer as unknown as Record<string, unknown>,
        latestLayer as unknown as Record<string, unknown>,
        `A camada ${layerId}`,
      );
    }

    for (const [layerId, incomingLayer] of incomingById) {
      if (baseById.has(layerId)) continue;
      const latestLayer = latestById.get(layerId);
      if (latestLayer && !valuesEqual(latestLayer, incomingLayer)) {
        throw new Error(`A nova camada ${layerId} conflita com outra operacao`);
      }
    }

    if (structureChanged && !valuesEqual(latestOrder, incomingOrder)) {
      const nextLayers = incomingOrder.map((layerId) => {
        const existing = latestById.get(layerId);
        if (existing) return existing;
        const created = incomingById.get(layerId);
        if (!created) throw new Error(`Camada estrutural ausente: ${layerId}`);
        return cloneValue(created);
      });
      latestPage.text_layers = nextLayers;
    }
    latestPage.textos = latestPage.text_layers;
  });

  if (!valuesEqual(base.font_assets, incoming.font_assets)) {
    if (!valuesEqual(latest.font_assets, incoming.font_assets)) {
      if (!valuesEqual(latest.font_assets, base.font_assets)) {
        throw new Error("Os recursos de fonte mudaram em outra operacao");
      }
      latest.font_assets = cloneValue(incoming.font_assets);
    }
  }
}

function toBitmapConfig(
  config: LegacyBitmapRegionConfig,
  layer_key: BitmapRegionConfig["layer_key"],
): BitmapRegionConfig {
  return {
    project_path: config.project_path,
    page_index: config.page_index,
    layer_key,
    width: config.width,
    height: config.height,
    png_data: requirePngData(config, layer_key),
    dirty_bbox: config.dirty_bbox ?? null,
  };
}

function sanitizeMaskLeakedIntoInpaint(page: StudioPage): StudioPage {
  const maskPath = page.image_layers.mask?.path;
  const inpaintPath = page.image_layers.inpaint?.path;
  const fallbackPath = page.image_layers.base?.path ?? page.arquivo_original ?? page.arquivo_traduzido ?? null;
  if (!maskPath || !inpaintPath || inpaintPath !== maskPath || !fallbackPath) return page;
  return {
    ...page,
    image_layers: {
      ...page.image_layers,
      inpaint: {
        ...(page.image_layers.inpaint ?? {}),
        key: "inpaint",
        path: fallbackPath,
        visible: true,
        locked: page.image_layers.inpaint?.locked === true,
      },
    },
  };
}

async function ensureRecoveryMetadata(backend: StudioEditorBackend, projectPath: string, pageIndex: number) {
  await backend.mutateProject({
    project_path: projectPath,
    mutate: (project) => {
      const page = project.paginas[pageIndex];
      if (!page) return;
      const existing = page.image_layers.recovery;
      page.image_layers.recovery = {
        key: "recovery",
        path: existing?.path ?? null,
        visible: false,
        locked: existing?.locked === true,
        opacity: existing?.opacity,
        order: existing?.order,
        technical: true,
      };
    },
  });
}

export function createLegacyEditorBackendAdapter(backend: StudioEditorBackend): LegacyEditorBackendApi {
  const baseProjects = projectBasesFor(backend);
  return {
    saveProjectJson: async ({ project_path, project_json }) => {
      const incoming = project_json as StudioProject;
      const base = baseProjects.get(project_path);
      if (!base) throw new Error("Base do editor indisponivel; recarregue a pagina antes de salvar alteracoes estruturais");
      const { project } = await backend.mutateProject({
        project_path,
        mutate: (latest) => {
          applyLegacyEditorProjectDelta(base, incoming, latest);
        },
      });
      baseProjects.set(project_path, cloneValue(project));
    },
    loadEditorPage: async (config) => {
      const payload = await backend.loadEditorPage(config);
      baseProjects.set(config.project_path, cloneValue(payload.project));
      return {
        ...payload,
        page: sanitizeMaskLeakedIntoInpaint(payload.page),
      };
    },
    createEditorTextLayer: async (config) => backend.createEditorTextLayer(config),
    patchEditorTextLayer: async (config) => backend.patchEditorTextLayer(config),
    deleteEditorTextLayer: async (config) => backend.deleteEditorTextLayer(config),
    setEditorLayerVisibility: async (config) => {
      await backend.setEditorLayerVisibility({
        project_path: config.project_path,
        page_index: config.page_index,
        layer_kind: config.layer_kind,
        layer_key: config.layer_key as never,
        layer_id: config.layer_id,
        visible: config.visible,
      });
    },
    snapshotImageLayer: async (config) => {
      if (config.source_path) return config.source_path;
      const loaded = await backend.loadEditorPage({ project_path: config.project_path, page_index: config.page_index });
      const layerKey = config.layer_key as keyof typeof loaded.page.image_layers;
      return loaded.page.image_layers[layerKey]?.path ?? null;
    },
    updateMaskRegion: async (config) => backend.updateBitmapLayer(toBitmapConfig(config, "mask")),
    updateBrushRegion: async (config) => backend.updateBitmapLayer(toBitmapConfig(config, "brush")),
    updateRecoveryRegion: async (config) => {
      const path = await backend.updateBitmapLayer(toBitmapConfig(config, "inpaint"));
      await ensureRecoveryMetadata(backend, config.project_path, config.page_index);
      return path;
    },
    updateReinpaintRegion: async (config) => backend.updateBitmapLayer(toBitmapConfig(config, "inpaint")),
    writeMaskFromPng: async (config) =>
      backend.updateBitmapLayer({
        project_path: config.project_path,
        page_index: config.page_index,
        layer_key: config.layer_key === "brush" ? "brush" : "mask",
        width: 1,
        height: 1,
        png_data: config.png_data,
        dirty_bbox: null,
      }),
    writeHealingMask: async (config) =>
      backend.updateBitmapLayer({
        project_path: config.project_path,
        page_index: config.page_index,
        layer_key: "mask",
        width: Math.max(1, config.bbox?.[2] ?? 1),
        height: Math.max(1, config.bbox?.[3] ?? 1),
        png_data: config.png_data,
        dirty_bbox: config.bbox ?? null,
      }),
    healInpaintRegion: async (config) => {
      const { result } = await backend.mutateProject({
        project_path: config.project_path,
        mutate: (project) => {
          const page = project.paginas[config.page_index];
          if (!page) throw new Error(`Pagina nao encontrada: ${config.page_index}`);
          const before = page.image_layers.inpaint?.path ?? null;
          const inpaintPath = before ?? page.image_layers.base?.path ?? page.arquivo_original ?? page.arquivo_traduzido ?? "";
          if (!inpaintPath) throw new Error("Imagem base indisponivel para preservar camada inpaint no Studio");
          page.image_layers.inpaint = {
            ...(page.image_layers.inpaint ?? {}),
            key: "inpaint",
            path: inpaintPath,
            visible: true,
            locked: page.image_layers.inpaint?.locked === true,
          };
          return { before, inpaintPath };
        },
      });
      return {
        page_index: config.page_index,
        inpaint_path: result.inpaintPath,
        before_inpaint_path: result.before,
        bbox: config.bbox,
      };
    },
    renderPreviewPage: async ({ project_path, page_index, page }) => {
      const path = await backend.updateBitmapLayer({
        project_path,
        page_index,
        layer_key: "rendered",
        width: 1,
        height: 1,
        png_data: page.image_layers.rendered?.path ?? page.arquivo_traduzido ?? "data:image/png;base64,",
        dirty_bbox: null,
      });
      return { output_path: path, path, preview_path: null, renderer_backend: "studio-local" };
    },
    runPageActionWithOptionalMask: async (config) => {
      if ((config.action === "detect" || config.action === "detect_boxes") && backend.studioLiteDetectPage) {
        const result = await backend.studioLiteDetectPage({
          project_path: config.project_path,
          page_index: config.page_index,
          boxes_only: config.action === "detect_boxes",
        });
        let maskPath = result.mask_path ?? null;
        if (config.action === "detect" && !maskPath && result.detections.length > 0 && backend.studioLiteBuildMask) {
          maskPath = await backend.studioLiteBuildMask({
            project_path: config.project_path,
            page_index: config.page_index,
            detections: result.detections,
          });
        }
        return {
          action: config.action,
          changed_assets: maskPath && config.action === "detect" ? ["mask", "project_json"] : ["project_json"],
          message: result.message ?? `Studio Lite detectou ${result.detections.length} regioes`,
        };
      }
      if (config.action === "inpaint" && backend.studioLiteInpaintRegion && (config.bbox || config.mask_path)) {
        await backend.studioLiteInpaintRegion({
          project_path: config.project_path,
          page_index: config.page_index,
          bbox: config.bbox ?? null,
          mask_path: config.mask_path ?? null,
        });
        return {
          action: config.action,
          changed_assets: ["inpaint", "project_json"],
          message: "Inpaint Studio Lite aplicado",
        };
      }
      return {
        action: config.action,
        changed_assets: ["project_json"],
        message: "Acao de pipeline ainda nao conectada no Studio local",
      };
    },
    runProcessRegion: async (config) => ({
      page_index: config.page_index,
      overlay: {
        id: `studio-process-${config.page_index + 1}`,
        page_index: config.page_index,
        bbox: config.bbox,
        crop_path: config.mask_path ?? "",
        text_layer_ids: [],
        visible: true,
        locked: false,
        order: 0,
      },
      changed_assets: ["project_json"],
      changed_layers: [],
      message: "Processamento regional ainda nao conectado no Studio local",
    }),
    retypesetPage: async (config) => {
      const page = await backend.loadEditorPage(config);
      return page.page.image_layers.rendered?.path ?? page.page.arquivo_traduzido ?? "";
    },
    detectPage: async (config) => {
      if (!backend.studioLiteDetectPage) return "Deteccao ainda nao conectada no Studio local";
      const result = await backend.studioLiteDetectPage({ ...config, boxes_only: false });
      return result.message ?? `Studio Lite detectou ${result.detections.length} regioes`;
    },
    detectBoxesPage: async (config) => {
      if (!backend.studioLiteDetectPage) return "Deteccao de caixas ainda nao conectada no Studio local";
      const result = await backend.studioLiteDetectPage({ ...config, boxes_only: true });
      return result.message ?? `Studio Lite encontrou ${result.detections.length} caixas`;
    },
    preloadEditorVisionPage: async () => "Preload visual desativado no Studio local",
    ocrPage: async () => "OCR ainda nao conectado no Studio local",
    translatePage: async () => "Traducao ainda nao conectada no Studio local",
    reinpaintPage: async (config) => {
      if (backend.studioLiteInpaintRegion && (config.bbox || config.mask_path)) {
        const result = await backend.studioLiteInpaintRegion(config);
        return result.inpaint_path;
      }
      const page = await backend.loadEditorPage(config);
      return page.page.image_layers.inpaint?.path ?? page.page.image_layers.base?.path ?? page.page.arquivo_original ?? "";
    },
    processBlock: async () => "Processamento de bloco ainda nao conectado no Studio local",
  };
}
