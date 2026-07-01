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
  const project = await backend.loadProject({ project_path: projectPath });
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
  await backend.saveProjectJson({ project_path: projectPath, project_json: project });
}

export function createLegacyEditorBackendAdapter(backend: StudioEditorBackend): LegacyEditorBackendApi {
  return {
    saveProjectJson: async ({ project_path, project_json }) => {
      await backend.saveProjectJson({ project_path, project_json: project_json as StudioProject });
    },
    loadEditorPage: async (config) => {
      const payload = await backend.loadEditorPage(config);
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
      const project = await backend.loadProject({ project_path: config.project_path });
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
      await backend.saveProjectJson({ project_path: config.project_path, project_json: project });
      return {
        page_index: config.page_index,
        inpaint_path: inpaintPath,
        before_inpaint_path: before,
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
