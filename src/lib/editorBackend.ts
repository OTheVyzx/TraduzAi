import type { PageData, ProcessRegionOverlay, TextEntry } from "./stores/appStore";

export type PageActionName = "detect" | "ocr" | "translate" | "inpaint";
export type PageActionChangedAsset = "brush" | "mask" | "inpaint" | "rendered" | "preview" | "project_json";

export interface EditorPagePayload {
  project_file?: string;
  project_dir?: string;
  page_index: number;
  total_pages?: number;
  page: PageData;
  project?: unknown;
}

export interface PageActionResult {
  action: PageActionName;
  mode?: "global" | "regional";
  bbox?: [number, number, number, number] | null;
  changed_assets: PageActionChangedAsset[];
  changed_layers?: string[];
  message?: string;
}

export interface RegionalInpaintResult {
  page_index: number;
  inpaint_path: string;
  before_inpaint_path?: string | null;
  bbox: [number, number, number, number];
}

export interface ProcessRegionResult {
  page_index: number;
  overlay: ProcessRegionOverlay;
  changed_assets: PageActionChangedAsset[];
  changed_layers: string[];
  message: string;
}

export interface EditorBackendApi {
  saveProjectJson(config: { project_path: string; project_json: unknown }): Promise<void>;
  loadEditorPage(config: { project_path: string; page_index: number }): Promise<EditorPagePayload>;
  createEditorTextLayer?(config: {
    project_path: string;
    page_index: number;
    layout_bbox: [number, number, number, number];
  }): Promise<TextEntry>;
  patchEditorTextLayer(config: {
    project_path: string;
    page_index: number;
    layer_id: string;
    patch: Record<string, unknown>;
  }): Promise<TextEntry>;
  deleteEditorTextLayer?(config: {
    project_path: string;
    page_index: number;
    layer_id: string;
  }): Promise<void>;
  setEditorLayerVisibility(config: {
    project_path: string;
    page_index: number;
    layer_kind: "image" | "text";
    layer_key?: string | null;
    layer_id?: string | null;
    visible: boolean;
  }): Promise<void>;
  updateMaskRegion(config: BitmapRegionConfig): Promise<string>;
  updateBrushRegion(config: BitmapRegionConfig): Promise<string>;
  updateRecoveryRegion(config: BitmapRegionConfig): Promise<string>;
  updateReinpaintRegion(config: BitmapRegionConfig): Promise<string>;
  writeMaskFromPng(config: {
    project_path: string;
    page_index: number;
    png_data: string;
    layer_key: string;
    op: "replace" | "add" | "subtract";
  }): Promise<string>;
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
    page: PageData;
    fingerprint: string;
  }): Promise<string>;
  runPageActionWithOptionalMask(config: {
    project_path: string;
    page_index: number;
    action: PageActionName;
    bbox?: [number, number, number, number] | null;
    mask_path?: string | null;
    engine_preset_id?: "manga" | "manhwa_manhua" | "default" | string;
    idioma_origem?: string;
    idioma_destino?: string;
  }): Promise<PageActionResult>;
  runProcessRegion(config: {
    project_path: string;
    page_index: number;
    bbox: [number, number, number, number];
    mask_path?: string | null;
    engine_preset_id?: "manga" | "manhwa_manhua" | "default" | string;
    idioma_origem?: string;
    idioma_destino?: string;
  }): Promise<ProcessRegionResult>;
  retypesetPage(args: { project_path: string; page_index: number }): Promise<string>;
  detectPage(args: { project_path: string; page_index: number; idioma_origem?: string }): Promise<string>;
  ocrPage(args: { project_path: string; page_index: number; idioma_origem?: string }): Promise<string>;
  translatePage(args: {
    project_path: string;
    page_index: number;
    idioma_origem?: string;
    idioma_destino?: string;
  }): Promise<string>;
  reinpaintPage(args: {
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
    idioma_origem?: string;
    idioma_destino?: string;
  }): Promise<string>;
}

export interface BitmapRegionConfig {
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

let configuredBackend: EditorBackendApi | null = null;
let defaultBackendPromise: Promise<EditorBackendApi> | null = null;

export function configureEditorBackend(backend: EditorBackendApi | null) {
  configuredBackend = backend;
}

export async function getEditorBackend(): Promise<EditorBackendApi> {
  if (configuredBackend) return configuredBackend;
  if (!defaultBackendPromise) {
    defaultBackendPromise = import("./editorBackends/tauriEditorBackend").then(
      (module) => module.tauriEditorBackend,
    );
  }
  return defaultBackendPromise;
}
