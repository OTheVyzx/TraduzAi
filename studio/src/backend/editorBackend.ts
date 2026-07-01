import type { StudioPage, StudioProject, StudioTextLayer, ImageLayerKey } from "../project/studioProject";

export type BitmapLayerKey = Exclude<ImageLayerKey, "base" | "rendered"> | "rendered";

export interface EditorPagePayload {
  project_file?: string;
  project_dir?: string;
  page_index: number;
  total_pages: number;
  page: StudioPage;
  project: StudioProject;
}

export interface BitmapRegionConfig {
  project_path: string;
  page_index: number;
  layer_key: BitmapLayerKey;
  width: number;
  height: number;
  png_data: string;
  dirty_bbox?: [number, number, number, number] | null;
}

export interface StudioLiteModelStatus {
  status: "ready" | "missing" | "error" | string;
  path?: string | null;
  message?: string | null;
  [key: string]: unknown;
}

export interface StudioLiteDetection {
  bbox: [number, number, number, number];
  score?: number | null;
  label?: string | null;
}

export interface StudioLiteDetectResult {
  detections: StudioLiteDetection[];
  mask_path?: string | null;
  message?: string | null;
  model?: StudioLiteModelStatus | null;
}

export interface StudioLiteInpaintResult {
  inpaint_path: string;
  before_inpaint_path?: string | null;
  bbox?: [number, number, number, number] | null;
  message?: string | null;
}

export interface StudioEditorBackend {
  loadProject(config: { project_path: string }): Promise<StudioProject>;
  saveProjectJson(config: { project_path: string; project_json: StudioProject }): Promise<void>;
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
    layer_key?: ImageLayerKey | null;
    layer_id?: string | null;
    visible: boolean;
  }): Promise<void>;
  updateBitmapLayer(config: BitmapRegionConfig): Promise<string>;
  studioLiteModelStatus?(): Promise<StudioLiteModelStatus>;
  studioLiteDetectPage?(config: {
    project_path: string;
    page_index: number;
    boxes_only?: boolean;
  }): Promise<StudioLiteDetectResult>;
  studioLiteBuildMask?(config: {
    project_path: string;
    page_index: number;
    detections?: StudioLiteDetection[];
    bboxes?: [number, number, number, number][];
    padding?: number;
  }): Promise<string>;
  studioLiteInpaintRegion?(config: {
    project_path: string;
    page_index: number;
    bbox?: [number, number, number, number] | null;
    mask_path?: string | null;
  }): Promise<StudioLiteInpaintResult>;
}

let configuredBackend: StudioEditorBackend | null = null;

export function configureStudioEditorBackend(backend: StudioEditorBackend | null) {
  configuredBackend = backend;
}

export function getStudioEditorBackend() {
  if (!configuredBackend) {
    throw new Error("Studio editor backend is not configured");
  }
  return configuredBackend;
}
