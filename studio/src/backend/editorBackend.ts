import type { StudioPage, StudioProject, StudioTextLayer, ImageLayerKey } from "../project/studioProject";
import type {
  FluxGenerateConfig,
  FluxGenerateResult,
  FluxProviderStatus,
} from "../ai/fluxContract";
import type { StudioRecoverySnapshot } from "../autosave/recovery";

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

export interface GeneratedAssetConfig {
  project_path: string;
  page_index: number;
  asset_id: string;
  png_data: string;
}

export interface DeleteGeneratedAssetsConfig {
  project_path: string;
  page_index: number;
  asset_ids: string[];
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
  mutateProject<T>(config: {
    project_path: string;
    mutate: (project: StudioProject) => T | Promise<T>;
  }): Promise<{ project: StudioProject; result: T }>;
  saveRecoverySnapshot(config: { project_path: string; snapshot: StudioRecoverySnapshot }): Promise<void>;
  loadRecoverySnapshot(config: { project_path: string }): Promise<StudioRecoverySnapshot | null>;
  clearRecoverySnapshot(config: { project_path: string }): Promise<void>;
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
  saveGeneratedAsset(config: GeneratedAssetConfig): Promise<string>;
  deleteGeneratedAssets(config: DeleteGeneratedAssetsConfig): Promise<void>;
  fluxProviderStatus(): Promise<FluxProviderStatus>;
  generateFluxFill(config: FluxGenerateConfig): Promise<FluxGenerateResult>;
  cancelFluxFill(jobId: string): Promise<boolean>;
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

const projectMutationTails = new Map<string, Promise<void>>();

function mutationKey(projectPath: string) {
  const normalized = projectPath.replace(/\\/g, "/");
  return /^[A-Za-z]:\//.test(normalized) ? normalized.toLocaleLowerCase() : normalized;
}

/**
 * Serializa todas as escritas do mesmo projeto dentro do processo do Studio.
 * A operacao recebe a vez mesmo quando a mutacao anterior falha.
 */
export async function runSerializedStudioProjectMutation<T>(
  projectPath: string,
  operation: () => Promise<T>,
): Promise<T> {
  const key = mutationKey(projectPath);
  const previous = projectMutationTails.get(key) ?? Promise.resolve();
  const running = previous.catch(() => undefined).then(operation);
  const tail = running.then(() => undefined, () => undefined);
  projectMutationTails.set(key, tail);
  try {
    return await running;
  } finally {
    if (projectMutationTails.get(key) === tail) projectMutationTails.delete(key);
  }
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
