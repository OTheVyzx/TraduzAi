import { invoke } from "@tauri-apps/api/core";
import { importStudioProject, toTraduzAiV2Compat } from "../project/adapters";
import type { ImageLayerKey, StudioProject, StudioTextLayer } from "../project/studioProject";
import {
  BitmapLayerKey,
  BitmapRegionConfig,
  DeleteGeneratedAssetsConfig,
  EditorPagePayload,
  GeneratedAssetConfig,
  StudioEditorBackend,
  StudioLiteDetectResult,
  StudioLiteDetection,
  StudioLiteInpaintResult,
  StudioLiteModelStatus,
  runSerializedStudioProjectMutation,
} from "./editorBackend";
import { MemoryStudioEditorBackend } from "./memoryBackend";
import type { FluxGenerateConfig, FluxGenerateResult, FluxProviderStatus } from "../ai/fluxContract";
import { parseStudioRecoverySnapshot, type StudioRecoverySnapshot } from "../autosave/recovery";

function isMemoryPath(projectPath: string) {
  return projectPath.startsWith("memory://");
}

function isTauriRuntime() {
  return typeof window !== "undefined" && ("__TAURI_INTERNALS__" in window || "__TAURI__" in window);
}

function normalizePath(path: string) {
  return path.replace(/\\/g, "/");
}

function isDirectOrAbsolutePath(path?: string | null) {
  if (!path) return true;
  return /^(data|blob|asset|file):/i.test(path) || /^https?:\/\//i.test(path) || /^[A-Za-z]:[\\/]/.test(path) || path.startsWith("/");
}

function resolveProjectAssetPath(projectPath: string, assetPath?: string | null) {
  if (!assetPath || isDirectOrAbsolutePath(assetPath)) return assetPath ?? null;
  const normalizedProject = normalizePath(projectPath);
  const projectDir = normalizedProject.toLowerCase().endsWith(".json")
    ? normalizedProject.slice(0, normalizedProject.lastIndexOf("/"))
    : normalizedProject;
  return `${projectDir}/${normalizePath(assetPath)}`;
}

function resolvePageAssetPaths(projectPath: string, page: ReturnType<typeof pageAt>) {
  const next = cloneProject({ paginas: [page] } as unknown as StudioProject).paginas[0];
  next.arquivo_original = resolveProjectAssetPath(projectPath, next.arquivo_original) ?? next.arquivo_original;
  next.arquivo_traduzido = resolveProjectAssetPath(projectPath, next.arquivo_traduzido) ?? next.arquivo_traduzido;
  if (typeof next.arquivo_final === "string") {
    next.arquivo_final = resolveProjectAssetPath(projectPath, next.arquivo_final) ?? next.arquivo_final;
  }
  if (typeof next.inpaint_path === "string") {
    next.inpaint_path = resolveProjectAssetPath(projectPath, next.inpaint_path) ?? next.inpaint_path;
  }
  for (const [key, layer] of Object.entries(next.image_layers ?? {})) {
    if (!layer) continue;
    next.image_layers[key as ImageLayerKey] = {
      ...layer,
      path: resolveProjectAssetPath(projectPath, layer.path) ?? layer.path,
    };
  }
  return next;
}

function cloneProject(project: StudioProject): StudioProject {
  return JSON.parse(JSON.stringify(project)) as StudioProject;
}

function normalizeProject(project: StudioProject) {
  return importStudioProject(toTraduzAiV2Compat(project)).project;
}

function pageAt(project: StudioProject, pageIndex: number) {
  const page = project.paginas[pageIndex];
  if (!page) throw new Error(`Pagina ${pageIndex + 1} nao encontrada`);
  return page;
}

function syncTextAliases(page: ReturnType<typeof pageAt>) {
  page.textos = page.text_layers;
}

function createTextLayer(index: number, bbox: [number, number, number, number]): StudioTextLayer {
  return {
    id: `studio-text-${crypto.randomUUID()}`,
    kind: "text",
    original: "",
    translated: "",
    traduzido: "",
    bbox,
    layout_bbox: bbox,
    style: {},
    estilo: {},
    visible: true,
    locked: false,
    order: index,
  };
}

export class TauriStudioEditorBackend implements StudioEditorBackend {
  async loadProject(config: { project_path: string }): Promise<StudioProject> {
    const raw = await invoke<unknown>("studio_load_project", { config });
    return importStudioProject(raw).project;
  }

  async saveProjectJson(config: { project_path: string; project_json: StudioProject }): Promise<void> {
    await runSerializedStudioProjectMutation(config.project_path, () => this.saveProjectJsonDirect(config));
  }

  async mutateProject<T>(config: {
    project_path: string;
    mutate: (project: StudioProject) => T | Promise<T>;
  }): Promise<{ project: StudioProject; result: T }> {
    return runSerializedStudioProjectMutation(config.project_path, async () => {
      const draft = await this.loadProject({ project_path: config.project_path });
      const result = await config.mutate(draft);
      await this.saveProjectJsonDirect({ project_path: config.project_path, project_json: draft });
      const project = normalizeProject(draft);
      return { project, result };
    });
  }

  private async saveProjectJsonDirect(config: { project_path: string; project_json: StudioProject }): Promise<void> {
    await invoke("studio_save_project", {
      config: {
        project_path: config.project_path,
        project_json: toTraduzAiV2Compat(config.project_json),
      },
    });
  }

  async saveRecoverySnapshot(config: { project_path: string; snapshot: StudioRecoverySnapshot }): Promise<void> {
    const snapshot = parseStudioRecoverySnapshot(config.snapshot, config.project_path);
    if (!snapshot) throw new Error("Snapshot de recuperacao pertence a outro projeto ou esta corrompido");
    await invoke("studio_save_recovery_snapshot", { config: { ...config, snapshot } });
  }

  async loadRecoverySnapshot(config: { project_path: string }): Promise<StudioRecoverySnapshot | null> {
    const raw = await invoke<unknown>("studio_load_recovery_snapshot", { config });
    return parseStudioRecoverySnapshot(raw, config.project_path);
  }

  async clearRecoverySnapshot(config: { project_path: string }): Promise<void> {
    await invoke("studio_clear_recovery_snapshot", { config });
  }

  async loadEditorPage(config: { project_path: string; page_index: number }): Promise<EditorPagePayload> {
    const project = await this.loadProject({ project_path: config.project_path });
    const page = resolvePageAssetPaths(config.project_path, pageAt(project, config.page_index));
    return {
      project_file: config.project_path.endsWith("project.json") ? config.project_path : `${config.project_path}/project.json`,
      project_dir: config.project_path,
      page_index: config.page_index,
      total_pages: project.paginas.length,
      page,
      project,
    };
  }

  async createEditorTextLayer(config: {
    project_path: string;
    page_index: number;
    layout_bbox: [number, number, number, number];
  }): Promise<StudioTextLayer> {
    const { result } = await this.mutateProject({
      project_path: config.project_path,
      mutate: (project) => {
        const page = pageAt(project, config.page_index);
        const layer = createTextLayer(page.text_layers.length, config.layout_bbox);
        page.text_layers.push(layer);
        syncTextAliases(page);
        return JSON.parse(JSON.stringify(layer)) as StudioTextLayer;
      },
    });
    return result;
  }

  async patchEditorTextLayer(config: {
    project_path: string;
    page_index: number;
    layer_id: string;
    patch: Record<string, unknown>;
  }): Promise<StudioTextLayer> {
    const { result } = await this.mutateProject({
      project_path: config.project_path,
      mutate: (project) => {
        const page = pageAt(project, config.page_index);
        const index = page.text_layers.findIndex((layer) => layer.id === config.layer_id);
        if (index < 0) throw new Error(`Camada nao encontrada: ${config.layer_id}`);
        const next = { ...page.text_layers[index], ...config.patch } as StudioTextLayer;
        const translated = next.translated ?? next.traduzido ?? "";
        next.translated = translated;
        next.traduzido = translated;
        const style = Object.keys(next.style ?? {}).length > 0 ? next.style : next.estilo;
        next.style = style ?? {};
        next.estilo = style ?? {};
        page.text_layers[index] = next;
        syncTextAliases(page);
        return JSON.parse(JSON.stringify(next)) as StudioTextLayer;
      },
    });
    return result;
  }

  async deleteEditorTextLayer(config: { project_path: string; page_index: number; layer_id: string }): Promise<void> {
    await this.mutateProject({
      project_path: config.project_path,
      mutate: (project) => {
        const page = pageAt(project, config.page_index);
        const next = page.text_layers.filter((layer) => layer.id !== config.layer_id);
        if (next.length === page.text_layers.length) throw new Error(`Camada nao encontrada: ${config.layer_id}`);
        page.text_layers = next.map((layer, index) => ({ ...layer, order: index }));
        syncTextAliases(page);
      },
    });
  }

  async setEditorLayerVisibility(config: {
    project_path: string;
    page_index: number;
    layer_kind: "image" | "text";
    layer_key?: ImageLayerKey | null;
    layer_id?: string | null;
    visible: boolean;
  }): Promise<void> {
    await this.mutateProject({
      project_path: config.project_path,
      mutate: (project) => {
        const page = pageAt(project, config.page_index);
        if (config.layer_kind === "image") {
          const key = config.layer_key;
          if (!key) throw new Error("layer_key e obrigatorio para camada de imagem");
          const layer = page.image_layers[key] ?? { key, path: null, locked: false, visible: true };
          page.image_layers[key] = { ...layer, visible: config.visible };
        } else {
          const layer = page.text_layers.find((item) => item.id === config.layer_id);
          if (!layer) throw new Error(`Camada nao encontrada: ${config.layer_id}`);
          layer.visible = config.visible;
          syncTextAliases(page);
        }
      },
    });
  }

  async updateBitmapLayer(config: BitmapRegionConfig): Promise<string> {
    const path = await invoke<string>("studio_write_bitmap_layer", { config });
    await this.mutateProject({
      project_path: config.project_path,
      mutate: (project) => {
        const page = pageAt(project, config.page_index);
        const key: BitmapLayerKey = config.layer_key;
        page.image_layers[key] = {
          key,
          path,
          visible: true,
          locked: page.image_layers[key]?.locked === true,
        };
        if (key === "rendered") page.arquivo_traduzido = path;
      },
    });
    return resolveProjectAssetPath(config.project_path, path) ?? path;
  }

  async saveGeneratedAsset(config: GeneratedAssetConfig): Promise<string> {
    return invoke<string>("studio_write_generated_asset", { config });
  }

  async deleteGeneratedAssets(config: DeleteGeneratedAssetsConfig): Promise<void> {
    await invoke("studio_delete_generated_assets", { config });
  }

  async fluxProviderStatus(): Promise<FluxProviderStatus> {
    return invoke<FluxProviderStatus>("studio_flux_status");
  }

  async generateFluxFill(config: FluxGenerateConfig): Promise<FluxGenerateResult> {
    return invoke<FluxGenerateResult>("studio_flux_generate", { config });
  }

  async cancelFluxFill(jobId: string): Promise<boolean> {
    return invoke<boolean>("studio_flux_cancel", { config: { job_id: jobId } });
  }

  async studioLiteModelStatus(): Promise<StudioLiteModelStatus> {
    const result = await invoke<{
      model?: StudioLiteModelStatus;
      status?: string;
      message?: string | null;
      [key: string]: unknown;
    }>(
      "studio_lite_model_status",
      { config: {} },
    );
    if (result.model) return result.model;
    return { ...result, status: result.status ?? "missing" };
  }

  async studioLiteDetectPage(config: {
    project_path: string;
    page_index: number;
    boxes_only?: boolean;
  }): Promise<StudioLiteDetectResult> {
    const image_path = await this.studioLiteImagePath(config.project_path, config.page_index, false);
    const result = await invoke<StudioLiteDetectResult>("studio_lite_detect_page", { config: { ...config, image_path } });
    if (!config.boxes_only && result.mask_path) {
      await this.setImageLayerPath(config.project_path, config.page_index, "mask", result.mask_path);
    }
    return {
      ...result,
      mask_path: resolveProjectAssetPath(config.project_path, result.mask_path) ?? result.mask_path ?? null,
    };
  }

  async studioLiteBuildMask(config: {
    project_path: string;
    page_index: number;
    detections?: StudioLiteDetection[];
    bboxes?: [number, number, number, number][];
    padding?: number;
  }): Promise<string> {
    const image_path = await this.studioLiteImagePath(config.project_path, config.page_index, false);
    const path = await invoke<string>("studio_lite_build_mask", { config: { ...config, image_path } });
    await this.setImageLayerPath(config.project_path, config.page_index, "mask", path);
    return resolveProjectAssetPath(config.project_path, path) ?? path;
  }

  async studioLiteInpaintRegion(config: {
    project_path: string;
    page_index: number;
    bbox?: [number, number, number, number] | null;
    mask_path?: string | null;
  }): Promise<StudioLiteInpaintResult> {
    const image_path = await this.studioLiteImagePath(config.project_path, config.page_index, true);
    const mask_path = await this.studioLiteMaskPath(config.project_path, config.page_index, config.mask_path);
    const raw = await invoke<StudioLiteInpaintResult | string>("studio_lite_inpaint_region", {
      config: { ...config, image_path, mask_path },
    });
    const result: StudioLiteInpaintResult = typeof raw === "string" ? { inpaint_path: raw, bbox: config.bbox ?? null } : raw;
    await this.setImageLayerPath(config.project_path, config.page_index, "inpaint", result.inpaint_path);
    return {
      ...result,
      inpaint_path: resolveProjectAssetPath(config.project_path, result.inpaint_path) ?? result.inpaint_path,
      before_inpaint_path:
        resolveProjectAssetPath(config.project_path, result.before_inpaint_path) ?? result.before_inpaint_path ?? null,
    };
  }

  private async setImageLayerPath(
    projectPath: string,
    pageIndex: number,
    key: BitmapLayerKey,
    path: string | null | undefined,
  ) {
    if (!path) return;
    await this.mutateProject({
      project_path: projectPath,
      mutate: (project) => {
        const page = pageAt(project, pageIndex);
        page.image_layers[key] = {
          ...(page.image_layers[key] ?? {}),
          key,
          path,
          visible: true,
          locked: page.image_layers[key]?.locked === true,
        };
        if (key === "rendered") page.arquivo_traduzido = path;
      },
    });
  }

  private async studioLiteImagePath(projectPath: string, pageIndex: number, preferInpaint: boolean) {
    const project = await this.loadProject({ project_path: projectPath });
    const page = pageAt(project, pageIndex);
    const raw =
      (preferInpaint ? page.image_layers.inpaint?.path : null) ??
      page.image_layers.base?.path ??
      page.arquivo_original ??
      page.arquivo_traduzido ??
      page.image_layers.inpaint?.path ??
      "";
    if (!raw) throw new Error("Imagem base indisponivel para Studio Lite");
    return resolveProjectAssetPath(projectPath, raw) ?? raw;
  }

  private async studioLiteMaskPath(projectPath: string, pageIndex: number, explicitPath?: string | null) {
    if (explicitPath) return resolveProjectAssetPath(projectPath, explicitPath) ?? explicitPath;
    const project = await this.loadProject({ project_path: projectPath });
    const page = pageAt(project, pageIndex);
    const raw = page.image_layers.mask?.path;
    if (!raw) throw new Error("Mascara indisponivel para Studio Lite");
    return resolveProjectAssetPath(projectPath, raw) ?? raw;
  }
}

export class HybridStudioEditorBackend implements StudioEditorBackend {
  constructor(
    private readonly memoryBackend: MemoryStudioEditorBackend,
    private readonly tauriBackend: TauriStudioEditorBackend | null,
  ) {}

  putProject(projectPath: string, project: StudioProject) {
    this.memoryBackend.putProject(projectPath, project);
  }

  private backendFor(projectPath: string): StudioEditorBackend {
    if (!this.tauriBackend || isMemoryPath(projectPath)) return this.memoryBackend;
    return this.tauriBackend;
  }

  loadProject(config: { project_path: string }) {
    return this.backendFor(config.project_path).loadProject(config);
  }

  saveProjectJson(config: { project_path: string; project_json: StudioProject }) {
    return this.backendFor(config.project_path).saveProjectJson(config);
  }

  mutateProject<T>(config: {
    project_path: string;
    mutate: (project: StudioProject) => T | Promise<T>;
  }) {
    return this.backendFor(config.project_path).mutateProject(config);
  }

  saveRecoverySnapshot(config: { project_path: string; snapshot: StudioRecoverySnapshot }) {
    return this.backendFor(config.project_path).saveRecoverySnapshot(config);
  }

  loadRecoverySnapshot(config: { project_path: string }) {
    return this.backendFor(config.project_path).loadRecoverySnapshot(config);
  }

  clearRecoverySnapshot(config: { project_path: string }) {
    return this.backendFor(config.project_path).clearRecoverySnapshot(config);
  }

  loadEditorPage(config: { project_path: string; page_index: number }) {
    return this.backendFor(config.project_path).loadEditorPage(config);
  }

  createEditorTextLayer(config: {
    project_path: string;
    page_index: number;
    layout_bbox: [number, number, number, number];
  }) {
    return this.backendFor(config.project_path).createEditorTextLayer(config);
  }

  patchEditorTextLayer(config: {
    project_path: string;
    page_index: number;
    layer_id: string;
    patch: Record<string, unknown>;
  }) {
    return this.backendFor(config.project_path).patchEditorTextLayer(config);
  }

  deleteEditorTextLayer(config: { project_path: string; page_index: number; layer_id: string }) {
    return this.backendFor(config.project_path).deleteEditorTextLayer(config);
  }

  setEditorLayerVisibility(config: {
    project_path: string;
    page_index: number;
    layer_kind: "image" | "text";
    layer_key?: ImageLayerKey | null;
    layer_id?: string | null;
    visible: boolean;
  }) {
    return this.backendFor(config.project_path).setEditorLayerVisibility(config);
  }

  updateBitmapLayer(config: BitmapRegionConfig) {
    return this.backendFor(config.project_path).updateBitmapLayer(config);
  }

  saveGeneratedAsset(config: GeneratedAssetConfig) {
    return this.backendFor(config.project_path).saveGeneratedAsset(config);
  }

  deleteGeneratedAssets(config: DeleteGeneratedAssetsConfig) {
    return this.backendFor(config.project_path).deleteGeneratedAssets(config);
  }

  fluxProviderStatus() {
    return this.tauriBackend?.fluxProviderStatus() ?? this.memoryBackend.fluxProviderStatus();
  }

  generateFluxFill(config: FluxGenerateConfig) {
    if (!this.tauriBackend) return this.memoryBackend.generateFluxFill(config);
    return this.tauriBackend.generateFluxFill(config);
  }

  cancelFluxFill(jobId: string) {
    return this.tauriBackend?.cancelFluxFill(jobId) ?? this.memoryBackend.cancelFluxFill(jobId);
  }

  studioLiteModelStatus() {
    return this.tauriBackend?.studioLiteModelStatus?.() ?? Promise.resolve({ status: "missing", message: "Tauri indisponivel" });
  }

  studioLiteDetectPage(config: { project_path: string; page_index: number; boxes_only?: boolean }) {
    return this.backendFor(config.project_path).studioLiteDetectPage?.(config) ?? Promise.resolve({ detections: [] });
  }

  studioLiteBuildMask(config: {
    project_path: string;
    page_index: number;
    detections?: StudioLiteDetection[];
    bboxes?: [number, number, number, number][];
    padding?: number;
  }) {
    const backend = this.backendFor(config.project_path);
    if (!backend.studioLiteBuildMask) throw new Error("Studio Lite indisponivel neste backend");
    return backend.studioLiteBuildMask(config);
  }

  studioLiteInpaintRegion(config: {
    project_path: string;
    page_index: number;
    bbox?: [number, number, number, number] | null;
    mask_path?: string | null;
  }) {
    const backend = this.backendFor(config.project_path);
    if (!backend.studioLiteInpaintRegion) throw new Error("Studio Lite indisponivel neste backend");
    return backend.studioLiteInpaintRegion(config);
  }
}

export function createDefaultStudioBackend() {
  const memory = new MemoryStudioEditorBackend();
  const tauri = isTauriRuntime() ? new TauriStudioEditorBackend() : null;
  return new HybridStudioEditorBackend(memory, tauri);
}
