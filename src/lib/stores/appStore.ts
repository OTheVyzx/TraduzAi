import { create } from "zustand";
import { sanitizeFavoriteWorks, upsertFavoriteWork } from "../favoriteWorks";
import type {
  AppProjectStatus,
  CompletionStatus,
  OutputReviewState,
  PipelineExportGateSummary,
} from "../pipelineCompletion";
import type { WorkContext } from "../workContext";

export type ImageLayerKey = "base" | "mask" | "inpaint" | "brush" | "recovery" | "rendered";

export interface TextLayerStyle {
  fonte: string;
  tamanho: number;
  cor: string;
  cor_gradiente: string[];
  contorno: string;
  contorno_px: number;
  glow: boolean;
  glow_cor: string;
  glow_px: number;
  sombra: boolean;
  sombra_cor: string;
  sombra_offset: [number, number];
  bold: boolean;
  italico: boolean;
  rotacao: number;
  curva?: boolean;
  curva_direcao?: "" | "arc_up" | "arc_down";
  curva_intensidade?: number;
  alinhamento: "left" | "center" | "right";
  force_upper?: boolean;
}

export type TextLayerStyleOrigin = "auto" | "editor" | "legacy" | "legacy_auto" | "source_detected";

export interface QaAction {
  flag_id: string;
  status: "ignored" | "resolved";
  ignored_reason?: string;
  ignored_at?: string;
}

export interface SfxMetadata {
  source_text?: string;
  adapted_text?: string;
  confidence?: number | null;
  translation_mode?: string;
  style_confidence?: number | null;
  inpaint_allowed?: boolean;
  review_required?: boolean;
  qa_flags?: string[];
  [key: string]: unknown;
}

export interface TextEntry {
  id: string;
  kind?: "text";
  content_class?: string;
  script?: string | null;
  route_action?: string | null;
  translation_route?: string | null;
  style_origin?: TextLayerStyleOrigin;
  style_confidence?: number | null;
  style_source?: string | null;
  style_evidence?: unknown;
  source_bbox?: [number, number, number, number];
  layout_bbox?: [number, number, number, number];
  render_bbox?: [number, number, number, number] | null;
  bbox: [number, number, number, number];
  tipo: "fala" | "narracao" | "sfx" | "pensamento";
  original: string;
  traduzido: string;
  translated?: string;
  confianca_ocr: number;
  ocr_confidence?: number;
  estilo: TextLayerStyle;
  style?: TextLayerStyle;
  visible?: boolean;
  locked?: boolean;
  order?: number;
  render_preview_path?: string | null;
  detector?: string | null;
  line_polygons?: unknown;
  source_direction?: string | null;
  rendered_direction?: string | null;
  source_language?: string | null;
  rotation_deg?: number;
  detected_font_size_px?: number | null;
  page_profile?: string | null;
  block_profile?: string | null;
  layout_profile?: string | null;
  balloon_bbox?: [number, number, number, number];
  balloon_subregions?: [number, number, number, number][];
  layout_group_size?: number;
  qa_flags?: string[];
  qa_actions?: QaAction[];
  sfx?: SfxMetadata;
}

export interface ImageLayer {
  key: ImageLayerKey;
  path: string | null;
  visible: boolean;
  locked: boolean;
  /** 0..1, padrão 1. Afeta renderização Konva e exportação. */
  opacity?: number;
  /** Posição na pilha (ascendente = mais abaixo). Menor = mais ao fundo. */
  order?: number;
  /** Se true, a camada é técnica (ex: mask) e não aparece no export final. */
  technical?: boolean;
}

export interface InpaintBlock {
  bbox: [number, number, number, number];
  confidence?: number;
}

export interface ProcessRegionOverlay {
  id: string;
  page_index: number;
  bbox: [number, number, number, number];
  crop_path: string;
  text_layer_ids: string[];
  visible: boolean;
  locked: boolean;
  order: number;
}

export interface EditorPageCache {
  inpaint?: string | null;
}

export interface PageData {
  numero: number;
  arquivo_original: string;
  arquivo_traduzido: string;
  image_layers?: Partial<Record<ImageLayerKey, ImageLayer>>;
  editor_cache?: EditorPageCache;
  inpaint_blocks?: InpaintBlock[];
  process_overlays?: ProcessRegionOverlay[];
  text_layers: TextEntry[];
  textos: TextEntry[];
}

export interface ContextSourceRef {
  fonte: string;
  titulo: string;
  url: string;
  trecho: string;
}

export interface ProjectContext {
  sinopse: string;
  genero: string[];
  personagens: string[];
  glossario: Record<string, string>;
  aliases: string[];
  termos: string[];
  relacoes: string[];
  faccoes: string[];
  resumo_por_arco: string[];
  memoria_lexical: Record<string, string>;
  fontes_usadas: ContextSourceRef[];
  internet_context?: unknown;
}

export interface SystemFontAsset {
  family: string;
  path: string;
  weight: string;
  style: string;
}

export interface ProjectFontAssets {
  system?: Record<string, SystemFontAsset>;
}

export interface WorkContextSummary {
  selected: boolean;
  work_id: string;
  title: string;
  context_loaded: boolean;
  glossary_loaded: boolean;
  glossary_entries_count: number;
  internet_context_loaded?: boolean;
  cover_url?: string;
  risk_level: "high" | "medium" | "low";
  user_ignored_warning: boolean;
}

export interface Project {
  id: string;
  obra: string;
  capitulo: number;
  idioma_origem: string;
  idioma_destino: string;
  engine_preset_id?: "manga" | "manhwa_manhua" | "default" | string;
  qualidade: ProjectQuality;
  contexto: ProjectContext;
  work_context?: WorkContextSummary | null;
  /** Contexto rico da obra para guiar a tradução (persiste no project.json) */
  translation_context?: WorkContext;
  font_assets?: ProjectFontAssets;
  qa?: unknown;
  output_review_state?: OutputReviewState;
  completion_status?: CompletionStatus;
  export_gate?: PipelineExportGateSummary;
  blocking_flags?: string[];
  review_flags?: string[];
  critical_issue_count?: number;
  review_issue_count?: number;
  paginas: PageData[];
  status: AppProjectStatus;
  source_path: string;
  output_path?: string;
  totalPages: number;
  mode: "auto" | "manual";
  preset?: unknown;
}

export type ProjectQuality = "rapida" | "normal" | "alta";

export type PipelineStep =
  | "extract"
  | "ocr"
  | "context"
  | "translate"
  | "inpaint"
  | "typeset";

export interface PipelineProgress {
  step: PipelineStep;
  step_progress: number;
  overall_progress: number;
  current_page: number;
  total_pages: number;
  message: string;
  eta_seconds: number;
}

export type PerformanceTier = "cpu_only" | "balanced" | "fast" | "workstation";

export interface QualityEstimateTable {
  rapida: number;
  normal: number;
  alta: number;
}

export interface SystemProfile {
  cpu_name: string;
  cpu_cores: number;
  cpu_threads: number;
  ram_gb: number;
  gpu_available: boolean;
  gpu_name: string;
  gpu_vram_gb: number | null;
  performance_tier: PerformanceTier;
  startup_seconds: number;
  seconds_per_page: QualityEstimateTable;
}

export interface PipelineTimeEstimate {
  total_pages: number;
  quality: ProjectQuality;
  total_seconds: number;
  seconds_per_page: number;
  startup_seconds: number;
  performance_tier: PerformanceTier;
  hardware_summary: string;
}

export type RecentProject = {
  id: string;
  obra: string;
  capitulo: number;
  pages: number;
  date: string;
  status: string;
  project_path?: string;
  output_path?: string;
  output_review_state?: OutputReviewState;
  critical_issue_count?: number;
  review_issue_count?: number;
};

export interface BatchCompletionChapter {
  id: string;
  obra: string;
  capitulo: number;
  pages: number;
  project_path: string;
  output_path: string;
  first_page_path: string | null;
  cover_url: string | null;
  paginas: PageData[];
  status?: "done" | "done_blocked" | "needs_review";
  critical_issue_count?: number;
  review_issue_count?: number;
}

export interface BatchCompletionSummary {
  id: string;
  obra: string;
  total_chapters: number;
  total_pages: number;
  elapsed_seconds: number;
  completed_at: string;
  chapters: BatchCompletionChapter[];
}

export type PipelineLogLevel = "info" | "step" | "progress" | "error" | "success";

export interface PipelineLogEntry {
  timestamp: number;
  level: PipelineLogLevel;
  message: string;
  step?: PipelineStep | null;
  current_page?: number | null;
  total_pages?: number | null;
  overall_progress?: number | null;
  step_progress?: number | null;
}

function sanitizeRecentProjects(value: unknown): RecentProject[] {
  if (!Array.isArray(value)) return [];

  const recent = value
    .filter((item): item is RecentProject => {
      if (!item || typeof item !== "object") return false;
      const candidate = item as Partial<RecentProject>;
      return typeof candidate.id === "string";
    })
    .map((item) => {
      const projectPath =
        typeof item.project_path === "string" && item.project_path.trim()
          ? item.project_path.trim().replace(/\\/g, "/")
          : undefined;
      const outputPath =
        typeof item.output_path === "string" && item.output_path.trim()
          ? item.output_path.trim().replace(/\\/g, "/")
          : undefined;

      return {
        id: item.id,
        obra: typeof item.obra === "string" ? item.obra : "Projeto sem nome",
        capitulo: typeof item.capitulo === "number" ? item.capitulo : 1,
        pages: typeof item.pages === "number" ? item.pages : 0,
        date: typeof item.date === "string" ? item.date : new Date(0).toISOString(),
        status: typeof item.status === "string" ? item.status : "done",
        output_review_state:
          item.output_review_state === "blocked_preview" ||
          item.output_review_state === "approved" ||
          item.output_review_state === "overridden"
            ? item.output_review_state
            : undefined,
        critical_issue_count:
          typeof item.critical_issue_count === "number" && Number.isFinite(item.critical_issue_count)
            ? item.critical_issue_count
            : undefined,
        review_issue_count:
          typeof item.review_issue_count === "number" && Number.isFinite(item.review_issue_count)
            ? item.review_issue_count
            : undefined,
        ...(projectPath ? { project_path: projectPath } : {}),
        ...(outputPath ? { output_path: outputPath } : {}),
      };
    })
    .filter((item) => !(item.obra === "Projeto sem nome" && item.pages <= 1));

  const deduped: RecentProject[] = [];
  for (const item of recent) {
    if (deduped.some((existing) => existing.id === item.id)) continue;
    const itemPath = (item.project_path ?? item.output_path ?? "").toLocaleLowerCase("pt-BR");
    const itemMeta = `${item.obra.toLocaleLowerCase("pt-BR")}::${item.capitulo}::${item.pages}`;
    if (
      deduped.some((existing) => {
        const existingPath = (existing.project_path ?? existing.output_path ?? "").toLocaleLowerCase("pt-BR");
        const existingMeta = `${existing.obra.toLocaleLowerCase("pt-BR")}::${existing.capitulo}::${existing.pages}`;
        return (
          (itemPath && existingPath && existingPath === itemPath) ||
          ((!itemPath || !existingPath) && existingMeta === itemMeta)
        );
      })
    ) {
      continue;
    }
    deduped.push(item);
  }
  return deduped.slice(0, 12);
}

function persistRecentProjects(projects: RecentProject[]) {
  localStorage.setItem("traduzai_recent", JSON.stringify(projects));
}

function persistFavoriteWorks(favoriteWorks: string[]) {
  localStorage.setItem("traduzai_favorite_works", JSON.stringify(favoriteWorks));
}

interface AppState {
  project: Project | null;
  recentProjects: RecentProject[];
  favoriteWorks: string[];
  pipeline: PipelineProgress | null;
  systemProfile: SystemProfile | null;
  setupEstimate: PipelineTimeEstimate | null;
  credits: number;
  weeklyFreeUsed: number;
  weeklyFreeLimit: number;
  gpuAvailable: boolean;
  gpuName: string;
  modelsReady: boolean;
  batchSources: string[];
  batchCompletion: BatchCompletionSummary | null;
  pipelineLog: PipelineLogEntry[];

  setProject: (project: Project | null) => void;
  updateProject: (updates: Partial<Project>) => void;
  setPipeline: (progress: PipelineProgress | null) => void;
  setSystemProfile: (profile: SystemProfile | null) => void;
  setSetupEstimate: (estimate: PipelineTimeEstimate | null) => void;
  setCredits: (credits: number, weeklyUsed?: number) => void;
  useCredits: (amount: number) => void;
  useFreePages: (pages: number) => void;
  setGpu: (available: boolean, name: string) => void;
  setModelsReady: (ready: boolean) => void;
  setBatchSources: (sources: string[]) => void;
  setBatchCompletion: (summary: BatchCompletionSummary | null) => void;
  clearBatchCompletion: () => void;
  addRecentProject: (p: RecentProject) => void;
  removeRecentProject: (id: string) => void;
  addFavoriteWork: (title: string) => void;
  removeFavoriteWork: (title: string) => void;
  appendPipelineLog: (entry: Omit<PipelineLogEntry, "timestamp"> & { timestamp?: number }) => void;
  clearPipelineLog: () => void;

  canTranslate: (pages: number) => boolean;
  freeRemaining: () => number;
}

// Persist recentProjects in localStorage between sessions
const savedRecent: RecentProject[] = (() => {
  try {
    const parsed = JSON.parse(localStorage.getItem("traduzai_recent") || "[]");
    const sanitized = sanitizeRecentProjects(parsed);
    persistRecentProjects(sanitized);
    return sanitized;
  } catch {
    return [];
  }
})();

const savedFavoriteWorks: string[] = (() => {
  try {
    const parsed = JSON.parse(localStorage.getItem("traduzai_favorite_works") || "[]");
    const sanitized = sanitizeFavoriteWorks(parsed);
    persistFavoriteWorks(sanitized);
    return sanitized;
  } catch {
    return [];
  }
})();

export const useAppStore = create<AppState>((set, get) => ({
  project: null,
  recentProjects: savedRecent,
  favoriteWorks: savedFavoriteWorks,
  pipeline: null,
  systemProfile: null,
  setupEstimate: null,
  credits: 1000,
  weeklyFreeUsed: 0,
  weeklyFreeLimit: 40,
  gpuAvailable: true,
  gpuName: "Verificando CUDA...",
  modelsReady: false,
  batchSources: [],
  batchCompletion: null,
  pipelineLog: [],

  setProject: (project) => set({ project, pipeline: null, setupEstimate: null, pipelineLog: [], batchCompletion: null }),
  updateProject: (updates) =>
    set((s) => ({
      project: s.project ? { ...s.project, ...updates } : null,
    })),
  setPipeline: (pipeline) => set({ pipeline }),
  setSystemProfile: (systemProfile) =>
    set({
      systemProfile,
      gpuAvailable: systemProfile?.gpu_available ?? true,
      gpuName: systemProfile?.gpu_name ?? "Verificando CUDA...",
    }),
  setSetupEstimate: (setupEstimate) => set({ setupEstimate }),
  setCredits: (credits, weeklyUsed) =>
    set((s) => ({
      credits,
      weeklyFreeUsed: weeklyUsed !== undefined ? weeklyUsed : s.weeklyFreeUsed,
    })),
  useCredits: (amount) => set((s) => ({ credits: Math.max(0, s.credits - amount) })),
  useFreePages: (pages) =>
    set((s) => ({ weeklyFreeUsed: s.weeklyFreeUsed + pages })),
  setGpu: (available, name) => set({ gpuAvailable: available, gpuName: name }),
  setModelsReady: (ready) => set({ modelsReady: ready }),
  setBatchSources: (batchSources) =>
    set({
      batchSources,
      ...(batchSources.length > 0 ? { batchCompletion: null } : {}),
    }),
  setBatchCompletion: (batchCompletion) => set({ batchCompletion }),
  clearBatchCompletion: () => set({ batchCompletion: null }),
  addRecentProject: (p) => {
    const updated = sanitizeRecentProjects([p, ...get().recentProjects]);
    persistRecentProjects(updated);
    set({ recentProjects: updated });
  },
  removeRecentProject: (id) =>
    set((s) => {
      const updated = s.recentProjects.filter((project) => project.id !== id);
      persistRecentProjects(updated);
      return { recentProjects: updated };
    }),
  addFavoriteWork: (title) => {
    const updated = upsertFavoriteWork(get().favoriteWorks, title);
    persistFavoriteWorks(updated);
    set({ favoriteWorks: updated });
  },
  removeFavoriteWork: (title) =>
    set((s) => {
      const normalized = title.trim().toLocaleLowerCase("pt-BR");
      const updated = sanitizeFavoriteWorks(s.favoriteWorks).filter(
        (item) => item.toLocaleLowerCase("pt-BR") !== normalized,
      );
      persistFavoriteWorks(updated);
      return { favoriteWorks: updated };
    }),
  appendPipelineLog: (entry) =>
    set((s) => {
      const next: PipelineLogEntry = {
        timestamp: entry.timestamp ?? Date.now(),
        level: entry.level,
        message: entry.message,
        step: entry.step ?? null,
        current_page: entry.current_page ?? null,
        total_pages: entry.total_pages ?? null,
        overall_progress: entry.overall_progress ?? null,
        step_progress: entry.step_progress ?? null,
      };
      const MAX = 5000;
      const appended = [...s.pipelineLog, next];
      return { pipelineLog: appended.length > MAX ? appended.slice(-MAX) : appended };
    }),
  clearPipelineLog: () => set({ pipelineLog: [] }),

  canTranslate: (pages) => {
    const { credits, weeklyFreeUsed, weeklyFreeLimit } = get();
    const free = Math.max(0, weeklyFreeLimit - weeklyFreeUsed);
    return free >= pages || credits >= pages;
  },

  freeRemaining: () => {
    const { weeklyFreeUsed, weeklyFreeLimit } = get();
    return Math.max(0, weeklyFreeLimit - weeklyFreeUsed);
  },
}));
