import { create } from "zustand";

export type ImageLayerKey = "base" | "mask" | "inpaint" | "brush" | "rendered";

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
  alinhamento: "left" | "center" | "right";
  force_upper?: boolean;
}

export interface TextEntry {
  id: string;
  kind?: "text";
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
  balloon_bbox?: [number, number, number, number];
  balloon_subregions?: [number, number, number, number][];
  layout_group_size?: number;
}

export interface ImageLayer {
  key: ImageLayerKey;
  path: string | null;
  visible: boolean;
  locked: boolean;
}

export interface InpaintBlock {
  bbox: [number, number, number, number];
  confidence?: number;
}

export interface PageData {
  numero: number;
  arquivo_original: string;
  arquivo_traduzido: string;
  image_layers?: Partial<Record<ImageLayerKey, ImageLayer>>;
  inpaint_blocks?: InpaintBlock[];
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
}

export interface Project {
  id: string;
  obra: string;
  capitulo: number;
  idioma_origem: string;
  idioma_destino: string;
  qualidade: ProjectQuality;
  contexto: ProjectContext;
  paginas: PageData[];
  status: "idle" | "setup" | "processing" | "done" | "error";
  source_path: string;
  output_path?: string;
  totalPages: number;
  mode: "auto" | "manual";
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

export type RecentProject = { id: string; obra: string; capitulo: number; pages: number; date: string; status: string };

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
    .map((item) => ({
      id: item.id,
      obra: typeof item.obra === "string" ? item.obra : "Projeto sem nome",
      capitulo: typeof item.capitulo === "number" ? item.capitulo : 1,
      pages: typeof item.pages === "number" ? item.pages : 0,
      date: typeof item.date === "string" ? item.date : new Date(0).toISOString(),
      status: typeof item.status === "string" ? item.status : "done",
    }))
    .filter((item) => !(item.obra === "Projeto sem nome" && item.pages <= 1));

  const deduped: RecentProject[] = [];
  for (const item of recent) {
    if (deduped.some((existing) => existing.id === item.id)) continue;
    deduped.push(item);
  }
  return deduped.slice(0, 12);
}

function persistRecentProjects(projects: RecentProject[]) {
  localStorage.setItem("traduzai_recent", JSON.stringify(projects));
}

interface AppState {
  project: Project | null;
  recentProjects: RecentProject[];
  pipeline: PipelineProgress | null;
  systemProfile: SystemProfile | null;
  setupEstimate: PipelineTimeEstimate | null;
  credits: number;
  weeklyFreeUsed: number;
  weeklyFreeLimit: number;
  gpuAvailable: boolean;
  gpuName: string;
  modelsReady: boolean;
  ollamaRunning: boolean;
  ollamaModels: string[];
  ollamaHasTranslator: boolean;
  batchSources: string[];
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
  setOllamaStatus: (running: boolean, models: string[], hasTranslator: boolean) => void;
  setBatchSources: (sources: string[]) => void;
  addRecentProject: (p: RecentProject) => void;
  removeRecentProject: (id: string) => void;
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

export const useAppStore = create<AppState>((set, get) => ({
  project: null,
  recentProjects: savedRecent,
  pipeline: null,
  systemProfile: null,
  setupEstimate: null,
  credits: 1000,
  weeklyFreeUsed: 0,
  weeklyFreeLimit: 40,
  gpuAvailable: true,
  gpuName: "Verificando CUDA...",
  modelsReady: false,
  ollamaRunning: false,
  ollamaModels: [],
  ollamaHasTranslator: false,
  batchSources: [],
  pipelineLog: [],

  setProject: (project) => set({ project, pipeline: null, setupEstimate: null, pipelineLog: [] }),
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
  setOllamaStatus: (ollamaRunning, ollamaModels, ollamaHasTranslator) =>
    set({ ollamaRunning, ollamaModels, ollamaHasTranslator }),
  setBatchSources: (batchSources) => set({ batchSources }),
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
