import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type {
  ContextSourceRef,
  PageData,
  PipelineProgress,
  ProjectContext,
  SystemProfile,
} from "./stores/appStore";

export interface ProjectJson {
  obra?: string;
  capitulo?: number;
  idioma_origem?: string;
  idioma_destino?: string;
  contexto?: Partial<ProjectContext>;
  paginas?: PageData[];
}

export interface WorkSearchCandidate {
  id: string;
  title: string;
  synopsis: string;
  source: "anilist" | "webnovel" | "fandom";
  source_url: string;
  cover_url?: string;
  score: number;
}

export interface WorkSearchResponse {
  query: string;
  candidates: WorkSearchCandidate[];
}

export interface EnrichedWorkContext {
  title: string;
  synopsis: string;
  genres: string[];
  characters: string[];
  aliases: string[];
  terms: string[];
  relationships: string[];
  factions: string[];
  arc_summaries: string[];
  lexical_memory: Record<string, string>;
  sources_used: ExternalContextSourceRef[];
  cover_url?: string;
}

export interface ExternalContextSourceRef {
  source: string;
  title: string;
  url: string;
  snippet: string;
}

export interface LabChapterPair {
  chapter_number: number;
  source_path: string;
  reference_path: string;
  source_pages: number;
  reference_pages: number;
  reference_group: string;
}

export type LabChapterScopeMode = "all" | "first_n" | "range";
export type LabGpuPolicy = "prefer_gpu" | "require_gpu";

export interface LabChapterScope {
  mode: LabChapterScopeMode;
  first_n?: number;
  start_chapter?: number;
  end_chapter?: number;
}

export interface StartLabRequest {
  chapter_scope: LabChapterScope;
  gpu_policy: LabGpuPolicy;
}

export interface LabAgentStatus {
  agent_id: string;
  label: string;
  layer: string;
  status: string;
  current_task: string;
  last_action: string;
  confidence: number;
  touched_domains: string[];
  proposal_id: string;
  updated_at_ms: number;
}

export interface LabReviewFinding {
  title: string;
  body: string;
  severity: string;
  file_path: string;
}

export interface LabReviewResult {
  proposal_id: string;
  reviewer_id: string;
  reviewer_label: string;
  verdict: string;
  findings: LabReviewFinding[];
  touched_domains: string[];
  reviewed_at_ms: number;
}

export interface LabBenchmarkMetrics {
  textual_similarity: number;
  term_consistency: number;
  layout_occupancy: number;
  readability: number;
  visual_cleanup: number;
  manual_edits_saved: number;
}

export interface LabBenchmarkResult {
  proposal_id: string;
  batch_id: string;
  score_before: number;
  score_after: number;
  green: boolean;
  summary: string;
  metrics: LabBenchmarkMetrics;
  git_available: boolean;
  pr_status: string;
  generated_at_ms: number;
}

export interface LabProposal {
  proposal_id: string;
  batch_id: string;
  title: string;
  summary: string;
  author: string;
  risk: string;
  touched_domains: string[];
  required_reviewers: string[];
  review_findings: LabReviewFinding[];
  integration_verdict: string;
  benchmark_batch_id: string;
  proposal_status: string;
  pr_status: string;
  git_available: boolean;
  created_at_ms: number;
}

export interface LabRunSummary {
  run_id: string;
  status: string;
  summary: string;
  total_pairs: number;
  processed_pairs: number;
  started_at_ms: number;
  finished_at_ms: number;
}

export interface LabSnapshot {
  status: string;
  run_id: string;
  current_stage: string;
  message: string;
  acceleration_summary: string;
  total_pairs: number;
  processed_pairs: number;
  eta_seconds: number;
  pending_proposals: number;
  active_batch_id: string;
  git_available: boolean;
  pr_ready: boolean;
  source_dir: string;
  reference_dir: string;
  chapter_pairs: LabChapterPair[];
  available_chapter_pairs: LabChapterPair[];
  scope_label: string;
  gpu_policy: LabGpuPolicy;
  agents: LabAgentStatus[];
  proposals: LabProposal[];
  reviews: LabReviewResult[];
  benchmarks: LabBenchmarkResult[];
  history: LabRunSummary[];
  updated_at_ms: number;
}

export interface LabControlResponse {
  run_id: string;
}

export interface LabReferencePreview {
  chapter_number: number;
  page_index: number;
  output_path: string;
  reference_path: string;
  source_path: string;
  output_kind: string;
}

export interface LabPromotionEvent {
  proposal_id: string;
  proposal_status: string;
  pr_status: string;
  summary: string;
}

// System info
export async function checkGpu(): Promise<{ available: boolean; name: string }> {
  return invoke("check_gpu");
}

export async function getSystemProfile(): Promise<SystemProfile> {
  return invoke("get_system_profile");
}

export async function warmupVisualStack(): Promise<string> {
  return invoke("warmup_visual_stack");
}

export async function checkModels(): Promise<{ ready: boolean; size_mb: number }> {
  return invoke("check_models");
}

export async function downloadModels(): Promise<void> {
  return invoke("download_models");
}

export async function onModelsProgress(
  callback: (data: { step: string; message: string }) => void
) {
  return listen<{ step: string; message: string }>("models-progress", (e) => callback(e.payload));
}

export async function onModelsReady(
  callback: (data: { success: boolean }) => void
) {
  return listen<{ success: boolean }>("models-ready", (e) => callback(e.payload));
}

// Project management
export async function openFiles(): Promise<string | null> {
  return invoke("open_source_dialog");
}

export async function openProjectDialog(): Promise<string | null> {
  return invoke("open_project_dialog");
}

export async function validateImport(path: string): Promise<{
  valid: boolean;
  pages: number;
  has_project_json: boolean;
  error?: string;
}> {
  return invoke("validate_import", { path });
}

export async function loadProjectJson(path: string): Promise<ProjectJson> {
  return invoke("load_project_json", { path });
}

export async function saveProjectJson(config: { project_path: string; project_json: any }): Promise<void> {
  return invoke("save_project_json", { config });
}

// Context lookup
export async function searchAnilist(query: string): Promise<{
  title: string;
  synopsis: string;
  genres: string[];
  characters: string[];
  cover_url?: string;
}> {
  return invoke("search_anilist", { query });
}

export async function searchWork(query: string): Promise<WorkSearchResponse> {
  return invoke("search_work", { query });
}

export async function enrichWorkContext(selection: WorkSearchCandidate): Promise<EnrichedWorkContext> {
  return invoke("enrich_work_context", { selection });
}

// Pipeline
export async function startPipeline(config: {
  source_path: string;
  obra: string;
  capitulo: number;
  idioma_destino: string;
  qualidade: "rapida" | "normal" | "alta";
  glossario: Record<string, string>;
  contexto: {
    sinopse: string;
    genero: string[];
    personagens: string[];
    aliases: string[];
    termos: string[];
    relacoes: string[];
    faccoes: string[];
    resumo_por_arco: string[];
    memoria_lexical: Record<string, string>;
    fontes_usadas: ContextSourceRef[];
  };
}): Promise<{ job_id: string }> {
  return invoke("start_pipeline", { config });
}

export async function retypesetPage(config: { project_path: string; page_index: number }): Promise<string> {
  return invoke("retypeset_page", { config });
}

export async function reinpaintPage(config: { project_path: string; page_index: number }): Promise<string> {
  return invoke("reinpaint_page", { config });
}

export async function cancelPipeline(): Promise<void> {
  return invoke("cancel_pipeline");
}

export async function pausePipeline(): Promise<void> {
  return invoke("pause_pipeline");
}

export async function resumePipeline(): Promise<void> {
  return invoke("resume_pipeline");
}

// Listen to pipeline progress events from Rust
export async function onPipelineProgress(
  callback: (progress: PipelineProgress) => void
) {
  return listen<PipelineProgress>("pipeline-progress", (event) => {
    callback(event.payload);
  });
}

export async function onPipelineComplete(
  callback: (result: { success: boolean; output_path: string; error?: string }) => void
) {
  return listen<{ success: boolean; output_path: string; error?: string }>("pipeline-complete", (event) => {
    callback(event.payload);
  });
}

// Lab
export async function getLabState(): Promise<LabSnapshot> {
  return invoke("get_lab_state");
}

export async function openLabWindow(): Promise<void> {
  return invoke("open_lab_window");
}

export async function startLab(request?: StartLabRequest): Promise<LabControlResponse> {
  return invoke("start_lab", { request });
}

export async function pauseLab(): Promise<void> {
  return invoke("pause_lab");
}

export async function resumeLab(): Promise<void> {
  return invoke("resume_lab");
}

export async function stopLab(): Promise<void> {
  return invoke("stop_lab");
}

export async function approveLabProposal(proposalId: string): Promise<LabProposal> {
  return invoke("approve_lab_proposal", { proposal_id: proposalId });
}

export async function rejectLabProposal(proposalId: string): Promise<LabProposal> {
  return invoke("reject_lab_proposal", { proposal_id: proposalId });
}

export async function approveLabBatch(batchId: string): Promise<LabProposal[]> {
  return invoke("approve_lab_batch", { batch_id: batchId });
}

export async function getLabReferencePreview(
  chapterNumber: number,
  pageIndex: number
): Promise<LabReferencePreview> {
  return invoke("get_lab_reference_preview", {
    chapterNumber,
    pageIndex,
  });
}

export async function onLabState(
  callback: (snapshot: LabSnapshot) => void
) {
  return listen<LabSnapshot>("lab_state", (event) => {
    callback(event.payload);
  });
}

export async function onLabAgentStatus(
  callback: (agent: LabAgentStatus) => void
) {
  return listen<LabAgentStatus>("agent_status", (event) => {
    callback(event.payload);
  });
}

export async function onLabReviewRequested(
  callback: (proposal: LabProposal) => void
) {
  return listen<LabProposal>("review_requested", (event) => {
    callback(event.payload);
  });
}

export async function onLabReviewResult(
  callback: (review: LabReviewResult) => void
) {
  return listen<LabReviewResult>("review_result", (event) => {
    callback(event.payload);
  });
}

export async function onLabBenchmarkResult(
  callback: (benchmark: LabBenchmarkResult) => void
) {
  return listen<LabBenchmarkResult>("benchmark_result", (event) => {
    callback(event.payload);
  });
}

export async function onLabProposalPromoted(
  callback: (event: LabPromotionEvent) => void
) {
  return listen<LabPromotionEvent>("proposal_promoted", (eventPayload) => {
    callback(eventPayload.payload);
  });
}

// Export
export async function exportProject(config: {
  project_path: string;
  format: "zip_full" | "jpg_only" | "cbz";
  output_path: string;
}): Promise<{ path: string }> {
  return invoke("export_project", { config });
}

export async function openExportDialog(format: "zip_full" | "jpg_only" | "cbz"): Promise<string | null> {
  return invoke("save_file_dialog", { format });
}

// Credits
export async function getCredits(): Promise<{ credits: number; weekly_used: number }> {
  return invoke("get_credits");
}

// Settings
export interface AppSettings {
  ollama_model: string;
  ollama_host: string;
  idioma_destino: string;
}

export async function saveSettings(settings: AppSettings): Promise<void> {
  return invoke("save_settings", { settings });
}

export async function loadSettings(): Promise<AppSettings> {
  return invoke("load_settings");
}

// Ollama
export interface OllamaStatus {
  running: boolean;
  models: string[];
  has_translator: boolean;
}

export async function checkOllama(): Promise<OllamaStatus> {
  return invoke("check_ollama");
}

export async function createTranslatorModel(): Promise<string> {
  return invoke("create_translator_model");
}

export async function restartApp(): Promise<void> {
  return invoke("restart_app");
}
