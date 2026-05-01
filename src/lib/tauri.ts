import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { tauriMock } from "./e2e/tauriMock";
import type {
  ContextSourceRef,
  ImageLayer,
  ImageLayerKey,
  PageData,
  PipelineProgress,
  ProjectContext,
  TextEntry,
  SystemProfile,
} from "./stores/appStore";

function isE2E() {
  const meta = import.meta as ImportMeta & { env?: Record<string, string | undefined> };
  if ((meta.env?.VITE_E2E ?? "") === "1") return true;
  return typeof window !== "undefined" && !("__TAURI_INTERNALS__" in window);
}

export interface ProjectJson {
  versao?: string;
  app?: string;
  obra?: string;
  capitulo?: number;
  idioma_origem?: string;
  idioma_destino?: string;
  contexto?: Partial<ProjectContext>;
  work_context?: Partial<WorkContextSummary>;
  paginas?: PageData[];
  estatisticas?: {
    total_paginas?: number;
    total_textos?: number;
    tempo_processamento_seg?: number;
    data_criacao?: string;
  };
}

export interface EditorPagePayload {
  project_file: string;
  project_dir: string;
  page_index: number;
  total_pages: number;
  page: PageData;
}

export interface StoragePaths {
  mode: "dev" | "production";
  root: string;
  works: string;
  memory: string;
  logs: string;
  exports: string;
  debug: string;
  fixtures: string;
  models: string;
  projects: string;
  settings: string;
}

export type WorkContextQuality = "empty" | "partial" | "reviewed";
export type WorkContextRisk = "high" | "medium" | "low";

export interface WorkContextTranslationStyle {
  tone: string;
  honorifics: string;
  names: string;
  lore_terms: string;
  sound_effects: string;
}

export interface WorkContextProfile {
  work_id: string;
  title: string;
  alt_titles: string[];
  source_language: string;
  target_language: string;
  status: string;
  context_quality: WorkContextQuality;
  synopsis: string;
  genre: string[];
  translation_style: WorkContextTranslationStyle;
  characters: unknown[];
  places: unknown[];
  factions: unknown[];
  terms: unknown[];
  forbidden_translations: unknown[];
  chapter_memory: unknown[];
  last_updated: string;
}

export interface WorkContextSummary {
  selected: boolean;
  work_id: string;
  title: string;
  context_loaded: boolean;
  glossary_loaded: boolean;
  glossary_entries_count: number;
  risk_level: WorkContextRisk;
  user_ignored_warning: boolean;
}

export interface GlossaryEntry {
  id: string;
  source: string;
  target: string;
  type: string;
  case_sensitive: boolean;
  protect: boolean;
  aliases: string[];
  forbidden: string[];
  confidence: number;
  status: string;
  notes: string;
  context_rule: string;
}

export interface Glossary {
  work_id: string;
  version: number;
  entries: GlossaryEntry[];
}

const IMAGE_LAYER_KEYS: ImageLayerKey[] = ["base", "mask", "inpaint", "brush", "rendered"];

function isAbsolutePath(path: string) {
  return /^[A-Za-z]:[\\/]/.test(path) || path.startsWith("/");
}

function projectBaseDir(baseDir: string) {
  return baseDir.replace(/\\/g, "/").replace(/\/project\.json$/i, "");
}

function joinProjectPath(baseDir: string, maybeRelative?: string | null) {
  if (!maybeRelative) return null;
  if (isAbsolutePath(maybeRelative)) return maybeRelative.replace(/\\/g, "/");
  return `${projectBaseDir(baseDir)}/${maybeRelative}`.replace(/\\/g, "/");
}

export function buildPlainPageCommandArgs(args: { project_path: string; page_index: number }) {
  return {
    projectPath: args.project_path,
    pageIndex: args.page_index,
  };
}

function hydrateTextLayer(layer: Partial<TextEntry>, baseDir: string): TextEntry {
  const rawStyle = (layer.style ?? layer.estilo ?? {
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
  }) as TextEntry["estilo"];
  const style: TextEntry["estilo"] = {
    ...rawStyle,
    cor: "#000000",
    contorno: "",
    contorno_px: 0,
  };
  const bbox =
    layer.render_bbox ?? layer.layout_bbox ?? layer.bbox ?? layer.source_bbox ?? layer.balloon_bbox ?? [0, 0, 32, 32];

  return {
    ...layer,
    kind: "text",
    id: layer.id ?? crypto.randomUUID(),
    bbox,
    source_bbox: layer.source_bbox ?? layer.bbox ?? bbox,
    layout_bbox: layer.layout_bbox ?? bbox,
    render_bbox: layer.render_bbox ?? null,
    tipo: (layer.tipo ?? "fala") as TextEntry["tipo"],
    original: layer.original ?? "",
    traduzido: layer.traduzido ?? layer.translated ?? "",
    translated: layer.translated ?? layer.traduzido ?? "",
    confianca_ocr: layer.confianca_ocr ?? layer.ocr_confidence ?? 0,
    ocr_confidence: layer.ocr_confidence ?? layer.confianca_ocr ?? 0,
    estilo: style,
    style,
    visible: layer.visible ?? true,
    locked: layer.locked ?? false,
    order: layer.order ?? 0,
    render_preview_path: joinProjectPath(baseDir, layer.render_preview_path ?? null),
    detector: layer.detector ?? null,
    line_polygons: layer.line_polygons ?? null,
    source_direction: layer.source_direction ?? null,
    rendered_direction: layer.rendered_direction ?? null,
    source_language: layer.source_language ?? null,
    rotation_deg: layer.rotation_deg ?? 0,
    detected_font_size_px: layer.detected_font_size_px ?? null,
    page_profile: layer.page_profile ?? null,
    block_profile: layer.block_profile ?? null,
    layout_profile: layer.layout_profile ?? layer.block_profile ?? null,
    balloon_bbox: layer.balloon_bbox ?? bbox,
    balloon_subregions: layer.balloon_subregions ?? [],
    layout_group_size: layer.layout_group_size ?? 1,
  };
}

export function hydratePageData(page: Partial<PageData>, baseDir: string): PageData {
  const imageLayers = Object.fromEntries(
    IMAGE_LAYER_KEYS.map((key) => {
      const layer = page.image_layers?.[key];
      const fallbackPath =
        key === "base"
          ? page.arquivo_original
          : key === "rendered"
            ? page.arquivo_traduzido
            : layer?.path ?? null;
      return [
        key,
        {
          key,
          path: joinProjectPath(baseDir, layer?.path ?? fallbackPath ?? null),
          visible: layer?.visible ?? (key === "base" || key === "rendered"),
          locked: layer?.locked ?? (key === "base" || key === "rendered"),
        } satisfies ImageLayer,
      ];
    }),
  ) as Partial<Record<ImageLayerKey, ImageLayer>>;

  const rawLayers = (page.text_layers?.length ? page.text_layers : page.textos) ?? [];
  const textLayers = rawLayers.map((layer) => hydrateTextLayer(layer, baseDir));

  return {
    ...page,
    numero: page.numero ?? 1,
    arquivo_original:
      imageLayers.base?.path ?? joinProjectPath(baseDir, page.arquivo_original) ?? "",
    arquivo_traduzido:
      imageLayers.rendered?.path ?? joinProjectPath(baseDir, page.arquivo_traduzido) ?? "",
    image_layers: imageLayers,
    inpaint_blocks: page.inpaint_blocks ?? [],
    text_layers: [...textLayers].sort((a, b) => (a.order ?? 0) - (b.order ?? 0)),
    textos: [...textLayers].sort((a, b) => (a.order ?? 0) - (b.order ?? 0)),
  };
}

export function hydrateProjectJson(raw: ProjectJson, projectDir: string): ProjectJson {
  return {
    ...raw,
    paginas: (raw.paginas ?? []).map((page) => hydratePageData(page, projectDir)),
  };
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
  work_id: string;
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
  context_quality: WorkContextQuality;
  risk_level: WorkContextRisk;
  glossary_entries_count: number;
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

export type LabChapterScopeMode = "all" | "first_n" | "range" | "explicit";
export type LabGpuPolicy = "prefer_gpu" | "require_gpu";

export interface LabChapterScope {
  mode: LabChapterScopeMode;
  first_n?: number;
  start_chapter?: number;
  end_chapter?: number;
  chapter_numbers?: number[];
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
  chapter_number: number;
  page_index: number;
  preview_path: string;
  selection_reason: string;
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

export interface LabVisualCriticPageProfile {
  page_index: number;
  text_count: number;
  total_area: number;
  max_area: number;
}

export interface LabVisualCriticRun {
  critic_id: string;
  chapter_number: number;
  selection_mode: string;
  selected_pages: number[];
  sample_limit: number;
  page_profiles: LabVisualCriticPageProfile[];
}

export interface LabPatchProposal {
  proposal_id: string;
  patch_unified_diff: string;
  files_affected: string[];
  rationale: string;
  author: string;
  confidence: number;
  model_used: string;
  generated_at_iso: string;
  dry_run: boolean;
  error: string;
  review_package_json_path?: string;
  review_package_markdown_path?: string;
}

export interface LabPatchApplyResult {
  proposal_id: string;
  applied: boolean;
  branch_created: string;
  commit_sha: string;
  error: string;
  files_patched: string[];
}

export type LabCoderStrategy = "local" | "ollama" | "nvidia_minimax" | "claude_code" | "claude_sdk";

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
  // Campos do Planner
  motivation?: string;
  target_file?: string;
  target_anchor?: string;
  change_kind?: string;
  needs_coder?: boolean;
  priority_score?: number;
  issue_type?: string;
  local_patch_hint?: Record<string, unknown>;
  expected_metric_gain?: Record<string, number>;
  // Preenchido apos coder gerar patch
  patch_proposal?: LabPatchProposal;
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
  visual_critic_summary: string;
  visual_critic_runs: LabVisualCriticRun[];
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

export interface LabHumanFeedbackEntry {
  id: string;
  scope_id: string;
  chapter_number: number;
  page_index: number;
  kind: string;
  status: string;
  bbox: number[];
  comment: string;
  created_at_ms: number;
  source_run_id: string;
}

export interface SaveLabHumanFeedbackRequest {
  chapter_number: number;
  page_index: number;
  kind: string;
  status: string;
  bbox: number[];
  comment: string;
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

export async function getStoragePaths(): Promise<StoragePaths> {
  if (isE2E()) {
    return {
      mode: "dev",
      root: "data",
      works: "data/works",
      memory: "data/memory",
      logs: "data/logs",
      exports: "data/exports",
      debug: "debug",
      fixtures: "fixtures",
      models: "data/models",
      projects: "data/projects",
      settings: "data/settings.json",
    };
  }
  return invoke("get_storage_paths");
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

export async function openMultipleSources(): Promise<string[]> {
  return invoke("open_multiple_sources_dialog");
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
  if (isE2E()) return tauriMock.loadProjectJson();
  const project = await invoke<ProjectJson>("load_project_json", { path });
  return hydrateProjectJson(project, path);
}

export async function saveProjectJson(config: { project_path: string; project_json: any }): Promise<void> {
  if (isE2E()) return tauriMock.saveProjectJson(config);
  return invoke("save_project_json", { config });
}

export async function loadEditorPage(config: {
  project_path: string;
  page_index: number;
}): Promise<EditorPagePayload> {
  if (isE2E()) return tauriMock.loadEditorPage(config);
  const payload = await invoke<EditorPagePayload>("load_editor_page", { config });
  return {
    ...payload,
    page: hydratePageData(payload.page, payload.project_dir),
  };
}

export async function createEditorTextLayer(config: {
  project_path: string;
  page_index: number;
  layout_bbox: [number, number, number, number];
}): Promise<TextEntry> {
  const layer = await invoke<Partial<TextEntry>>("create_text_layer", { config });
  return hydrateTextLayer(layer, config.project_path);
}

export async function patchEditorTextLayer(config: {
  project_path: string;
  page_index: number;
  layer_id: string;
  patch: Record<string, unknown>;
}): Promise<TextEntry> {
  if (isE2E()) return tauriMock.patchEditorTextLayer(config);
  const layer = await invoke<Partial<TextEntry>>("patch_text_layer", { config });
  return hydrateTextLayer(layer, config.project_path);
}

export async function deleteEditorTextLayer(config: {
  project_path: string;
  page_index: number;
  layer_id: string;
}): Promise<void> {
  return invoke("delete_text_layer", { config });
}

export async function setEditorLayerVisibility(config: {
  project_path: string;
  page_index: number;
  layer_kind: "image" | "text";
  layer_key?: string | null;
  layer_id?: string | null;
  visible: boolean;
}): Promise<void> {
  if (isE2E()) return tauriMock.setEditorLayerVisibility(config);
  return invoke("set_layer_visibility", { config });
}

export async function updateMaskRegion(config: {
  project_path: string;
  page_index: number;
  width: number;
  height: number;
  brush_size: number;
  clear?: boolean;
  erase?: boolean;
  strokes: [number, number][][];
}): Promise<string> {
  if (isE2E()) return tauriMock.updateMaskRegion();
  return invoke("update_mask_region", { config });
}

export async function updateBrushRegion(config: {
  project_path: string;
  page_index: number;
  width: number;
  height: number;
  brush_size: number;
  clear?: boolean;
  erase?: boolean;
  strokes: [number, number][][];
}): Promise<string> {
  if (isE2E()) return tauriMock.updateBrushRegion();
  return invoke("update_brush_region", { config });
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

export async function loadOrCreateWorkContext(request: {
  title: string;
  source_language: string;
  target_language: string;
  synopsis?: string;
  genre?: string[];
  characters?: string[];
  terms?: string[];
  factions?: string[];
}): Promise<WorkContextProfile> {
  if (isE2E()) {
    const workId =
      request.title.toLocaleLowerCase("pt-BR").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") ||
      "obra-sem-titulo";
    return {
      work_id: workId,
      title: request.title,
      alt_titles: [],
      source_language: request.source_language,
      target_language: request.target_language,
      status: "active",
      context_quality: request.synopsis ? "partial" : "empty",
      synopsis: request.synopsis ?? "",
      genre: request.genre ?? [],
      translation_style: {
        tone: "natural Brazilian Portuguese",
        honorifics: "adapted",
        names: "preserve proper names",
        lore_terms: "use glossary",
        sound_effects: "review",
      },
      characters: request.characters ?? [],
      places: [],
      factions: request.factions ?? [],
      terms: request.terms ?? [],
      forbidden_translations: [],
      chapter_memory: [],
      last_updated: new Date(0).toISOString(),
    };
  }
  return invoke("load_or_create_work_context", { request });
}

function browserGlossaryKey(workId: string) {
  return `traduzai_e2e_glossary_${workId}`;
}

export async function loadGlossary(workId: string): Promise<Glossary> {
  if (isE2E()) {
    const raw = localStorage.getItem(browserGlossaryKey(workId));
    return raw ? JSON.parse(raw) : { work_id: workId, version: 1, entries: [] };
  }
  return invoke("load_glossary", { workId });
}

export async function saveGlossary(glossary: Glossary): Promise<Glossary> {
  if (isE2E()) {
    localStorage.setItem(browserGlossaryKey(glossary.work_id), JSON.stringify(glossary));
    return glossary;
  }
  return invoke("save_glossary", { glossary });
}

export async function upsertGlossaryEntry(workId: string, entry: GlossaryEntry): Promise<Glossary> {
  if (isE2E()) {
    const glossary = await loadGlossary(workId);
    const index = glossary.entries.findIndex((item) => item.id === entry.id);
    const entries = [...glossary.entries];
    if (index >= 0) entries[index] = entry;
    else entries.push(entry);
    return saveGlossary({ ...glossary, entries });
  }
  return invoke("upsert_glossary_entry", { workId, entry });
}

export async function removeGlossaryEntry(workId: string, entryId: string): Promise<Glossary> {
  if (isE2E()) {
    const glossary = await loadGlossary(workId);
    return saveGlossary({
      ...glossary,
      entries: glossary.entries.filter((entry) => entry.id !== entryId),
    });
  }
  return invoke("remove_glossary_entry", { workId, entryId });
}

// Pipeline
export async function startPipeline(config: {
  source_path: string;
  mode: "auto" | "manual";
  obra: string;
  capitulo: number;
  idioma_origem: string;
  idioma_destino: string;
  qualidade: "rapida" | "normal" | "alta";
  glossario: Record<string, string>;
  work_context?: WorkContextSummary | null;
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

export async function retypesetPage(args: { project_path: string; page_index: number }): Promise<string> {
  return await invoke("retypeset_page", { config: args });
}

export async function renderPreviewPage(args: {
  project_path: string;
  page_index: number;
  page: PageData;
  fingerprint: string;
}): Promise<string> {
  if (isE2E()) return tauriMock.renderPreviewPage();
  return await invoke("render_preview_page", { config: args });
}

export async function detectPage(args: { project_path: string; page_index: number }): Promise<string> {
  return await invoke("detect_page", buildPlainPageCommandArgs(args));
}

export async function ocrPage(args: { project_path: string; page_index: number }): Promise<string> {
  return await invoke("ocr_page", buildPlainPageCommandArgs(args));
}

export async function translatePage(args: { project_path: string; page_index: number }): Promise<string> {
  return await invoke("translate_page", buildPlainPageCommandArgs(args));
}

export async function reinpaintPage(args: { project_path: string; page_index: number }): Promise<string> {
  return invoke("reinpaint_page", { config: args });
}

export async function processBlock(config: {
  project_path: string;
  page_index: number;
  block_id: string;
  mode: "ocr" | "translate" | "inpaint";
}): Promise<string> {
  return invoke("process_block", { config });
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
  // Tauri uses camelCase for multi-word command args.
  return invoke("approve_lab_proposal", { proposalId });
}

export async function rejectLabProposal(proposalId: string): Promise<LabProposal> {
  return invoke("reject_lab_proposal", { proposalId });
}

export async function approveLabBatch(batchId: string): Promise<LabProposal[]> {
  return invoke("approve_lab_batch", { batchId });
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

export async function listLabHumanFeedback(
  chapterNumber: number,
  pageIndex?: number
): Promise<LabHumanFeedbackEntry[]> {
  return invoke("list_lab_human_feedback", {
    chapterNumber,
    pageIndex: pageIndex ?? null,
  });
}

export async function saveLabHumanFeedback(
  request: SaveLabHumanFeedbackRequest
): Promise<LabHumanFeedbackEntry> {
  return invoke("save_lab_human_feedback", { request });
}

export async function pickLabSourceDir(): Promise<string | null> {
  return invoke("pick_lab_source_dir");
}

export async function pickLabReferenceDir(): Promise<string | null> {
  return invoke("pick_lab_reference_dir");
}

export async function pickLabSourceFiles(): Promise<string[]> {
  return invoke("pick_lab_source_files");
}

export async function pickLabReferenceFiles(): Promise<string[]> {
  return invoke("pick_lab_reference_files");
}

export async function setLabDirs(
  sourceDir: string,
  referenceDir: string
): Promise<LabSnapshot> {
  return invoke("set_lab_dirs", { sourceDir, referenceDir });
}

export async function proposeLabPatch(
  proposalId: string,
  coderStrategy: LabCoderStrategy = "nvidia_minimax",
  ollamaHost?: string
): Promise<LabPatchProposal> {
  return invoke("propose_lab_patch", {
    proposalId,
    coderStrategy,
    ollamaHost: ollamaHost ?? null,
  });
}

export async function applyLabPatch(
  proposalId: string,
  patchUnifiedDiff: string,
  createBranch: boolean,
  commit: boolean,
  commitMessage?: string
): Promise<LabPatchApplyResult> {
  return invoke("apply_lab_patch", {
    proposalId,
    patchUnifiedDiff,
    createBranch,
    commit,
    commitMessage: commitMessage ?? null,
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
  format: "zip_full" | "jpg_only" | "cbz" | "psd";
  output_path: string;
  export_mode?: "clean" | "with_warnings" | "debug";
}): Promise<{ path: string }> {
  return invoke("export_project", { config });
}

export async function exportPagePsd(config: {
  project_path: string;
  page_index: number;
  output_path: string;
}): Promise<string> {
  return invoke("export_page_psd", { config });
}

export async function openExportDialog(format: "zip_full" | "jpg_only" | "cbz" | "psd"): Promise<string | null> {
  return invoke("save_file_dialog", { format });
}

export async function openLogSaveDialog(suggestedName?: string): Promise<string | null> {
  return invoke("save_file_dialog", {
    format: "log",
    suggestedName: suggestedName ?? null,
  });
}

export async function exportTextFile(outputPath: string, content: string): Promise<string> {
  return invoke("export_text_file", { outputPath, content });
}

export async function exportLocalMemory(): Promise<unknown> {
  return invoke("export_local_memory");
}

export async function importLocalMemory(payload: unknown): Promise<void> {
  return invoke("import_local_memory", { payload });
}

export async function upsertMemoryWork(workId: string, title: string): Promise<void> {
  return invoke("upsert_memory_work", { workId, title });
}

export async function recordTranslationMemory(input: {
  work_id: string;
  source_text: string;
  target_text: string;
  context_json: string;
  confidence: number;
  confirmed_by_user: boolean;
}): Promise<void> {
  return invoke("record_translation_memory", { input });
}

export async function recordUserCorrection(input: {
  work_id: string;
  page: number;
  region_id: string;
  before_text: string;
  after_text: string;
  correction_type: string;
}): Promise<void> {
  return invoke("record_user_correction", { input });
}

export async function recordOcrCorrection(input: {
  work_id: string;
  raw_text: string;
  normalized_text: string;
  reason: string;
  confidence: number;
}): Promise<void> {
  return invoke("record_ocr_correction", { input });
}

export async function suggestMemoryTranslation(args: {
  work_id: string;
  source_text: string;
  glossary_reviewed: boolean;
}): Promise<{ target_text: string; source: string; confidence: number } | null> {
  return invoke("suggest_memory_translation", {
    workId: args.work_id,
    sourceText: args.source_text,
    glossaryReviewed: args.glossary_reviewed,
  });
}

export async function openLabPatchJsonDialog(proposalId: string): Promise<string | null> {
  return invoke("save_file_dialog", {
    format: "lab_patch_json",
    suggestedName: `lab-patch-${proposalId}.json`,
  });
}

export async function exportLabPatchJson(outputPath: string, content: string): Promise<string> {
  return invoke("export_lab_patch_json", { outputPath, content });
}

// Credits
export async function getCredits(): Promise<{ credits: number; weekly_used: number }> {
  return invoke("get_credits");
}

// Settings
export interface AppSettings {
  ollama_model: string;
  ollama_host: string;
  idioma_origem: string;
  idioma_destino: string;
}

export interface SupportedLanguage {
  code: string;
  label: string;
  ocr_strategy: "dedicated" | "best_effort";
}

export async function saveSettings(settings: AppSettings): Promise<void> {
  return invoke("save_settings", { settings });
}

export async function loadSettings(): Promise<AppSettings> {
  return invoke("load_settings");
}

export async function loadSupportedLanguages(): Promise<SupportedLanguage[]> {
  if (isE2E()) return [];
  return invoke("load_supported_languages");
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
