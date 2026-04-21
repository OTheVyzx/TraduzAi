use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tauri::{AppHandle, Emitter, Manager, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_dialog::DialogExt;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;
use zip::ZipArchive;

static LAB_CANCEL: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));
static LAB_PAUSE_MARKER: Lazy<Mutex<Option<PathBuf>>> = Lazy::new(|| Mutex::new(None));
static LAB_SNAPSHOT: Lazy<Mutex<LabSnapshot>> =
    Lazy::new(|| Mutex::new(LabSnapshot::default()));
const LAB_STOPPING_STALE_MS: u64 = 5_000;
const LAB_WINDOW_LABEL: &str = "lab";
const LAB_WINDOW_APP_PATH: &str = "index.html?window=lab";
const LAB_WINDOW_TITLE: &str = "TraduzAi Lab";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabChapterPair {
    pub chapter_number: u32,
    pub source_path: String,
    pub reference_path: String,
    pub source_pages: u32,
    pub reference_pages: u32,
    pub reference_group: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabChapterScope {
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub first_n: Option<u32>,
    #[serde(default)]
    pub start_chapter: Option<u32>,
    #[serde(default)]
    pub end_chapter: Option<u32>,
    #[serde(default)]
    pub chapter_numbers: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabPreferences {
    #[serde(default)]
    pub last_source_dir: String,
    #[serde(default)]
    pub last_reference_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StartLabRequest {
    #[serde(default)]
    pub chapter_scope: LabChapterScope,
    #[serde(default)]
    pub gpu_policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabAgentStatus {
    pub agent_id: String,
    pub label: String,
    pub layer: String,
    pub status: String,
    pub current_task: String,
    pub last_action: String,
    pub confidence: f64,
    #[serde(default)]
    pub touched_domains: Vec<String>,
    #[serde(default)]
    pub proposal_id: String,
    #[serde(default)]
    pub updated_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabReviewFinding {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub body: String,
    #[serde(default)]
    pub severity: String,
    #[serde(default)]
    pub file_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabReviewResult {
    pub proposal_id: String,
    pub reviewer_id: String,
    pub reviewer_label: String,
    pub verdict: String,
    #[serde(default)]
    pub findings: Vec<LabReviewFinding>,
    #[serde(default)]
    pub touched_domains: Vec<String>,
    #[serde(default)]
    pub reviewed_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabBenchmarkMetrics {
    pub textual_similarity: f64,
    pub term_consistency: f64,
    pub layout_occupancy: f64,
    pub readability: f64,
    pub visual_cleanup: f64,
    pub manual_edits_saved: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabBenchmarkResult {
    pub proposal_id: String,
    pub batch_id: String,
    pub score_before: f64,
    pub score_after: f64,
    pub green: bool,
    pub summary: String,
    pub metrics: LabBenchmarkMetrics,
    pub git_available: bool,
    pub pr_status: String,
    #[serde(default)]
    pub generated_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabProposal {
    pub proposal_id: String,
    pub batch_id: String,
    pub title: String,
    pub summary: String,
    pub author: String,
    pub risk: String,
    #[serde(default)]
    pub touched_domains: Vec<String>,
    #[serde(default)]
    pub required_reviewers: Vec<String>,
    #[serde(default)]
    pub review_findings: Vec<LabReviewFinding>,
    #[serde(default)]
    pub integration_verdict: String,
    #[serde(default)]
    pub benchmark_batch_id: String,
    #[serde(default)]
    pub proposal_status: String,
    #[serde(default)]
    pub pr_status: String,
    #[serde(default)]
    pub git_available: bool,
    #[serde(default)]
    pub created_at_ms: u64,
    // Campos do Planner (backwards-compat via serde(default))
    #[serde(default)]
    pub motivation: String,
    #[serde(default)]
    pub target_file: String,
    #[serde(default)]
    pub target_anchor: String,
    #[serde(default)]
    pub change_kind: String,
    #[serde(default)]
    pub needs_coder: bool,
    #[serde(default)]
    pub priority_score: f64,
    #[serde(default)]
    pub issue_type: String,
    #[serde(default)]
    pub local_patch_hint: serde_json::Value,
    #[serde(default)]
    pub expected_metric_gain: serde_json::Value,
    // Preenchido apos coder gerar patch
    #[serde(default)]
    pub patch_proposal: Option<LabPatchProposal>,
}

/// Resultado do coder — unified diff + metadados. Dry-run sempre.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabPatchProposal {
    pub proposal_id: String,
    pub patch_unified_diff: String,
    #[serde(default)]
    pub files_affected: Vec<String>,
    #[serde(default)]
    pub rationale: String,
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub model_used: String,
    #[serde(default)]
    pub generated_at_iso: String,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default)]
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabRunSummary {
    pub run_id: String,
    pub status: String,
    pub summary: String,
    pub total_pairs: u32,
    pub processed_pairs: u32,
    #[serde(default)]
    pub started_at_ms: u64,
    #[serde(default)]
    pub finished_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabSnapshot {
    pub status: String,
    pub run_id: String,
    pub current_stage: String,
    pub message: String,
    #[serde(default)]
    pub acceleration_summary: String,
    pub total_pairs: u32,
    pub processed_pairs: u32,
    pub eta_seconds: f64,
    pub pending_proposals: u32,
    pub active_batch_id: String,
    pub git_available: bool,
    pub pr_ready: bool,
    pub source_dir: String,
    pub reference_dir: String,
    #[serde(default)]
    pub chapter_pairs: Vec<LabChapterPair>,
    #[serde(default)]
    pub available_chapter_pairs: Vec<LabChapterPair>,
    #[serde(default)]
    pub scope_label: String,
    #[serde(default)]
    pub gpu_policy: String,
    #[serde(default)]
    pub agents: Vec<LabAgentStatus>,
    #[serde(default)]
    pub proposals: Vec<LabProposal>,
    #[serde(default)]
    pub reviews: Vec<LabReviewResult>,
    #[serde(default)]
    pub benchmarks: Vec<LabBenchmarkResult>,
    #[serde(default)]
    pub history: Vec<LabRunSummary>,
    #[serde(default)]
    pub updated_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabReferencePreview {
    pub chapter_number: u32,
    pub page_index: u32,
    pub output_path: String,
    pub reference_path: String,
    pub source_path: String,
    pub output_kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabControlResponse {
    pub run_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LabPromotionEvent {
    pub proposal_id: String,
    pub proposal_status: String,
    pub pr_status: String,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LabRunnerConfig {
    pub run_id: String,
    pub source_dir: String,
    pub reference_dir: String,
    pub pause_file: String,
    pub git_available: bool,
    pub vision_worker_path: String,
    pub selected_chapters: Vec<u32>,
    pub scope_label: String,
    pub gpu_policy: String,
}

#[derive(Debug)]
struct LabSidecarInfo {
    program: String,
    script: String,
    cwd: PathBuf,
}

#[derive(Debug, Deserialize, Default)]
struct LabProjectJson {
    #[serde(default)]
    paginas: Vec<LabProjectPage>,
}

#[derive(Debug, Deserialize, Default)]
struct LabProjectPage {
    #[serde(default)]
    arquivo_traduzido: String,
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn project_root() -> Result<PathBuf, String> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    if cwd.join("package.json").exists() && cwd.join("src-tauri").exists() {
        return Ok(cwd);
    }

    if let Some(parent) = cwd.parent() {
        let parent = parent.to_path_buf();
        if parent.join("package.json").exists() && parent.join("src-tauri").exists() {
            return Ok(parent);
        }
    }

    Ok(cwd)
}

fn git_available(root: &Path) -> bool {
    root.join(".git").exists()
}

fn extract_chapter_number(file_name: &str) -> Option<u32> {
    let re = Regex::new(r"(?i)(?:chapter|cap[ií]tulo)\s*(\d+)").expect("regex valido");
    let captures = re.captures(file_name)?;
    captures.get(1)?.as_str().parse::<u32>().ok()
}

fn extract_reference_group(file_name: &str) -> String {
    let stem = Path::new(file_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or_default();

    if let Some((group, _)) = stem.split_once("_Cap") {
        return group.to_string();
    }

    if let Some((group, _)) = stem.split_once("_cap") {
        return group.to_string();
    }

    "Referencia".to_string()
}

fn is_image_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".png")
        || lower.ends_with(".webp")
}

fn count_cbz_pages(path: &Path) -> Result<u32, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(file).map_err(|e| e.to_string())?;
    let mut count = 0_u32;

    for index in 0..archive.len() {
        let entry = archive.by_index(index).map_err(|e| e.to_string())?;
        if is_image_name(entry.name()) {
            count += 1;
        }
    }

    Ok(count)
}

fn discover_reference_pairs(source_dir: &Path, reference_dir: &Path) -> Result<Vec<LabChapterPair>, String> {
    let mut source_by_chapter: BTreeMap<u32, PathBuf> = BTreeMap::new();
    let mut reference_by_chapter: BTreeMap<u32, PathBuf> = BTreeMap::new();

    if source_dir.exists() {
        for entry in std::fs::read_dir(source_dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            let file_name = path.file_name().and_then(|value| value.to_str()).unwrap_or_default();
            if path.extension().and_then(|value| value.to_str()).unwrap_or_default().eq_ignore_ascii_case("cbz") {
                if let Some(chapter) = extract_chapter_number(file_name) {
                    source_by_chapter.insert(chapter, path);
                }
            }
        }
    }

    if reference_dir.exists() {
        for entry in std::fs::read_dir(reference_dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            let file_name = path.file_name().and_then(|value| value.to_str()).unwrap_or_default();
            if path.extension().and_then(|value| value.to_str()).unwrap_or_default().eq_ignore_ascii_case("cbz") {
                if let Some(chapter) = extract_chapter_number(file_name) {
                    reference_by_chapter.insert(chapter, path);
                }
            }
        }
    }

    let mut pairs = Vec::new();
    for (chapter_number, source_path) in source_by_chapter {
        let Some(reference_path) = reference_by_chapter.get(&chapter_number) else {
            continue;
        };

        let reference_name = reference_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_default();

        pairs.push(LabChapterPair {
            chapter_number,
            source_path: source_path.to_string_lossy().to_string(),
            reference_path: reference_path.to_string_lossy().to_string(),
            source_pages: count_cbz_pages(&source_path).unwrap_or(0),
            reference_pages: count_cbz_pages(reference_path).unwrap_or(0),
            reference_group: extract_reference_group(reference_name),
        });
    }

    Ok(pairs)
}

fn normalized_gpu_policy(value: &str) -> String {
    if value.trim().eq_ignore_ascii_case("require_gpu") {
        "require_gpu".into()
    } else {
        "prefer_gpu".into()
    }
}

fn apply_chapter_scope(
    chapter_pairs: &[LabChapterPair],
    scope: &LabChapterScope,
) -> Result<Vec<LabChapterPair>, String> {
    let mode = scope.mode.trim().to_ascii_lowercase();
    if mode.is_empty() || mode == "all" {
        return Ok(chapter_pairs.to_vec());
    }

    match mode.as_str() {
        "first_n" => {
            let count = scope
                .first_n
                .ok_or_else(|| "Informe quantos capitulos iniciais deseja processar.".to_string())?;
            if count == 0 {
                return Err("A quantidade inicial de capitulos precisa ser maior que zero.".into());
            }
            Ok(chapter_pairs.iter().take(count as usize).cloned().collect())
        }
        "range" => {
            let start = scope
                .start_chapter
                .ok_or_else(|| "Informe o capitulo inicial do intervalo.".to_string())?;
            let end = scope
                .end_chapter
                .ok_or_else(|| "Informe o capitulo final do intervalo.".to_string())?;
            if end < start {
                return Err("O capitulo final precisa ser maior ou igual ao inicial.".into());
            }
            Ok(chapter_pairs
                .iter()
                .filter(|pair| pair.chapter_number >= start && pair.chapter_number <= end)
                .cloned()
                .collect())
        }
        "explicit" => {
            if scope.chapter_numbers.is_empty() {
                return Err("Selecione ao menos um capitulo para rodar no modo explicito.".into());
            }
            let wanted: BTreeSet<u32> = scope.chapter_numbers.iter().copied().collect();
            let selected: Vec<LabChapterPair> = chapter_pairs
                .iter()
                .filter(|pair| wanted.contains(&pair.chapter_number))
                .cloned()
                .collect();
            if selected.is_empty() {
                return Err(
                    "Nenhum dos capitulos informados esta disponivel no corpus pareado do Lab."
                        .into(),
                );
            }
            Ok(selected)
        }
        other => Err(format!("Modo de selecao de capitulos nao suportado: {other}")),
    }
}

fn describe_chapter_scope(
    all_pairs: &[LabChapterPair],
    selected_pairs: &[LabChapterPair],
) -> String {
    if selected_pairs.is_empty() {
        return "Nenhum capitulo selecionado".into();
    }
    if selected_pairs.len() == all_pairs.len() {
        return "Todos os capitulos".into();
    }
    if selected_pairs.len() == 1 {
        return format!("Capitulo {}", selected_pairs[0].chapter_number);
    }

    let numbers: Vec<u32> = selected_pairs.iter().map(|pair| pair.chapter_number).collect();
    let contiguous = numbers
        .iter()
        .enumerate()
        .all(|(index, chapter)| *chapter == numbers[0] + index as u32);
    if contiguous {
        return format!("Capitulos {}-{}", numbers[0], numbers[numbers.len() - 1]);
    }

    format!("{} capitulos selecionados", selected_pairs.len())
}

fn should_preserve_runner_error(snapshot: &LabSnapshot) -> bool {
    snapshot.status == "error"
        && !snapshot.message.trim().is_empty()
        && snapshot.message.trim() != "Inicializando Improvement Lab"
}

fn default_lab_dirs() -> Result<(PathBuf, PathBuf, PathBuf, bool), String> {
    let root = project_root()?;
    let has_git = git_available(&root);
    let (source_dir, reference_dir) = resolve_preferred_lab_dirs(&root);
    Ok((root, source_dir, reference_dir, has_git))
}

fn resolve_preferred_lab_dirs(root: &Path) -> (PathBuf, PathBuf) {
    let fallback_source = root.join("exemplos").join("exemploen");
    let fallback_reference = root.join("exemplos").join("exemploptbr");

    let prefs = load_lab_preferences().unwrap_or_default();
    let source = if !prefs.last_source_dir.trim().is_empty()
        && Path::new(&prefs.last_source_dir).exists()
    {
        PathBuf::from(prefs.last_source_dir)
    } else {
        fallback_source
    };
    let reference = if !prefs.last_reference_dir.trim().is_empty()
        && Path::new(&prefs.last_reference_dir).exists()
    {
        PathBuf::from(prefs.last_reference_dir)
    } else {
        fallback_reference
    };
    (source, reference)
}

fn lab_preferences_path() -> PathBuf {
    data_root().join("lab_preferences.json")
}

fn load_lab_preferences() -> Option<LabPreferences> {
    let path = lab_preferences_path();
    if !path.exists() {
        return None;
    }
    let raw = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<LabPreferences>(&raw).ok()
}

fn persist_lab_preferences(prefs: &LabPreferences) -> Result<(), String> {
    let path = lab_preferences_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let payload = serde_json::to_string_pretty(prefs).map_err(|e| e.to_string())?;
    std::fs::write(path, payload).map_err(|e| e.to_string())
}

fn data_root() -> PathBuf {
    PathBuf::from("D:\\traduzai_data").join("lab")
}

fn lab_window_label() -> &'static str {
    LAB_WINDOW_LABEL
}

fn lab_window_app_path() -> &'static str {
    LAB_WINDOW_APP_PATH
}

fn snapshot_path() -> PathBuf {
    data_root().join("snapshot.json")
}

fn run_root(run_id: &str) -> PathBuf {
    data_root().join("runs").join(run_id)
}

fn persist_snapshot(snapshot: &LabSnapshot) -> Result<(), String> {
    let path = snapshot_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let payload = serde_json::to_string_pretty(snapshot).map_err(|e| e.to_string())?;
    std::fs::write(path, payload).map_err(|e| e.to_string())
}

fn load_persisted_snapshot() -> Option<LabSnapshot> {
    let path = snapshot_path();
    if !path.exists() {
        return None;
    }
    let raw = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<LabSnapshot>(&raw).ok()
}

fn reconcile_stale_snapshot(snapshot: &mut LabSnapshot, now: u64) -> bool {
    let terminal_status =
        matches!(snapshot.status.as_str(), "idle" | "stopped" | "completed" | "error");
    if terminal_status && !snapshot.agents.is_empty() {
        snapshot.agents.clear();
        snapshot.updated_at_ms = now;
        return true;
    }

    let stale_stopping = snapshot.status == "stopping"
        && snapshot.updated_at_ms > 0
        && now.saturating_sub(snapshot.updated_at_ms) > LAB_STOPPING_STALE_MS;

    if stale_stopping {
        snapshot.status = "stopped".into();
        snapshot.current_stage = "interrompido".into();
        snapshot.message = "Laboratorio interrompido. Pronto para iniciar nova rodada.".into();
        snapshot.updated_at_ms = now;

        if let Some(history_entry) = snapshot.history.iter_mut().rev().find(|entry| entry.run_id == snapshot.run_id) {
            history_entry.status = "stopped".into();
            history_entry.summary = "Rodada anterior interrompida".into();
            if history_entry.finished_at_ms == 0 {
                history_entry.finished_at_ms = now;
            }
        }

        return true;
    }

    false
}

fn seed_lab_snapshot() -> Result<LabSnapshot, String> {
    let (_root, source_dir, reference_dir, has_git) = default_lab_dirs()?;
    let chapter_pairs = discover_reference_pairs(&source_dir, &reference_dir)?;

    Ok(LabSnapshot {
        status: "idle".into(),
        run_id: String::new(),
        current_stage: "aguardando".into(),
        message: "Laboratorio pronto para iniciar".into(),
        acceleration_summary: String::new(),
        total_pairs: chapter_pairs.len() as u32,
        processed_pairs: 0,
        eta_seconds: 0.0,
        pending_proposals: 0,
        active_batch_id: String::new(),
        git_available: has_git,
        pr_ready: false,
        source_dir: source_dir.to_string_lossy().to_string(),
        reference_dir: reference_dir.to_string_lossy().to_string(),
        chapter_pairs: chapter_pairs.clone(),
        available_chapter_pairs: chapter_pairs,
        scope_label: "Todos os capitulos".into(),
        gpu_policy: "require_gpu".into(),
        agents: vec![],
        proposals: vec![],
        reviews: vec![],
        benchmarks: vec![],
        history: vec![],
        updated_at_ms: now_ms(),
    })
}

async fn ensure_snapshot_seeded() -> Result<LabSnapshot, String> {
    let mut snapshot = LAB_SNAPSHOT.lock().await;
    if snapshot.source_dir.is_empty() {
        *snapshot = load_persisted_snapshot().unwrap_or(seed_lab_snapshot()?);
        if reconcile_stale_snapshot(&mut snapshot, now_ms()) {
            let _ = persist_snapshot(&snapshot);
        }
    } else if let Some(persisted) = load_persisted_snapshot() {
        if persisted.updated_at_ms >= snapshot.updated_at_ms || persisted.run_id != snapshot.run_id {
            *snapshot = persisted;
            if reconcile_stale_snapshot(&mut snapshot, now_ms()) {
                let _ = persist_snapshot(&snapshot);
            }
        }
    }
    if snapshot.available_chapter_pairs.is_empty() {
        snapshot.available_chapter_pairs = snapshot.chapter_pairs.clone();
    }
    if snapshot.scope_label.is_empty() {
        snapshot.scope_label =
            describe_chapter_scope(&snapshot.available_chapter_pairs, &snapshot.chapter_pairs);
    }
    if snapshot.gpu_policy.is_empty() {
        snapshot.gpu_policy = "require_gpu".into();
    }
    Ok(snapshot.clone())
}

fn resolve_lab_output_preview(
    snapshot: &LabSnapshot,
    chapter_number: u32,
    page_index: usize,
) -> Option<PathBuf> {
    if snapshot.run_id.trim().is_empty() {
        return None;
    }

    let project_json_path = run_root(&snapshot.run_id)
        .join("chapters")
        .join(format!("chapter-{chapter_number:04}"))
        .join("output")
        .join("project.json");

    if !project_json_path.exists() {
        return None;
    }

    let raw = std::fs::read_to_string(&project_json_path).ok()?;
    let parsed = serde_json::from_str::<LabProjectJson>(&raw).ok()?;
    let page = parsed.paginas.get(page_index)?;
    if page.arquivo_traduzido.trim().is_empty() {
        return None;
    }

    let candidate = project_json_path
        .parent()?
        .join(page.arquivo_traduzido.replace('/', std::path::MAIN_SEPARATOR_STR));

    if candidate.exists() {
        Some(candidate)
    } else {
        None
    }
}

async fn set_pause_marker(path: &Path, paused: bool) -> Result<(), String> {
    if paused {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, "paused").map_err(|e| e.to_string())
    } else if path.exists() {
        std::fs::remove_file(path).map_err(|e| e.to_string())
    } else {
        Ok(())
    }
}

async fn clear_pause_marker(path: &Path) {
    let mut current = LAB_PAUSE_MARKER.lock().await;
    if current.as_deref() == Some(path) {
        let _ = set_pause_marker(path, false).await;
        *current = None;
    }
}

fn python_program_for_lab() -> Result<String, String> {
    let root = project_root()?;

    #[cfg(windows)]
    let candidate = root
        .join("pipeline")
        .join("venv")
        .join("Scripts")
        .join("python.exe");
    #[cfg(not(windows))]
    let candidate = root
        .join("pipeline")
        .join("venv")
        .join("bin")
        .join("python3");

    if candidate.exists() {
        Ok(candidate.to_string_lossy().to_string())
    } else {
        #[cfg(windows)]
        {
            Ok("python".to_string())
        }
        #[cfg(not(windows))]
        {
            Ok("python3".to_string())
        }
    }
}

fn get_lab_sidecar_info() -> Result<LabSidecarInfo, String> {
    let root = project_root()?;
    Ok(LabSidecarInfo {
        program: python_program_for_lab()?,
        script: root.join("lab").join("runner.py").to_string_lossy().to_string(),
        cwd: root,
    })
}

fn focus_existing_lab_window(app: &AppHandle) -> Result<bool, String> {
    let Some(window) = app.get_webview_window(lab_window_label()) else {
        return Ok(false);
    };

    let _ = window.unminimize();
    let _ = window.show();
    window.set_focus().map_err(|e| e.to_string())?;
    Ok(true)
}

fn build_lab_window(app: &AppHandle) -> Result<(), String> {
    WebviewWindowBuilder::new(
        app,
        lab_window_label(),
        WebviewUrl::App(lab_window_app_path().into()),
    )
    .title(LAB_WINDOW_TITLE)
    .inner_size(1540.0, 940.0)
    .min_inner_size(1180.0, 760.0)
    .resizable(true)
    .center()
    .build()
    .map_err(|e| e.to_string())?;

    Ok(())
}

fn route_reviewers_for_domains_set(touched_domains: &[String]) -> BTreeSet<String> {
    let mut reviewers = BTreeSet::new();
    reviewers.insert("integration_architect".to_string());

    for domain in touched_domains {
        let normalized = domain.to_ascii_lowercase();
        if normalized.starts_with("pipeline/")
            || normalized.starts_with("pipeline**")
            || normalized.contains("pipeline/**")
        {
            reviewers.insert("python_senior_reviewer".to_string());
        }
        if normalized.starts_with("src-tauri/")
            || normalized.starts_with("src-tauri**")
            || normalized.contains("src-tauri/**")
        {
            reviewers.insert("rust_senior_reviewer".to_string());
        }
        if normalized.starts_with("src/")
            || normalized.starts_with("src**")
            || normalized.contains("src/**")
        {
            reviewers.insert("react_ts_senior_reviewer".to_string());
        }
        if normalized.contains("ipc")
            || normalized.contains("event")
            || normalized.contains("artifact")
            || normalized.contains("src/lib/tauri.ts")
            || normalized.contains("tauri_boundary")
        {
            reviewers.insert("tauri_boundary_reviewer".to_string());
        }
    }

    reviewers
}

pub fn route_reviewers_for_domains(touched_domains: &[String]) -> Vec<String> {
    route_reviewers_for_domains_set(touched_domains)
        .into_iter()
        .collect()
}

fn recompute_pending_proposals(snapshot: &mut LabSnapshot) {
    snapshot.pending_proposals = snapshot
        .proposals
        .iter()
        .filter(|proposal| {
            proposal.proposal_status == "needs_approval"
                || proposal.proposal_status == "reviewing"
                || proposal.proposal_status == "benchmark_passed"
        })
        .count() as u32;
}

fn upsert_agent(snapshot: &mut LabSnapshot, agent: LabAgentStatus) {
    if let Some(existing) = snapshot
        .agents
        .iter_mut()
        .find(|candidate| candidate.agent_id == agent.agent_id)
    {
        *existing = agent;
    } else {
        snapshot.agents.push(agent);
    }
}

fn upsert_proposal(snapshot: &mut LabSnapshot, mut proposal: LabProposal) {
    if proposal.required_reviewers.is_empty() {
        proposal.required_reviewers = route_reviewers_for_domains(&proposal.touched_domains);
    }

    if let Some(existing) = snapshot
        .proposals
        .iter_mut()
        .find(|candidate| candidate.proposal_id == proposal.proposal_id)
    {
        *existing = proposal;
    } else {
        snapshot.proposals.push(proposal);
    }
    recompute_pending_proposals(snapshot);
}

fn upsert_review(snapshot: &mut LabSnapshot, review: LabReviewResult) {
    if let Some(existing) = snapshot.reviews.iter_mut().find(|candidate| {
        candidate.proposal_id == review.proposal_id && candidate.reviewer_id == review.reviewer_id
    }) {
        *existing = review.clone();
    } else {
        snapshot.reviews.push(review.clone());
    }

    if let Some(proposal) = snapshot
        .proposals
        .iter_mut()
        .find(|candidate| candidate.proposal_id == review.proposal_id)
    {
        proposal.review_findings.extend(review.findings.clone());
        if review.reviewer_id == "integration_architect" {
            proposal.integration_verdict = review.verdict.clone();
        }
    }
}

fn upsert_benchmark(snapshot: &mut LabSnapshot, benchmark: LabBenchmarkResult) {
    if let Some(existing) = snapshot
        .benchmarks
        .iter_mut()
        .find(|candidate| candidate.proposal_id == benchmark.proposal_id)
    {
        *existing = benchmark.clone();
    } else {
        snapshot.benchmarks.push(benchmark.clone());
    }

    if let Some(proposal) = snapshot
        .proposals
        .iter_mut()
        .find(|candidate| candidate.proposal_id == benchmark.proposal_id)
    {
        proposal.benchmark_batch_id = benchmark.batch_id.clone();
        proposal.proposal_status = if benchmark.green {
            "benchmark_passed".into()
        } else {
            "benchmark_failed".into()
        };
        proposal.pr_status = benchmark.pr_status.clone();
    }

    recompute_pending_proposals(snapshot);
}

fn apply_promotion(snapshot: &mut LabSnapshot, event: &LabPromotionEvent) {
    if let Some(proposal) = snapshot
        .proposals
        .iter_mut()
        .find(|candidate| candidate.proposal_id == event.proposal_id)
    {
        proposal.proposal_status = event.proposal_status.clone();
        proposal.pr_status = event.pr_status.clone();
        snapshot.message = event.summary.clone();
    }
    snapshot.pr_ready = snapshot
        .proposals
        .iter()
        .any(|candidate| candidate.pr_status == "ready_for_local_pr");
    recompute_pending_proposals(snapshot);
}

async fn emit_snapshot(app: &AppHandle) {
    let snapshot = LAB_SNAPSHOT.lock().await.clone();
    let _ = persist_snapshot(&snapshot);
    app.emit("lab_state", snapshot).ok();
}

async fn handle_lab_message(app: &AppHandle, message: serde_json::Value) -> Result<(), String> {
    let message_type = message
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or_default();

    match message_type {
        "lab_state" => {
            let mut snapshot = LAB_SNAPSHOT.lock().await;
            snapshot.status = message["status"].as_str().unwrap_or("running").to_string();
            snapshot.run_id = message["run_id"]
                .as_str()
                .unwrap_or(snapshot.run_id.as_str())
                .to_string();
            snapshot.current_stage = message["current_stage"]
                .as_str()
                .unwrap_or(snapshot.current_stage.as_str())
                .to_string();
            snapshot.message = message["message"]
                .as_str()
                .unwrap_or(snapshot.message.as_str())
                .to_string();
            snapshot.acceleration_summary = message["acceleration_summary"]
                .as_str()
                .unwrap_or(snapshot.acceleration_summary.as_str())
                .to_string();
            snapshot.total_pairs = message["total_pairs"]
                .as_u64()
                .unwrap_or(snapshot.total_pairs as u64) as u32;
            snapshot.processed_pairs = message["processed_pairs"]
                .as_u64()
                .unwrap_or(snapshot.processed_pairs as u64) as u32;
            snapshot.eta_seconds = message["eta_seconds"].as_f64().unwrap_or(snapshot.eta_seconds);
            snapshot.active_batch_id = message["active_batch_id"]
                .as_str()
                .unwrap_or(snapshot.active_batch_id.as_str())
                .to_string();
            snapshot.git_available = message["git_available"]
                .as_bool()
                .unwrap_or(snapshot.git_available);
            snapshot.source_dir = message["source_dir"]
                .as_str()
                .unwrap_or(snapshot.source_dir.as_str())
                .to_string();
            snapshot.reference_dir = message["reference_dir"]
                .as_str()
                .unwrap_or(snapshot.reference_dir.as_str())
                .to_string();
            snapshot.scope_label = message["scope_label"]
                .as_str()
                .unwrap_or(snapshot.scope_label.as_str())
                .to_string();
            snapshot.gpu_policy = message["gpu_policy"]
                .as_str()
                .unwrap_or(snapshot.gpu_policy.as_str())
                .to_string();

            if let Some(chapter_pairs) = message.get("chapter_pairs") {
                if let Ok(parsed) =
                    serde_json::from_value::<Vec<LabChapterPair>>(chapter_pairs.clone())
                {
                    snapshot.chapter_pairs = parsed;
                }
            }

            if let Some(chapter_pairs) = message.get("available_chapter_pairs") {
                if let Ok(parsed) =
                    serde_json::from_value::<Vec<LabChapterPair>>(chapter_pairs.clone())
                {
                    snapshot.available_chapter_pairs = parsed;
                }
            }

            snapshot.updated_at_ms = now_ms();
            let _ = persist_snapshot(&snapshot);
            app.emit("lab_state", snapshot.clone()).ok();
        }
        "agent_status" => {
            let mut agent = serde_json::from_value::<LabAgentStatus>(message["agent"].clone())
                .map_err(|e| e.to_string())?;
            agent.updated_at_ms = now_ms();

            let mut snapshot = LAB_SNAPSHOT.lock().await;
            upsert_agent(&mut snapshot, agent.clone());
            snapshot.updated_at_ms = now_ms();
            let _ = persist_snapshot(&snapshot);
            app.emit("agent_status", agent).ok();
            app.emit("lab_state", snapshot.clone()).ok();
        }
        "review_requested" => {
            let mut proposal = serde_json::from_value::<LabProposal>(message["proposal"].clone())
                .map_err(|e| e.to_string())?;
            proposal.created_at_ms = now_ms();
            if proposal.proposal_status.is_empty() {
                proposal.proposal_status = "reviewing".into();
            }

            let mut snapshot = LAB_SNAPSHOT.lock().await;
            upsert_proposal(&mut snapshot, proposal.clone());
            snapshot.updated_at_ms = now_ms();
            let _ = persist_snapshot(&snapshot);
            app.emit("review_requested", proposal).ok();
            app.emit("lab_state", snapshot.clone()).ok();
        }
        "review_result" => {
            let mut review =
                serde_json::from_value::<LabReviewResult>(message["review"].clone())
                    .map_err(|e| e.to_string())?;
            review.reviewed_at_ms = now_ms();

            let mut snapshot = LAB_SNAPSHOT.lock().await;
            upsert_review(&mut snapshot, review.clone());
            snapshot.updated_at_ms = now_ms();
            let _ = persist_snapshot(&snapshot);
            app.emit("review_result", review).ok();
            app.emit("lab_state", snapshot.clone()).ok();
        }
        "benchmark_result" => {
            let mut benchmark =
                serde_json::from_value::<LabBenchmarkResult>(message["benchmark"].clone())
                    .map_err(|e| e.to_string())?;
            benchmark.generated_at_ms = now_ms();

            let mut snapshot = LAB_SNAPSHOT.lock().await;
            upsert_benchmark(&mut snapshot, benchmark.clone());
            snapshot.updated_at_ms = now_ms();
            let _ = persist_snapshot(&snapshot);
            app.emit("benchmark_result", benchmark).ok();
            app.emit("lab_state", snapshot.clone()).ok();
        }
        "proposal_promoted" => {
            let event =
                serde_json::from_value::<LabPromotionEvent>(message["promotion"].clone())
                    .map_err(|e| e.to_string())?;

            let mut snapshot = LAB_SNAPSHOT.lock().await;
            apply_promotion(&mut snapshot, &event);
            snapshot.updated_at_ms = now_ms();
            let _ = persist_snapshot(&snapshot);
            app.emit("proposal_promoted", event).ok();
            app.emit("lab_state", snapshot.clone()).ok();
        }
        _ => {}
    }

    Ok(())
}

async fn run_lab_sidecar(
    app: &AppHandle,
    sidecar: &LabSidecarInfo,
    config_path: &Path,
) -> Result<(), String> {
    let log_path = config_path
        .parent()
        .unwrap_or(config_path)
        .join("lab.log");
    let log_file = File::create(&log_path).map_err(|e| format!("Erro ao criar log do lab: {e}"))?;

    let mut command = Command::new(&sidecar.program);
    command
        .arg(&sidecar.script)
        .arg(config_path.to_string_lossy().to_string())
        .current_dir(&sidecar.cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::from(log_file));
    for (key, value) in crate::commands::pipeline::sidecar_env_overrides(&sidecar.program) {
        command.env(key, value);
    }

    let mut child = command
        .spawn()
        .map_err(|e| format!("Erro ao iniciar laboratorio: {e}"))?;

    let stdout = child.stdout.take().expect("stdout capturado");
    let mut reader = BufReader::new(stdout).lines();

    while let Ok(Some(line)) = reader.next_line().await {
        if *LAB_CANCEL.lock().await {
            child.kill().await.ok();
            return Err("Laboratorio encerrado manualmente".into());
        }

        if let Ok(message) = serde_json::from_str::<serde_json::Value>(&line) {
            handle_lab_message(app, message).await?;
        }
    }

    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !status.success() {
        let log_content = std::fs::read_to_string(&log_path).unwrap_or_default();
        let detail = if log_content.trim().is_empty() {
            format!("codigo {status}")
        } else if log_content.len() > 2000 {
            log_content[log_content.len() - 2000..].to_string()
        } else {
            log_content
        };
        return Err(format!("Laboratorio encerrou com {detail}"));
    }

    Ok(())
}

fn image_entries_from_archive(path: &Path) -> Result<Vec<String>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(file).map_err(|e| e.to_string())?;
    let mut images = Vec::new();

    for index in 0..archive.len() {
        let entry = archive.by_index(index).map_err(|e| e.to_string())?;
        if is_image_name(entry.name()) {
            images.push(entry.name().to_string());
        }
    }

    images.sort();
    Ok(images)
}

fn extract_page_preview(
    archive_path: &Path,
    entry_name: &str,
    cache_dir: &Path,
    file_prefix: &str,
) -> Result<PathBuf, String> {
    std::fs::create_dir_all(cache_dir).map_err(|e| e.to_string())?;

    let extension = Path::new(entry_name)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("jpg");

    let target_path = cache_dir.join(format!("{file_prefix}.{extension}"));
    if target_path.exists() {
        return Ok(target_path);
    }

    let file = File::open(archive_path).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(file).map_err(|e| e.to_string())?;
    let mut entry = archive.by_name(entry_name).map_err(|e| e.to_string())?;
    let mut output = File::create(&target_path).map_err(|e| e.to_string())?;
    std::io::copy(&mut entry, &mut output).map_err(|e| e.to_string())?;
    Ok(target_path)
}

#[tauri::command]
pub async fn get_lab_state() -> Result<LabSnapshot, String> {
    ensure_snapshot_seeded().await
}

#[tauri::command]
pub async fn open_lab_window(app: AppHandle) -> Result<(), String> {
    if !focus_existing_lab_window(&app)? {
        build_lab_window(&app)?;
    }
    Ok(())
}

#[tauri::command]
pub async fn start_lab(
    app: AppHandle,
    request: Option<StartLabRequest>,
) -> Result<LabControlResponse, String> {
    let mut snapshot = ensure_snapshot_seeded().await?;
    *LAB_CANCEL.lock().await = false;
    let request = request.unwrap_or_default();
    let all_pairs = if snapshot.available_chapter_pairs.is_empty() {
        snapshot.chapter_pairs.clone()
    } else {
        snapshot.available_chapter_pairs.clone()
    };
    let selected_pairs = apply_chapter_scope(&all_pairs, &request.chapter_scope)?;
    if selected_pairs.is_empty() {
        return Err("A selecao atual nao encontrou capitulos pareados no corpus do Lab.".into());
    }
    let scope_label = describe_chapter_scope(&all_pairs, &selected_pairs);
    let gpu_policy = normalized_gpu_policy(&request.gpu_policy);

    let run_id = uuid::Uuid::new_v4().to_string();
    let run_dir = data_root().join("runs").join(&run_id);
    std::fs::create_dir_all(&run_dir).map_err(|e| e.to_string())?;
    let pause_marker = run_dir.join("lab.pause");
    set_pause_marker(&pause_marker, false).await?;

    {
        let mut current = LAB_PAUSE_MARKER.lock().await;
        if let Some(previous) = current.take() {
            let _ = set_pause_marker(&previous, false).await;
        }
        *current = Some(pause_marker.clone());
    }

    snapshot.status = "starting".into();
    snapshot.run_id = run_id.clone();
    snapshot.current_stage = "boot".into();
    snapshot.message = "Inicializando Improvement Lab".into();
    snapshot.chapter_pairs = selected_pairs.clone();
    snapshot.available_chapter_pairs = all_pairs;
    snapshot.scope_label = scope_label.clone();
    snapshot.gpu_policy = gpu_policy.clone();
    snapshot.total_pairs = snapshot.chapter_pairs.len() as u32;
    snapshot.processed_pairs = 0;
    snapshot.pending_proposals = 0;
    snapshot.pr_ready = false;
    snapshot.active_batch_id = format!("batch-{}", &run_id[..8]);
    snapshot.agents.clear();
    // Preserva propostas ainda nao decididas pelo operador (e seus reviews/benchmarks).
    // Propostas ja aceitas/rejeitadas/aplicadas sao descartadas.
    let pending_proposals: Vec<LabProposal> = snapshot
        .proposals
        .drain(..)
        .filter(|p| {
            !matches!(
                p.proposal_status.as_str(),
                "approved" | "rejected" | "patch_applied" | "closed" | "dismissed"
            )
        })
        .collect();
    let pending_ids: BTreeSet<String> = pending_proposals
        .iter()
        .map(|p| p.proposal_id.clone())
        .collect();
    let preserved_reviews: Vec<LabReviewResult> = snapshot
        .reviews
        .drain(..)
        .filter(|r| pending_ids.contains(&r.proposal_id))
        .collect();
    let preserved_benchmarks: Vec<LabBenchmarkResult> = snapshot
        .benchmarks
        .drain(..)
        .filter(|b| pending_ids.contains(&b.proposal_id))
        .collect();
    snapshot.proposals = pending_proposals;
    snapshot.reviews = preserved_reviews;
    snapshot.benchmarks = preserved_benchmarks;
    recompute_pending_proposals(&mut snapshot);
    snapshot.updated_at_ms = now_ms();
    snapshot.history.push(LabRunSummary {
        run_id: run_id.clone(),
        status: "starting".into(),
        summary: format!("Rodada iniciada pelo operador ({scope_label})"),
        total_pairs: snapshot.total_pairs,
        processed_pairs: 0,
        started_at_ms: now_ms(),
        finished_at_ms: 0,
    });

    {
        let mut shared = LAB_SNAPSHOT.lock().await;
        *shared = snapshot.clone();
    }
    let _ = persist_snapshot(&snapshot);
    app.emit("lab_state", snapshot.clone()).ok();

    let vision_worker_path = match crate::commands::pipeline::get_vision_worker_path(&app) {
        Ok(path) => path,
        Err(err) => {
            eprintln!("[TraduzAi Lab] Vision worker indisponivel, seguindo com fallback atual: {err}");
            String::new()
        }
    };

    let config = LabRunnerConfig {
        run_id: run_id.clone(),
        source_dir: snapshot.source_dir.clone(),
        reference_dir: snapshot.reference_dir.clone(),
        pause_file: pause_marker.to_string_lossy().to_string(),
        git_available: snapshot.git_available,
        vision_worker_path,
        selected_chapters: snapshot
            .chapter_pairs
            .iter()
            .map(|pair| pair.chapter_number)
            .collect(),
        scope_label,
        gpu_policy,
    };
    let config_path = run_dir.join("lab_config.json");
    std::fs::write(
        &config_path,
        serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    let sidecar = get_lab_sidecar_info()?;
    let app_clone = app.clone();
    let run_id_clone = run_id.clone();
    let pause_marker_clone = pause_marker.clone();

    tokio::spawn(async move {
        let result = run_lab_sidecar(&app_clone, &sidecar, &config_path).await;
        clear_pause_marker(&pause_marker_clone).await;

        let mut snapshot = LAB_SNAPSHOT.lock().await;
        if let Some(history_entry) = snapshot
            .history
            .iter_mut()
            .rev()
            .find(|entry| entry.run_id == run_id_clone)
        {
            history_entry.finished_at_ms = now_ms();
        }

        match result {
            Ok(()) => {
                if snapshot.status != "completed" {
                    snapshot.status = "completed".into();
                    snapshot.current_stage = "finalizado".into();
                    snapshot.message = "Laboratorio concluido".into();
                }
                snapshot.agents.clear();
                let processed_pairs = snapshot.processed_pairs;
                if let Some(history_entry) = snapshot
                    .history
                    .iter_mut()
                    .rev()
                    .find(|entry| entry.run_id == run_id_clone)
                {
                    history_entry.status = "completed".into();
                    history_entry.summary = "Rodada concluida".into();
                    history_entry.processed_pairs = processed_pairs;
                }
            }
            Err(err) => {
                if *LAB_CANCEL.lock().await {
                    snapshot.status = "stopped".into();
                    snapshot.current_stage = "interrompido".into();
                    snapshot.message = "Laboratorio pausado/encerrado pelo operador".into();
                    snapshot.agents.clear();
                    let processed_pairs = snapshot.processed_pairs;
                    if let Some(history_entry) = snapshot
                        .history
                        .iter_mut()
                        .rev()
                        .find(|entry| entry.run_id == run_id_clone)
                    {
                        history_entry.status = "stopped".into();
                        history_entry.summary = "Rodada interrompida manualmente".into();
                        history_entry.processed_pairs = processed_pairs;
                    }
                } else {
                    if !should_preserve_runner_error(&snapshot) {
                        snapshot.status = "error".into();
                        snapshot.current_stage = "erro".into();
                        snapshot.message = err;
                    }
                    snapshot.agents.clear();
                    let processed_pairs = snapshot.processed_pairs;
                    if let Some(history_entry) = snapshot
                        .history
                        .iter_mut()
                        .rev()
                        .find(|entry| entry.run_id == run_id_clone)
                    {
                        history_entry.status = "error".into();
                        history_entry.summary = "Rodada terminou com erro".into();
                        history_entry.processed_pairs = processed_pairs;
                    }
                }
            }
        }
        snapshot.updated_at_ms = now_ms();
        let _ = persist_snapshot(&snapshot);
        app_clone.emit("lab_state", snapshot.clone()).ok();
    });

    Ok(LabControlResponse { run_id })
}

#[tauri::command]
pub async fn pause_lab() -> Result<(), String> {
    let pause_path = LAB_PAUSE_MARKER
        .lock()
        .await
        .clone()
        .ok_or_else(|| "Nenhum laboratorio em andamento para pausar.".to_string())?;

    set_pause_marker(&pause_path, true).await?;
    let mut snapshot = LAB_SNAPSHOT.lock().await;
    snapshot.status = "paused".into();
    snapshot.current_stage = "pausado".into();
    snapshot.message = "Laboratorio pausado em ponto seguro".into();
    snapshot.updated_at_ms = now_ms();
    let _ = persist_snapshot(&snapshot);
    Ok(())
}

#[tauri::command]
pub async fn resume_lab(app: AppHandle) -> Result<(), String> {
    let pause_path = LAB_PAUSE_MARKER
        .lock()
        .await
        .clone()
        .ok_or_else(|| "Nenhum laboratorio em andamento para continuar.".to_string())?;

    set_pause_marker(&pause_path, false).await?;
    {
        let mut snapshot = LAB_SNAPSHOT.lock().await;
        snapshot.status = "running".into();
        snapshot.current_stage = "retomado".into();
        snapshot.message = "Laboratorio retomado".into();
        snapshot.updated_at_ms = now_ms();
    }
    emit_snapshot(&app).await;
    Ok(())
}

#[tauri::command]
pub async fn stop_lab(app: AppHandle) -> Result<(), String> {
    *LAB_CANCEL.lock().await = true;
    if let Some(path) = LAB_PAUSE_MARKER.lock().await.take() {
        let _ = set_pause_marker(&path, false).await;
    }
    {
        let mut snapshot = LAB_SNAPSHOT.lock().await;
        snapshot.status = "stopping".into();
        snapshot.current_stage = "encerrando".into();
        snapshot.message = "Encerrando laboratorio".into();
        snapshot.updated_at_ms = now_ms();
    }
    emit_snapshot(&app).await;
    Ok(())
}

#[tauri::command]
pub async fn approve_lab_proposal(
    app: AppHandle,
    proposal_id: String,
) -> Result<LabProposal, String> {
    let mut snapshot = LAB_SNAPSHOT.lock().await;
    let benchmark_green = snapshot
        .benchmarks
        .iter()
        .find(|benchmark| benchmark.proposal_id == proposal_id)
        .map(|benchmark| benchmark.green)
        .unwrap_or(false);
    let git_available = snapshot.git_available;

    let proposal_index = snapshot
        .proposals
        .iter()
        .position(|candidate| candidate.proposal_id == proposal_id)
        .ok_or_else(|| "Proposta nao encontrada.".to_string())?;

    if !benchmark_green {
        return Err("A proposta ainda nao possui benchmark verde.".into());
    }
    if snapshot.proposals[proposal_index].integration_verdict != "approve" {
        return Err("A proposta ainda nao passou pelo integration_architect.".into());
    }

    snapshot.proposals[proposal_index].proposal_status = "approved".into();
    snapshot.proposals[proposal_index].pr_status = if git_available {
        "ready_for_local_pr".into()
    } else {
        "blocked_no_git".into()
    };
    snapshot.pr_ready = git_available;
    snapshot.message = format!("Proposta {} aprovada para promocao", proposal_id);
    recompute_pending_proposals(&mut snapshot);
    snapshot.updated_at_ms = now_ms();
    let _ = persist_snapshot(&snapshot);
    let approved_proposal = snapshot.proposals[proposal_index].clone();
    let summary = snapshot.message.clone();

    let event = LabPromotionEvent {
        proposal_id: approved_proposal.proposal_id.clone(),
        proposal_status: approved_proposal.proposal_status.clone(),
        pr_status: approved_proposal.pr_status.clone(),
        summary,
    };

    app.emit("proposal_promoted", event).ok();
    app.emit("lab_state", snapshot.clone()).ok();
    Ok(approved_proposal)
}

#[tauri::command]
pub async fn reject_lab_proposal(
    app: AppHandle,
    proposal_id: String,
) -> Result<LabProposal, String> {
    let mut snapshot = LAB_SNAPSHOT.lock().await;
    let proposal_index = snapshot
        .proposals
        .iter()
        .position(|candidate| candidate.proposal_id == proposal_id)
        .ok_or_else(|| "Proposta nao encontrada.".to_string())?;

    snapshot.proposals[proposal_index].proposal_status = "rejected".into();
    snapshot.proposals[proposal_index].pr_status = "not_applicable".into();
    let mut rejected_proposal = snapshot.proposals[proposal_index].clone();
    rejected_proposal.proposal_status = "rejected".into();
    rejected_proposal.pr_status = "not_applicable".into();

    // Remove a proposta e artefatos associados do snapshot ativo — o operador
    // rejeitou, entao nao deve mais aparecer na UI.
    snapshot.proposals.remove(proposal_index);
    snapshot
        .reviews
        .retain(|review| review.proposal_id != proposal_id);
    snapshot
        .benchmarks
        .retain(|benchmark| benchmark.proposal_id != proposal_id);

    snapshot.message = format!("Proposta {} rejeitada", proposal_id);
    recompute_pending_proposals(&mut snapshot);
    snapshot.updated_at_ms = now_ms();
    let _ = persist_snapshot(&snapshot);
    let summary = snapshot.message.clone();
    app.emit(
        "proposal_promoted",
        LabPromotionEvent {
            proposal_id: rejected_proposal.proposal_id.clone(),
            proposal_status: rejected_proposal.proposal_status.clone(),
            pr_status: rejected_proposal.pr_status.clone(),
            summary,
        },
    )
    .ok();
    app.emit("lab_state", snapshot.clone()).ok();
    Ok(rejected_proposal)
}

#[tauri::command]
pub async fn approve_lab_batch(app: AppHandle, batch_id: String) -> Result<Vec<LabProposal>, String> {
    let proposal_ids = {
        let snapshot = LAB_SNAPSHOT.lock().await;
        snapshot
            .proposals
            .iter()
            .filter(|proposal| proposal.batch_id == batch_id)
            .map(|proposal| proposal.proposal_id.clone())
            .collect::<Vec<_>>()
    };

    if proposal_ids.is_empty() {
        return Err("Nenhuma proposta encontrada para o lote informado.".into());
    }

    let mut approved = Vec::new();
    for proposal_id in proposal_ids {
        if let Ok(proposal) = approve_lab_proposal(app.clone(), proposal_id).await {
            approved.push(proposal);
        }
    }

    if approved.is_empty() {
        return Err("Nenhuma proposta do lote ficou elegivel para promocao.".into());
    }

    Ok(approved)
}

#[tauri::command]
pub async fn get_lab_reference_preview(
    chapter_number: u32,
    page_index: u32,
) -> Result<LabReferencePreview, String> {
    let snapshot = ensure_snapshot_seeded().await?;
    let pair = snapshot
        .chapter_pairs
        .iter()
        .find(|candidate| candidate.chapter_number == chapter_number)
        .cloned()
        .or_else(|| {
            snapshot
                .available_chapter_pairs
                .iter()
                .find(|candidate| candidate.chapter_number == chapter_number)
                .cloned()
        })
        .ok_or_else(|| "Capitulo nao encontrado no corpus do Lab.".to_string())?;

    let cache_dir = data_root().join("preview-cache").join(format!("chapter-{chapter_number}"));
    let source_entries = image_entries_from_archive(Path::new(&pair.source_path))?;
    let reference_entries = image_entries_from_archive(Path::new(&pair.reference_path))?;
    let page_index = page_index as usize;

    if source_entries.is_empty() || reference_entries.is_empty() {
        return Err("Nao foi possivel localizar paginas nos CBZs selecionados.".into());
    }

    let source_entry = source_entries
        .get(page_index.min(source_entries.len().saturating_sub(1)))
        .ok_or_else(|| "Pagina de origem indisponivel.".to_string())?;
    let reference_entry = reference_entries
        .get(page_index.min(reference_entries.len().saturating_sub(1)))
        .ok_or_else(|| "Pagina de referencia indisponivel.".to_string())?;

    let source_preview = extract_page_preview(
        Path::new(&pair.source_path),
        source_entry,
        &cache_dir,
        &format!("chapter-{chapter_number}-page-{page_index}-source"),
    )?;
    let reference_preview = extract_page_preview(
        Path::new(&pair.reference_path),
        reference_entry,
        &cache_dir,
        &format!("chapter-{chapter_number}-page-{page_index}-reference"),
    )?;

    let (output_path, output_kind) = if let Some(real_output) =
        resolve_lab_output_preview(&snapshot, chapter_number, page_index)
    {
        (real_output.to_string_lossy().to_string(), "lab_output".to_string())
    } else {
        (
            source_preview.to_string_lossy().to_string(),
            "source_fallback".to_string(),
        )
    };

    Ok(LabReferencePreview {
        chapter_number,
        page_index: page_index as u32,
        output_path,
        reference_path: reference_preview.to_string_lossy().to_string(),
        source_path: source_preview.to_string_lossy().to_string(),
        output_kind,
    })
}

#[tauri::command]
pub async fn pick_lab_source_dir(app: AppHandle) -> Result<Option<String>, String> {
    Ok(app
        .dialog()
        .file()
        .blocking_pick_folder()
        .map(|value| value.to_string()))
}

#[tauri::command]
pub async fn pick_lab_reference_dir(app: AppHandle) -> Result<Option<String>, String> {
    Ok(app
        .dialog()
        .file()
        .blocking_pick_folder()
        .map(|value| value.to_string()))
}

#[tauri::command]
pub async fn pick_lab_source_files(app: AppHandle) -> Result<Vec<String>, String> {
    let files = app
        .dialog()
        .file()
        .add_filter("CBZ/ZIP", &["cbz", "zip"])
        .blocking_pick_files()
        .unwrap_or_default();
    Ok(files.into_iter().map(|file| file.to_string()).collect())
}

#[tauri::command]
pub async fn pick_lab_reference_files(app: AppHandle) -> Result<Vec<String>, String> {
    let files = app
        .dialog()
        .file()
        .add_filter("CBZ/ZIP", &["cbz", "zip"])
        .blocking_pick_files()
        .unwrap_or_default();
    Ok(files.into_iter().map(|file| file.to_string()).collect())
}

#[tauri::command]
pub async fn set_lab_dirs(
    source_dir: String,
    reference_dir: String,
) -> Result<LabSnapshot, String> {
    let source_path = crate::commands::project::normalize_path(&source_dir);
    let reference_path = crate::commands::project::normalize_path(&reference_dir);

    if !source_path.exists() {
        return Err(format!(
            "Pasta de origem (EN) nao encontrada: {}",
            source_path.to_string_lossy()
        ));
    }
    if !reference_path.exists() {
        return Err(format!(
            "Pasta de referencia (PT-BR) nao encontrada: {}",
            reference_path.to_string_lossy()
        ));
    }

    let chapter_pairs = discover_reference_pairs(&source_path, &reference_path)?;

    let prefs = LabPreferences {
        last_source_dir: source_path.to_string_lossy().to_string(),
        last_reference_dir: reference_path.to_string_lossy().to_string(),
    };
    persist_lab_preferences(&prefs)?;

    let mut snapshot = LAB_SNAPSHOT.lock().await;
    snapshot.source_dir = source_path.to_string_lossy().to_string();
    snapshot.reference_dir = reference_path.to_string_lossy().to_string();
    snapshot.chapter_pairs = chapter_pairs.clone();
    snapshot.available_chapter_pairs = chapter_pairs.clone();
    snapshot.total_pairs = chapter_pairs.len() as u32;
    snapshot.scope_label =
        describe_chapter_scope(&snapshot.available_chapter_pairs, &snapshot.chapter_pairs);
    if snapshot.status.is_empty() {
        snapshot.status = "idle".into();
    }
    if snapshot.current_stage.is_empty() {
        snapshot.current_stage = "aguardando".into();
    }
    if snapshot.message.is_empty() || snapshot.message == "Inicializando Improvement Lab" {
        snapshot.message = "Corpus do Lab atualizado".into();
    }
    snapshot.updated_at_ms = now_ms();
    let _ = persist_snapshot(&snapshot);
    Ok(snapshot.clone())
}

// ---------------------------------------------------------------------------
// propose_lab_patch — invoca coder Python para gerar diff de uma proposta
// ---------------------------------------------------------------------------

/// Pequeno script Python inlined que carrega o coder certo e devolve o
/// PatchProposal como JSON para o Rust.
const PATCH_BRIDGE_SCRIPT: &str = r#"
import sys, json
from pathlib import Path

args = json.loads(sys.argv[1])
proposal = args["proposal"]
coder_strategy = args.get("coder_strategy", "local")
repo_root = Path(args.get("repo_root", ".")).resolve()
sys.path.insert(0, str(repo_root))

if coder_strategy == "ollama":
    from lab.coders.ollama_coder import OllamaCoder
    coder = OllamaCoder(host=args.get("ollama_host", "http://localhost:11434"))
elif coder_strategy == "claude_code":
    from lab.coders.claude_code_coder import ClaudeCodeCoder
    coder = ClaudeCodeCoder()
elif coder_strategy == "claude_sdk":
    from lab.agents.coder_agent import ClaudeSDKCoder
    coder = ClaudeSDKCoder()
else:
    # "local" — apenas build_local_patch_from_hint, sem LLM
    from lab.coders.base import build_local_patch_from_hint, PatchProposal
    result = build_local_patch_from_hint(proposal, repo_root)
    if result is None:
        result = PatchProposal(
            proposal_id=proposal.get("proposal_id","?"),
            patch_unified_diff="",
            error="change_kind nao suportado por patch local; use coder ollama ou claude_code.",
        )
    print(json.dumps(result.to_dict(), ensure_ascii=False))
    sys.exit(0)

result = coder.propose_patch(proposal, repo_root)
print(json.dumps(result.to_dict(), ensure_ascii=False))
"#;

fn find_proposals_json(run_dir: &Path) -> Option<Vec<serde_json::Value>> {
    let path = run_dir.join("proposals.json");
    if path.exists() {
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(arr) = obj.get("proposals").and_then(|v| v.as_array()) {
                    return Some(arr.clone());
                }
            }
        }
    }
    // Fallback: proposal.json singular
    let single = run_dir.join("proposal.json");
    if single.exists() {
        if let Ok(text) = std::fs::read_to_string(&single) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&text) {
                return Some(vec![val]);
            }
        }
    }
    None
}

fn find_run_dir_for_proposal(proposal_id: &str) -> Option<PathBuf> {
    let lab_root = PathBuf::from("D:/traduzai_data/lab");
    let runs_dir = lab_root.join("runs");
    if !runs_dir.exists() {
        return None;
    }
    // Percorre runs mais recentes primeiro (sort desc por nome)
    let mut run_dirs: Vec<PathBuf> = std::fs::read_dir(&runs_dir)
        .ok()?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    run_dirs.sort_by(|a, b| b.cmp(a));

    for run_dir in run_dirs {
        if let Some(proposals) = find_proposals_json(&run_dir) {
            for p in &proposals {
                if p.get("proposal_id").and_then(|v| v.as_str()) == Some(proposal_id) {
                    return Some(run_dir);
                }
            }
        }
    }
    None
}

#[tauri::command]
pub async fn propose_lab_patch(
    proposal_id: String,
    coder_strategy: Option<String>,
    ollama_host: Option<String>,
) -> Result<LabPatchProposal, String> {
    let strategy = coder_strategy.unwrap_or_else(|| "local".to_string());
    let host = ollama_host.unwrap_or_else(|| "http://localhost:11434".to_string());

    // Localiza a proposta no disco
    let run_dir = find_run_dir_for_proposal(&proposal_id)
        .ok_or_else(|| format!("Proposta '{}' nao encontrada em nenhuma rodada do Lab.", proposal_id))?;

    let proposals = find_proposals_json(&run_dir)
        .ok_or_else(|| format!("proposals.json nao encontrado em {:?}", run_dir))?;

    let proposal_val = proposals
        .into_iter()
        .find(|p| p.get("proposal_id").and_then(|v| v.as_str()) == Some(&proposal_id))
        .ok_or_else(|| format!("proposal_id '{}' nao encontrado.", proposal_id))?;

    let root = project_root()?;

    // Monta args para o script bridge
    let bridge_args = serde_json::json!({
        "proposal": proposal_val,
        "coder_strategy": strategy,
        "repo_root": root.to_string_lossy(),
        "ollama_host": host,
    });
    let args_json = serde_json::to_string(&bridge_args).map_err(|e| e.to_string())?;

    let sidecar = get_lab_sidecar_info()?;

    // Escreve script em arquivo temp (evita problemas com quotes no Windows)
    let script_path = std::env::temp_dir().join(format!("lab_patch_bridge_{}.py", now_ms()));
    std::fs::write(&script_path, PATCH_BRIDGE_SCRIPT).map_err(|e| e.to_string())?;

    let output = tokio::process::Command::new(&sidecar.program)
        .arg(script_path.to_string_lossy().as_ref())
        .arg(&args_json)
        .current_dir(&sidecar.cwd)
        .output()
        .await
        .map_err(|e| format!("Falha ao iniciar coder: {e}"))?;

    let _ = std::fs::remove_file(&script_path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "Coder falhou (exit {}): {}",
            output.status.code().unwrap_or(-1),
            &stderr[stderr.len().saturating_sub(800)..],
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let patch: LabPatchProposal = serde_json::from_str(stdout.trim())
        .map_err(|e| format!("Resposta invalida do coder: {e}\n---\n{}", &stdout[..stdout.len().min(500)]))?;

    // Salva patch no disco para referencia futura
    let patch_path = run_dir.join(format!("patch_{}.json", proposal_id));
    if let Ok(patch_json) = serde_json::to_string_pretty(&patch) {
        let _ = std::fs::write(&patch_path, patch_json.as_bytes());
    }

    // Actualiza proposta no snapshot em memoria
    {
        let mut snapshot = LAB_SNAPSHOT.lock().await;
        for proposal_entry in snapshot.proposals.iter_mut() {
            if proposal_entry.proposal_id == proposal_id {
                proposal_entry.patch_proposal = Some(patch.clone());
                break;
            }
        }
        snapshot.updated_at_ms = now_ms();
        let _ = persist_snapshot(&snapshot);
    }

    Ok(patch)
}

// ---------------------------------------------------------------------------
// apply_lab_patch — aplica o unified diff via git apply e cria branch/commit
// ---------------------------------------------------------------------------

/// Resultado da aplicacao de um patch.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabPatchApplyResult {
    pub proposal_id: String,
    /// true se o patch foi aplicado com sucesso
    pub applied: bool,
    /// Nome da branch criada (vazio se create_branch=false ou git indisponivel)
    pub branch_created: String,
    /// SHA do commit criado (vazio se sem commit ou git indisponivel)
    pub commit_sha: String,
    /// Mensagem de erro (vazio se sucesso)
    pub error: String,
    /// Arquivos tocados (extraido do diff)
    pub files_patched: Vec<String>,
}

/// Extrai os caminhos dos arquivos afetados de um unified diff.
/// Linhas `+++ b/<path>` indicam o arquivo destino.
fn extract_files_from_diff(diff: &str) -> Vec<String> {
    let mut files: Vec<String> = Vec::new();
    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("+++ b/") {
            let path = rest.trim().to_string();
            if !path.is_empty() && path != "/dev/null" && !files.contains(&path) {
                files.push(path);
            }
        }
    }
    files
}

/// Sanitiza uma string para uso como nome de branch git (sem chars invalidos).
fn sanitize_branch_name(raw: &str) -> String {
    raw.chars()
        .map(|ch| {
            if ch.is_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

/// Gera nome de branch para a proposta: `lab/proposal-{id_curto}`.
pub fn branch_name_for_proposal(proposal_id: &str) -> String {
    // proposal_id tipico: "proposal-abcd1234-00"
    // Produz: "lab/proposal-abcd1234-00"
    let safe = sanitize_branch_name(proposal_id);
    format!("lab/{safe}")
}

async fn run_git(args: &[&str], cwd: &Path) -> Result<String, String> {
    let output = tokio::process::Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .await
        .map_err(|e| format!("git nao encontrado: {e}"))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(stderr.trim().to_string())
    }
}

#[tauri::command]
pub async fn export_lab_patch_json(
    output_path: String,
    content: String,
) -> Result<String, String> {
    let path = PathBuf::from(&output_path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Falha ao criar pasta de destino: {e}"))?;
        }
    }
    std::fs::write(&path, content.as_bytes())
        .map_err(|e| format!("Falha ao gravar {}: {e}", path.display()))?;
    Ok(path.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn apply_lab_patch(
    proposal_id: String,
    patch_unified_diff: String,
    create_branch: bool,
    commit: bool,
    commit_message: Option<String>,
) -> Result<LabPatchApplyResult, String> {
    if patch_unified_diff.trim().is_empty() {
        return Ok(LabPatchApplyResult {
            proposal_id,
            error: "Nenhum diff fornecido.".into(),
            ..Default::default()
        });
    }

    let root = project_root()?;
    let is_git = git_available(&root);
    let files_patched = extract_files_from_diff(&patch_unified_diff);

    // Sem git: escreve os arquivos diretamente aplicando o diff linha por linha
    if !is_git {
        match apply_diff_without_git(&patch_unified_diff, &root) {
            Ok(()) => {
                return Ok(LabPatchApplyResult {
                    proposal_id,
                    applied: true,
                    files_patched,
                    ..Default::default()
                });
            }
            Err(e) => {
                return Ok(LabPatchApplyResult {
                    proposal_id,
                    error: format!("Aplicacao manual do diff falhou: {e}"),
                    files_patched,
                    ..Default::default()
                });
            }
        }
    }

    // Com git disponivel
    // 1. Verifica se ha mudancas nao comitadas que conflitariam
    let status_out = run_git(&["status", "--porcelain"], &root).await.unwrap_or_default();
    let _has_unstaged = status_out.lines().any(|l| l.starts_with(" M") || l.starts_with("??"));

    // 2. Cria branch se solicitado
    let mut branch_created = String::new();
    if create_branch {
        let branch = branch_name_for_proposal(&proposal_id);
        match run_git(&["checkout", "-b", &branch], &root).await {
            Ok(_) => branch_created = branch,
            Err(e) => {
                // Branch ja existe? Tenta usar sem criar
                if e.contains("already exists") {
                    if let Ok(_) = run_git(&["checkout", &branch], &root).await {
                        branch_created = branch;
                    } else {
                        return Ok(LabPatchApplyResult {
                            proposal_id,
                            error: format!("Nao foi possivel criar ou acessar branch: {e}"),
                            files_patched,
                            ..Default::default()
                        });
                    }
                } else {
                    return Ok(LabPatchApplyResult {
                        proposal_id,
                        error: format!("git checkout -b falhou: {e}"),
                        files_patched,
                        ..Default::default()
                    });
                }
            }
        }
    }

    // 3. Escreve diff em arquivo temp e aplica
    let patch_tmp = std::env::temp_dir().join(format!("lab_patch_{}.diff", now_ms()));
    std::fs::write(&patch_tmp, patch_unified_diff.as_bytes())
        .map_err(|e| format!("Falha ao escrever patch temp: {e}"))?;

    // git apply --check primeiro (valida sem modificar)
    let check_result = run_git(
        &["apply", "--check", patch_tmp.to_string_lossy().as_ref()],
        &root,
    )
    .await;

    if let Err(e) = check_result {
        let _ = std::fs::remove_file(&patch_tmp);
        // Desfaz branch se criamos
        if !branch_created.is_empty() {
            let _ = run_git(&["checkout", "-"], &root).await;
            let _ = run_git(&["branch", "-D", &branch_created], &root).await;
        }
        return Ok(LabPatchApplyResult {
            proposal_id,
            branch_created,
            error: format!("git apply --check falhou (diff invalido ou conflito): {e}"),
            files_patched,
            ..Default::default()
        });
    }

    // Aplica de verdade
    let apply_result = run_git(
        &["apply", patch_tmp.to_string_lossy().as_ref()],
        &root,
    )
    .await;
    let _ = std::fs::remove_file(&patch_tmp);

    if let Err(e) = apply_result {
        if !branch_created.is_empty() {
            let _ = run_git(&["checkout", "-"], &root).await;
            let _ = run_git(&["branch", "-D", &branch_created], &root).await;
        }
        return Ok(LabPatchApplyResult {
            proposal_id,
            branch_created,
            error: format!("git apply falhou: {e}"),
            files_patched,
            ..Default::default()
        });
    }

    // 4. Commit opcional
    let mut commit_sha = String::new();
    if commit {
        // Stage os arquivos tocados
        let _ = run_git(&["add", "--all"], &root).await;

        let msg = commit_message.unwrap_or_else(|| {
            format!(
                "Lab: patch automatico para {}\n\nGerado por TraduzAi Lab (dry-run aprovado pelo usuario).",
                proposal_id
            )
        });
        match run_git(&["commit", "-m", &msg], &root).await {
            Ok(_) => {
                commit_sha = run_git(&["rev-parse", "--short", "HEAD"], &root)
                    .await
                    .unwrap_or_default();
            }
            Err(e) => {
                // Patch ja aplicado mas commit falhou — nao e fatal
                return Ok(LabPatchApplyResult {
                    proposal_id,
                    applied: true,
                    branch_created,
                    error: format!("Patch aplicado mas commit falhou: {e}"),
                    files_patched,
                    ..Default::default()
                });
            }
        }
    }

    // 5. Atualiza snapshot em memoria
    {
        let mut snapshot = LAB_SNAPSHOT.lock().await;
        for p in snapshot.proposals.iter_mut() {
            if p.proposal_id == proposal_id {
                p.proposal_status = "patch_applied".into();
                if let Some(ref mut pp) = p.patch_proposal {
                    pp.dry_run = false;
                }
                break;
            }
        }
        snapshot.updated_at_ms = now_ms();
        let _ = persist_snapshot(&snapshot);
    }

    Ok(LabPatchApplyResult {
        proposal_id,
        applied: true,
        branch_created,
        commit_sha,
        files_patched,
        ..Default::default()
    })
}

/// Aplica um unified diff manualmente (sem git) lendo e reescrevendo cada arquivo.
/// Suporta apenas diffs simples (sem fuzzy matching). Retorna Err se qualquer
/// hunk falhar.
fn apply_diff_without_git(diff: &str, repo_root: &Path) -> Result<(), String> {
    let mut current_file: Option<PathBuf> = None;
    let mut in_hunk = false;

    // Primeiro passo: coleta arquivos e seus conteudos originais
    struct FilePatch {
        path: PathBuf,
        hunks: Vec<(usize, Vec<String>, Vec<String>)>, // (orig_start, removes, adds)
    }

    let mut file_patches: Vec<FilePatch> = Vec::new();
    let mut cur_removes: Vec<String> = Vec::new();
    let mut cur_adds: Vec<String> = Vec::new();
    let mut cur_hunk_start: usize = 0;

    for line in diff.lines() {
        if line.starts_with("+++ b/") {
            // Novo arquivo
            if let Some(ref path) = current_file {
                if !cur_removes.is_empty() || !cur_adds.is_empty() {
                    if let Some(fp) = file_patches.iter_mut().find(|f| &f.path == path) {
                        fp.hunks.push((cur_hunk_start, cur_removes.clone(), cur_adds.clone()));
                    }
                    cur_removes.clear();
                    cur_adds.clear();
                }
            }
            let rel = line.trim_start_matches("+++ b/").trim();
            let abs_path = repo_root.join(rel);
            current_file = Some(abs_path.clone());
            file_patches.push(FilePatch { path: abs_path, hunks: Vec::new() });
            in_hunk = false;
        } else if line.starts_with("@@ ") {
            // Flush hunk anterior
            if !cur_removes.is_empty() || !cur_adds.is_empty() {
                if let Some(ref path) = current_file {
                    if let Some(fp) = file_patches.iter_mut().find(|f| &f.path == path) {
                        fp.hunks.push((cur_hunk_start, cur_removes.clone(), cur_adds.clone()));
                    }
                }
                cur_removes.clear();
                cur_adds.clear();
            }
            // Parse "@@ -L,N +L,N @@"
            let parts: Vec<&str> = line.splitn(5, ' ').collect();
            if parts.len() >= 2 {
                let orig_part = parts[1].trim_start_matches('-');
                cur_hunk_start = orig_part.split(',').next()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(1)
                    .saturating_sub(1); // converte para 0-indexed
            }
            in_hunk = true;
        } else if in_hunk {
            if line.starts_with('-') && !line.starts_with("---") {
                cur_removes.push(line[1..].to_string());
            } else if line.starts_with('+') && !line.starts_with("+++") {
                cur_adds.push(line[1..].to_string());
            }
        }
    }
    // Flush ultimo hunk
    if !cur_removes.is_empty() || !cur_adds.is_empty() {
        if let Some(ref path) = current_file {
            if let Some(fp) = file_patches.iter_mut().find(|f| &f.path == path) {
                fp.hunks.push((cur_hunk_start, cur_removes.clone(), cur_adds.clone()));
            }
        }
    }

    // Segundo passo: aplica hunks em cada arquivo
    for fp in file_patches {
        if !fp.path.exists() {
            return Err(format!("Arquivo nao encontrado: {}", fp.path.display()));
        }
        let original = std::fs::read_to_string(&fp.path)
            .map_err(|e| format!("Falha ao ler {}: {e}", fp.path.display()))?;
        let mut file_lines: Vec<String> = original.lines().map(|l| l.to_string()).collect();

        // Aplica hunks de tras pra frente para preservar indices
        let mut hunks_sorted = fp.hunks;
        hunks_sorted.sort_by(|a, b| b.0.cmp(&a.0));

        for (start, removes, adds) in hunks_sorted {
            let end = start + removes.len();
            if end > file_lines.len() {
                return Err(format!(
                    "Hunk fora dos limites em {} (linha {end} > {})",
                    fp.path.display(), file_lines.len()
                ));
            }
            // Verifica contexto
            let existing = &file_lines[start..end];
            for (i, (exp, got)) in removes.iter().zip(existing.iter()).enumerate() {
                if exp != got {
                    return Err(format!(
                        "Contexto nao bate em {} linha {}: esperado {:?}, encontrado {:?}",
                        fp.path.display(), start + i + 1, exp, got
                    ));
                }
            }
            file_lines.splice(start..end, adds.into_iter());
        }

        let new_content = file_lines.join("\n") + "\n";
        std::fs::write(&fp.path, new_content.as_bytes())
            .map_err(|e| format!("Falha ao escrever {}: {e}", fp.path.display()))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sort(mut values: Vec<String>) -> Vec<String> {
        values.sort();
        values
    }

    #[test]
    fn chapter_scope_all_returns_everything() {
        let pairs = vec![
            LabChapterPair {
                chapter_number: 1,
                ..LabChapterPair::default()
            },
            LabChapterPair {
                chapter_number: 2,
                ..LabChapterPair::default()
            },
            LabChapterPair {
                chapter_number: 3,
                ..LabChapterPair::default()
            },
        ];

        let scoped = apply_chapter_scope(
            &pairs,
            &LabChapterScope {
                mode: "all".into(),
                ..LabChapterScope::default()
            },
        )
        .expect("scope all");

        assert_eq!(scoped.len(), 3);
        assert_eq!(describe_chapter_scope(&pairs, &scoped), "Todos os capitulos");
    }

    #[test]
    fn chapter_scope_first_n_limits_the_run() {
        let pairs = vec![
            LabChapterPair {
                chapter_number: 1,
                ..LabChapterPair::default()
            },
            LabChapterPair {
                chapter_number: 2,
                ..LabChapterPair::default()
            },
            LabChapterPair {
                chapter_number: 3,
                ..LabChapterPair::default()
            },
        ];

        let scoped = apply_chapter_scope(
            &pairs,
            &LabChapterScope {
                mode: "first_n".into(),
                first_n: Some(2),
                ..LabChapterScope::default()
            },
        )
        .expect("scope first_n");

        assert_eq!(
            scoped.iter().map(|pair| pair.chapter_number).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(describe_chapter_scope(&pairs, &scoped), "Capitulos 1-2");
    }

    #[test]
    fn chapter_scope_explicit_selects_arbitrary_chapters() {
        let pairs = vec![
            LabChapterPair {
                chapter_number: 1,
                ..LabChapterPair::default()
            },
            LabChapterPair {
                chapter_number: 5,
                ..LabChapterPair::default()
            },
            LabChapterPair {
                chapter_number: 23,
                ..LabChapterPair::default()
            },
            LabChapterPair {
                chapter_number: 80,
                ..LabChapterPair::default()
            },
        ];

        let scoped = apply_chapter_scope(
            &pairs,
            &LabChapterScope {
                mode: "explicit".into(),
                chapter_numbers: vec![5, 23, 999],
                ..LabChapterScope::default()
            },
        )
        .expect("scope explicit");

        assert_eq!(
            scoped.iter().map(|pair| pair.chapter_number).collect::<Vec<_>>(),
            vec![5, 23]
        );
    }

    #[test]
    fn chapter_scope_explicit_requires_numbers() {
        let pairs = vec![LabChapterPair {
            chapter_number: 1,
            ..LabChapterPair::default()
        }];

        let result = apply_chapter_scope(
            &pairs,
            &LabChapterScope {
                mode: "explicit".into(),
                ..LabChapterScope::default()
            },
        );

        assert!(result.is_err());
    }

    #[test]
    fn chapter_scope_explicit_errors_when_no_match() {
        let pairs = vec![LabChapterPair {
            chapter_number: 1,
            ..LabChapterPair::default()
        }];

        let result = apply_chapter_scope(
            &pairs,
            &LabChapterScope {
                mode: "explicit".into(),
                chapter_numbers: vec![999],
                ..LabChapterScope::default()
            },
        );

        assert!(result.is_err());
    }

    #[test]
    fn chapter_scope_range_selects_contiguous_chapters() {
        let pairs = vec![
            LabChapterPair {
                chapter_number: 1,
                ..LabChapterPair::default()
            },
            LabChapterPair {
                chapter_number: 2,
                ..LabChapterPair::default()
            },
            LabChapterPair {
                chapter_number: 3,
                ..LabChapterPair::default()
            },
            LabChapterPair {
                chapter_number: 4,
                ..LabChapterPair::default()
            },
        ];

        let scoped = apply_chapter_scope(
            &pairs,
            &LabChapterScope {
                mode: "range".into(),
                start_chapter: Some(2),
                end_chapter: Some(3),
                ..LabChapterScope::default()
            },
        )
        .expect("scope range");

        assert_eq!(
            scoped.iter().map(|pair| pair.chapter_number).collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert_eq!(describe_chapter_scope(&pairs, &scoped), "Capitulos 2-3");
    }

    #[test]
    fn preserve_runner_error_keeps_detailed_lab_message() {
        let snapshot = LabSnapshot {
            status: "error".into(),
            current_stage: "gpu_guard".into(),
            message: "ONNX GPU indisponivel: o inpainting ainda cairia para CPU.".into(),
            ..LabSnapshot::default()
        };

        assert!(should_preserve_runner_error(&snapshot));
    }

    #[test]
    fn preserve_runner_error_ignores_boot_message() {
        let snapshot = LabSnapshot {
            status: "starting".into(),
            current_stage: "boot".into(),
            message: "Inicializando Improvement Lab".into(),
            ..LabSnapshot::default()
        };

        assert!(!should_preserve_runner_error(&snapshot));
    }

    #[test]
    fn pipeline_domains_require_python_and_integrator() {
        let reviewers = route_reviewers_for_domains(&["pipeline/**".into()]);
        assert_eq!(
            sort(reviewers),
            sort(vec![
                "integration_architect".into(),
                "python_senior_reviewer".into(),
            ])
        );
    }

    #[test]
    fn src_tauri_domains_require_rust_and_integrator() {
        let reviewers = route_reviewers_for_domains(&["src-tauri/**".into()]);
        assert_eq!(
            sort(reviewers),
            sort(vec![
                "integration_architect".into(),
                "rust_senior_reviewer".into(),
            ])
        );
    }

    #[test]
    fn src_domains_require_react_and_integrator() {
        let reviewers = route_reviewers_for_domains(&["src/**".into()]);
        assert_eq!(
            sort(reviewers),
            sort(vec![
                "integration_architect".into(),
                "react_ts_senior_reviewer".into(),
            ])
        );
    }

    #[test]
    fn tauri_boundary_domains_require_boundary_reviewer() {
        let reviewers = route_reviewers_for_domains(&[
            "src/**".into(),
            "ipc_contract".into(),
            "events".into(),
            "artifacts".into(),
        ]);

        assert_eq!(
            sort(reviewers),
            sort(vec![
                "integration_architect".into(),
                "react_ts_senior_reviewer".into(),
                "tauri_boundary_reviewer".into(),
            ])
        );
    }

    #[test]
    fn mixed_domains_require_all_specialists_once() {
        let reviewers = route_reviewers_for_domains(&[
            "pipeline/**".into(),
            "src-tauri/**".into(),
            "src/**".into(),
            "ipc_contract".into(),
        ]);

        assert_eq!(
            sort(reviewers),
            sort(vec![
                "integration_architect".into(),
                "python_senior_reviewer".into(),
                "react_ts_senior_reviewer".into(),
                "rust_senior_reviewer".into(),
                "tauri_boundary_reviewer".into(),
            ])
        );
    }

    #[test]
    fn lab_window_uses_dedicated_label() {
        assert_eq!(lab_window_label(), "lab");
    }

    #[test]
    fn lab_window_url_targets_standalone_shell() {
        assert_eq!(lab_window_app_path(), "index.html?window=lab");
    }

    #[test]
    fn stale_stopping_snapshot_is_reconciled() {
        let mut snapshot = LabSnapshot {
            status: "stopping".into(),
            run_id: "run-123".into(),
            current_stage: "encerrando".into(),
            message: "Encerrando laboratorio".into(),
            updated_at_ms: 10_000,
            history: vec![LabRunSummary {
                run_id: "run-123".into(),
                status: "stopping".into(),
                summary: "Encerrando".into(),
                total_pairs: 83,
                processed_pairs: 0,
                started_at_ms: 5_000,
                finished_at_ms: 0,
            }],
            ..LabSnapshot::default()
        };

        let changed = reconcile_stale_snapshot(
            &mut snapshot,
            10_000 + LAB_STOPPING_STALE_MS + 1,
        );

        assert!(changed);
        assert_eq!(snapshot.status, "stopped");
        assert_eq!(snapshot.current_stage, "interrompido");
        assert!(snapshot.message.contains("Pronto para iniciar"));
        assert_eq!(snapshot.history[0].status, "stopped");
        assert!(snapshot.history[0].finished_at_ms > 0);
    }

    #[test]
    fn terminal_snapshot_clears_stale_agents() {
        let mut snapshot = LabSnapshot {
            status: "error".into(),
            run_id: "run-123".into(),
            current_stage: "erro".into(),
            message: "Falha detalhada".into(),
            updated_at_ms: 10_000,
            agents: vec![LabAgentStatus {
                agent_id: "runtime_orchestrator".into(),
                label: "Runtime Orchestrator".into(),
                layer: "runtime".into(),
                status: "running".into(),
                current_task: "Processando".into(),
                last_action: "Capitulo 14".into(),
                confidence: 0.7,
                touched_domains: vec![],
                proposal_id: String::new(),
                updated_at_ms: 9_000,
            }],
            ..LabSnapshot::default()
        };

        let changed = reconcile_stale_snapshot(&mut snapshot, 10_500);

        assert!(changed);
        assert!(snapshot.agents.is_empty());
        assert_eq!(snapshot.status, "error");
    }

    #[test]
    fn chapter_number_parser_handles_english_and_ptbr_names() {
        assert_eq!(extract_chapter_number("Chapter 10_5b99a4.cbz"), Some(10));
        assert_eq!(
            extract_chapter_number("ArinVale_Capítulo 10_e04b44.cbz"),
            Some(10)
        );
        assert_eq!(
            extract_chapter_number("WorldScan_Capítulo 80_e73190.cbz"),
            Some(80)
        );
    }

    #[test]
    fn reference_group_parser_uses_scanlator_prefix() {
        assert_eq!(
            extract_reference_group("ArinVale_Capítulo 10_e04b44.cbz"),
            "ArinVale"
        );
        assert_eq!(
            extract_reference_group("WorldScan_Capítulo 80_e73190.cbz"),
            "WorldScan"
        );
    }

    // --- apply_lab_patch helpers ---

    #[test]
    fn branch_name_uses_proposal_id() {
        assert_eq!(
            branch_name_for_proposal("proposal-abcd1234-00"),
            "lab/proposal-abcd1234-00"
        );
    }

    #[test]
    fn branch_name_sanitizes_special_chars() {
        let branch = branch_name_for_proposal("proposal/abc def@2!");
        assert!(!branch.contains(' '));
        assert!(!branch.contains('@'));
        assert!(branch.starts_with("lab/"));
    }

    #[test]
    fn extract_files_from_diff_parses_plus_plus_lines() {
        let diff = "\
--- a/pipeline/ocr/postprocess.py\n\
+++ b/pipeline/ocr/postprocess.py\n\
@@ -1,3 +1,3 @@\n\
 x = 1\n\
-y = 2\n\
+y = 99\n";
        let files = extract_files_from_diff(diff);
        assert_eq!(files, vec!["pipeline/ocr/postprocess.py"]);
    }

    #[test]
    fn extract_files_from_diff_deduplicates() {
        let diff = "\
+++ b/pipeline/a.py\n\
+++ b/pipeline/a.py\n\
+++ b/pipeline/b.py\n";
        let files = extract_files_from_diff(diff);
        assert_eq!(files.len(), 2);
        assert!(files.contains(&"pipeline/a.py".to_string()));
        assert!(files.contains(&"pipeline/b.py".to_string()));
    }

    #[test]
    fn extract_files_ignores_dev_null() {
        let diff = "+++ /dev/null\n+++ b/real/file.py\n";
        let files = extract_files_from_diff(diff);
        assert_eq!(files, vec!["real/file.py"]);
    }

    #[test]
    fn sanitize_branch_name_strips_leading_trailing_dashes() {
        let result = sanitize_branch_name("--my-branch--");
        assert!(!result.starts_with('-'));
        assert!(!result.ends_with('-'));
    }

    #[test]
    fn apply_diff_without_git_patches_simple_file() {
        use std::fs;
        let dir = std::env::temp_dir().join(format!("lab_test_{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("f.py");
        fs::write(&file_path, "a = 1\nb = 2\nc = 3\n").unwrap();

        let diff = format!(
            "--- a/f.py\n+++ b/f.py\n@@ -2,1 +2,1 @@\n-b = 2\n+b = 99\n"
        );
        apply_diff_without_git(&diff, &dir).unwrap();
        let result = fs::read_to_string(&file_path).unwrap();
        assert!(result.contains("b = 99"), "esperava b = 99, obteve: {result}");
        assert!(!result.contains("b = 2"), "nao deveria conter b = 2");
    }

    #[test]
    fn patch_apply_result_default_is_not_applied() {
        let result = LabPatchApplyResult::default();
        assert!(!result.applied);
        assert!(result.branch_created.is_empty());
        assert!(result.commit_sha.is_empty());
    }
}
