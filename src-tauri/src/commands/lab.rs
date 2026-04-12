use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tauri::{AppHandle, Emitter, Manager, WebviewUrl, WebviewWindowBuilder};
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
    pub title: String,
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
    let source_dir = root.join("exemplos").join("exemploen");
    let reference_dir = root.join("exemplos").join("exemploptbr");
    Ok((root.clone(), source_dir, reference_dir, git_available(&root)))
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
    snapshot.proposals.clear();
    snapshot.reviews.clear();
    snapshot.benchmarks.clear();
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
    snapshot.message = format!("Proposta {} rejeitada", proposal_id);
    recompute_pending_proposals(&mut snapshot);
    snapshot.updated_at_ms = now_ms();
    let _ = persist_snapshot(&snapshot);
    let rejected_proposal = snapshot.proposals[proposal_index].clone();
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
}
