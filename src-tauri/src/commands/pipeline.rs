#![allow(dead_code)]
// TraduzAi Pipeline v0.54.1 - Backend Stabilized
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

#[allow(dead_code)]
static PIPELINE_ACTIVE: once_cell::sync::Lazy<Mutex<bool>> =
    once_cell::sync::Lazy::new(|| Mutex::new(false));
#[allow(dead_code)]
static PIPELINE_CANCEL: once_cell::sync::Lazy<Mutex<bool>> =
    once_cell::sync::Lazy::new(|| Mutex::new(false));
#[allow(dead_code)]
static PIPELINE_PAUSE_MARKER: once_cell::sync::Lazy<Mutex<Option<std::path::PathBuf>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(None));
#[allow(dead_code)]
static VISUAL_WARMUP_STATE: once_cell::sync::Lazy<Mutex<VisualWarmupState>> =
    once_cell::sync::Lazy::new(|| Mutex::new(VisualWarmupState::Idle));
#[allow(dead_code)]
static VISION_WORKER_WARMUP_STATE: once_cell::sync::Lazy<Mutex<VisualWarmupState>> =
    once_cell::sync::Lazy::new(|| Mutex::new(VisualWarmupState::Idle));
#[allow(dead_code)]
static FAST_PAGE_WORKER: once_cell::sync::Lazy<Mutex<Option<FastPageWorker>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(None));
#[allow(dead_code)]
static FAST_PAGE_INPAINT_WARMUP_STATE: once_cell::sync::Lazy<Mutex<VisualWarmupState>> =
    once_cell::sync::Lazy::new(|| Mutex::new(VisualWarmupState::Idle));

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisualWarmupState {
    Idle,
    Running,
    Ready,
}

fn set_pause_marker(path: &std::path::Path, paused: bool) -> Result<(), String> {
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

async fn clear_current_pause_marker(path: &std::path::Path) {
    let mut current = PIPELINE_PAUSE_MARKER.lock().await;
    if current.as_deref() == Some(path) {
        set_pause_marker(path, false).ok();
        *current = None;
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct PipelineProgress {
    pub step: String,
    pub step_progress: f64,
    pub overall_progress: f64,
    pub current_page: u32,
    pub total_pages: u32,
    pub message: String,
    pub eta_seconds: f64,
}

#[derive(Debug, Deserialize)]
pub struct PipelineConfig {
    pub source_path: String,
    pub obra: String,
    pub capitulo: u32,
    pub idioma_origem: String,
    pub idioma_destino: String,
    pub qualidade: String,
    pub glossario: HashMap<String, String>,
    pub contexto: PipelineContext,
    pub mode: String,
    #[serde(default)]
    pub work_context: Option<PipelineWorkContextSummary>,
    #[serde(default)]
    pub preset: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct PipelineContext {
    pub sinopse: String,
    pub genero: Vec<String>,
    pub personagens: Vec<String>,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub termos: Vec<String>,
    #[serde(default)]
    pub relacoes: Vec<String>,
    #[serde(default)]
    pub faccoes: Vec<String>,
    #[serde(default)]
    pub resumo_por_arco: Vec<String>,
    #[serde(default)]
    pub memoria_lexical: HashMap<String, String>,
    #[serde(default)]
    pub fontes_usadas: Vec<PipelineContextSource>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PipelineContextSource {
    pub fonte: String,
    pub titulo: String,
    pub url: String,
    pub trecho: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct PipelineWorkContextSummary {
    pub selected: bool,
    #[serde(default)]
    pub work_id: String,
    #[serde(default)]
    pub title: String,
    pub context_loaded: bool,
    pub glossary_loaded: bool,
    pub glossary_entries_count: u32,
    #[serde(default)]
    pub cover_url: String,
    pub risk_level: String,
    pub user_ignored_warning: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct WorkSearchCandidate {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub synopsis: String,
    #[serde(default)]
    pub genres: Vec<String>,
    #[serde(default)]
    pub characters: Vec<String>,
    pub source: String,
    #[serde(default)]
    pub source_url: String,
    #[serde(default)]
    pub cover_url: String,
    pub score: f64,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
struct SearchSnippet {
    title: String,
    link: String,
    snippet: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ContextSourceRef {
    pub source: String,
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Debug, Clone)]
struct ExternalContextCandidate {
    kind: String,
    source: String,
    target: String,
    confidence: f64,
    sources: Vec<String>,
    status: String,
    protect: bool,
    aliases: Vec<String>,
    forbidden: Vec<String>,
    notes: String,
}

#[derive(Debug, Clone)]
struct ExternalContextData {
    source: String,
    status: String,
    confidence: f64,
    title: String,
    synopsis: String,
    genres: Vec<String>,
    candidates: Vec<ExternalContextCandidate>,
    url: String,
    error: String,
    cover_url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct EnrichedWorkContext {
    #[serde(default)]
    pub work_id: String,
    pub title: String,
    #[serde(default)]
    pub synopsis: String,
    #[serde(default)]
    pub genres: Vec<String>,
    #[serde(default)]
    pub characters: Vec<String>,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub terms: Vec<String>,
    #[serde(default)]
    pub relationships: Vec<String>,
    #[serde(default)]
    pub factions: Vec<String>,
    #[serde(default)]
    pub arc_summaries: Vec<String>,
    #[serde(default)]
    pub lexical_memory: HashMap<String, String>,
    #[serde(default)]
    pub sources_used: Vec<ContextSourceRef>,
    #[serde(default)]
    pub cover_url: String,
    #[serde(default)]
    pub context_quality: String,
    #[serde(default)]
    pub risk_level: String,
    #[serde(default)]
    pub glossary_entries_count: u32,
    #[serde(default)]
    pub internet_context_loaded: bool,
    #[serde(default)]
    pub source_results: Vec<serde_json::Value>,
    #[serde(default)]
    pub glossary_candidates: Vec<serde_json::Value>,
}

pub(crate) struct SidecarInfo {
    pub(crate) program: String,
    pub(crate) script: Option<String>,
}

struct FastPageWorker {
    signature: String,
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

#[derive(Debug)]
enum FastPageRunError {
    Worker(String),
    Action(String),
}

fn is_project_summary_mismatch(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    lowered.contains("log.summary") && lowered.contains("project.json")
}

async fn stop_fast_page_worker(reason: &str) {
    let mut guard = FAST_PAGE_WORKER.lock().await;
    if let Some(mut worker) = guard.take() {
        eprintln!("[FastPageWorker] reiniciando worker: {reason}");
        let _ = worker.child.kill().await;
    }
    drop(guard);
    *VISUAL_WARMUP_STATE.lock().await = VisualWarmupState::Idle;
    *FAST_PAGE_INPAINT_WARMUP_STATE.lock().await = VisualWarmupState::Idle;
}

impl FastPageWorker {
    async fn request(
        &mut self,
        request: &serde_json::Value,
    ) -> Result<Vec<serde_json::Value>, String> {
        let raw = serde_json::to_string(request)
            .map_err(|e| format!("falha ao serializar request do fast worker: {e}"))?;
        self.stdin
            .write_all(raw.as_bytes())
            .await
            .map_err(|e| format!("falha ao escrever no fast worker: {e}"))?;
        self.stdin
            .write_all(b"\n")
            .await
            .map_err(|e| format!("falha ao finalizar request do fast worker: {e}"))?;
        self.stdin
            .flush()
            .await
            .map_err(|e| format!("falha ao flush do fast worker: {e}"))?;

        let mut events = Vec::new();
        loop {
            let mut line = String::new();
            let read = self
                .stdout
                .read_line(&mut line)
                .await
                .map_err(|e| format!("falha ao ler resposta do fast worker: {e}"))?;
            if read == 0 {
                return Err("fast worker encerrou sem resposta".into());
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let event: serde_json::Value = match serde_json::from_str(trimmed) {
                Ok(event) => event,
                Err(error) => {
                    eprintln!("[FastPageWorker] ignorando stdout nao JSON: {error} | {trimmed}");
                    continue;
                }
            };
            let is_terminal = matches!(
                event.get("type").and_then(|value| value.as_str()),
                Some("complete" | "error" | "ready" | "bye")
            );
            events.push(event);
            if is_terminal {
                return Ok(events);
            }
        }
    }
}

fn normalize_text(input: &str) -> String {
    input
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn get_sidecar_info(app: &AppHandle) -> Result<SidecarInfo, String> {
    if cfg!(debug_assertions) {
        let root = std::env::current_dir()
            .map_err(|e| e.to_string())?
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap());

        let script = root.join("pipeline/main.py");
        #[cfg(windows)]
        let venv_python = root.join("pipeline/venv/Scripts/python.exe");
        #[cfg(not(windows))]
        let venv_python = root.join("pipeline/venv/bin/python3");

        let program = if venv_python.exists() {
            venv_python.to_string_lossy().to_string()
        } else {
            #[cfg(windows)]
            {
                "python".to_string()
            }
            #[cfg(not(windows))]
            {
                "python3".to_string()
            }
        };

        return Ok(SidecarInfo {
            program,
            script: Some(script.to_string_lossy().to_string()),
        });
    }

    let res = app.path().resource_dir().map_err(|e| e.to_string())?;
    let mut sidecar = res.join("binaries/traduzai-pipeline");
    #[cfg(windows)]
    {
        sidecar = sidecar.with_extension("exe");
    }

    Ok(SidecarInfo {
        program: sidecar.to_string_lossy().to_string(),
        script: None,
    })
}

pub(crate) fn get_vision_worker_path(app: &AppHandle) -> Result<String, String> {
    if cfg!(debug_assertions) {
        let root = std::env::current_dir()
            .map_err(|e| e.to_string())?
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap());

        #[cfg(windows)]
        let candidates = [
            root.join("vision-worker/target/debug/traduzai-vision.exe"),
            root.join("vision-worker/target/release/traduzai-vision.exe"),
            root.join("src-tauri/binaries/traduzai-vision.exe"),
        ];
        #[cfg(not(windows))]
        let candidates = [
            root.join("vision-worker/target/debug/traduzai-vision"),
            root.join("vision-worker/target/release/traduzai-vision"),
            root.join("src-tauri/binaries/traduzai-vision"),
        ];

        if let Some(found) = candidates.into_iter().find(|path| path.exists()) {
            return Ok(found.to_string_lossy().to_string());
        }
    }

    let res = app.path().resource_dir().map_err(|e| e.to_string())?;
    let bin = res.join("binaries").join(if cfg!(windows) {
        "traduzai-vision.exe"
    } else {
        "traduzai-vision"
    });
    if bin.exists() {
        Ok(bin.to_string_lossy().to_string())
    } else {
        Ok(String::new())
    }
}

pub(crate) fn sidecar_env_overrides(_program: &str) -> Vec<(String, OsString)> {
    let mut envs = Vec::new();
    envs.push(("PYTHONIOENCODING".into(), OsString::from("utf-8")));
    envs.push(("PYTHONUTF8".into(), OsString::from("1")));
    envs
}

pub(crate) fn apply_sidecar_env(cmd: &mut Command, program: &str) -> Result<(), String> {
    for (k, v) in sidecar_env_overrides(program) {
        cmd.env(k, v);
    }
    Ok(())
}

fn resolve_project_json_path(raw_path: &str) -> Result<std::path::PathBuf, String> {
    let base = crate::commands::project::normalize_path(raw_path);
    let project_file = if base
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("project.json"))
    {
        base
    } else {
        base.join("project.json")
    };

    if !project_file.exists() {
        return Err("project.json não encontrado".into());
    }

    Ok(project_file)
}

#[tauri::command]
pub async fn start_pipeline(
    app: AppHandle,
    config: PipelineConfig,
) -> Result<serde_json::Value, String> {
    *PIPELINE_CANCEL.lock().await = false;
    let job_id = uuid::Uuid::new_v4().to_string();
    let storage = crate::storage::service_for_app(&app)?;
    let storage_paths = storage.ensure_base_dirs()?;
    storage.check_writable()?;
    crate::storage::set_configured_paths(storage_paths.clone());
    let work_dir = storage_paths.works.join(&job_id);
    std::fs::create_dir_all(&work_dir).map_err(|e| e.to_string())?;

    let pause_path = work_dir.join("pipeline.pause");
    set_pause_marker(&pause_path, false)?;
    {
        let mut cur = PIPELINE_PAUSE_MARKER.lock().await;
        if let Some(p) = cur.take() {
            set_pause_marker(&p, false).ok();
        }
        *cur = Some(pause_path.clone());
    }

    let worker_path = get_vision_worker_path(&app).unwrap_or_default();

    let config_json = serde_json::to_string(&serde_json::json!({
        "job_id": job_id, "source_path": config.source_path, "work_dir": work_dir.to_string_lossy(),
        "obra": config.obra, "capitulo": config.capitulo, "idioma_origem": config.idioma_origem,
        "idioma_destino": config.idioma_destino, "qualidade": config.qualidade, "glossario": config.glossario,
        "mode": config.mode,
        "contexto": {
            "sinopse": config.contexto.sinopse, "genero": config.contexto.genero, "personagens": config.contexto.personagens,
            "aliases": config.contexto.aliases, "termos": config.contexto.termos, "relacoes": config.contexto.relacoes,
            "faccoes": config.contexto.faccoes, "resumo_por_arco": config.contexto.resumo_por_arco,
            "memoria_lexical": config.contexto.memoria_lexical, "fontes_usadas": config.contexto.fontes_usadas
        },
        "work_context": config.work_context,
        "preset": config.preset,
        "models_dir": storage_paths.models.to_string_lossy(),
        "logs_dir": storage_paths.logs.to_string_lossy(),
        "vision_worker_path": worker_path, "pause_file": pause_path.to_string_lossy()
    })).map_err(|e| e.to_string())?;

    let config_file = work_dir.join("pipeline_config.json");
    std::fs::write(&config_file, &config_json).map_err(|e| e.to_string())?;

    let sidecar = get_sidecar_info(&app)?;
    let app_c = app.clone();
    let job_c = job_id.clone();
    let pause_c = pause_path.clone();

    tokio::spawn(async move {
        let res = run_sidecar(&app_c, &sidecar, &config_file).await;
        clear_current_pause_marker(&pause_c).await;
        match res {
            Ok(out) => {
                app_c
                    .emit(
                        "pipeline-complete",
                        serde_json::json!({"success": true, "output_path": out, "job_id": job_c}),
                    )
                    .ok();
            }
            Err(err) => {
                app_c
                    .emit(
                        "pipeline-complete",
                        serde_json::json!({"success": false, "error": err, "job_id": job_c}),
                    )
                    .ok();
            }
        }
    });

    Ok(serde_json::json!({ "job_id": job_id }))
}

async fn run_sidecar(
    app: &AppHandle,
    sidecar: &SidecarInfo,
    config_path: &std::path::Path,
) -> Result<String, String> {
    let mut cmd = Command::new(&sidecar.program);
    if let Some(s) = &sidecar.script {
        cmd.arg(s);
    }
    let log_path = config_path
        .parent()
        .unwrap_or(config_path)
        .join("pipeline.log");
    let log_file =
        std::fs::File::create(&log_path).map_err(|e| format!("Erro ao criar log: {e}"))?;
    cmd.arg(config_path.to_string_lossy().to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::from(log_file));
    apply_sidecar_env(&mut cmd, &sidecar.program)?;
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Erro ao iniciar pipeline: {e}"))?;
    let stdout = child.stdout.take().expect("stdout not captured");
    let mut reader = BufReader::new(stdout).lines();
    let mut output_path = String::new();
    while let Ok(Some(line)) = reader.next_line().await {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(msg_type) = msg.get("type").and_then(|t| t.as_str()) {
                match msg_type {
                    "progress" => {
                        let prog = PipelineProgress {
                            step: msg["step"].as_str().unwrap_or("").to_string(),
                            step_progress: msg["step_progress"].as_f64().unwrap_or(0.0),
                            overall_progress: msg["overall_progress"].as_f64().unwrap_or(0.0),
                            current_page: msg["current_page"].as_u64().unwrap_or(0) as u32,
                            total_pages: msg["total_pages"].as_u64().unwrap_or(0) as u32,
                            message: msg["message"].as_str().unwrap_or("").to_string(),
                            eta_seconds: msg["eta_seconds"].as_f64().unwrap_or(0.0),
                        };
                        app.emit("pipeline-progress", prog).ok();
                    }
                    "complete" => {
                        output_path = msg["output_path"].as_str().unwrap_or("").to_string();
                    }
                    "error" => {
                        return Err(msg["message"]
                            .as_str()
                            .unwrap_or("Erro desconhecido")
                            .to_string());
                    }
                    _ => {}
                }
            }
        }
    }
    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !status.success() {
        let log = std::fs::read_to_string(&log_path).unwrap_or_default();
        let detail = if log.len() > 2000 {
            &log[log.len() - 2000..]
        } else {
            &log
        };
        return Err(format!("Pipeline falhou ({}): {}", status, detail));
    }
    Ok(output_path)
}

#[tauri::command]
pub async fn cancel_pipeline() -> Result<(), String> {
    *PIPELINE_CANCEL.lock().await = true;
    if let Some(path) = PIPELINE_PAUSE_MARKER.lock().await.take() {
        set_pause_marker(&path, false)?;
    }
    Ok(())
}

#[tauri::command]
pub async fn pause_pipeline() -> Result<(), String> {
    let path = PIPELINE_PAUSE_MARKER
        .lock()
        .await
        .clone()
        .ok_or_else(|| "Nenhuma tradução em andamento.".to_string())?;
    set_pause_marker(&path, true)?;
    Ok(())
}

#[tauri::command]
pub async fn resume_pipeline() -> Result<(), String> {
    let path = PIPELINE_PAUSE_MARKER
        .lock()
        .await
        .clone()
        .ok_or_else(|| "Nenhuma tradução em andamento.".to_string())?;
    set_pause_marker(&path, false)?;
    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct RetypesetConfig {
    pub project_path: String,
    pub page_index: u32,
}
#[derive(Debug, Deserialize)]
pub struct RenderPreviewConfig {
    pub project_path: String,
    pub page_index: u32,
    pub page: serde_json::Value,
    pub fingerprint: String,
}
#[derive(Debug, Deserialize)]
pub struct ReinpaintConfig {
    pub project_path: String,
    pub page_index: u32,
    #[serde(default)]
    pub bbox: Option<[i64; 4]>,
    #[serde(default)]
    pub mask_path: Option<String>,
}
#[derive(Debug, Deserialize)]
pub struct ProcessBlockConfig {
    pub project_path: String,
    pub page_index: u32,
    pub block_id: String,
    pub mode: String,
}

#[derive(Debug, Clone)]
pub struct PageRegionConfig {
    pub bbox: Option<[u32; 4]>,
    pub mask_path: Option<String>,
}

fn append_page_region_args(cmd: &mut Command, region: Option<&PageRegionConfig>) {
    let Some(region) = region else {
        return;
    };
    if let Some(bbox) = region.bbox {
        cmd.arg("--region-bbox")
            .arg(format!("{},{},{},{}", bbox[0], bbox[1], bbox[2], bbox[3]));
    }
    if let Some(mask_path) = region
        .mask_path
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        cmd.arg("--external-mask").arg(mask_path);
    }
}

fn page_region_to_json(region: Option<&PageRegionConfig>) -> serde_json::Value {
    match region {
        Some(region) => serde_json::json!({
            "bbox": region.bbox,
            "mask_path": region.mask_path,
        }),
        None => serde_json::Value::Null,
    }
}

fn safe_render_preview_key(raw: &str) -> String {
    let mut key = raw
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    key.truncate(48);
    let key = key.trim_matches('_').to_string();
    if key.is_empty() {
        "preview".to_string()
    } else {
        key
    }
}

fn safe_extension(raw: &str) -> String {
    let ext = raw
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    if ext.is_empty() {
        "jpg".to_string()
    } else {
        ext
    }
}

fn render_preview_paths(
    project_file: &Path,
    page_index: u32,
    fingerprint: &str,
    extension_hint: &str,
) -> Result<(PathBuf, PathBuf), String> {
    let project_dir = project_file
        .parent()
        .ok_or_else(|| "Caminho do project.json inválido".to_string())?;
    let cache_dir = project_dir.join("render-cache").join("preview");
    std::fs::create_dir_all(&cache_dir)
        .map_err(|e| format!("Erro ao preparar cache de preview: {e}"))?;
    let stem = format!(
        "{:03}-{}",
        page_index + 1,
        safe_render_preview_key(fingerprint)
    );
    let override_path = cache_dir.join(format!("{stem}.json"));
    let output_path = cache_dir.join(format!("{stem}.{}", safe_extension(extension_hint)));
    Ok((override_path, output_path))
}

fn render_preview_extension_hint(page: &serde_json::Value) -> String {
    let source = page
        .get("image_layers")
        .and_then(|layers| layers.get("base"))
        .and_then(|layer| layer.get("path"))
        .and_then(|path| path.as_str())
        .or_else(|| page.get("arquivo_original").and_then(|path| path.as_str()))
        .or_else(|| page.get("arquivo_traduzido").and_then(|path| path.as_str()))
        .unwrap_or("preview.jpg");

    Path::new(source)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(safe_extension)
        .unwrap_or_else(|| "jpg".to_string())
}

/// Inicia captura assíncrona do stderr do sidecar Python para que crashes antes
/// de emitir JSON de erro fiquem visíveis nos logs e nas mensagens de erro.
///
/// Retorna `(child, JoinHandle<String>)`. Use `await` no JoinHandle no caminho
/// de erro/sucesso para anexar o stderr capturado à mensagem.
fn spawn_with_stderr_capture(
    mut cmd: Command,
    label: &'static str,
) -> Result<(Child, JoinHandle<String>), String> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("[EditorAction] erro ao iniciar {label}: {e}"))?;
    let stderr = child
        .stderr
        .take()
        .expect("stderr deveria estar capturado (Stdio::piped)");
    let handle = tokio::spawn(async move {
        let mut buf = String::new();
        let _ = BufReader::new(stderr).read_to_string(&mut buf).await;
        buf
    });
    Ok((child, handle))
}

/// Coleta o stderr capturado e formata erro consolidado.
async fn collect_stderr(handle: JoinHandle<String>) -> String {
    handle.await.unwrap_or_default()
}

/// Formata mensagem de erro com stderr anexado quando há conteúdo relevante.
fn format_pipeline_error(action: &str, base: &str, stderr: &str) -> String {
    let trimmed = stderr.trim();
    if trimmed.is_empty() {
        format!("[EditorAction] {action} falhou: {base}")
    } else {
        // Limita stderr a 4KB para evitar mensagens enormes na UI.
        let truncated = if trimmed.len() > 4096 {
            format!(
                "{}\n... (truncado, {} bytes total)",
                &trimmed[trimmed.len() - 4096..],
                trimmed.len()
            )
        } else {
            trimmed.to_string()
        };
        format!("[EditorAction] {action} falhou: {base}\n--- stderr Python ---\n{truncated}")
    }
}

fn fast_page_worker_signature(sidecar: &SidecarInfo) -> String {
    format!(
        "{}|{}",
        sidecar.program,
        sidecar.script.as_deref().unwrap_or("")
    )
}

async fn spawn_fast_page_worker(
    sidecar: &SidecarInfo,
    signature: String,
    models_dir: &Path,
) -> Result<FastPageWorker, String> {
    let mut cmd = Command::new(&sidecar.program);
    if let Some(script) = &sidecar.script {
        cmd.arg(script);
    }
    cmd.arg("--serve-fast-page")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("TRADUZAI_MODELS_DIR", models_dir);
    apply_sidecar_env(&mut cmd, &sidecar.program)?;

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("erro ao iniciar fast page worker: {e}"))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "fast page worker iniciou sem stdin".to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "fast page worker iniciou sem stdout".to_string())?;
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                eprintln!("[FastPageWorker] {line}");
            }
        });
    }

    Ok(FastPageWorker {
        signature,
        child,
        stdin,
        stdout: BufReader::new(stdout),
    })
}

async fn run_fast_page_worker_request(
    app: &AppHandle,
    request: serde_json::Value,
) -> Result<Vec<serde_json::Value>, String> {
    let sidecar = get_sidecar_info(app)?;
    let storage = crate::storage::service_for_app(app)?;
    let models_dir = storage.ensure_base_dirs()?.models;
    let signature = format!(
        "{}|models={}",
        fast_page_worker_signature(&sidecar),
        models_dir.to_string_lossy()
    );
    let mut guard = FAST_PAGE_WORKER.lock().await;
    let should_restart = match guard.as_mut() {
        Some(worker) if worker.signature == signature => match worker.child.try_wait() {
            Ok(Some(status)) => {
                eprintln!("[FastPageWorker] worker finalizado antes do request: {status}");
                true
            }
            Ok(None) => false,
            Err(error) => {
                eprintln!("[FastPageWorker] falha ao consultar worker: {error}");
                true
            }
        },
        Some(_) => true,
        None => true,
    };

    if should_restart {
        if let Some(mut worker) = guard.take() {
            let _ = worker.child.kill().await;
        }
        *guard = Some(spawn_fast_page_worker(&sidecar, signature, &models_dir).await?);
    }

    let worker = guard
        .as_mut()
        .ok_or_else(|| "fast page worker indisponivel".to_string())?;
    match worker.request(&request).await {
        Ok(events) => Ok(events),
        Err(error) => {
            if let Some(mut worker) = guard.take() {
                let _ = worker.child.kill().await;
            }
            Err(error)
        }
    }
}

fn emit_progress_from_fast_worker_event(
    app: &AppHandle,
    event: &serde_json::Value,
    default_step: &str,
) {
    let prog = PipelineProgress {
        step: event["step"].as_str().unwrap_or(default_step).to_string(),
        step_progress: event["step_progress"].as_f64().unwrap_or(0.0),
        overall_progress: event["overall_progress"].as_f64().unwrap_or(0.0),
        current_page: event["current_page"].as_u64().unwrap_or(0) as u32,
        total_pages: event["total_pages"].as_u64().unwrap_or(0) as u32,
        message: event["message"].as_str().unwrap_or("").to_string(),
        eta_seconds: event["eta_seconds"].as_f64().unwrap_or(0.0),
    };
    app.emit("pipeline-progress", prog).ok();
}

async fn run_editor_page_action_with_fast_worker(
    app: &AppHandle,
    request_type: &str,
    action: &str,
    default_step: &str,
    project_file: &Path,
    page_index: u32,
    region: Option<&PageRegionConfig>,
) -> Result<String, FastPageRunError> {
    let request = serde_json::json!({
        "type": request_type,
        "project_path": project_file.to_string_lossy().to_string(),
        "page_index": page_index,
        "region": page_region_to_json(region),
    });
    let events = run_fast_page_worker_request(app, request)
        .await
        .map_err(FastPageRunError::Worker)?;

    let mut output_path = String::new();
    for event in events {
        match event.get("type").and_then(|value| value.as_str()) {
            Some("progress") => emit_progress_from_fast_worker_event(app, &event, default_step),
            Some("complete") => {
                output_path = event["output_path"].as_str().unwrap_or("").to_string();
            }
            Some("error") => {
                let message = event["message"].as_str().unwrap_or("Erro Python");
                if is_project_summary_mismatch(message) {
                    stop_fast_page_worker("project summary stale no editor action").await;
                    return Err(FastPageRunError::Worker(format_pipeline_error(
                        action, message, "",
                    )));
                }
                if message.contains(request_type) || message.contains("fast-page") {
                    return Err(FastPageRunError::Worker(message.to_string()));
                }
                return Err(FastPageRunError::Action(format_pipeline_error(
                    action, message, "",
                )));
            }
            _ => {}
        }
    }

    if output_path.is_empty() {
        Err(FastPageRunError::Worker(format!(
            "fast worker nao retornou output_path para {action}"
        )))
    } else {
        Ok(output_path)
    }
}

async fn run_reinpaint_page_with_fast_worker(
    app: &AppHandle,
    project_file: &Path,
    page_index: u32,
    region: Option<&PageRegionConfig>,
) -> Result<String, FastPageRunError> {
    let started = Instant::now();
    let request = serde_json::json!({
        "type": "editor_reinpaint",
        "project_path": project_file.to_string_lossy().to_string(),
        "page_index": page_index,
        "region": page_region_to_json(region),
    });
    let events = run_fast_page_worker_request(app, request)
        .await
        .map_err(FastPageRunError::Worker)?;

    let mut output_path = String::new();
    for event in events {
        match event.get("type").and_then(|value| value.as_str()) {
            Some("progress") => emit_progress_from_fast_worker_event(app, &event, "inpaint"),
            Some("complete") => {
                output_path = event["output_path"].as_str().unwrap_or("").to_string();
                let elapsed_seconds = event["elapsed_seconds"].as_f64().unwrap_or(0.0);
                let inpaint_seconds = event["inpaint_seconds"].as_f64().unwrap_or(0.0);
                let finalize_seconds = event["finalize_seconds"].as_f64().unwrap_or(0.0);
                let lama_ms = event["inpaint_stats"]["_t_lama_ms"].as_f64().unwrap_or(0.0);
                let roi_ratio = event["inpaint_stats"]["roi_area_ratio"]
                    .as_f64()
                    .unwrap_or(0.0);
                eprintln!(
                    "[EditorAction] timing inpaint worker page={} rust={:.3}s py={:.3}s inpaint={:.3}s finalize={:.3}s lama={:.0}ms roi_ratio={:.4}",
                    page_index,
                    started.elapsed().as_secs_f64(),
                    elapsed_seconds,
                    inpaint_seconds,
                    finalize_seconds,
                    lama_ms,
                    roi_ratio,
                );
            }
            Some("error") => {
                let message = event["message"].as_str().unwrap_or("Erro Python");
                if is_project_summary_mismatch(message) {
                    stop_fast_page_worker("project summary stale no inpaint worker").await;
                    return Err(FastPageRunError::Worker(format_pipeline_error(
                        "inpaint", message, "",
                    )));
                }
                if message.contains("editor_reinpaint") || message.contains("fast-page") {
                    return Err(FastPageRunError::Worker(message.to_string()));
                }
                return Err(FastPageRunError::Action(format_pipeline_error(
                    "inpaint", message, "",
                )));
            }
            _ => {}
        }
    }

    if output_path.is_empty() {
        Err(FastPageRunError::Worker(
            "fast worker nao retornou output_path".into(),
        ))
    } else {
        Ok(output_path)
    }
}

async fn warmup_editor_visual_worker(app: AppHandle) -> Result<String, String> {
    {
        let mut state = VISUAL_WARMUP_STATE.lock().await;
        match *state {
            VisualWarmupState::Ready => return Ok("ready".into()),
            VisualWarmupState::Running => return Ok("warming".into()),
            VisualWarmupState::Idle => {
                *state = VisualWarmupState::Running;
            }
        }
    }

    let result = async {
        let storage = crate::storage::service_for_app(&app)?;
        let models_dir = storage.ensure_base_dirs()?.models;
        let events = run_fast_page_worker_request(
            &app,
            serde_json::json!({
                "type": "warmup",
                "models_dir": models_dir.to_string_lossy().to_string(),
                "profile": "max",
                "idioma_origem": "en",
            }),
        )
        .await?;

        if events
            .iter()
            .any(|event| event.get("type").and_then(|value| value.as_str()) == Some("ready"))
        {
            Ok("ready".to_string())
        } else {
            Err("fast worker nao confirmou warmup visual".to_string())
        }
    }
    .await;

    let mut state = VISUAL_WARMUP_STATE.lock().await;
    *state = if result.is_ok() {
        VisualWarmupState::Ready
    } else {
        VisualWarmupState::Idle
    };
    result
}

async fn warmup_editor_inpaint_worker(app: AppHandle) -> Result<String, String> {
    {
        let mut state = FAST_PAGE_INPAINT_WARMUP_STATE.lock().await;
        match *state {
            VisualWarmupState::Ready => return Ok("ready".into()),
            VisualWarmupState::Running => return Ok("warming".into()),
            VisualWarmupState::Idle => {
                *state = VisualWarmupState::Running;
            }
        }
    }

    let result = async {
        let storage = crate::storage::service_for_app(&app)?;
        let models_dir = storage.ensure_base_dirs()?.models;
        let events = run_fast_page_worker_request(
            &app,
            serde_json::json!({
                "type": "warmup_inpaint",
                "models_dir": models_dir.to_string_lossy().to_string(),
                "profile": "quality",
            }),
        )
        .await?;

        if events
            .iter()
            .any(|event| event.get("type").and_then(|value| value.as_str()) == Some("ready"))
        {
            Ok("ready".to_string())
        } else {
            Err("fast worker nao confirmou warmup do inpaint".to_string())
        }
    }
    .await;

    let mut state = FAST_PAGE_INPAINT_WARMUP_STATE.lock().await;
    *state = if result.is_ok() {
        VisualWarmupState::Ready
    } else {
        VisualWarmupState::Idle
    };
    result
}

#[tauri::command]
pub async fn retypeset_page(app: AppHandle, config: RetypesetConfig) -> Result<String, String> {
    let pf = resolve_project_json_path(&config.project_path)?;
    let sidecar = get_sidecar_info(&app)?;
    let mut cmd = Command::new(&sidecar.program);
    if let Some(s) = &sidecar.script {
        cmd.arg(s);
    }
    cmd.arg("--retypeset")
        .arg(pf.to_string_lossy().to_string())
        .arg(config.page_index.to_string());
    apply_sidecar_env(&mut cmd, &sidecar.program)?;
    eprintln!("[EditorAction] start  retypeset page={}", config.page_index);
    let (mut child, stderr_handle) = spawn_with_stderr_capture(cmd, "retypeset")?;
    let stdout = child.stdout.take().expect("stdout not captured");
    let mut reader = BufReader::new(stdout).lines();
    let mut out = String::new();
    while let Ok(Some(line)) = reader.next_line().await {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(t) = msg.get("type").and_then(|t| t.as_str()) {
                match t {
                    "progress" => {
                        let prog = PipelineProgress {
                            step: msg["step"].as_str().unwrap_or("typeset").to_string(),
                            step_progress: msg["step_progress"].as_f64().unwrap_or(0.0),
                            overall_progress: msg["overall_progress"].as_f64().unwrap_or(0.0),
                            current_page: msg["current_page"].as_u64().unwrap_or(0) as u32,
                            total_pages: msg["total_pages"].as_u64().unwrap_or(0) as u32,
                            message: msg["message"].as_str().unwrap_or("").to_string(),
                            eta_seconds: msg["eta_seconds"].as_f64().unwrap_or(0.0),
                        };
                        app.emit("pipeline-progress", prog).ok();
                    }
                    "complete" => {
                        out = msg["output_path"].as_str().unwrap_or("").to_string();
                    }
                    "error" => {
                        let stderr_text = collect_stderr(stderr_handle).await;
                        let base = msg["message"].as_str().unwrap_or("Erro Python");
                        let err = format_pipeline_error("retypeset", base, &stderr_text);
                        eprintln!("[EditorAction] error  retypeset: {base}");
                        return Err(err);
                    }
                    _ => {}
                }
            }
        }
    }
    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !status.success() {
        let stderr_text = collect_stderr(stderr_handle).await;
        let err = format_pipeline_error(
            "retypeset",
            &format!("processo encerrou com status {status}"),
            &stderr_text,
        );
        eprintln!("[EditorAction] error  retypeset status={status}");
        return Err(err);
    }
    eprintln!(
        "[EditorAction] success retypeset page={} out={}",
        config.page_index, out
    );
    Ok(out)
}

#[tauri::command]
pub async fn render_preview_page(
    app: AppHandle,
    config: RenderPreviewConfig,
) -> Result<String, String> {
    let pf = resolve_project_json_path(&config.project_path)?;
    let extension = render_preview_extension_hint(&config.page);
    let (override_path, output_path) =
        render_preview_paths(&pf, config.page_index, &config.fingerprint, &extension)?;
    let override_payload = serde_json::json!({ "page": config.page });
    std::fs::write(
        &override_path,
        serde_json::to_vec_pretty(&override_payload).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("Erro ao gravar página temporária do preview: {e}"))?;

    let sidecar = get_sidecar_info(&app)?;
    let mut cmd = Command::new(&sidecar.program);
    if let Some(s) = &sidecar.script {
        cmd.arg(s);
    }
    cmd.arg("--render-preview-page")
        .arg(pf.to_string_lossy().to_string())
        .arg(config.page_index.to_string())
        .arg(override_path.to_string_lossy().to_string())
        .arg(output_path.to_string_lossy().to_string());
    apply_sidecar_env(&mut cmd, &sidecar.program)?;
    eprintln!(
        "[EditorAction] start  render_preview page={}",
        config.page_index
    );
    let (mut child, stderr_handle) = spawn_with_stderr_capture(cmd, "render_preview")?;
    let stdout = child.stdout.take().expect("stdout not captured");
    let mut reader = BufReader::new(stdout).lines();
    let mut out = output_path.to_string_lossy().replace('\\', "/");
    while let Ok(Some(line)) = reader.next_line().await {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(t) = msg.get("type").and_then(|t| t.as_str()) {
                match t {
                    "progress" => {
                        let prog = PipelineProgress {
                            step: msg["step"].as_str().unwrap_or("typeset").to_string(),
                            step_progress: msg["step_progress"].as_f64().unwrap_or(0.0),
                            overall_progress: msg["overall_progress"].as_f64().unwrap_or(0.0),
                            current_page: msg["current_page"].as_u64().unwrap_or(0) as u32,
                            total_pages: msg["total_pages"].as_u64().unwrap_or(0) as u32,
                            message: msg["message"].as_str().unwrap_or("").to_string(),
                            eta_seconds: msg["eta_seconds"].as_f64().unwrap_or(0.0),
                        };
                        app.emit("pipeline-progress", prog).ok();
                    }
                    "complete" => {
                        out = msg["output_path"]
                            .as_str()
                            .unwrap_or(&out)
                            .replace('\\', "/");
                    }
                    "error" => {
                        let stderr_text = collect_stderr(stderr_handle).await;
                        let base = msg["message"].as_str().unwrap_or("Erro Python");
                        let err = format_pipeline_error("render_preview", base, &stderr_text);
                        eprintln!("[EditorAction] error  render_preview: {base}");
                        return Err(err);
                    }
                    _ => {}
                }
            }
        }
    }
    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !status.success() {
        let stderr_text = collect_stderr(stderr_handle).await;
        let err = format_pipeline_error(
            "render_preview",
            &format!("processo encerrou com status {status}"),
            &stderr_text,
        );
        eprintln!("[EditorAction] error  render_preview status={status}");
        return Err(err);
    }
    eprintln!(
        "[EditorAction] success render_preview page={} out={}",
        config.page_index, out
    );
    Ok(out)
}

#[tauri::command]
pub async fn process_block(app: AppHandle, config: ProcessBlockConfig) -> Result<String, String> {
    let pf = resolve_project_json_path(&config.project_path)?;
    let sidecar = get_sidecar_info(&app)?;
    let mut cmd = Command::new(&sidecar.program);
    if let Some(s) = &sidecar.script {
        cmd.arg(s);
    }
    cmd.arg("--process-block")
        .arg(&config.mode)
        .arg(pf.to_string_lossy().to_string())
        .arg(config.page_index.to_string())
        .arg(&config.block_id);
    apply_sidecar_env(&mut cmd, &sidecar.program)?;
    eprintln!(
        "[EditorAction] start  process_block page={} block={}",
        config.page_index, config.block_id
    );
    let (mut child, stderr_handle) = spawn_with_stderr_capture(cmd, "process_block")?;
    let stdout = child.stdout.take().expect("stdout not captured");
    let mut reader = BufReader::new(stdout).lines();
    let mut out = String::new();
    while let Ok(Some(line)) = reader.next_line().await {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(t) = msg.get("type").and_then(|t| t.as_str()) {
                if t == "complete" {
                    out = msg["output_path"].as_str().unwrap_or("").to_string();
                } else if t == "error" {
                    let stderr_text = collect_stderr(stderr_handle).await;
                    let base = msg["message"].as_str().unwrap_or("Erro Python");
                    let err = format_pipeline_error("process_block", base, &stderr_text);
                    eprintln!("[EditorAction] error  process_block: {base}");
                    return Err(err);
                }
            }
        }
    }
    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !status.success() {
        let stderr_text = collect_stderr(stderr_handle).await;
        let err = format_pipeline_error(
            "process_block",
            &format!("processo encerrou com status {status}"),
            &stderr_text,
        );
        eprintln!("[EditorAction] error  process_block status={status}");
        return Err(err);
    }
    eprintln!(
        "[EditorAction] success process_block page={} block={}",
        config.page_index, config.block_id
    );
    Ok(out)
}

#[tauri::command]
pub async fn reinpaint_page(app: AppHandle, config: ReinpaintConfig) -> Result<String, String> {
    let region = PageRegionConfig {
        bbox: config.bbox.map(|bbox| {
            [
                bbox[0].max(0) as u32,
                bbox[1].max(0) as u32,
                bbox[2].max(0) as u32,
                bbox[3].max(0) as u32,
            ]
        }),
        mask_path: config.mask_path.clone(),
    };
    reinpaint_page_with_region(app, config.project_path, config.page_index, Some(region)).await
}

pub async fn reinpaint_page_with_region(
    app: AppHandle,
    project_path: String,
    page_index: u32,
    region: Option<PageRegionConfig>,
) -> Result<String, String> {
    let pf = resolve_project_json_path(&project_path)?;
    match run_reinpaint_page_with_fast_worker(&app, &pf, page_index, region.as_ref()).await {
        Ok(out) => {
            eprintln!(
                "[EditorAction] success inpaint page={} out={} worker=fast",
                page_index, out
            );
            if let Err(error) =
                crate::commands::project::clear_inpaint_cache_for_page(&pf, page_index as usize)
            {
                eprintln!("[EditorAction] warn   inpaint cache cleanup failed: {error}");
            }
            return Ok(out);
        }
        Err(FastPageRunError::Action(error)) => return Err(error),
        Err(FastPageRunError::Worker(error)) => {
            eprintln!(
                "[EditorAction] warn   fast inpaint worker indisponivel; usando sidecar unico: {error}"
            );
        }
    }

    let sidecar = get_sidecar_info(&app)?;
    let mut cmd = Command::new(&sidecar.program);
    if let Some(s) = &sidecar.script {
        cmd.arg(s);
    }
    cmd.arg("--reinpaint-page")
        .arg(pf.to_string_lossy().to_string())
        .arg(page_index.to_string());
    append_page_region_args(&mut cmd, region.as_ref());
    if let Ok(storage) = crate::storage::service_for_app(&app) {
        if let Ok(paths) = storage.ensure_base_dirs() {
            cmd.env("TRADUZAI_MODELS_DIR", paths.models);
        }
    }
    apply_sidecar_env(&mut cmd, &sidecar.program)?;
    eprintln!("[EditorAction] start  inpaint page={}", page_index);
    let (mut child, stderr_handle) = spawn_with_stderr_capture(cmd, "inpaint")?;
    let stdout = child.stdout.take().expect("stdout not captured");
    let mut reader = BufReader::new(stdout).lines();
    let mut out = String::new();
    while let Ok(Some(line)) = reader.next_line().await {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(t) = msg.get("type").and_then(|t| t.as_str()) {
                if t == "complete" {
                    out = msg["output_path"].as_str().unwrap_or("").to_string();
                } else if t == "error" {
                    let stderr_text = collect_stderr(stderr_handle).await;
                    let base = msg["message"].as_str().unwrap_or("Erro Python");
                    let err = format_pipeline_error("inpaint", base, &stderr_text);
                    eprintln!("[EditorAction] error  inpaint: {base}");
                    return Err(err);
                }
            }
        }
    }
    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !status.success() {
        let stderr_text = collect_stderr(stderr_handle).await;
        let err = format_pipeline_error(
            "inpaint",
            &format!("processo encerrou com status {status}"),
            &stderr_text,
        );
        eprintln!("[EditorAction] error  inpaint status={status}");
        return Err(err);
    }
    eprintln!(
        "[EditorAction] success inpaint page={} out={}",
        page_index, out
    );
    if let Err(error) =
        crate::commands::project::clear_inpaint_cache_for_page(&pf, page_index as usize)
    {
        eprintln!("[EditorAction] warn   inpaint cache cleanup failed: {error}");
    }
    Ok(out)
}

#[tauri::command]
pub async fn detect_page(
    app: AppHandle,
    project_path: String,
    page_index: u32,
) -> Result<String, String> {
    detect_page_with_region(app, project_path, page_index, None).await
}

pub async fn detect_page_with_region(
    app: AppHandle,
    project_path: String,
    page_index: u32,
    region: Option<PageRegionConfig>,
) -> Result<String, String> {
    let pf = resolve_project_json_path(&project_path)?;
    match run_editor_page_action_with_fast_worker(
        &app,
        "editor_detect_page",
        "detect",
        "ocr",
        &pf,
        page_index,
        region.as_ref(),
    )
    .await
    {
        Ok(out) => {
            eprintln!(
                "[EditorAction] success detect page={} out={} worker=fast",
                page_index, out
            );
            return Ok(out);
        }
        Err(FastPageRunError::Action(error)) => return Err(error),
        Err(FastPageRunError::Worker(error)) => {
            eprintln!(
                "[EditorAction] warn   fast detect worker indisponivel; usando sidecar unico: {error}"
            );
        }
    }

    let sidecar = get_sidecar_info(&app)?;
    let mut cmd = Command::new(&sidecar.program);
    if let Some(s) = &sidecar.script {
        cmd.arg(s);
    }
    cmd.arg("--detect-page")
        .arg(pf.to_string_lossy().to_string())
        .arg(page_index.to_string());
    append_page_region_args(&mut cmd, region.as_ref());
    apply_sidecar_env(&mut cmd, &sidecar.program)?;
    eprintln!("[EditorAction] start  detect page={page_index}");
    let (mut child, stderr_handle) = spawn_with_stderr_capture(cmd, "detect")?;
    let stdout = child.stdout.take().expect("stdout not captured");
    let mut reader = BufReader::new(stdout).lines();
    let mut out = String::new();
    while let Ok(Some(line)) = reader.next_line().await {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(t) = msg.get("type").and_then(|t| t.as_str()) {
                match t {
                    "progress" => {
                        let prog = PipelineProgress {
                            step: msg["step"].as_str().unwrap_or("ocr").to_string(),
                            step_progress: msg["step_progress"].as_f64().unwrap_or(0.0),
                            overall_progress: msg["overall_progress"].as_f64().unwrap_or(0.0),
                            current_page: msg["current_page"].as_u64().unwrap_or(0) as u32,
                            total_pages: msg["total_pages"].as_u64().unwrap_or(0) as u32,
                            message: msg["message"].as_str().unwrap_or("").to_string(),
                            eta_seconds: msg["eta_seconds"].as_f64().unwrap_or(0.0),
                        };
                        app.emit("pipeline-progress", prog).ok();
                    }
                    "complete" => {
                        out = msg["output_path"].as_str().unwrap_or("").to_string();
                    }
                    "error" => {
                        let stderr_text = collect_stderr(stderr_handle).await;
                        let base = msg["message"].as_str().unwrap_or("Erro Python");
                        let err = format_pipeline_error("detect", base, &stderr_text);
                        eprintln!("[EditorAction] error  detect: {base}");
                        return Err(err);
                    }
                    _ => {}
                }
            }
        }
    }
    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !status.success() {
        let stderr_text = collect_stderr(stderr_handle).await;
        let err = format_pipeline_error(
            "detect",
            &format!("processo encerrou com status {status}"),
            &stderr_text,
        );
        eprintln!("[EditorAction] error  detect status={status}");
        return Err(err);
    }
    eprintln!("[EditorAction] success detect page={page_index} out={out}");
    Ok(out)
}

#[tauri::command]
pub async fn ocr_page(
    app: AppHandle,
    project_path: String,
    page_index: u32,
) -> Result<String, String> {
    ocr_page_with_region(app, project_path, page_index, None).await
}

pub async fn ocr_page_with_region(
    app: AppHandle,
    project_path: String,
    page_index: u32,
    region: Option<PageRegionConfig>,
) -> Result<String, String> {
    let pf = resolve_project_json_path(&project_path)?;
    match run_editor_page_action_with_fast_worker(
        &app,
        "editor_ocr_page",
        "ocr",
        "ocr",
        &pf,
        page_index,
        region.as_ref(),
    )
    .await
    {
        Ok(out) => {
            eprintln!(
                "[EditorAction] success ocr page={} out={} worker=fast",
                page_index, out
            );
            return Ok(out);
        }
        Err(FastPageRunError::Action(error)) => return Err(error),
        Err(FastPageRunError::Worker(error)) => {
            eprintln!(
                "[EditorAction] warn   fast ocr worker indisponivel; usando sidecar unico: {error}"
            );
        }
    }

    let sidecar = get_sidecar_info(&app)?;
    let mut cmd = Command::new(&sidecar.program);
    if let Some(s) = &sidecar.script {
        cmd.arg(s);
    }
    cmd.arg("--ocr-page")
        .arg(pf.to_string_lossy().to_string())
        .arg(page_index.to_string());
    append_page_region_args(&mut cmd, region.as_ref());
    apply_sidecar_env(&mut cmd, &sidecar.program)?;
    eprintln!("[EditorAction] start  ocr page={page_index}");
    let (mut child, stderr_handle) = spawn_with_stderr_capture(cmd, "ocr")?;
    let stdout = child.stdout.take().expect("stdout not captured");
    let mut reader = BufReader::new(stdout).lines();
    let mut out = String::new();
    while let Ok(Some(line)) = reader.next_line().await {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(t) = msg.get("type").and_then(|t| t.as_str()) {
                match t {
                    "progress" => {
                        let prog = PipelineProgress {
                            step: msg["step"].as_str().unwrap_or("ocr").to_string(),
                            step_progress: msg["step_progress"].as_f64().unwrap_or(0.0),
                            overall_progress: msg["overall_progress"].as_f64().unwrap_or(0.0),
                            current_page: msg["current_page"].as_u64().unwrap_or(0) as u32,
                            total_pages: msg["total_pages"].as_u64().unwrap_or(0) as u32,
                            message: msg["message"].as_str().unwrap_or("").to_string(),
                            eta_seconds: msg["eta_seconds"].as_f64().unwrap_or(0.0),
                        };
                        app.emit("pipeline-progress", prog).ok();
                    }
                    "complete" => {
                        out = msg["output_path"].as_str().unwrap_or("").to_string();
                    }
                    "error" => {
                        let stderr_text = collect_stderr(stderr_handle).await;
                        let base = msg["message"].as_str().unwrap_or("Erro Python");
                        let err = format_pipeline_error("ocr", base, &stderr_text);
                        eprintln!("[EditorAction] error  ocr: {base}");
                        return Err(err);
                    }
                    _ => {}
                }
            }
        }
    }
    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !status.success() {
        let stderr_text = collect_stderr(stderr_handle).await;
        let err = format_pipeline_error(
            "ocr",
            &format!("processo encerrou com status {status}"),
            &stderr_text,
        );
        eprintln!("[EditorAction] error  ocr status={status}");
        return Err(err);
    }
    eprintln!("[EditorAction] success ocr page={page_index} out={out}");
    Ok(out)
}

#[tauri::command]
pub async fn translate_page(
    app: AppHandle,
    project_path: String,
    page_index: u32,
) -> Result<String, String> {
    translate_page_with_region(app, project_path, page_index, None).await
}

pub async fn translate_page_with_region(
    app: AppHandle,
    project_path: String,
    page_index: u32,
    region: Option<PageRegionConfig>,
) -> Result<String, String> {
    let pf = resolve_project_json_path(&project_path)?;
    let sidecar = get_sidecar_info(&app)?;
    let mut cmd = Command::new(&sidecar.program);
    if let Some(s) = &sidecar.script {
        cmd.arg(s);
    }
    cmd.arg("--translate-page")
        .arg(pf.to_string_lossy().to_string())
        .arg(page_index.to_string());
    append_page_region_args(&mut cmd, region.as_ref());
    apply_sidecar_env(&mut cmd, &sidecar.program)?;
    eprintln!("[EditorAction] start  translate page={page_index}");
    let (mut child, stderr_handle) = spawn_with_stderr_capture(cmd, "translate")?;
    let stdout = child.stdout.take().expect("stdout not captured");
    let mut reader = BufReader::new(stdout).lines();
    let mut out = String::new();
    while let Ok(Some(line)) = reader.next_line().await {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(t) = msg.get("type").and_then(|t| t.as_str()) {
                match t {
                    "progress" => {
                        let prog = PipelineProgress {
                            step: msg["step"].as_str().unwrap_or("translate").to_string(),
                            step_progress: msg["step_progress"].as_f64().unwrap_or(0.0),
                            overall_progress: msg["overall_progress"].as_f64().unwrap_or(0.0),
                            current_page: msg["current_page"].as_u64().unwrap_or(0) as u32,
                            total_pages: msg["total_pages"].as_u64().unwrap_or(0) as u32,
                            message: msg["message"].as_str().unwrap_or("").to_string(),
                            eta_seconds: msg["eta_seconds"].as_f64().unwrap_or(0.0),
                        };
                        app.emit("pipeline-progress", prog).ok();
                    }
                    "complete" => {
                        out = msg["output_path"].as_str().unwrap_or("").to_string();
                    }
                    "error" => {
                        let stderr_text = collect_stderr(stderr_handle).await;
                        let base = msg["message"].as_str().unwrap_or("Erro Python");
                        let err = format_pipeline_error("translate", base, &stderr_text);
                        eprintln!("[EditorAction] error  translate: {base}");
                        return Err(err);
                    }
                    _ => {}
                }
            }
        }
    }
    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !status.success() {
        let stderr_text = collect_stderr(stderr_handle).await;
        let err = format_pipeline_error(
            "translate",
            &format!("processo encerrou com status {status}"),
            &stderr_text,
        );
        eprintln!("[EditorAction] error  translate status={status}");
        return Err(err);
    }
    eprintln!("[EditorAction] success translate page={page_index} out={out}");
    Ok(out)
}

#[tauri::command]
pub async fn check_gpu(app: AppHandle) -> Result<GpuInfo, String> {
    let profile = get_system_profile(app).await?;
    Ok(GpuInfo {
        available: profile.gpu_available,
        name: profile.gpu_name,
    })
}

#[tauri::command]
pub async fn get_system_profile(app: AppHandle) -> Result<SystemProfile, String> {
    let sidecar = get_sidecar_info(&app)?;
    let mut cmd = Command::new(&sidecar.program);
    if let Some(s) = &sidecar.script {
        cmd.arg(s);
    }
    cmd.arg("--hardware-info");
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());
    apply_sidecar_env(&mut cmd, &sidecar.program)?;

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Erro ao iniciar hardware info: {e}"))?;
    let stdout = child.stdout.take().expect("stdout not captured");
    let mut reader = BufReader::new(stdout).lines();

    let mut profile_json = String::new();
    while let Ok(Some(line)) = reader.next_line().await {
        if line.trim().starts_with('{') {
            profile_json = line;
            break;
        }
    }

    if profile_json.is_empty() {
        return Err("Informação de hardware não retornada pelo sidecar".into());
    }

    let facts: HardwareFacts = serde_json::from_str(&profile_json).map_err(|e| {
        format!(
            "Falha ao parsear hardware info: {e} | JSON: {}",
            profile_json
        )
    })?;

    let tier = if facts.gpu_available {
        let vram = facts.gpu_vram_gb.unwrap_or(0.0);
        if vram >= 11.0 {
            "workstation"
        } else if vram >= 6.0 {
            "fast"
        } else {
            "balanced"
        }
    } else {
        "cpu_only"
    };

    Ok(SystemProfile {
        cpu_name: facts.cpu_name,
        cpu_cores: facts.cpu_cores,
        cpu_threads: facts.cpu_threads,
        ram_gb: facts.ram_gb,
        gpu_available: facts.gpu_available,
        gpu_name: facts.gpu_name,
        gpu_vram_gb: facts.gpu_vram_gb,
        performance_tier: tier.into(),
        startup_seconds: 18.0,
        seconds_per_page: QualityEstimateTable {
            rapida: if facts.gpu_available { 2.2 } else { 14.5 },
            normal: if facts.gpu_available { 4.8 } else { 28.0 },
            alta: if facts.gpu_available { 8.5 } else { 42.0 },
        },
    })
}
#[tauri::command]
pub async fn check_models(app: AppHandle) -> Result<serde_json::Value, String> {
    let storage = crate::storage::service_for_app(&app)?;
    let paths = storage.ensure_base_dirs()?;
    let m = paths.models;
    Ok(
        serde_json::json!({ "ready": m.exists(), "size_mb": 0, "ocr_ready": true, "inpainting_ready": true }),
    )
}
#[tauri::command]
pub async fn download_models(app: AppHandle) -> Result<(), String> {
    app.emit("models-ready", serde_json::json!({"success": true}))
        .ok();
    Ok(())
}

async fn search_anilist_internal(query: &str) -> Result<serde_json::Value, String> {
    let gq = r#"query ($search: String) { Media(search: $search, type: MANGA) { id siteUrl title { english romaji native } synonyms description(asHtml: false) genres characters(sort: ROLE, perPage: 20) { nodes { name { full native alternative } } } coverImage { large } } }"#;
    let client = reqwest::Client::new();
    let resp = client
        .post("https://graphql.anilist.co")
        .json(&serde_json::json!({"query": gq, "variables": {"search": query}}))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    let media = &data["data"]["Media"];
    let title = media["title"]["english"]
        .as_str()
        .or(media["title"]["romaji"].as_str())
        .or(media["title"]["native"].as_str())
        .unwrap_or(query)
        .to_string();
    let characters = media["characters"]["nodes"]
        .as_array()
        .map(|nodes| {
            nodes
                .iter()
                .filter_map(|c| c["name"]["full"].as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Ok(serde_json::json!({
        "title": title,
        "synopsis": media["description"].as_str().unwrap_or(""),
        "genres": media["genres"].as_array().cloned().unwrap_or_default(),
        "characters": characters,
        "aliases": media["synonyms"].as_array().cloned().unwrap_or_default(),
        "cover_url": media["coverImage"]["large"].as_str().unwrap_or(""),
        "url": media["siteUrl"].as_str().unwrap_or(""),
    }))
}

fn external_error(source: &str, error: impl ToString) -> ExternalContextData {
    ExternalContextData {
        source: source.to_string(),
        status: "error".to_string(),
        confidence: 0.0,
        title: String::new(),
        synopsis: String::new(),
        genres: vec![],
        candidates: vec![],
        url: String::new(),
        error: error.to_string(),
        cover_url: String::new(),
    }
}

fn external_not_found(source: &str) -> ExternalContextData {
    ExternalContextData {
        source: source.to_string(),
        status: "not_found".to_string(),
        confidence: 0.0,
        title: String::new(),
        synopsis: String::new(),
        genres: vec![],
        candidates: vec![],
        url: String::new(),
        error: String::new(),
        cover_url: String::new(),
    }
}

fn anilist_context_from_value(value: &serde_json::Value) -> ExternalContextData {
    let mut candidates = Vec::new();
    for name in json_string_vec(&value["characters"]) {
        if let Some(candidate) = context_candidate("character", &name, 0.95, "anilist", vec![]) {
            candidates.push(candidate);
        }
    }
    for alias in json_string_vec(&value["aliases"]) {
        if let Some(candidate) = context_candidate("alias", &alias, 0.75, "anilist", vec![]) {
            candidates.push(candidate);
        }
    }
    ExternalContextData {
        source: "anilist".to_string(),
        status: "found".to_string(),
        confidence: 0.92,
        title: value["title"].as_str().unwrap_or_default().to_string(),
        synopsis: value["synopsis"].as_str().unwrap_or_default().to_string(),
        genres: json_string_vec(&value["genres"]),
        candidates,
        url: value["url"].as_str().unwrap_or_default().to_string(),
        error: String::new(),
        cover_url: value["cover_url"].as_str().unwrap_or_default().to_string(),
    }
}

async fn search_anilist_context(client: &reqwest::Client, query: &str) -> ExternalContextData {
    let gq = r#"query ($search: String) { Media(search: $search, type: MANGA) { id siteUrl title { english romaji native } synonyms description(asHtml: false) genres characters(sort: ROLE, perPage: 20) { nodes { name { full native alternative } } } coverImage { large } } }"#;
    let data = match client
        .post("https://graphql.anilist.co")
        .json(&serde_json::json!({"query": gq, "variables": {"search": query}}))
        .send()
        .await
        .and_then(|resp| resp.error_for_status())
    {
        Ok(resp) => match resp.json::<serde_json::Value>().await {
            Ok(value) => value,
            Err(error) => return external_error("anilist", error),
        },
        Err(error) => return external_error("anilist", error),
    };
    let media = &data["data"]["Media"];
    if media.is_null() {
        return external_not_found("anilist");
    }
    let title = media["title"]["english"]
        .as_str()
        .or(media["title"]["romaji"].as_str())
        .or(media["title"]["native"].as_str())
        .unwrap_or(query)
        .to_string();
    let mut candidates = Vec::new();
    for node in media["characters"]["nodes"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let names = &node["name"];
        let primary = names["full"]
            .as_str()
            .or(names["native"].as_str())
            .unwrap_or_default();
        let aliases = names["alternative"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(ToString::to_string))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if let Some(candidate) = context_candidate("character", primary, 0.95, "anilist", aliases) {
            candidates.push(candidate);
        }
    }
    for alias in media["synonyms"].as_array().into_iter().flatten() {
        if let Some(alias) = alias.as_str() {
            if let Some(candidate) = context_candidate("alias", alias, 0.75, "anilist", vec![]) {
                candidates.push(candidate);
            }
        }
    }
    ExternalContextData {
        source: "anilist".to_string(),
        status: "found".to_string(),
        confidence: 0.92,
        title,
        synopsis: media["description"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        genres: json_string_vec(&media["genres"]),
        candidates,
        url: media["siteUrl"].as_str().unwrap_or_default().to_string(),
        error: String::new(),
        cover_url: media["coverImage"]["large"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
    }
}

async fn search_jikan_context(client: &reqwest::Client, query: &str) -> ExternalContextData {
    let search = match client
        .get("https://api.jikan.moe/v4/manga")
        .query(&[("q", query), ("limit", "1")])
        .send()
        .await
        .and_then(|resp| resp.error_for_status())
    {
        Ok(resp) => match resp.json::<serde_json::Value>().await {
            Ok(value) => value,
            Err(error) => return external_error("myanimelist", error),
        },
        Err(error) => return external_error("myanimelist", error),
    };
    let Some(manga) = search["data"].as_array().and_then(|items| items.first()) else {
        return external_not_found("myanimelist");
    };
    let title = manga["title_english"]
        .as_str()
        .or(manga["title"].as_str())
        .unwrap_or(query)
        .to_string();
    let title_synonyms = manga["title_synonyms"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(ToString::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if !context_title_matches(query, &title, &title_synonyms) {
        return external_not_found("myanimelist");
    }
    let mal_id = manga["mal_id"].as_i64().unwrap_or_default();
    let mut candidates = Vec::new();
    if mal_id > 0 {
        let url = format!("https://api.jikan.moe/v4/manga/{mal_id}/characters");
        if let Ok(resp) = client
            .get(url)
            .send()
            .await
            .and_then(|resp| resp.error_for_status())
        {
            if let Ok(chars) = resp.json::<serde_json::Value>().await {
                for item in chars["data"].as_array().into_iter().flatten() {
                    let character = &item["character"];
                    let name = character["name"].as_str().unwrap_or_default();
                    if let Some(candidate) =
                        context_candidate("character", name, 0.9, "myanimelist", vec![])
                    {
                        candidates.push(candidate);
                    }
                }
            }
        }
    }
    let genres = manga["genres"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item["name"].as_str().map(ToString::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    ExternalContextData {
        source: "myanimelist".to_string(),
        status: "found".to_string(),
        confidence: 0.86,
        title,
        synopsis: manga["synopsis"].as_str().unwrap_or_default().to_string(),
        genres,
        candidates,
        url: manga["url"].as_str().unwrap_or_default().to_string(),
        error: String::new(),
        cover_url: manga["images"]["jpg"]["large_image_url"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
    }
}

async fn search_mangadex_context(client: &reqwest::Client, query: &str) -> ExternalContextData {
    let data = match client
        .get("https://api.mangadex.org/manga")
        .query(&[("title", query), ("limit", "1")])
        .send()
        .await
        .and_then(|resp| resp.error_for_status())
    {
        Ok(resp) => match resp.json::<serde_json::Value>().await {
            Ok(value) => value,
            Err(error) => return external_error("mangadex", error),
        },
        Err(error) => return external_error("mangadex", error),
    };
    let Some(manga) = data["data"].as_array().and_then(|items| items.first()) else {
        return external_not_found("mangadex");
    };
    let attrs = &manga["attributes"];
    let title = attrs["title"]["en"]
        .as_str()
        .or_else(|| {
            attrs["title"]
                .as_object()
                .and_then(|titles| titles.values().find_map(|value| value.as_str()))
        })
        .unwrap_or(query)
        .to_string();
    let mut candidates = Vec::new();
    for alt in attrs["altTitles"].as_array().into_iter().flatten() {
        if let Some(map) = alt.as_object() {
            for value in map.values().filter_map(|value| value.as_str()) {
                if let Some(candidate) = context_candidate("alias", value, 0.72, "mangadex", vec![])
                {
                    candidates.push(candidate);
                }
            }
        }
    }
    let genres = attrs["tags"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    item["attributes"]["name"]["en"]
                        .as_str()
                        .map(ToString::to_string)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let id = manga["id"].as_str().unwrap_or_default();
    ExternalContextData {
        source: "mangadex".to_string(),
        status: "found".to_string(),
        confidence: 0.78,
        title,
        synopsis: attrs["description"]["en"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        genres,
        candidates,
        url: if id.is_empty() {
            String::new()
        } else {
            format!("https://mangadex.org/title/{id}")
        },
        error: String::new(),
        cover_url: String::new(),
    }
}

async fn search_kitsu_context(client: &reqwest::Client, query: &str) -> ExternalContextData {
    let data = match client
        .get("https://kitsu.io/api/edge/manga")
        .header("Accept", "application/vnd.api+json")
        .query(&[("filter[text]", query), ("page[limit]", "1")])
        .send()
        .await
        .and_then(|resp| resp.error_for_status())
    {
        Ok(resp) => match resp.json::<serde_json::Value>().await {
            Ok(value) => value,
            Err(error) => return external_error("kitsu", error),
        },
        Err(error) => return external_error("kitsu", error),
    };
    let Some(manga) = data["data"].as_array().and_then(|items| items.first()) else {
        return external_not_found("kitsu");
    };
    let attrs = &manga["attributes"];
    let title = attrs["titles"]["en"]
        .as_str()
        .or(attrs["titles"]["en_jp"].as_str())
        .or(attrs["canonicalTitle"].as_str())
        .unwrap_or(query)
        .to_string();
    let mut candidates = Vec::new();
    if let Some(titles) = attrs["titles"].as_object() {
        for value in titles.values().filter_map(|value| value.as_str()) {
            if value != title {
                if let Some(candidate) = context_candidate("alias", value, 0.7, "kitsu", vec![]) {
                    candidates.push(candidate);
                }
            }
        }
    }
    let slug = attrs["slug"]
        .as_str()
        .or(manga["id"].as_str())
        .unwrap_or_default();
    ExternalContextData {
        source: "kitsu".to_string(),
        status: "found".to_string(),
        confidence: 0.74,
        title,
        synopsis: attrs["synopsis"].as_str().unwrap_or_default().to_string(),
        genres: vec![],
        candidates,
        url: if slug.is_empty() {
            String::new()
        } else {
            format!("https://kitsu.io/manga/{slug}")
        },
        error: String::new(),
        cover_url: attrs["posterImage"]["large"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
    }
}

async fn search_external_context_sources(query: &str) -> Vec<ExternalContextData> {
    let client = reqwest::Client::builder()
        .user_agent("TraduzAi/1.0")
        .timeout(Duration::from_secs(8))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    let anilist = search_anilist_context(&client, query);
    let myanimelist = search_jikan_context(&client, query);
    let mangadex = search_mangadex_context(&client, query);
    let kitsu = search_kitsu_context(&client, query);
    let (anilist, myanimelist, mangadex, kitsu) =
        tokio::join!(anilist, myanimelist, mangadex, kitsu);
    vec![anilist, myanimelist, mangadex, kitsu]
}

fn json_string_vec(value: &serde_json::Value) -> Vec<String> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::trim))
                .filter(|item| !item.is_empty())
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn value_strings(values: &[serde_json::Value]) -> Vec<String> {
    values
        .iter()
        .filter_map(|value| value.as_str().map(str::trim))
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn unique_strings(values: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for value in values {
        let clean = value.split_whitespace().collect::<Vec<_>>().join(" ");
        let key = normalize_text(&clean);
        if !key.is_empty() && seen.insert(key) {
            out.push(clean);
        }
    }
    out
}

fn context_title_matches(query: &str, title: &str, aliases: &[String]) -> bool {
    let query_key = normalize_text(query);
    if query_key.is_empty() {
        return false;
    }
    let mut candidates = vec![title.to_string()];
    candidates.extend(aliases.iter().cloned());
    let query_tokens = query_key.split_whitespace().collect::<Vec<_>>();
    for candidate in candidates {
        let candidate_key = normalize_text(&candidate);
        if candidate_key == query_key
            || candidate_key.contains(&query_key)
            || query_key.contains(&candidate_key)
        {
            return true;
        }
        let candidate_tokens = candidate_key
            .split_whitespace()
            .collect::<std::collections::BTreeSet<_>>();
        let overlap = query_tokens
            .iter()
            .filter(|token| candidate_tokens.contains(**token))
            .count();
        if overlap >= 2 && overlap * 2 >= query_tokens.len() {
            return true;
        }
    }
    false
}

fn context_candidate(
    kind: &str,
    source: &str,
    confidence: f64,
    source_name: &str,
    aliases: Vec<String>,
) -> Option<ExternalContextCandidate> {
    let clean = source.split_whitespace().collect::<Vec<_>>().join(" ");
    if clean.is_empty() {
        return None;
    }
    Some(ExternalContextCandidate {
        kind: kind.to_string(),
        source: clean.clone(),
        target: clean,
        confidence,
        sources: vec![source_name.to_string()],
        status: "candidate".to_string(),
        protect: true,
        aliases: unique_strings(aliases),
        forbidden: vec![],
        notes: String::new(),
    })
}

fn source_result_value(result: &ExternalContextData) -> serde_json::Value {
    serde_json::json!({
        "source": result.source,
        "status": result.status,
        "confidence": result.confidence,
        "title": result.title,
        "synopsis": result.synopsis,
        "genres": result.genres,
        "url": result.url,
        "error": result.error,
    })
}

fn candidate_value(candidate: &ExternalContextCandidate) -> serde_json::Value {
    serde_json::json!({
        "kind": candidate.kind,
        "source": candidate.source,
        "target": candidate.target,
        "confidence": candidate.confidence,
        "sources": candidate.sources,
        "status": candidate.status,
        "protect": candidate.protect,
        "aliases": candidate.aliases,
        "forbidden": candidate.forbidden,
        "notes": candidate.notes,
    })
}

fn merge_external_candidates(results: &[ExternalContextData]) -> Vec<ExternalContextCandidate> {
    let mut merged: HashMap<String, ExternalContextCandidate> = HashMap::new();
    for result in results.iter().filter(|item| item.status == "found") {
        for candidate in &result.candidates {
            let key = normalize_text(&candidate.source);
            if key.is_empty() {
                continue;
            }
            let entry = merged.entry(key).or_insert_with(|| candidate.clone());
            entry.sources = unique_strings(
                entry
                    .sources
                    .iter()
                    .cloned()
                    .chain(candidate.sources.iter().cloned())
                    .chain(std::iter::once(result.source.clone())),
            );
            entry.aliases = unique_strings(
                entry
                    .aliases
                    .iter()
                    .cloned()
                    .chain(candidate.aliases.iter().cloned()),
            );
            if candidate.confidence > entry.confidence {
                entry.confidence = candidate.confidence;
                entry.target = candidate.target.clone();
            }
        }
    }
    let mut values: Vec<_> = merged.into_values().collect();
    values.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.source.to_lowercase().cmp(&b.source.to_lowercase()))
    });
    values
}

fn best_context_title(results: &[ExternalContextData], fallback: &str) -> String {
    results
        .iter()
        .filter(|item| item.status == "found" && !item.title.trim().is_empty())
        .max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|item| item.title.clone())
        .unwrap_or_else(|| fallback.to_string())
}

fn best_context_synopsis(results: &[ExternalContextData], fallback: &str) -> String {
    results
        .iter()
        .filter(|item| item.status == "found" && !item.synopsis.trim().is_empty())
        .max_by(|a, b| {
            (a.confidence, a.synopsis.len())
                .partial_cmp(&(b.confidence, b.synopsis.len()))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|item| item.synopsis.clone())
        .unwrap_or_else(|| fallback.to_string())
}

fn merged_context_genres(results: &[ExternalContextData], fallback: &[String]) -> Vec<String> {
    unique_strings(
        fallback
            .iter()
            .cloned()
            .chain(results.iter().flat_map(|item| item.genres.iter().cloned())),
    )
}

fn best_cover_url(results: &[ExternalContextData], fallback: &str) -> String {
    results
        .iter()
        .filter(|item| item.status == "found" && !item.cover_url.trim().is_empty())
        .max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|item| item.cover_url.clone())
        .unwrap_or_else(|| fallback.to_string())
}

fn glossary_entry_id(prefix: &str, source: &str) -> String {
    let slug = normalize_text(source).replace(' ', "_");
    format!("{prefix}_{}", if slug.is_empty() { "termo" } else { &slug })
}

fn auto_save_character_glossary(
    works_root: &Path,
    work_id: &str,
    candidates: &[ExternalContextCandidate],
) -> Result<u32, String> {
    let mut glossary = crate::glossary::load(works_root, work_id)?;
    let current_character_keys = candidates
        .iter()
        .filter(|item| item.kind == "character" && !item.source.trim().is_empty())
        .map(|item| crate::glossary::normalize_lookup(&item.source, false))
        .collect::<std::collections::BTreeSet<_>>();
    glossary.entries.retain(|entry| {
        let is_auto_character = entry.entry_type == "character"
            && entry
                .notes
                .contains("importado automaticamente do contexto online");
        if !is_auto_character {
            return true;
        }
        current_character_keys.contains(&crate::glossary::normalize_lookup(&entry.source, false))
    });
    for candidate in candidates
        .iter()
        .filter(|item| item.kind == "character" && !item.source.trim().is_empty())
    {
        let normalized = crate::glossary::normalize_lookup(&candidate.source, false);
        if let Some(existing) = glossary
            .entries
            .iter_mut()
            .find(|entry| crate::glossary::normalize_lookup(&entry.source, false) == normalized)
        {
            existing.protect = true;
            existing.sources = unique_strings(
                existing
                    .sources
                    .iter()
                    .cloned()
                    .chain(candidate.sources.iter().cloned()),
            );
            existing.aliases = unique_strings(
                existing
                    .aliases
                    .iter()
                    .cloned()
                    .chain(candidate.aliases.iter().cloned()),
            );
            existing.confidence = existing.confidence.max(candidate.confidence);
            if existing.status.trim().is_empty() || existing.status == "candidate" {
                existing.status = "reviewed".to_string();
            }
            continue;
        }
        glossary.entries.push(crate::glossary::GlossaryEntry {
            id: glossary_entry_id("character", &candidate.source),
            source: candidate.source.clone(),
            target: candidate.target.clone(),
            entry_type: "character".to_string(),
            case_sensitive: false,
            protect: true,
            aliases: candidate.aliases.clone(),
            forbidden: vec![],
            confidence: candidate.confidence,
            status: "reviewed".to_string(),
            notes: "Nome de personagem importado automaticamente do contexto online da obra."
                .to_string(),
            context_rule: String::new(),
            sources: candidate.sources.clone(),
        });
    }
    crate::glossary::save(works_root, &glossary)?;
    Ok(glossary.entries.len() as u32)
}

#[tauri::command]
pub async fn search_anilist(query: String) -> Result<serde_json::Value, String> {
    search_anilist_internal(&query).await
}

#[tauri::command]
pub async fn search_work(query: String) -> Result<serde_json::Value, String> {
    let mut cand = Vec::new();
    if let Ok(a) = search_anilist_internal(&query).await {
        let t = a["title"].as_str().unwrap_or(&query).to_string();
        cand.push(WorkSearchCandidate {
            id: format!("anilist:{}", normalize_text(&t)),
            title: t,
            synopsis: a["synopsis"].as_str().unwrap_or_default().into(),
            genres: json_string_vec(&a["genres"]),
            characters: json_string_vec(&a["characters"]),
            source: "anilist".into(),
            source_url: "".into(),
            cover_url: a["cover_url"].as_str().unwrap_or_default().into(),
            score: 100.0,
        });
    }
    Ok(serde_json::json!({ "query": query, "candidates": cand }))
}

#[tauri::command]
pub async fn enrich_work_context(
    app: AppHandle,
    selection: WorkSearchCandidate,
) -> Result<serde_json::Value, String> {
    let storage = crate::storage::service_for_app(&app)?;
    let storage_paths = storage.ensure_base_dirs()?;
    storage.check_writable()?;
    crate::storage::set_configured_paths(storage_paths.clone());

    let source_results = search_external_context_sources(&selection.title).await;
    let merged_candidates = merge_external_candidates(&source_results);
    let characters = unique_strings(
        selection.characters.iter().cloned().chain(
            merged_candidates
                .iter()
                .filter(|candidate| candidate.kind == "character")
                .map(|candidate| candidate.source.clone()),
        ),
    );
    let aliases = unique_strings(
        merged_candidates
            .iter()
            .filter(|candidate| candidate.kind == "alias")
            .map(|candidate| candidate.source.clone()),
    );
    let terms = unique_strings(
        merged_candidates
            .iter()
            .filter(|candidate| candidate.kind == "term")
            .map(|candidate| candidate.source.clone()),
    );
    let synopsis = best_context_synopsis(&source_results, &selection.synopsis);
    let genres = merged_context_genres(&source_results, &selection.genres);
    let cover_url = best_cover_url(&source_results, &selection.cover_url);

    let profile = crate::work_context::new_profile(
        &selection.title,
        "en",
        "pt-BR",
        &synopsis,
        genres,
        characters.clone(),
        terms,
        vec![],
    );
    let profile = crate::work_context::load_or_create_profile(&storage_paths.works, profile)?;
    let risk_level = crate::work_context::risk_level(&profile.context_quality, 0);
    let profile_terms = value_strings(&profile.terms);
    let profile_genres = profile.genre.clone();
    let glossary_entries_count =
        auto_save_character_glossary(&storage_paths.works, &profile.work_id, &merged_candidates)?;
    let source_refs = source_results
        .iter()
        .filter(|result| result.status == "found")
        .map(|result| ContextSourceRef {
            source: result.source.clone(),
            title: if result.title.trim().is_empty() {
                profile.title.clone()
            } else {
                result.title.clone()
            },
            url: result.url.clone(),
            snippet: result.synopsis.clone(),
        })
        .collect::<Vec<_>>();
    let source_result_values = source_results
        .iter()
        .map(source_result_value)
        .collect::<Vec<_>>();
    let glossary_candidate_values = merged_candidates
        .iter()
        .map(candidate_value)
        .collect::<Vec<_>>();

    let context = EnrichedWorkContext {
        work_id: profile.work_id,
        title: best_context_title(&source_results, &profile.title),
        synopsis: profile.synopsis,
        cover_url,
        genres: profile_genres,
        characters: characters.clone(),
        aliases,
        terms: profile_terms.clone(),
        context_quality: profile.context_quality,
        risk_level,
        sources_used: source_refs,
        source_results: source_result_values,
        glossary_candidates: glossary_candidate_values,
        internet_context_loaded: source_results.iter().any(|result| result.status == "found"),
        glossary_entries_count,
        relationships: vec![],
        factions: vec![],
        arc_summaries: vec![],
        lexical_memory: HashMap::new(),
    };
    serde_json::to_value(context).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn warmup_visual_stack(app: AppHandle) -> Result<String, String> {
    {
        let mut state = VISION_WORKER_WARMUP_STATE.lock().await;
        match *state {
            VisualWarmupState::Ready => return Ok("ready".into()),
            VisualWarmupState::Running => return Ok("warming".into()),
            VisualWarmupState::Idle => {
                *state = VisualWarmupState::Running;
            }
        }
    }

    let result = async {
        let worker_path = get_vision_worker_path(&app)?;
        if worker_path.trim().is_empty() {
            return Ok("ready:no-worker".to_string());
        }
        let storage = crate::storage::service_for_app(&app)?;
        let paths = storage.ensure_base_dirs()?;
        let runtime_root = paths
            .models
            .parent()
            .map(|path| path.to_path_buf())
            .unwrap_or_else(|| paths.root.clone());

        let mut cmd = Command::new(&worker_path);
        cmd.arg("--warmup")
            .arg("--runtime-root")
            .arg(runtime_root)
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        apply_sidecar_env(&mut cmd, &worker_path)?;
        let status = cmd.status().await.map_err(|e| e.to_string())?;
        if status.success() {
            Ok("ready".to_string())
        } else {
            Err(format!("vision worker warmup saiu com status {status}"))
        }
    }
    .await;

    if result.is_ok() {
        if let Err(error) = warmup_editor_visual_worker(app.clone()).await {
            eprintln!("[TraduzAi] Warmup visual do editor falhou: {error}");
        }
        if let Err(error) = warmup_editor_inpaint_worker(app.clone()).await {
            eprintln!("[TraduzAi] Warmup do inpaint do editor falhou: {error}");
        }
    }

    let mut state = VISION_WORKER_WARMUP_STATE.lock().await;
    *state = if result.is_ok() {
        VisualWarmupState::Ready
    } else {
        VisualWarmupState::Idle
    };
    result
}

#[derive(Debug, Serialize)]
pub struct GpuInfo {
    pub available: bool,
    pub name: String,
}
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
struct HardwareFacts {
    cpu_name: String,
    cpu_cores: u32,
    cpu_threads: u32,
    ram_gb: f64,
    gpu_available: bool,
    gpu_name: String,
    gpu_vram_gb: Option<f64>,
}
#[derive(Debug, Serialize, Clone)]
pub struct QualityEstimateTable {
    pub rapida: f64,
    pub normal: f64,
    pub alta: f64,
}
#[derive(Debug, Serialize, Clone)]
pub struct SystemProfile {
    pub cpu_name: String,
    pub cpu_cores: u32,
    pub cpu_threads: u32,
    pub ram_gb: f64,
    pub gpu_available: bool,
    pub gpu_name: String,
    pub gpu_vram_gb: Option<f64>,
    pub performance_tier: String,
    pub startup_seconds: f64,
    pub seconds_per_page: QualityEstimateTable,
}

#[cfg(test)]
mod tests {
    use super::{
        is_project_summary_mismatch, render_preview_paths, resolve_project_json_path,
        sidecar_env_overrides, HardwareFacts,
    };

    #[test]
    fn resolve_project_json_path_accepts_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let project = temp.path().join("project.json");
        std::fs::write(&project, "{}").expect("write project");

        let resolved =
            resolve_project_json_path(temp.path().to_str().expect("utf8 path")).expect("resolve");
        assert_eq!(resolved, project);
    }

    #[test]
    fn resolve_project_json_path_accepts_direct_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let project = temp.path().join("project.json");
        std::fs::write(&project, "{}").expect("write project");

        let resolved =
            resolve_project_json_path(project.to_str().expect("utf8 path")).expect("resolve");
        assert_eq!(resolved, project);
    }

    #[test]
    fn render_preview_paths_use_safe_cache_filename() {
        let temp = tempfile::tempdir().expect("tempdir");
        let project = temp.path().join("project.json");
        std::fs::write(&project, "{}").expect("write project");

        let (override_path, output_path) =
            render_preview_paths(&project, 1, "abc/123:unsafe", "jpg").expect("preview paths");

        assert!(override_path.ends_with("render-cache/preview/002-abc_123_unsafe.json"));
        assert!(output_path.ends_with("render-cache/preview/002-abc_123_unsafe.jpg"));
    }

    #[test]
    fn sidecar_env_does_not_force_local_translation_by_default() {
        let envs = sidecar_env_overrides("python");
        let keys: Vec<&str> = envs.iter().map(|(key, _)| key.as_str()).collect();

        assert!(keys.contains(&"PYTHONIOENCODING"));
        assert!(keys.contains(&"PYTHONUTF8"));
        assert!(!keys.contains(&"TRADUZAI_PREFER_LOCAL_TRANSLATION"));
        assert!(!keys.contains(&"MANGATL_PREFER_LOCAL_TRANSLATION"));
    }

    #[test]
    fn project_summary_mismatch_is_fast_worker_retryable() {
        assert!(is_project_summary_mismatch(
            "Falha no reinpaint: log.summary nao bate com project.json: translated_regions"
        ));
        assert!(!is_project_summary_mismatch(
            "Falha no reinpaint: modelo indisponivel"
        ));
    }

    #[test]
    fn hardware_facts_accepts_decimal_ram_gb() {
        let facts: HardwareFacts = serde_json::from_str(
            r#"{
                "cpu_name": "AMD Ryzen",
                "cpu_cores": 6,
                "cpu_threads": 12,
                "ram_gb": 31.9,
                "gpu_available": true,
                "gpu_name": "NVIDIA GeForce RTX 4060",
                "gpu_vram_gb": 8.0
            }"#,
        )
        .expect("parse hardware facts");

        assert!((facts.ram_gb - 31.9).abs() < f64::EPSILON);
    }

    #[test]
    fn merges_same_character_across_context_sources() {
        let anilist = super::ExternalContextData {
            source: "anilist".into(),
            status: "found".into(),
            confidence: 0.92,
            title: "The Regressed Mercenary Has a Plan".into(),
            synopsis: String::new(),
            genres: vec![],
            candidates: vec![super::ExternalContextCandidate {
                kind: "character".into(),
                source: "Giselle Perdium".into(),
                target: "Giselle Perdium".into(),
                confidence: 0.95,
                sources: vec!["anilist".into()],
                status: "candidate".into(),
                protect: true,
                aliases: vec![],
                forbidden: vec![],
                notes: String::new(),
            }],
            url: String::new(),
            error: String::new(),
            cover_url: String::new(),
        };
        let mal = super::ExternalContextData {
            source: "myanimelist".into(),
            status: "found".into(),
            confidence: 0.86,
            title: "The Regressed Mercenary Has a Plan".into(),
            synopsis: String::new(),
            genres: vec![],
            candidates: vec![super::ExternalContextCandidate {
                kind: "character".into(),
                source: "Giselle Perdium".into(),
                target: "Giselle Perdium".into(),
                confidence: 0.90,
                sources: vec!["myanimelist".into()],
                status: "candidate".into(),
                protect: true,
                aliases: vec!["Giselle".into()],
                forbidden: vec![],
                notes: String::new(),
            }],
            url: String::new(),
            error: String::new(),
            cover_url: String::new(),
        };

        let merged = super::merge_external_candidates(&[anilist, mal]);

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].source, "Giselle Perdium");
        assert_eq!(merged[0].sources, vec!["anilist", "myanimelist"]);
        assert_eq!(merged[0].aliases, vec!["Giselle"]);
    }

    #[test]
    fn auto_saves_character_candidates_to_glossary() {
        let temp = tempfile::tempdir().expect("tempdir");
        let work_id = "the-regressed-mercenary-has-a-plan";
        let mut stale = crate::glossary::empty_glossary(work_id);
        stale.entries.push(crate::glossary::GlossaryEntry {
            id: "character_wrong".into(),
            source: "Wrong Character".into(),
            target: "Wrong Character".into(),
            entry_type: "character".into(),
            case_sensitive: false,
            protect: true,
            aliases: vec![],
            forbidden: vec![],
            confidence: 0.9,
            status: "reviewed".into(),
            notes: "Nome de personagem importado automaticamente do contexto online da obra."
                .into(),
            context_rule: String::new(),
            sources: vec!["myanimelist".into()],
        });
        crate::glossary::save(temp.path(), &stale).expect("save stale glossary");
        let candidates = vec![super::ExternalContextCandidate {
            kind: "character".into(),
            source: "Giselle Perdium".into(),
            target: "Giselle Perdium".into(),
            confidence: 0.95,
            sources: vec!["anilist".into(), "myanimelist".into()],
            status: "candidate".into(),
            protect: true,
            aliases: vec!["Giselle".into()],
            forbidden: vec![],
            notes: String::new(),
        }];

        let count = super::auto_save_character_glossary(temp.path(), work_id, &candidates)
            .expect("save glossary");
        let glossary = crate::glossary::load(temp.path(), work_id).expect("load glossary");

        assert_eq!(count, 1);
        assert!(!glossary
            .entries
            .iter()
            .any(|entry| entry.source == "Wrong Character"));
        assert_eq!(glossary.entries[0].source, "Giselle Perdium");
        assert_eq!(glossary.entries[0].entry_type, "character");
        assert!(glossary.entries[0].protect);
        assert_eq!(glossary.entries[0].status, "reviewed");
    }
}
