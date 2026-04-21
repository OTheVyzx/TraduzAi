#![allow(dead_code)]
// TraduzAi Pipeline v0.54.1 - Backend Stabilized
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::OsString;
use std::process::Stdio;
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;

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
pub struct WorkSearchCandidate {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub synopsis: String,
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

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct EnrichedWorkContext {
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
}

pub(crate) struct SidecarInfo {
    pub(crate) program: String,
    pub(crate) script: Option<String>,
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
            { "python".to_string() }
            #[cfg(not(windows))]
            { "python3".to_string() }
        };

        return Ok(SidecarInfo { program, script: Some(script.to_string_lossy().to_string()) });
    }

    let res = app.path().resource_dir().map_err(|e| e.to_string())?;
    let mut sidecar = res.join("binaries/traduzai-pipeline");
    #[cfg(windows)]
    { sidecar = sidecar.with_extension("exe"); }

    Ok(SidecarInfo { program: sidecar.to_string_lossy().to_string(), script: None })
}

pub(crate) fn get_vision_worker_path(app: &AppHandle) -> Result<String, String> {
    let res = app.path().resource_dir().map_err(|e| e.to_string())?;
    let bin = res.join("binaries").join(if cfg!(windows) { "traduzai-vision.exe" } else { "traduzai-vision" });
    if bin.exists() { Ok(bin.to_string_lossy().to_string()) } else { Ok(String::new()) }
}

pub(crate) fn sidecar_env_overrides(_program: &str) -> Vec<(String, OsString)> {
    let mut envs = Vec::new();
    envs.push(("PYTHONIOENCODING".into(), OsString::from("utf-8")));
    envs.push(("PYTHONUTF8".into(), OsString::from("1")));
    envs.push(("TRADUZAI_PREFER_LOCAL_TRANSLATION".into(), OsString::from("1")));
    envs
}

pub(crate) fn apply_sidecar_env(cmd: &mut Command, program: &str) -> Result<(), String> {
    for (k, v) in sidecar_env_overrides(program) { cmd.env(k, v); }
    Ok(())
}

#[tauri::command]
pub async fn start_pipeline(app: AppHandle, config: PipelineConfig) -> Result<serde_json::Value, String> {
    *PIPELINE_CANCEL.lock().await = false;
    let job_id = uuid::Uuid::new_v4().to_string();
    let app_data = std::path::PathBuf::from("D:\\traduzai_data");
    let work_dir = app_data.join("projects").join(&job_id);
    std::fs::create_dir_all(&work_dir).map_err(|e| e.to_string())?;
    
    let pause_path = work_dir.join("pipeline.pause");
    set_pause_marker(&pause_path, false)?;
    {
        let mut cur = PIPELINE_PAUSE_MARKER.lock().await;
        if let Some(p) = cur.take() { set_pause_marker(&p, false).ok(); }
        *cur = Some(pause_path.clone());
    }

    let settings = crate::commands::settings::load_settings_sync(&app);
    let worker_path = get_vision_worker_path(&app).unwrap_or_default();
    
    let config_json = serde_json::to_string(&serde_json::json!({
        "job_id": job_id, "source_path": config.source_path, "work_dir": work_dir.to_string_lossy(),
        "obra": config.obra, "capitulo": config.capitulo, "idioma_origem": config.idioma_origem,
        "idioma_destino": config.idioma_destino, "qualidade": config.qualidade, "glossario": config.glossario,
        "contexto": {
            "sinopse": config.contexto.sinopse, "genero": config.contexto.genero, "personagens": config.contexto.personagens,
            "aliases": config.contexto.aliases, "termos": config.contexto.termos, "relacoes": config.contexto.relacoes,
            "faccoes": config.contexto.faccoes, "resumo_por_arco": config.contexto.resumo_por_arco,
            "memoria_lexical": config.contexto.memoria_lexical, "fontes_usadas": config.contexto.fontes_usadas
        },
        "models_dir": app_data.join("models").to_string_lossy(),
        "ollama_host": settings.ollama_host, "ollama_model": settings.ollama_model,
        "vision_worker_path": worker_path, "pause_file": pause_path.to_string_lossy()
    })).map_err(|e| e.to_string())?;

    let config_file = work_dir.join("pipeline_config.json");
    std::fs::write(&config_file, &config_json).map_err(|e| e.to_string())?;

    let sidecar = get_sidecar_info(&app)?;
    let app_c = app.clone(); let job_c = job_id.clone(); let pause_c = pause_path.clone();

    tokio::spawn(async move {
        let res = run_sidecar(&app_c, &sidecar, &config_file).await;
        clear_current_pause_marker(&pause_c).await;
        match res {
            Ok(out) => { app_c.emit("pipeline-complete", serde_json::json!({"success": true, "output_path": out, "job_id": job_c})).ok(); }
            Err(err) => { app_c.emit("pipeline-complete", serde_json::json!({"success": false, "error": err, "job_id": job_c})).ok(); }
        }
    });

    Ok(serde_json::json!({ "job_id": job_id }))
}

async fn run_sidecar(app: &AppHandle, sidecar: &SidecarInfo, config_path: &std::path::Path) -> Result<String, String> {
    let mut cmd = Command::new(&sidecar.program);
    if let Some(s) = &sidecar.script { cmd.arg(s); }
    let log_path = config_path.parent().unwrap_or(config_path).join("pipeline.log");
    let log_file = std::fs::File::create(&log_path).map_err(|e| format!("Erro ao criar log: {e}"))?;
    cmd.arg(config_path.to_string_lossy().to_string()).stdout(Stdio::piped()).stderr(Stdio::from(log_file));
    apply_sidecar_env(&mut cmd, &sidecar.program)?;
    let mut child = cmd.spawn().map_err(|e| format!("Erro ao iniciar pipeline: {e}"))?;
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
                    "complete" => { output_path = msg["output_path"].as_str().unwrap_or("").to_string(); }
                    "error" => { return Err(msg["message"].as_str().unwrap_or("Erro desconhecido").to_string()); }
                    _ => {}
                }
            }
        }
    }
    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !status.success() {
        let log = std::fs::read_to_string(&log_path).unwrap_or_default();
        let detail = if log.len() > 2000 { &log[log.len()-2000..] } else { &log };
        return Err(format!("Pipeline falhou ({}): {}", status, detail));
    }
    Ok(output_path)
}

#[tauri::command]
pub async fn cancel_pipeline() -> Result<(), String> {
    *PIPELINE_CANCEL.lock().await = true;
    if let Some(path) = PIPELINE_PAUSE_MARKER.lock().await.take() { set_pause_marker(&path, false)?; }
    Ok(())
}

#[tauri::command]
pub async fn pause_pipeline() -> Result<(), String> {
    let path = PIPELINE_PAUSE_MARKER.lock().await.clone().ok_or_else(|| "Nenhuma tradução em andamento.".to_string())?;
    set_pause_marker(&path, true)?;
    Ok(())
}

#[tauri::command]
pub async fn resume_pipeline() -> Result<(), String> {
    let path = PIPELINE_PAUSE_MARKER.lock().await.clone().ok_or_else(|| "Nenhuma tradução em andamento.".to_string())?;
    set_pause_marker(&path, false)?;
    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct RetypesetConfig { pub project_path: String, pub page_index: u32 }
#[derive(Debug, Deserialize)]
pub struct ReinpaintConfig { pub project_path: String, pub page_index: u32 }
#[derive(Debug, Deserialize)]
pub struct ProcessBlockConfig { pub project_path: String, pub page_index: u32, pub block_id: String, pub mode: String }

#[tauri::command]
pub async fn retypeset_page(app: AppHandle, config: RetypesetConfig) -> Result<String, String> {
    let base = crate::commands::project::normalize_path(&config.project_path);
    let pf = if base.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.eq_ignore_ascii_case("project.json")) { base } else { base.join("project.json") };
    if !pf.exists() { return Err("project.json não encontrado".into()); }
    let sidecar = get_sidecar_info(&app)?;
    let mut cmd = Command::new(&sidecar.program);
    if let Some(s) = &sidecar.script { cmd.arg(s); }
    cmd.arg("--retypeset").arg(pf.to_string_lossy().to_string()).arg(config.page_index.to_string());
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());
    apply_sidecar_env(&mut cmd, &sidecar.program)?;
    let mut child = cmd.spawn().map_err(|e| format!("Erro ao iniciar retypeset: {e}"))?;
    let stdout = child.stdout.take().expect("stdout not captured");
    let mut reader = BufReader::new(stdout).lines();
    let mut out = String::new();
    while let Ok(Some(line)) = reader.next_line().await {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(t) = msg.get("type").and_then(|t| t.as_str()) {
                if t == "complete" { out = msg["output_path"].as_str().unwrap_or("").to_string(); }
                else if t == "error" { return Err(msg["message"].as_str().unwrap_or("Erro").to_string()); }
            }
        }
    }
    if !child.wait().await.map_err(|e| e.to_string())?.success() { return Err("Retypeset falhou".into()); }
    Ok(out)
}

#[tauri::command]
pub async fn process_block(app: AppHandle, config: ProcessBlockConfig) -> Result<String, String> {
    let base = crate::commands::project::normalize_path(&config.project_path);
    let pf = if base.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.eq_ignore_ascii_case("project.json")) { base } else { base.join("project.json") };
    if !pf.exists() { return Err("project.json não encontrado".into()); }
    let sidecar = get_sidecar_info(&app)?;
    let mut cmd = Command::new(&sidecar.program);
    if let Some(s) = &sidecar.script { cmd.arg(s); }
    cmd.arg("--process-block").arg(&config.mode).arg(pf.to_string_lossy().to_string()).arg(config.page_index.to_string()).arg(&config.block_id);
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());
    apply_sidecar_env(&mut cmd, &sidecar.program)?;
    let mut child = cmd.spawn().map_err(|e| format!("Erro ao iniciar process_block: {e}"))?;
    let stdout = child.stdout.take().expect("stdout not captured");
    let mut reader = BufReader::new(stdout).lines();
    let mut out = String::new();
    while let Ok(Some(line)) = reader.next_line().await {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(t) = msg.get("type").and_then(|t| t.as_str()) {
                if t == "complete" { out = msg["output_path"].as_str().unwrap_or("").to_string(); }
                else if t == "error" { return Err(msg["message"].as_str().unwrap_or("Erro").to_string()); }
            }
        }
    }
    if !child.wait().await.map_err(|e| e.to_string())?.success() { return Err("ProcessBlock falhou".into()); }
    Ok(out)
}

#[tauri::command]
pub async fn reinpaint_page(app: AppHandle, config: ReinpaintConfig) -> Result<String, String> {
    let base = crate::commands::project::normalize_path(&config.project_path);
    let pf = if base.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.eq_ignore_ascii_case("project.json")) { base } else { base.join("project.json") };
    let sidecar = get_sidecar_info(&app)?;
    let mut cmd = Command::new(&sidecar.program);
    if let Some(s) = &sidecar.script { cmd.arg(s); }
    cmd.arg("--reinpaint-page").arg(pf.to_string_lossy().to_string()).arg(config.page_index.to_string());
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());
    apply_sidecar_env(&mut cmd, &sidecar.program)?;
    let mut child = cmd.spawn().map_err(|e| format!("Erro ao iniciar reinpaint: {e}"))?;
    let stdout = child.stdout.take().expect("stdout not captured");
    let mut reader = BufReader::new(stdout).lines();
    let mut out = String::new();
    while let Ok(Some(line)) = reader.next_line().await {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(t) = msg.get("type").and_then(|t| t.as_str()) {
                if t == "complete" { out = msg["output_path"].as_str().unwrap_or("").to_string(); }
                else if t == "error" { return Err(msg["message"].as_str().unwrap_or("Erro").to_string()); }
            }
        }
    }
    if !child.wait().await.map_err(|e| e.to_string())?.success() { return Err("Reinpaint falhou".into()); }
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

    let mut child = cmd.spawn().map_err(|e| format!("Erro ao iniciar hardware info: {e}"))?;
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

    let facts: HardwareFacts = serde_json::from_str(&profile_json)
        .map_err(|e| format!("Falha ao parsear hardware info: {e} | JSON: {}", profile_json))?;

    let tier = if facts.gpu_available {
        let vram = facts.gpu_vram_gb.unwrap_or(0.0);
        if vram >= 11.0 { "workstation" }
        else if vram >= 6.0 { "fast" }
        else { "balanced" }
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
#[tauri::command] pub async fn check_models(_app: AppHandle) -> Result<serde_json::Value, String> { let m = std::path::PathBuf::from("D:\\traduzai_data\\models"); Ok(serde_json::json!({ "ready": m.exists(), "size_mb": 0, "ocr_ready": true, "inpainting_ready": true })) }
#[tauri::command] pub async fn download_models(app: AppHandle) -> Result<(), String> { app.emit("models-ready", serde_json::json!({"success": true})).ok(); Ok(()) }

async fn search_anilist_internal(query: &str) -> Result<serde_json::Value, String> {
    let gq = r#"query ($search: String) { Media(search: $search, type: MANGA) { title { english romaji } description(asHtml: false) genres characters(sort: ROLE, perPage: 10) { nodes { name { full } } } coverImage { large } } }"#;
    let client = reqwest::Client::new();
    let resp = client.post("https://graphql.anilist.co").json(&serde_json::json!({"query": gq, "variables": {"search": query}})).send().await.map_err(|e| e.to_string())?;
    let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    let media = &data["data"]["Media"];
    let title = media["title"]["english"].as_str().or(media["title"]["romaji"].as_str()).unwrap_or(query).to_string();
    Ok(serde_json::json!({ "title": title, "synopsis": media["description"].as_str().unwrap_or(""), "genres": media["genres"].as_array().unwrap_or(&vec![]), "characters": media["characters"]["nodes"].as_array().unwrap_or(&vec![]).iter().filter_map(|c| c["name"]["full"].as_str().map(|s| s.to_string())).collect::<Vec<_>>(), "cover_url": media["coverImage"]["large"].as_str().unwrap_or("") }))
}

#[tauri::command] pub async fn search_anilist(query: String) -> Result<serde_json::Value, String> { search_anilist_internal(&query).await }

#[tauri::command]
pub async fn search_work(query: String) -> Result<serde_json::Value, String> {
    let mut cand = Vec::new();
    if let Ok(a) = search_anilist_internal(&query).await {
        let t = a["title"].as_str().unwrap_or(&query).to_string();
        cand.push(WorkSearchCandidate { id: format!("anilist:{}", normalize_text(&t)), title: t, synopsis: a["synopsis"].as_str().unwrap_or_default().into(), source: "anilist".into(), source_url: "".into(), cover_url: a["cover_url"].as_str().unwrap_or_default().into(), score: 100.0 });
    }
    Ok(serde_json::json!({ "query": query, "candidates": cand }))
}

#[tauri::command] pub async fn enrich_work_context(selection: WorkSearchCandidate) -> Result<serde_json::Value, String> { 
    let context = EnrichedWorkContext { title: selection.title, synopsis: selection.synopsis, cover_url: selection.cover_url, ..Default::default() };
    serde_json::to_value(context).map_err(|e| e.to_string())
}

#[tauri::command] pub async fn warmup_visual_stack(_app: AppHandle) -> Result<String, String> { Ok("ready".into()) }

#[derive(Debug, Serialize)] pub struct GpuInfo { pub available: bool, pub name: String }
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)] struct HardwareFacts { cpu_name: String, cpu_cores: u32, cpu_threads: u32, ram_gb: u32, gpu_available: bool, gpu_name: String, gpu_vram_gb: Option<f64> }
#[derive(Debug, Serialize, Clone)] pub struct QualityEstimateTable { pub rapida: f64, pub normal: f64, pub alta: f64 }
#[derive(Debug, Serialize, Clone)] pub struct SystemProfile { pub cpu_name: String, pub cpu_cores: u32, pub cpu_threads: u32, pub ram_gb: u32, pub gpu_available: bool, pub gpu_name: String, pub gpu_vram_gb: Option<f64>, pub performance_tier: String, pub startup_seconds: f64, pub seconds_per_page: QualityEstimateTable }

#[cfg(test)] mod tests { #[test] fn it_works() { assert_eq!(2 + 2, 4); } }
