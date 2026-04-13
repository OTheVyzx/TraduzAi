use html_escape::decode_html_entities;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::process::Stdio;
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;

static PIPELINE_CANCEL: once_cell::sync::Lazy<Mutex<bool>> =
    once_cell::sync::Lazy::new(|| Mutex::new(false));
static PIPELINE_PAUSE_MARKER: once_cell::sync::Lazy<Mutex<Option<std::path::PathBuf>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(None));
static VISUAL_WARMUP_STATE: once_cell::sync::Lazy<Mutex<VisualWarmupState>> =
    once_cell::sync::Lazy::new(|| Mutex::new(VisualWarmupState::Idle));
static VISION_WORKER_WARMUP_STATE: once_cell::sync::Lazy<Mutex<VisualWarmupState>> =
    once_cell::sync::Lazy::new(|| Mutex::new(VisualWarmupState::Idle));

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
    pub glossario: std::collections::HashMap<String, String>,
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

/// Resolved paths for spawning the Python pipeline.
struct SidecarInfo {
    /// The executable to run (venv python or system python or bundled binary).
    program: String,
    /// Optional script path (used when program is Python interpreter).
    script: Option<String>,
}

fn build_http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 TraduzAi/0.1")
        .build()
        .map_err(|e| e.to_string())
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

fn clean_title_for_source(raw_title: &str, source: &str) -> String {
    let mut title = decode_html_entities(raw_title).to_string();
    title = title.replace("&#39;", "'");
    if source == "webnovel" {
        title = title.replace(" - WebNovel", "");
    }
    if source == "fandom" {
        title = title.replace(" Wiki | Fandom", "");
        title = title.replace(" | Fandom", "");
    }
    title.trim().to_string()
}

fn score_title_match(query: &str, title: &str, source: &str) -> f64 {
    let normalized_query = normalize_text(query);
    let normalized_title = normalize_text(title);
    if normalized_query.is_empty() || normalized_title.is_empty() {
        return 0.0;
    }

    if normalized_query == normalized_title {
        return 100.0;
    }

    if normalized_title.contains(&normalized_query) || normalized_query.contains(&normalized_title)
    {
        return 88.0;
    }

    let q_tokens: HashSet<_> = normalized_query.split_whitespace().collect();
    let t_tokens: HashSet<_> = normalized_title.split_whitespace().collect();
    let overlap = q_tokens.intersection(&t_tokens).count() as f64;
    let union = q_tokens.union(&t_tokens).count() as f64;
    let token_score = if union > 0.0 {
        (overlap / union) * 70.0
    } else {
        0.0
    };
    let source_boost = match source {
        "anilist" => 8.0,
        "webnovel" => 5.0,
        "fandom" => 3.0,
        _ => 0.0,
    };

    token_score + source_boost
}

fn parse_bing_rss_items(xml: &str) -> Vec<SearchSnippet> {
    let item_re = Regex::new(
        r"(?s)<item>\s*<title>(.*?)</title>\s*<link>(.*?)</link>\s*<description>(.*?)</description>",
    )
    .expect("valid regex");

    item_re
        .captures_iter(xml)
        .map(|cap| SearchSnippet {
            title: decode_html_entities(cap.get(1).map(|m| m.as_str()).unwrap_or("")).to_string(),
            link: decode_html_entities(cap.get(2).map(|m| m.as_str()).unwrap_or("")).to_string(),
            snippet: decode_html_entities(cap.get(3).map(|m| m.as_str()).unwrap_or("")).to_string(),
        })
        .collect()
}

async fn fetch_bing_site_results(
    query: &str,
    site_filter: &str,
) -> Result<Vec<SearchSnippet>, String> {
    let client = build_http_client()?;
    let search_query = format!("site:{site_filter} \"{query}\"");
    let url = reqwest::Url::parse_with_params(
        "https://www.bing.com/search",
        &[("format", "rss"), ("q", search_query.as_str())],
    )
    .map_err(|e| e.to_string())?;

    let xml = client
        .get(url)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())?;

    Ok(parse_bing_rss_items(&xml))
}

fn filter_source_results(items: Vec<SearchSnippet>, source: &str) -> Vec<SearchSnippet> {
    items
        .into_iter()
        .filter(|item| match source {
            "webnovel" => item.link.contains("webnovel.com/") && item.link.contains("/book/"),
            "fandom" => item.link.contains(".fandom.com/"),
            _ => true,
        })
        .collect()
}

fn build_candidate_from_snippet(
    query: &str,
    source: &str,
    item: SearchSnippet,
) -> WorkSearchCandidate {
    let cleaned_title = clean_title_for_source(&item.title, source);
    WorkSearchCandidate {
        id: format!(
            "{source}:{}",
            normalize_text(&cleaned_title).replace(' ', "-")
        ),
        title: cleaned_title.clone(),
        synopsis: item.snippet.clone(),
        source: source.to_string(),
        source_url: item.link,
        cover_url: String::new(),
        score: score_title_match(query, &cleaned_title, source),
    }
}

fn dedupe_candidates(mut candidates: Vec<WorkSearchCandidate>) -> Vec<WorkSearchCandidate> {
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut seen = HashSet::new();
    candidates
        .into_iter()
        .filter(|candidate| {
            let key = if !candidate.source_url.is_empty() {
                candidate.source_url.clone()
            } else {
                format!("{}:{}", candidate.source, normalize_text(&candidate.title))
            };
            seen.insert(key)
        })
        .collect()
}

fn dedupe_strings(values: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut output = Vec::new();
    for value in values {
        let cleaned = value.trim().to_string();
        if cleaned.is_empty() {
            continue;
        }
        let key = normalize_text(&cleaned);
        if seen.insert(key) {
            output.push(cleaned);
        }
    }
    output
}

fn build_context_sources(items: &[SearchSnippet], source: &str) -> Vec<ContextSourceRef> {
    items
        .iter()
        .take(6)
        .map(|item| ContextSourceRef {
            source: source.to_string(),
            title: clean_title_for_source(&item.title, source),
            url: item.link.clone(),
            snippet: item.snippet.clone(),
        })
        .collect()
}

fn build_enriched_context(
    selection: &WorkSearchCandidate,
    anilist: Option<serde_json::Value>,
    webnovel_results: Vec<SearchSnippet>,
    fandom_results: Vec<SearchSnippet>,
) -> EnrichedWorkContext {
    let fallback_title = selection.title.clone();
    let title = anilist
        .as_ref()
        .and_then(|value| value.get("title"))
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .unwrap_or(&fallback_title)
        .to_string();

    let synopsis_parts = dedupe_strings(
        anilist
            .as_ref()
            .and_then(|value| value.get("synopsis"))
            .and_then(|value| value.as_str())
            .into_iter()
            .chain(std::iter::once(selection.synopsis.as_str()))
            .chain(
                webnovel_results
                    .iter()
                    .take(4)
                    .map(|item| item.snippet.as_str()),
            )
            .chain(
                fandom_results
                    .iter()
                    .take(3)
                    .map(|item| item.snippet.as_str()),
            )
            .map(|value| value.to_string()),
    );

    let synopsis = synopsis_parts
        .into_iter()
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");

    let genres = anilist
        .as_ref()
        .and_then(|value| value.get("genres"))
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(|value| value.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let characters = anilist
        .as_ref()
        .and_then(|value| value.get("characters"))
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(|value| value.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let aliases = dedupe_strings(
        webnovel_results
            .iter()
            .chain(fandom_results.iter())
            .map(|item| {
                clean_title_for_source(
                    &item.title,
                    if item.link.contains("webnovel.com") {
                        "webnovel"
                    } else {
                        "fandom"
                    },
                )
            })
            .filter(|candidate_title| normalize_text(candidate_title) != normalize_text(&title)),
    );

    let arc_summaries = dedupe_strings(
        webnovel_results
            .iter()
            .take(5)
            .map(|item| item.snippet.clone()),
    );
    let sources_used = build_context_sources(&webnovel_results, "webnovel")
        .into_iter()
        .chain(build_context_sources(&fandom_results, "fandom"))
        .collect::<Vec<_>>();

    EnrichedWorkContext {
        title,
        synopsis: synopsis.chars().take(1600).collect(),
        genres: dedupe_strings(genres),
        characters: dedupe_strings(characters),
        aliases,
        terms: Vec::new(),
        relationships: Vec::new(),
        factions: Vec::new(),
        arc_summaries,
        lexical_memory: HashMap::new(),
        sources_used,
        cover_url: anilist
            .as_ref()
            .and_then(|value| value.get("cover_url"))
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string(),
    }
}

async fn search_anilist_internal(query: &str) -> Result<serde_json::Value, String> {
    let graphql_query = r#"
        query ($search: String) {
            Media(search: $search, type: MANGA) {
                title { english romaji }
                description(asHtml: false)
                genres
                characters(sort: ROLE, perPage: 10) {
                    nodes { name { full } }
                }
                coverImage { large }
            }
        }
    "#;

    let client = reqwest::Client::new();
    let resp = client
        .post("https://graphql.anilist.co")
        .json(&serde_json::json!({
            "query": graphql_query,
            "variables": { "search": query }
        }))
        .send()
        .await
        .map_err(|e| format!("Erro na busca: {e}"))?;

    let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    let media = &data["data"]["Media"];

    let title = media["title"]["english"]
        .as_str()
        .or(media["title"]["romaji"].as_str())
        .unwrap_or(query)
        .to_string();

    let synopsis = media["description"].as_str().unwrap_or("").to_string();

    let genres: Vec<String> = media["genres"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|g| g.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let characters: Vec<String> = media["characters"]["nodes"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|c| c["name"]["full"].as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let cover_url = media["coverImage"]["large"]
        .as_str()
        .unwrap_or("")
        .to_string();

    Ok(serde_json::json!({
        "title": title,
        "synopsis": synopsis,
        "genres": genres,
        "characters": characters,
        "cover_url": cover_url
    }))
}

fn get_sidecar_info(app: &AppHandle) -> Result<SidecarInfo, String> {
    if cfg!(debug_assertions) {
        // Dev mode: find pipeline/main.py relative to the project root.
        // The Rust process runs from src-tauri/, so parent() = project root.
        let project_root = std::env::current_dir()
            .map_err(|e| e.to_string())?
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap());

        let script = project_root.join("pipeline").join("main.py");

        // Prefer the venv Python so installed packages (PIL, cv2, etc.) are available.
        #[cfg(windows)]
        let venv_python = project_root
            .join("pipeline")
            .join("venv")
            .join("Scripts")
            .join("python.exe");
        #[cfg(not(windows))]
        let venv_python = project_root
            .join("pipeline")
            .join("venv")
            .join("bin")
            .join("python3");

        let program = if venv_python.exists() {
            venv_python.to_string_lossy().to_string()
        } else {
            // Fall back to system Python
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

    // Production: use the bundled sidecar binary.
    let resource_dir = app.path().resource_dir().map_err(|e| e.to_string())?;

    let mut sidecar = resource_dir.join("binaries").join("traduzai-pipeline");

    #[cfg(windows)]
    {
        sidecar = sidecar.with_extension("exe");
    }

    Ok(SidecarInfo {
        program: sidecar.to_string_lossy().to_string(),
        script: None,
    })
}

fn find_cuda_toolkit_root() -> Option<std::path::PathBuf> {
    for key in [
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
    ] {
        if let Some(value) = std::env::var_os(key) {
            let candidate = std::path::PathBuf::from(value);
            if candidate
                .join("bin")
                .join(if cfg!(windows) { "nvcc.exe" } else { "nvcc" })
                .exists()
            {
                return Some(candidate);
            }
        }
    }

    #[cfg(windows)]
    {
        let base = std::path::PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
        if let Ok(entries) = std::fs::read_dir(&base) {
            let mut versions = entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| path.join("bin").join("nvcc.exe").exists())
                .collect::<Vec<_>>();
            versions.sort();
            versions.reverse();
            if let Some(path) = versions.into_iter().next() {
                return Some(path);
            }
        }
    }

    None
}

fn infer_cudarc_cuda_version(cuda_root: &std::path::Path) -> Option<String> {
    let folder = cuda_root.file_name()?.to_string_lossy();
    let trimmed = folder.strip_prefix('v').unwrap_or(&folder);
    let mut parts = trimmed.split('.');
    let major = parts.next()?.parse::<u32>().ok()?;
    let minor = parts.next()?.parse::<u32>().ok()?;
    Some(format!("{major}0{minor}0"))
}

fn base_cuda_env_overrides() -> Vec<(String, OsString)> {
    let mut envs = Vec::new();
    let Some(cuda_root) = find_cuda_toolkit_root() else {
        return envs;
    };

    let root_os = cuda_root.as_os_str().to_os_string();
    envs.push(("CUDA_PATH".into(), root_os.clone()));
    envs.push(("CUDA_HOME".into(), root_os.clone()));
    envs.push(("CUDA_ROOT".into(), root_os.clone()));
    envs.push(("CUDA_TOOLKIT_ROOT_DIR".into(), root_os.clone()));

    if let Some(version) = infer_cudarc_cuda_version(&cuda_root) {
        envs.push(("CUDARC_CUDA_VERSION".into(), OsString::from(version)));
    }

    envs
}

fn dedupe_path_prefixes(prefixes: Vec<std::path::PathBuf>) -> Vec<std::path::PathBuf> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for prefix in prefixes {
        if !prefix.exists() {
            continue;
        }
        let key = prefix.to_string_lossy().to_lowercase();
        if seen.insert(key) {
            unique.push(prefix);
        }
    }
    unique
}

fn merged_path_with_prefixes(prefixes: Vec<std::path::PathBuf>) -> OsString {
    let mut merged = OsString::new();
    let mut first = true;
    for prefix in dedupe_path_prefixes(prefixes) {
        if !first {
            merged.push(";");
        }
        merged.push(prefix.as_os_str());
        first = false;
    }

    if let Some(existing) = std::env::var_os("PATH") {
        if !existing.is_empty() {
            if !first {
                merged.push(";");
            }
            merged.push(existing);
        }
    }

    merged
}

fn python_site_packages_roots(program: &str) -> Vec<std::path::PathBuf> {
    let program_path = std::path::PathBuf::from(program);
    let mut roots = Vec::new();

    #[cfg(windows)]
    {
        if let Some(venv_root) = program_path.parent().and_then(|path| path.parent()) {
            roots.push(venv_root.join("Lib").join("site-packages"));
        }
    }

    #[cfg(not(windows))]
    {
        if let Some(venv_root) = program_path.parent().and_then(|path| path.parent()) {
            if let Ok(entries) = std::fs::read_dir(venv_root.join("lib")) {
                for entry in entries.filter_map(|entry| entry.ok()) {
                    let path = entry.path().join("site-packages");
                    if path.exists() {
                        roots.push(path);
                    }
                }
            }
        }
    }

    dedupe_path_prefixes(roots)
}

fn python_cuda_runtime_dirs(program: &str) -> Vec<std::path::PathBuf> {
    let mut dirs = Vec::new();
    let relative_dirs = [
        std::path::PathBuf::from("nvidia").join("cuda_runtime").join("bin"),
        std::path::PathBuf::from("nvidia").join("cublas").join("bin"),
        std::path::PathBuf::from("nvidia").join("cufft").join("bin"),
        std::path::PathBuf::from("nvidia").join("curand").join("bin"),
        std::path::PathBuf::from("nvidia").join("cusolver").join("bin"),
        std::path::PathBuf::from("nvidia").join("cusparse").join("bin"),
        std::path::PathBuf::from("nvidia").join("cudnn").join("bin"),
        std::path::PathBuf::from("tensorrt_libs"),
    ];

    for root in python_site_packages_roots(program) {
        for relative in &relative_dirs {
            dirs.push(root.join(relative));
        }
    }

    dedupe_path_prefixes(dirs)
}

fn cuda_env_overrides() -> Vec<(String, OsString)> {
    let mut envs = base_cuda_env_overrides();
    let mut path_prefixes = Vec::new();
    if let Some(cuda_root) = find_cuda_toolkit_root() {
        let cuda_bin = cuda_root.join("bin");
        if cuda_bin.exists() {
            path_prefixes.push(cuda_bin);
        }
    }
    if !path_prefixes.is_empty() {
        envs.push(("PATH".into(), merged_path_with_prefixes(path_prefixes)));
    }
    envs
}

pub(crate) fn sidecar_env_overrides(program: &str) -> Vec<(String, OsString)> {
    let mut envs = base_cuda_env_overrides();
    let mut path_prefixes = Vec::new();
    if let Some(cuda_root) = find_cuda_toolkit_root() {
        let cuda_bin = cuda_root.join("bin");
        if cuda_bin.exists() {
            path_prefixes.push(cuda_bin);
        }
    }
    path_prefixes.extend(python_cuda_runtime_dirs(program));
    if !path_prefixes.is_empty() {
        envs.push(("PATH".into(), merged_path_with_prefixes(path_prefixes)));
    }
    envs.push(("PYTHONIOENCODING".into(), OsString::from("utf-8")));
    envs.push(("PYTHONUTF8".into(), OsString::from("1")));
    envs.push((
        "TRADUZAI_PREFER_LOCAL_TRANSLATION".into(),
        OsString::from("1"),
    ));
    envs
}

#[cfg(windows)]
fn find_vcvars64_bat() -> Option<std::path::PathBuf> {
    let candidates = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    ];
    candidates
        .iter()
        .map(std::path::PathBuf::from)
        .find(|path| path.exists())
}

fn maybe_seed_vision_worker_build_env(
    cmd: &mut std::process::Command,
    worker_dir: &std::path::Path,
) {
    let tool_venv = worker_dir.join(".toolvenv");

    #[cfg(windows)]
    {
        let cmake = tool_venv.join("Scripts").join("cmake.exe");
        if cmake.exists() {
            cmd.env("CMAKE", &cmake);
        }

        let libclang = tool_venv
            .join("Lib")
            .join("site-packages")
            .join("clang")
            .join("native");
        if libclang.exists() {
            cmd.env("LIBCLANG_PATH", &libclang);
        }

        if std::env::var_os("NVCC_PREPEND_FLAGS").is_none() {
            cmd.env(
                "NVCC_PREPEND_FLAGS",
                "-std=c++17 -Xcompiler=/Zc:preprocessor",
            );
        }
    }

    if std::env::var_os("LLAMA_CPP_TAG").is_none() {
        cmd.env("LLAMA_CPP_TAG", "b8665");
    }

    for (key, value) in cuda_env_overrides() {
        cmd.env(key, value);
    }
}

fn ensure_dev_vision_worker(project_root: &std::path::Path) -> Result<std::path::PathBuf, String> {
    let worker_dir = project_root.join("vision-worker");
    let manifest_path = worker_dir.join("Cargo.toml");
    #[cfg(windows)]
    let worker_exe_candidates = vec![
        worker_dir.join("target").join("debug").join("traduzai-vision.exe"),
        worker_dir.join("target").join("debug").join("mangatl-vision.exe"),
    ];
    #[cfg(not(windows))]
    let worker_exe_candidates = vec![worker_dir.join("target").join("debug").join("traduzai-vision")];

    for worker_exe in worker_exe_candidates.iter() {
        if worker_exe.exists() {
            return Ok(worker_exe.clone());
        }
    }

    if !manifest_path.exists() {
        return Err("Cargo.toml do vision-worker não encontrado".into());
    }

    #[cfg(windows)]
    let output = {
        let cargo_program = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
        if let Some(vcvars64) = find_vcvars64_bat() {
            let command = format!(
                "call \"{}\" >nul && \"{}\" build --manifest-path \"{}\"",
                vcvars64.display(),
                cargo_program,
                manifest_path.display()
            );
            let mut cmd = std::process::Command::new("cmd");
            cmd.arg("/C").arg(command).current_dir(&worker_dir);
            maybe_seed_vision_worker_build_env(&mut cmd, &worker_dir);
            cmd.output()
        } else {
            let mut cmd = std::process::Command::new(&cargo_program);
            cmd.arg("build")
                .arg("--manifest-path")
                .arg(&manifest_path)
                .current_dir(&worker_dir);
            maybe_seed_vision_worker_build_env(&mut cmd, &worker_dir);
            cmd.output()
        }
    }
    .map_err(|e| format!("Erro ao compilar vision-worker: {e}"))?;

    #[cfg(not(windows))]
    let output = {
        let mut cmd = std::process::Command::new("cargo");
        cmd.arg("build")
            .arg("--manifest-path")
            .arg(&manifest_path)
            .current_dir(&worker_dir);
        maybe_seed_vision_worker_build_env(&mut cmd, &worker_dir);
        cmd.output()
    }
    .map_err(|e| format!("Erro ao compilar vision-worker: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let detail = if !stderr.trim().is_empty() {
            stderr.to_string()
        } else {
            stdout.to_string()
        };
        return Err(format!("Falha ao compilar vision-worker:\n{detail}"));
    }

    for worker_exe in worker_exe_candidates.iter() {
        if worker_exe.exists() {
            return Ok(worker_exe.clone());
        }
    }

    Err("Vision-worker compilado, mas executável não foi encontrado".into())
}

pub(crate) fn get_vision_worker_path(app: &AppHandle) -> Result<String, String> {
    if cfg!(debug_assertions) {
        let project_root = std::env::current_dir()
            .map_err(|e| e.to_string())?
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap());

        return ensure_dev_vision_worker(&project_root)
            .map(|path| path.to_string_lossy().to_string());
    }

    let resource_dir = app.path().resource_dir().map_err(|e| e.to_string())?;
    let worker = resource_dir.join("binaries").join(if cfg!(windows) {
        "traduzai-vision.exe"
    } else {
        "traduzai-vision"
    });

    if worker.exists() {
        Ok(worker.to_string_lossy().to_string())
    } else {
        Ok(String::new())
    }
}

fn apply_sidecar_env(cmd: &mut Command, program: &str) -> Result<(), String> {
    for (key, value) in sidecar_env_overrides(program) {
        cmd.env(key, value);
    }
    Ok(())
}

#[tauri::command]
pub async fn start_pipeline(
    app: AppHandle,
    config: PipelineConfig,
) -> Result<serde_json::Value, String> {
    *PIPELINE_CANCEL.lock().await = false;

    let job_id = uuid::Uuid::new_v4().to_string();
    let app_data = std::path::PathBuf::from("D:\\traduzai_data");

    let work_dir = app_data.join("projects").join(&job_id);
    std::fs::create_dir_all(&work_dir).map_err(|e| e.to_string())?;
    let pause_marker_path = work_dir.join("pipeline.pause");
    set_pause_marker(&pause_marker_path, false)?;
    {
        let mut current = PIPELINE_PAUSE_MARKER.lock().await;
        if let Some(previous) = current.take() {
            set_pause_marker(&previous, false).ok();
        }
        *current = Some(pause_marker_path.clone());
    }

    let settings = crate::commands::settings::load_settings_sync(&app);
    let vision_worker_path = match get_vision_worker_path(&app) {
        Ok(path) => path,
        Err(err) => {
            eprintln!("[TraduzAi] Vision worker indisponível, usando fallback atual: {err}");
            String::new()
        }
    };

    let config_json = serde_json::to_string(&serde_json::json!({
        "job_id": job_id,
        "source_path": config.source_path,
        "work_dir": work_dir.to_string_lossy(),
        "obra": config.obra,
        "capitulo": config.capitulo,
        "idioma_origem": config.idioma_origem,
        "idioma_destino": config.idioma_destino,
        "qualidade": config.qualidade,
        "glossario": config.glossario,
        "contexto": {
            "sinopse": config.contexto.sinopse,
            "genero": config.contexto.genero,
            "personagens": config.contexto.personagens,
            "aliases": config.contexto.aliases,
            "termos": config.contexto.termos,
            "relacoes": config.contexto.relacoes,
            "faccoes": config.contexto.faccoes,
            "resumo_por_arco": config.contexto.resumo_por_arco,
            "memoria_lexical": config.contexto.memoria_lexical,
            "fontes_usadas": config.contexto.fontes_usadas
        },
        "models_dir": app_data.join("models").to_string_lossy(),
        "ollama_host": settings.ollama_host,
        "ollama_model": settings.ollama_model,
        "vision_worker_path": vision_worker_path,
        "pause_file": pause_marker_path.to_string_lossy(),
    }))
    .map_err(|e| e.to_string())?;

    let config_path = work_dir.join("pipeline_config.json");
    std::fs::write(&config_path, &config_json).map_err(|e| e.to_string())?;

    let sidecar = get_sidecar_info(&app)?;

    let app_clone = app.clone();
    let job_id_clone = job_id.clone();
    let pause_marker_path_clone = pause_marker_path.clone();

    tokio::spawn(async move {
        let result = run_sidecar(&app_clone, &sidecar, &config_path).await;
        clear_current_pause_marker(&pause_marker_path_clone).await;

        match result {
            Ok(output_path) => {
                app_clone
                    .emit(
                        "pipeline-complete",
                        serde_json::json!({
                            "success": true,
                            "output_path": output_path,
                            "job_id": job_id_clone
                        }),
                    )
                    .ok();
            }
            Err(err) => {
                app_clone
                    .emit(
                        "pipeline-complete",
                        serde_json::json!({
                            "success": false,
                            "error": err,
                            "job_id": job_id_clone
                        }),
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

    if let Some(script) = &sidecar.script {
        cmd.arg(script);
    }

    // Write stderr to a log file (avoids pipe-buffer deadlock AND keeps crash info visible).
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
        if *PIPELINE_CANCEL.lock().await {
            child.kill().await.ok();
            return Err("Cancelado pelo usuário".into());
        }

        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(msg_type) = msg.get("type").and_then(|t| t.as_str()) {
                match msg_type {
                    "progress" => {
                        let progress = PipelineProgress {
                            step: msg["step"].as_str().unwrap_or("").to_string(),
                            step_progress: msg["step_progress"].as_f64().unwrap_or(0.0),
                            overall_progress: msg["overall_progress"].as_f64().unwrap_or(0.0),
                            current_page: msg["current_page"].as_u64().unwrap_or(0) as u32,
                            total_pages: msg["total_pages"].as_u64().unwrap_or(0) as u32,
                            message: msg["message"].as_str().unwrap_or("").to_string(),
                            eta_seconds: msg["eta_seconds"].as_f64().unwrap_or(0.0),
                        };
                        app.emit("pipeline-progress", progress).ok();
                    }
                    "complete" => {
                        output_path = msg["output_path"].as_str().unwrap_or("").to_string();
                    }
                    "error" => {
                        let err = msg["message"].as_str().unwrap_or("Erro desconhecido");
                        return Err(err.to_string());
                    }
                    _ => {}
                }
            }
        }
    }

    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !status.success() {
        // Read the log file to surface the Python traceback in the error message.
        let log_content = std::fs::read_to_string(&log_path).unwrap_or_default();
        let detail = if log_content.trim().is_empty() {
            format!("código {status}")
        } else {
            // Last 2000 chars to stay within reasonable size.
            let trimmed = if log_content.len() > 2000 {
                &log_content[log_content.len() - 2000..]
            } else {
                &log_content
            };
            format!("código {status}\n\n{trimmed}")
        };
        return Err(format!("Pipeline encerrou com {detail}"));
    }

    Ok(output_path)
}

async fn run_visual_warmup_sidecar(
    sidecar: &SidecarInfo,
    models_dir: &std::path::Path,
    log_path: &std::path::Path,
) -> Result<(), String> {
    let mut cmd = Command::new(&sidecar.program);

    if let Some(script) = &sidecar.script {
        cmd.arg(script);
    }

    let log_file =
        std::fs::File::create(log_path).map_err(|e| format!("Erro ao criar log do warmup: {e}"))?;

    cmd.arg("--warmup-visual")
        .arg("--models-dir")
        .arg(models_dir.to_string_lossy().to_string())
        .arg("--profile")
        .arg("normal")
        .stdout(Stdio::null())
        .stderr(Stdio::from(log_file));
    apply_sidecar_env(&mut cmd, &sidecar.program)?;

    let status = cmd
        .status()
        .await
        .map_err(|e| format!("Erro ao iniciar warmup visual: {e}"))?;

    if !status.success() {
        let log_content = std::fs::read_to_string(log_path).unwrap_or_default();
        let detail = if log_content.trim().is_empty() {
            format!("cÃ³digo {status}")
        } else if log_content.len() > 2000 {
            log_content[log_content.len() - 2000..].to_string()
        } else {
            log_content
        };
        return Err(format!("Warmup visual falhou com {detail}"));
    }

    Ok(())
}

async fn run_vision_worker_warmup(
    worker_path: &std::path::Path,
    runtime_root: &std::path::Path,
    log_path: &std::path::Path,
) -> Result<(), String> {
    let log_file = std::fs::File::create(log_path)
        .map_err(|e| format!("Erro ao criar log do warmup do vision-worker: {e}"))?;

    let mut cmd = Command::new(worker_path);
    cmd.arg("--warmup")
        .arg("--runtime-root")
        .arg(runtime_root.to_string_lossy().to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::from(log_file));

    for (key, value) in cuda_env_overrides() {
        cmd.env(key, value);
    }

    let status = cmd
        .status()
        .await
        .map_err(|e| format!("Erro ao iniciar warmup do vision-worker: {e}"))?;

    if !status.success() {
        let log_content = std::fs::read_to_string(log_path).unwrap_or_default();
        let detail = if log_content.trim().is_empty() {
            format!("código {status}")
        } else if log_content.len() > 2000 {
            log_content[log_content.len() - 2000..].to_string()
        } else {
            log_content
        };
        return Err(format!("Warmup do vision-worker falhou com {detail}"));
    }

    Ok(())
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
        .ok_or_else(|| "Nenhuma tradução em andamento para pausar.".to_string())?;
    set_pause_marker(&path, true)?;
    Ok(())
}

#[tauri::command]
pub async fn resume_pipeline() -> Result<(), String> {
    let path = PIPELINE_PAUSE_MARKER
        .lock()
        .await
        .clone()
        .ok_or_else(|| "Nenhuma tradução em andamento para continuar.".to_string())?;
    set_pause_marker(&path, false)?;
    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct RetypesetConfig {
    pub project_path: String,
    pub page_index: u32,
}

#[derive(Debug, Deserialize)]
pub struct ReinpaintConfig {
    pub project_path: String,
    pub page_index: u32,
}

#[tauri::command]
pub async fn retypeset_page(app: AppHandle, config: RetypesetConfig) -> Result<String, String> {
    let base_path = crate::commands::project::normalize_path(&config.project_path);
    let project_file = if base_path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("project.json"))
    {
        base_path
    } else {
        base_path.join("project.json")
    };

    if !project_file.exists() {
        return Err("project.json não encontrado".into());
    }

    let sidecar = get_sidecar_info(&app)?;
    let mut cmd = Command::new(&sidecar.program);

    if let Some(script) = &sidecar.script {
        cmd.arg(script);
    }

    cmd.arg("--retypeset")
        .arg(project_file.to_string_lossy().to_string())
        .arg(config.page_index.to_string());

    let log_path = project_file
        .parent()
        .unwrap_or(&project_file)
        .join("retypeset.log");

    let log_file =
        std::fs::File::create(&log_path).map_err(|e| format!("Erro ao criar log: {e}"))?;

    cmd.stdout(Stdio::piped()).stderr(Stdio::from(log_file));
    apply_sidecar_env(&mut cmd, &sidecar.program)?;

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Erro ao iniciar retypeset: {e}"))?;

    let stdout = child.stdout.take().expect("stdout not captured");
    let mut reader = BufReader::new(stdout).lines();
    let mut output_path = String::new();

    while let Ok(Some(line)) = reader.next_line().await {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(msg_type) = msg.get("type").and_then(|t| t.as_str()) {
                if msg_type == "complete" {
                    output_path = msg["output_path"].as_str().unwrap_or("").to_string();
                } else if msg_type == "error" {
                    let err = msg["message"].as_str().unwrap_or("Erro desconhecido");
                    return Err(err.to_string());
                }
            }
        }
    }

    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !status.success() {
        let log_content = std::fs::read_to_string(&log_path).unwrap_or_default();
        return Err(format!("Retypeset falhou: {status}\n\n{log_content}"));
    }

    Ok(output_path)
}

#[tauri::command]
pub async fn reinpaint_page(app: AppHandle, config: ReinpaintConfig) -> Result<String, String> {
    let base_path = crate::commands::project::normalize_path(&config.project_path);
    let project_file = if base_path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("project.json"))
    {
        base_path
    } else {
        base_path.join("project.json")
    };

    if !project_file.exists() {
        return Err("project.json não encontrado".into());
    }

    let sidecar = get_sidecar_info(&app)?;
    let mut cmd = Command::new(&sidecar.program);

    if let Some(script) = &sidecar.script {
        cmd.arg(script);
    }

    cmd.arg("--reinpaint-page")
        .arg(project_file.to_string_lossy().to_string())
        .arg(config.page_index.to_string());

    let log_path = project_file
        .parent()
        .unwrap_or(&project_file)
        .join("reinpaint.log");
    let log_file = std::fs::File::create(&log_path)
        .map_err(|e| format!("Erro ao criar log do reinpaint: {e}"))?;

    cmd.stdout(Stdio::piped()).stderr(Stdio::from(log_file));
    apply_sidecar_env(&mut cmd, &sidecar.program)?;

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Erro ao iniciar reinpaint: {e}"))?;

    let stdout = child.stdout.take().expect("stdout not captured");
    let mut reader = BufReader::new(stdout).lines();
    let mut output_path = String::new();

    while let Ok(Some(line)) = reader.next_line().await {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(msg_type) = msg.get("type").and_then(|t| t.as_str()) {
                if msg_type == "complete" {
                    output_path = msg["output_path"].as_str().unwrap_or("").to_string();
                } else if msg_type == "error" {
                    let err = msg["message"].as_str().unwrap_or("Erro desconhecido");
                    return Err(err.to_string());
                }
            }
        }
    }

    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !status.success() {
        let log_content = std::fs::read_to_string(&log_path).unwrap_or_default();
        return Err(format!("Reinpaint falhou: {status}\n\n{log_content}"));
    }

    Ok(output_path)
}

#[tauri::command]
pub async fn warmup_visual_stack(app: AppHandle) -> Result<String, String> {
    {
        let visual_state = *VISUAL_WARMUP_STATE.lock().await;
        let worker_state = *VISION_WORKER_WARMUP_STATE.lock().await;
        if matches!(visual_state, VisualWarmupState::Running)
            || matches!(worker_state, VisualWarmupState::Running)
        {
            return Ok("running".into());
        }
        if matches!(visual_state, VisualWarmupState::Ready)
            && matches!(worker_state, VisualWarmupState::Ready)
        {
            return Ok("ready".into());
        }
    }

    let sidecar = get_sidecar_info(&app)?;
    let vision_worker_path = match get_vision_worker_path(&app) {
        Ok(path) if !path.trim().is_empty() => Some(std::path::PathBuf::from(path)),
        Ok(_) => None,
        Err(err) => {
            eprintln!("[TraduzAi] Vision worker indisponível no warmup: {err}");
            None
        }
    };
    let app_data = std::path::PathBuf::from("D:\\traduzai_data");
    let models_dir = app_data.join("models");
    let runtime_root = app_data.clone();
    let warmup_dir = app_data.join("warmup");
    std::fs::create_dir_all(&warmup_dir).map_err(|e| e.to_string())?;
    let visual_log_path = warmup_dir.join("visual-stack.log");
    let worker_log_path = warmup_dir.join("vision-worker.log");

    {
        let mut visual_state = VISUAL_WARMUP_STATE.lock().await;
        let mut worker_state = VISION_WORKER_WARMUP_STATE.lock().await;
        if matches!(*visual_state, VisualWarmupState::Running)
            || matches!(*worker_state, VisualWarmupState::Running)
        {
            return Ok("running".into());
        }
        if matches!(*visual_state, VisualWarmupState::Ready)
            && matches!(*worker_state, VisualWarmupState::Ready)
        {
            return Ok("ready".into());
        }
        if !matches!(*visual_state, VisualWarmupState::Ready) {
            *visual_state = VisualWarmupState::Running;
        }
        if vision_worker_path.is_some() && !matches!(*worker_state, VisualWarmupState::Ready) {
            *worker_state = VisualWarmupState::Running;
        }
    }

    tokio::spawn(async move {
        if let Some(worker_path) = vision_worker_path {
            let next_worker_state =
                match run_vision_worker_warmup(&worker_path, &runtime_root, &worker_log_path).await
                {
                    Ok(_) => VisualWarmupState::Ready,
                    Err(err) => {
                        eprintln!("[TraduzAi] Warmup do vision-worker falhou: {err}");
                        VisualWarmupState::Idle
                    }
                };
            *VISION_WORKER_WARMUP_STATE.lock().await = next_worker_state;
        } else {
            *VISION_WORKER_WARMUP_STATE.lock().await = VisualWarmupState::Ready;
        }

        let next_visual_state =
            match run_visual_warmup_sidecar(&sidecar, &models_dir, &visual_log_path).await {
                Ok(_) => VisualWarmupState::Ready,
                Err(err) => {
                    eprintln!("[TraduzAi] Warmup visual falhou: {err}");
                    VisualWarmupState::Idle
                }
            };
        *VISUAL_WARMUP_STATE.lock().await = next_visual_state;
    });

    Ok("started".into())
}

#[derive(Debug, Serialize)]
pub struct GpuInfo {
    pub available: bool,
    pub name: String,
}

#[derive(Debug, Clone, Deserialize)]
struct HardwareFacts {
    cpu_name: String,
    cpu_cores: u32,
    cpu_threads: u32,
    ram_gb: u32,
    gpu_available: bool,
    gpu_name: String,
    gpu_vram_gb: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct WindowsHardwareSnapshot {
    #[serde(default)]
    cpu_name: String,
    cpu_cores: Option<u32>,
    cpu_threads: Option<u32>,
    ram_gb: Option<f64>,
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
    pub ram_gb: u32,
    pub gpu_available: bool,
    pub gpu_name: String,
    pub gpu_vram_gb: Option<f64>,
    pub performance_tier: String,
    pub startup_seconds: f64,
    pub seconds_per_page: QualityEstimateTable,
}

fn round_to_tenth(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

fn fallback_hardware_facts() -> HardwareFacts {
    let cpu_threads = std::thread::available_parallelism()
        .map(|threads| threads.get() as u32)
        .unwrap_or(8);
    let cpu_cores = (cpu_threads / 2).max(1);
    let cpu_name = std::env::var("PROCESSOR_IDENTIFIER")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "CPU local".into());

    HardwareFacts {
        cpu_name,
        cpu_cores,
        cpu_threads,
        ram_gb: 16,
        gpu_available: false,
        gpu_name: "CPU (sem CUDA detectada)".into(),
        gpu_vram_gb: None,
    }
}

#[cfg(target_os = "windows")]
fn query_windows_cpu_ram_snapshot() -> Option<WindowsHardwareSnapshot> {
    let script = r#"
        $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1 Name, NumberOfCores, NumberOfLogicalProcessors
        $ram = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
        [pscustomobject]@{
            cpu_name = $cpu.Name
            cpu_cores = $cpu.NumberOfCores
            cpu_threads = $cpu.NumberOfLogicalProcessors
            ram_gb = [math]::Round([double]$ram / 1GB, 0)
        } | ConvertTo-Json -Compress
    "#;

    let output = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", script])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str::<WindowsHardwareSnapshot>(stdout.trim()).ok()
}

#[cfg(not(target_os = "windows"))]
fn query_windows_cpu_ram_snapshot() -> Option<WindowsHardwareSnapshot> {
    None
}

fn query_cuda_gpu_info() -> Option<(String, Option<f64>)> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let first_line = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .to_string();
    if first_line.is_empty() {
        return None;
    }

    let mut parts = first_line.split(',').map(|part| part.trim());
    let gpu_name = parts.next()?.to_string();
    let gpu_vram_gb = parts
        .next()
        .and_then(|value| value.parse::<f64>().ok())
        .map(|memory_mb| round_to_tenth(memory_mb / 1024.0));

    Some((gpu_name, gpu_vram_gb))
}

fn gather_hardware_facts() -> HardwareFacts {
    let mut facts = fallback_hardware_facts();

    if let Some(snapshot) = query_windows_cpu_ram_snapshot() {
        if !snapshot.cpu_name.trim().is_empty() {
            facts.cpu_name = snapshot.cpu_name;
        }
        if let Some(cpu_cores) = snapshot.cpu_cores {
            facts.cpu_cores = cpu_cores.max(1);
        }
        if let Some(cpu_threads) = snapshot.cpu_threads {
            facts.cpu_threads = cpu_threads.max(1);
        }
        if let Some(ram_gb) = snapshot.ram_gb {
            facts.ram_gb = ram_gb.max(4.0).round() as u32;
        }
    }

    if let Some((gpu_name, gpu_vram_gb)) = query_cuda_gpu_info() {
        facts.gpu_available = true;
        facts.gpu_name = gpu_name;
        facts.gpu_vram_gb = gpu_vram_gb;
    }

    facts
}

fn classify_performance_tier(facts: &HardwareFacts) -> &'static str {
    if !facts.gpu_available {
        return "cpu_only";
    }

    let gpu_name = facts.gpu_name.to_lowercase();
    let vram_gb = facts.gpu_vram_gb.unwrap_or(0.0);
    let mut score = 0;

    if facts.cpu_threads >= 12 {
        score += 1;
    }
    if facts.cpu_threads >= 20 {
        score += 1;
    }
    if facts.ram_gb >= 32 {
        score += 1;
    }
    if vram_gb >= 8.0 {
        score += 1;
    }
    if vram_gb >= 12.0 {
        score += 1;
    }
    if gpu_name.contains("4090")
        || gpu_name.contains("5090")
        || gpu_name.contains("5080")
        || gpu_name.contains("4080")
    {
        score += 1;
    }

    if score >= 5 {
        "workstation"
    } else if score >= 3 {
        "fast"
    } else {
        "balanced"
    }
}

fn build_system_profile(facts: HardwareFacts) -> SystemProfile {
    let performance_tier = classify_performance_tier(&facts).to_string();
    let ram_penalty = if facts.ram_gb < 16 {
        1.12
    } else if facts.ram_gb < 24 {
        1.05
    } else {
        1.0
    };
    let thread_bonus = if facts.cpu_threads >= 24 {
        0.92
    } else if facts.cpu_threads >= 16 {
        0.97
    } else {
        1.0
    };
    let vram_penalty = if facts.gpu_available && facts.gpu_vram_gb.unwrap_or(0.0) < 8.0 {
        1.08
    } else {
        1.0
    };

    let (startup_seconds, quick_base, normal_base, high_base) = match performance_tier.as_str() {
        "workstation" => (11.0, 0.8, 1.2, 1.7),
        "fast" => (13.0, 1.1, 1.6, 2.2),
        "balanced" => (16.0, 1.6, 2.3, 3.1),
        _ => (22.0, 3.4, 4.8, 6.5),
    };

    let multiplier: f64 = ram_penalty * thread_bonus * vram_penalty;

    SystemProfile {
        cpu_name: facts.cpu_name,
        cpu_cores: facts.cpu_cores,
        cpu_threads: facts.cpu_threads,
        ram_gb: facts.ram_gb,
        gpu_available: facts.gpu_available,
        gpu_name: facts.gpu_name,
        gpu_vram_gb: facts.gpu_vram_gb.map(round_to_tenth),
        performance_tier,
        startup_seconds: round_to_tenth(startup_seconds * multiplier.min(1.08)),
        seconds_per_page: QualityEstimateTable {
            rapida: round_to_tenth(quick_base * multiplier),
            normal: round_to_tenth(normal_base * multiplier),
            alta: round_to_tenth(high_base * multiplier),
        },
    }
}

#[tauri::command]
pub async fn check_gpu() -> Result<GpuInfo, String> {
    let facts = gather_hardware_facts();
    Ok(GpuInfo {
        available: facts.gpu_available,
        name: facts.gpu_name,
    })
}

#[tauri::command]
pub async fn get_system_profile() -> Result<SystemProfile, String> {
    Ok(build_system_profile(gather_hardware_facts()))
}

#[tauri::command]
pub async fn check_models(_app: AppHandle) -> Result<serde_json::Value, String> {
    let app_data = std::path::PathBuf::from("D:\\traduzai_data");
    let models_dir = app_data.join("models");

    // EasyOCR is the only component that needs a first-run warmup/download now.
    let ocr_ready = models_dir.join("easyocr").join(".ready").exists();
    let inpainting_ready = true;

    let total_size: u64 = if models_dir.exists() {
        walkdir::WalkDir::new(&models_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum()
    } else {
        0
    };

    Ok(serde_json::json!({
        "ready": ocr_ready && inpainting_ready,
        "size_mb": total_size / (1024 * 1024),
        "ocr_ready": ocr_ready,
        "inpainting_ready": inpainting_ready
    }))
}

fn get_venv_python(_app: &AppHandle) -> String {
    let project_root = std::env::current_dir()
        .ok()
        .and_then(|d| d.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    #[cfg(windows)]
    let candidate = project_root
        .join("pipeline")
        .join("venv")
        .join("Scripts")
        .join("python.exe");
    #[cfg(not(windows))]
    let candidate = project_root
        .join("pipeline")
        .join("venv")
        .join("bin")
        .join("python3");

    if candidate.exists() {
        candidate.to_string_lossy().to_string()
    } else {
        #[cfg(windows)]
        {
            "python".to_string()
        }
        #[cfg(not(windows))]
        {
            "python3".to_string()
        }
    }
}

#[tauri::command]
pub async fn download_models(app: AppHandle) -> Result<(), String> {
    let app_data = std::path::PathBuf::from("D:\\traduzai_data");
    let models_dir = app_data.join("models");
    std::fs::create_dir_all(&models_dir).map_err(|e| e.to_string())?;

    let python_exe = get_venv_python(&app);
    let app_clone = app.clone();
    let models_dir_clone = models_dir.clone();

    tokio::spawn(async move {
        macro_rules! progress {
            ($step:expr, $msg:expr) => {
                app_clone.emit("models-progress", serde_json::json!({
                    "step": $step,
                    "message": $msg
                })).ok();
            };
        }

        // Step 1 — warm EasyOCR so its weights are downloaded before the first pipeline run.
        progress!(
            "download_ocr",
            "Preparando EasyOCR (pode baixar modelos na primeira vez)..."
        );

        let ocr_dir = models_dir_clone.join("easyocr");
        std::fs::create_dir_all(&ocr_dir).ok();
        let init_script = "import easyocr; easyocr.Reader(['en', 'ko'], gpu=False, verbose=False)";

        let ocr_dl = Command::new(&python_exe)
            .args(["-c", &init_script])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;

        match ocr_dl {
            Ok(s) if s.success() => {
                std::fs::write(ocr_dir.join(".ready"), "ok").ok();
            }
            Ok(s) => {
                app_clone.emit("models-progress", serde_json::json!({
                    "step": "error",
                    "message": format!("Falha ao inicializar EasyOCR (código {}). Verifique o venv do pipeline.", s)
                })).ok();
                app_clone
                    .emit("models-ready", serde_json::json!({ "success": false }))
                    .ok();
                return;
            }
            Err(e) => {
                app_clone
                    .emit(
                        "models-progress",
                        serde_json::json!({
                            "step": "error",
                            "message": format!("Não foi possível preparar EasyOCR: {e}")
                        }),
                    )
                    .ok();
                app_clone
                    .emit("models-ready", serde_json::json!({ "success": false }))
                    .ok();
                return;
            }
        }

        // Step 2 — the current inpainting backend is local OpenCV-based and needs no download.
        progress!(
            "download_inpaint",
            "Inpainting local pronto (OpenCV TELEA)..."
        );

        let inpaint_dir = models_dir_clone.join("inpainting");
        std::fs::create_dir_all(&inpaint_dir).ok();
        std::fs::write(inpaint_dir.join(".ready"), "ok").ok();

        progress!("done", "Modelos prontos!");
        app_clone
            .emit("models-ready", serde_json::json!({ "success": true }))
            .ok();
    });

    Ok(())
}

#[tauri::command]
pub async fn search_anilist(query: String) -> Result<serde_json::Value, String> {
    search_anilist_internal(&query).await
}

#[tauri::command]
pub async fn search_work(query: String) -> Result<serde_json::Value, String> {
    let mut candidates = Vec::new();

    if let Ok(anilist) = search_anilist_internal(&query).await {
        let title = anilist["title"].as_str().unwrap_or(&query).to_string();
        candidates.push(WorkSearchCandidate {
            id: format!("anilist:{}", normalize_text(&title).replace(' ', "-")),
            title: title.clone(),
            synopsis: anilist["synopsis"].as_str().unwrap_or_default().to_string(),
            source: "anilist".to_string(),
            source_url: String::new(),
            cover_url: anilist["cover_url"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            score: score_title_match(&query, &title, "anilist"),
        });
    }

    if let Ok(items) = fetch_bing_site_results(&query, "webnovel.com/book").await {
        candidates.extend(
            filter_source_results(items, "webnovel")
                .into_iter()
                .take(4)
                .map(|item| build_candidate_from_snippet(&query, "webnovel", item)),
        );
    }

    if let Ok(items) = fetch_bing_site_results(&query, "fandom.com").await {
        candidates.extend(
            filter_source_results(items, "fandom")
                .into_iter()
                .take(4)
                .map(|item| build_candidate_from_snippet(&query, "fandom", item)),
        );
    }

    let candidates = dedupe_candidates(candidates)
        .into_iter()
        .filter(|candidate| candidate.score >= 20.0)
        .take(6)
        .collect::<Vec<_>>();

    Ok(serde_json::json!({
        "query": query,
        "candidates": candidates
    }))
}

#[tauri::command]
pub async fn enrich_work_context(
    selection: WorkSearchCandidate,
) -> Result<serde_json::Value, String> {
    let anilist = search_anilist_internal(&selection.title).await.ok();
    let webnovel_results = fetch_bing_site_results(&selection.title, "webnovel.com/book")
        .await
        .map(|items| filter_source_results(items, "webnovel"))
        .unwrap_or_default();
    let fandom_results = fetch_bing_site_results(&selection.title, "fandom.com")
        .await
        .map(|items| filter_source_results(items, "fandom"))
        .unwrap_or_default();

    let enriched = build_enriched_context(&selection, anilist, webnovel_results, fandom_results);
    serde_json::to_value(enriched).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bing_rss_extracts_items() {
        let xml = r#"
            <rss><channel>
                <item>
                    <title>The Regressed Mercenary’s Machinations - WebNovel</title>
                    <link>https://www.webnovel.com/pt/book/31039231200309705</link>
                    <description>Free chapters and synopsis.</description>
                </item>
            </channel></rss>
        "#;

        let items = parse_bing_rss_items(xml);
        assert_eq!(items.len(), 1);
        assert!(items[0].link.contains("webnovel.com"));
        assert!(items[0].title.contains("Machinations"));
    }

    #[test]
    fn dedupe_candidates_prefers_exact_match_first() {
        let candidates = dedupe_candidates(vec![
            WorkSearchCandidate {
                id: "1".into(),
                title: "Mercenary Plans".into(),
                synopsis: String::new(),
                source: "fandom".into(),
                source_url: "https://a".into(),
                cover_url: String::new(),
                score: 35.0,
            },
            WorkSearchCandidate {
                id: "2".into(),
                title: "The Regressed Mercenary's Machinations".into(),
                synopsis: String::new(),
                source: "anilist".into(),
                source_url: "https://b".into(),
                cover_url: String::new(),
                score: 100.0,
            },
        ]);

        assert_eq!(
            candidates[0].title,
            "The Regressed Mercenary's Machinations"
        );
    }

    #[test]
    fn build_enriched_context_keeps_unique_values() {
        let selection = WorkSearchCandidate {
            id: "anilist:test".into(),
            title: "The Regressed Mercenary's Machinations".into(),
            synopsis: "A regressed mercenary takes revenge.".into(),
            source: "anilist".into(),
            source_url: String::new(),
            cover_url: String::new(),
            score: 100.0,
        };

        let anilist = serde_json::json!({
            "title": "The Regressed Mercenary's Machinations",
            "synopsis": "Canon synopsis.",
            "genres": ["Action", "Fantasy"],
            "characters": ["Ghislain", "Vanessa"],
            "cover_url": "https://cover"
        });
        let webnovel = vec![
            SearchSnippet {
                title: "The Regressed Mercenary’s Machinations - WebNovel".into(),
                link: "https://www.webnovel.com/book/123".into(),
                snippet: "Ghislain returns to change his fate.".into(),
            },
            SearchSnippet {
                title: "The Regressed Mercenary’s Machinations - WebNovel".into(),
                link: "https://www.webnovel.com/book/123".into(),
                snippet: "Ghislain returns to change his fate.".into(),
            },
        ];
        let fandom = vec![SearchSnippet {
            title: "The Regressed Mercenary's Machinations Wiki | Fandom".into(),
            link: "https://regressed.fandom.com/wiki/Main".into(),
            snippet: "Community glossary and character notes.".into(),
        }];

        let enriched = build_enriched_context(&selection, Some(anilist), webnovel, fandom);
        assert_eq!(
            enriched.characters,
            vec!["Ghislain".to_string(), "Vanessa".to_string()]
        );
        assert!(enriched.sources_used.len() >= 2);
        assert!(enriched.synopsis.contains("Canon synopsis."));
    }

    #[test]
    fn build_system_profile_makes_gpu_profiles_faster() {
        let cpu_only = build_system_profile(HardwareFacts {
            cpu_name: "Ryzen 5".into(),
            cpu_cores: 6,
            cpu_threads: 12,
            ram_gb: 16,
            gpu_available: false,
            gpu_name: "CPU only".into(),
            gpu_vram_gb: None,
        });

        let gpu_fast = build_system_profile(HardwareFacts {
            cpu_name: "Ryzen 7".into(),
            cpu_cores: 8,
            cpu_threads: 16,
            ram_gb: 32,
            gpu_available: true,
            gpu_name: "NVIDIA GeForce RTX 4070".into(),
            gpu_vram_gb: Some(12.0),
        });

        assert_eq!(cpu_only.performance_tier, "cpu_only");
        assert_eq!(gpu_fast.performance_tier, "fast");
        assert!(gpu_fast.seconds_per_page.normal < cpu_only.seconds_per_page.normal);
        assert!(gpu_fast.startup_seconds <= cpu_only.startup_seconds);
    }

    #[test]
    fn build_system_profile_increases_cost_with_quality() {
        let profile = build_system_profile(HardwareFacts {
            cpu_name: "Ryzen 7".into(),
            cpu_cores: 8,
            cpu_threads: 16,
            ram_gb: 32,
            gpu_available: true,
            gpu_name: "NVIDIA GeForce RTX 4060".into(),
            gpu_vram_gb: Some(8.0),
        });

        assert!(profile.seconds_per_page.rapida < profile.seconds_per_page.normal);
        assert!(profile.seconds_per_page.normal < profile.seconds_per_page.alta);
    }

    #[test]
    fn infer_cudarc_cuda_version_parses_folder_name() {
        assert_eq!(
            infer_cudarc_cuda_version(std::path::Path::new(
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
            )),
            Some("13020".into())
        );
        assert_eq!(
            infer_cudarc_cuda_version(std::path::Path::new(r"C:\CUDA\v12.8")),
            Some("12080".into())
        );
    }

    #[test]
    fn python_cuda_runtime_dirs_detects_venv_nvidia_bins() {
        let temp_root = std::env::temp_dir().join(format!(
            "traduzai-python-cuda-env-{}",
            uuid::Uuid::new_v4()
        ));
        let python_exe = temp_root.join("Scripts").join("python.exe");
        let package_root = temp_root.join("Lib").join("site-packages");
        let expected_dirs = vec![
            package_root.join("nvidia").join("cuda_runtime").join("bin"),
            package_root.join("nvidia").join("cublas").join("bin"),
            package_root.join("tensorrt_libs"),
        ];

        std::fs::create_dir_all(python_exe.parent().unwrap()).expect("scripts dir");
        std::fs::write(&python_exe, "").expect("fake python");
        for dir in &expected_dirs {
            std::fs::create_dir_all(dir).expect("runtime dir");
        }

        let resolved = python_cuda_runtime_dirs(&python_exe.to_string_lossy());
        for dir in &expected_dirs {
            assert!(resolved.contains(dir), "expected {:?} in {:?}", dir, resolved);
        }

        std::fs::remove_dir_all(&temp_root).ok();
    }

    #[test]
    fn set_pause_marker_toggles_file_on_disk() {
        let pause_path = std::env::temp_dir().join(format!(
            "traduzai-pipeline-pause-test-{}",
            uuid::Uuid::new_v4()
        ));

        set_pause_marker(&pause_path, true).expect("pause marker should be created");
        assert!(pause_path.exists());

        set_pause_marker(&pause_path, false).expect("pause marker should be removed");
        assert!(!pause_path.exists());
    }
}
