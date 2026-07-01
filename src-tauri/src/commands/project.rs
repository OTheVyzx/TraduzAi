use crate::commands::project_schema;
use crate::export::psd::engine_data::{TextEngineSpec, TextJustification, TextOrientation};
use crate::export::psd::{export_psd, PsdLayer};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use chrono::Utc;
use dafont::{FcFontCache, PatternMatch};
use image::imageops::FilterType;
use image::{
    open, DynamicImage, GrayImage, ImageBuffer, ImageFormat, ImageReader, Luma, RgbaImage,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};
use tauri_plugin_dialog::DialogExt;

#[derive(Debug, Serialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub pages: usize,
    pub has_project_json: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CacheGoogleFontRequest {
    pub family: String,
    pub css_family: String,
    pub variant: String,
    pub url: String,
    pub filename: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CachedGoogleFont {
    pub family: String,
    pub css_family: String,
    pub variant: String,
    pub filename: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GoogleFontSearchResult {
    pub family: String,
    pub css_family: String,
    pub variant: String,
    pub filename: String,
    pub download_url: String,
    pub category: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SystemFontInfo {
    pub family: String,
    pub full_name: String,
    pub filename: String,
    pub path: String,
    pub weight: String,
    pub style: String,
    pub monospace: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct GoogleFontsMetadataResponse {
    #[serde(rename = "familyMetadataList", default)]
    family_metadata_list: Vec<GoogleFontFamilyMetadata>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct GoogleFontFamilyMetadata {
    pub family: String,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub popularity: Option<i64>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct GoogleFontRepoEntry {
    pub name: String,
    #[serde(default)]
    pub download_url: Option<String>,
    #[serde(rename = "type")]
    pub entry_type: String,
}

const GOOGLE_FONTS_METADATA_URL: &str = "https://fonts.google.com/metadata/fonts";
const GOOGLE_FONTS_REPO_CONTENTS_URL: &str = "https://api.github.com/repos/google/fonts/contents";
const GOOGLE_FONT_LICENSE_DIRS: [&str; 3] = ["ofl", "apache", "ufl"];
const GOOGLE_FONT_SEARCH_LIMIT: usize = 12;

#[tauri::command]
pub async fn cache_google_font(
    family: String,
    css_family: String,
    variant: String,
    url: String,
    filename: String,
) -> Result<CachedGoogleFont, String> {
    cache_google_font_in_dir(
        CacheGoogleFontRequest {
            family,
            css_family,
            variant,
            url,
            filename,
        },
        &google_fonts_cache_dir()?,
    )
    .await
}

#[tauri::command]
pub async fn search_google_fonts(query: String) -> Result<Vec<GoogleFontSearchResult>, String> {
    let normalized_query = normalize_google_font_query(&query);
    if normalized_query.len() < 2 {
        return Ok(Vec::new());
    }

    let client = reqwest::Client::new();
    let metadata = client
        .get(GOOGLE_FONTS_METADATA_URL)
        .header(reqwest::header::USER_AGENT, "TraduzAi")
        .send()
        .await
        .map_err(|e| format!("Falha ao consultar Google Fonts: {e}"))?;

    if !metadata.status().is_success() {
        return Err(format!(
            "Falha ao consultar Google Fonts: HTTP {}",
            metadata.status()
        ));
    }

    let metadata_text = metadata
        .text()
        .await
        .map_err(|e| format!("Falha ao ler resposta do Google Fonts: {e}"))?;
    let families =
        search_google_fonts_metadata_json(&metadata_text, &query, GOOGLE_FONT_SEARCH_LIMIT)?;
    let mut results = Vec::new();

    for family in families {
        if let Ok(repo_file) = fetch_google_font_repo_file(&client, &family.family).await {
            if let Some(download_url) = repo_file.download_url {
                let family_name = family.family;
                let extension = Path::new(&repo_file.name)
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("ttf");
                results.push(GoogleFontSearchResult {
                    family: family_name.clone(),
                    css_family: family_name.clone(),
                    variant: "regular".to_string(),
                    filename: google_font_cache_filename(&family_name, extension),
                    download_url,
                    category: family.category,
                });
            }
        }
    }

    Ok(results)
}

#[tauri::command]
pub async fn list_system_fonts(query: Option<String>) -> Result<Vec<SystemFontInfo>, String> {
    let normalized_query = normalize_system_font_query(query.as_deref().unwrap_or(""));
    let cache = FcFontCache::build();
    let mut fonts = Vec::new();

    for (pattern, font_path) in cache.list() {
        let family = pattern
            .family
            .clone()
            .unwrap_or_default()
            .trim()
            .to_string();
        if family.is_empty() {
            continue;
        }
        let full_name = pattern
            .name
            .clone()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| family.clone());
        let haystack = normalize_system_font_query(&format!("{} {}", family, full_name));
        if normalized_query.len() >= 2 && !haystack.contains(&normalized_query) {
            continue;
        }
        let path = font_path.path.clone();
        let extension = Path::new(&path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("ttf");
        let style_name = system_font_style_name(pattern.bold.clone(), pattern.italic.clone());
        let filename = match system_font_cache_filename(&family, &style_name, extension) {
            Ok(filename) => filename,
            Err(_) => continue,
        };
        fonts.push(SystemFontInfo {
            family,
            full_name,
            filename,
            path,
            weight: system_font_weight(pattern.bold.clone()),
            style: system_font_style(pattern.italic.clone()),
            monospace: pattern.monospace == PatternMatch::True,
        });
    }

    fonts.sort_by(|a, b| {
        a.family
            .to_lowercase()
            .cmp(&b.family.to_lowercase())
            .then(a.full_name.to_lowercase().cmp(&b.full_name.to_lowercase()))
            .then(a.filename.cmp(&b.filename))
    });
    fonts.dedup_by(|a, b| a.filename == b.filename);
    Ok(fonts)
}

#[tauri::command]
pub async fn resolve_system_font(filename: String) -> Result<Option<SystemFontInfo>, String> {
    let wanted = sanitize_system_font_filename(&filename)?;
    Ok(list_system_fonts(None)
        .await?
        .into_iter()
        .find(|font| font.filename == wanted))
}

fn google_fonts_cache_dir() -> Result<PathBuf, String> {
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .ok_or_else(|| {
            "Nao foi possivel localizar a pasta do usuario para cache de fontes".to_string()
        })?;

    Ok(PathBuf::from(home)
        .join(".traduzai")
        .join("fonts")
        .join("google"))
}

fn sanitize_google_font_filename(filename: &str) -> Result<String, String> {
    let trimmed = filename.trim();
    if trimmed.is_empty() {
        return Err("Nome de fonte vazio".to_string());
    }
    if trimmed == "." || trimmed == ".." || trimmed.contains("..") {
        return Err("Nome de fonte invalido".to_string());
    }
    if trimmed.chars().any(|ch| {
        matches!(
            ch,
            '/' | '\\' | ':' | '\0' | '<' | '>' | '"' | '|' | '?' | '*'
        ) || ch.is_control()
    }) {
        return Err("Nome de fonte deve ser um arquivo simples".to_string());
    }

    let lower = trimmed.to_ascii_lowercase();
    if !lower.ends_with(".ttf") && !lower.ends_with(".otf") {
        return Err("Fonte Google deve terminar em .ttf ou .otf".to_string());
    }

    Ok(trimmed.to_string())
}

fn normalize_google_font_query(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn google_font_cache_slug(family: &str) -> String {
    let mut slug = String::new();
    let mut needs_separator = false;

    for ch in family.chars() {
        if ch.is_ascii_alphanumeric() {
            if needs_separator && !slug.is_empty() {
                slug.push('_');
            }
            slug.push(ch);
            needs_separator = false;
        } else {
            needs_separator = true;
        }
    }

    if slug.is_empty() {
        "Google_Font".to_string()
    } else {
        slug
    }
}

fn google_font_cache_filename(family: &str, extension: &str) -> String {
    let normalized_extension = match extension.to_ascii_lowercase().as_str() {
        "otf" => "otf",
        _ => "ttf",
    };
    format!(
        "GoogleFont__{}__regular.{}",
        google_font_cache_slug(family),
        normalized_extension
    )
}

fn normalize_system_font_query(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn system_font_cache_slug(value: &str) -> Result<String, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed == "." || trimmed == ".." || trimmed.contains("..") {
        return Err("Nome de fonte do sistema invalido".to_string());
    }
    let mut slug = String::new();
    let mut needs_separator = false;
    for ch in trimmed.chars() {
        if ch.is_ascii_alphanumeric() {
            if needs_separator && !slug.is_empty() {
                slug.push('_');
            }
            slug.push(ch);
            needs_separator = false;
        } else if ch.is_whitespace() || matches!(ch, '-' | '_') {
            needs_separator = true;
        } else {
            return Err("Nome de fonte do sistema contem caracteres invalidos".to_string());
        }
    }
    if slug.is_empty() {
        Err("Nome de fonte do sistema invalido".to_string())
    } else {
        Ok(slug)
    }
}

fn system_font_cache_filename(
    family: &str,
    style: &str,
    extension: &str,
) -> Result<String, String> {
    let normalized_extension = match extension.to_ascii_lowercase().as_str() {
        "otf" => "otf",
        _ => "ttf",
    };
    Ok(format!(
        "SystemFont__{}__{}.{}",
        system_font_cache_slug(family)?,
        system_font_cache_slug(style)?,
        normalized_extension
    ))
}

fn sanitize_system_font_filename(filename: &str) -> Result<String, String> {
    let trimmed = filename.trim();
    if trimmed.is_empty()
        || trimmed == "."
        || trimmed == ".."
        || trimmed.contains("..")
        || !trimmed.starts_with("SystemFont__")
        || trimmed.chars().any(|ch| {
            matches!(
                ch,
                '/' | '\\' | ':' | '\0' | '<' | '>' | '"' | '|' | '?' | '*'
            ) || ch.is_control()
        })
    {
        return Err("Nome de fonte do sistema invalido".to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    if !lower.ends_with(".ttf") && !lower.ends_with(".otf") {
        return Err("Fonte do sistema deve terminar em .ttf ou .otf".to_string());
    }
    Ok(trimmed.to_string())
}

fn system_font_weight(bold: PatternMatch) -> String {
    if bold == PatternMatch::True {
        "700".to_string()
    } else {
        "400".to_string()
    }
}

fn system_font_style(italic: PatternMatch) -> String {
    if italic == PatternMatch::True {
        "italic".to_string()
    } else {
        "normal".to_string()
    }
}

fn system_font_style_name(bold: PatternMatch, italic: PatternMatch) -> String {
    match (bold == PatternMatch::True, italic == PatternMatch::True) {
        (true, true) => "Bold Italic".to_string(),
        (true, false) => "Bold".to_string(),
        (false, true) => "Italic".to_string(),
        (false, false) => "Regular".to_string(),
    }
}

fn search_google_fonts_metadata_json(
    metadata_json: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<GoogleFontFamilyMetadata>, String> {
    let parsed: GoogleFontsMetadataResponse = serde_json::from_str(metadata_json)
        .map_err(|e| format!("Resposta invalida do Google Fonts: {e}"))?;
    let normalized_query = normalize_google_font_query(query);
    if normalized_query.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    let query_tokens: Vec<&str> = normalized_query.split_whitespace().collect();
    let mut matches: Vec<GoogleFontFamilyMetadata> = parsed
        .family_metadata_list
        .into_iter()
        .filter(|font| {
            let family = normalize_google_font_query(&font.family);
            query_tokens.iter().all(|token| family.contains(token))
        })
        .collect();

    matches.sort_by(|a, b| {
        let a_family = normalize_google_font_query(&a.family);
        let b_family = normalize_google_font_query(&b.family);
        google_font_match_rank(&a_family, &normalized_query)
            .cmp(&google_font_match_rank(&b_family, &normalized_query))
            .then_with(|| {
                a.popularity
                    .unwrap_or(i64::MAX)
                    .cmp(&b.popularity.unwrap_or(i64::MAX))
            })
            .then_with(|| a.family.cmp(&b.family))
    });
    matches.truncate(limit);
    Ok(matches)
}

fn google_font_match_rank(family: &str, query: &str) -> i32 {
    if family == query {
        0
    } else if family.starts_with(query) {
        1
    } else if family.contains(query) {
        2
    } else {
        3
    }
}

fn google_font_repo_slug(family: &str) -> String {
    family
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

fn select_google_font_repo_file(entries: &[GoogleFontRepoEntry]) -> Option<&GoogleFontRepoEntry> {
    entries
        .iter()
        .filter(|entry| {
            entry.entry_type == "file"
                && entry.download_url.is_some()
                && matches!(
                    Path::new(&entry.name)
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| ext.to_ascii_lowercase())
                        .as_deref(),
                    Some("ttf") | Some("otf")
                )
        })
        .min_by(|a, b| {
            google_font_repo_file_rank(&a.name).cmp(&google_font_repo_file_rank(&b.name))
        })
}

fn google_font_repo_file_rank(name: &str) -> (i32, usize, String) {
    let lower = name.to_ascii_lowercase();
    let regular = lower.contains("regular");
    let italic = lower.contains("italic");
    let variable = lower.contains('[') && lower.contains(']');
    let rank = if regular && !italic {
        0
    } else if variable && !italic {
        1
    } else if !italic {
        2
    } else if regular {
        3
    } else {
        4
    };
    (rank, name.len(), name.to_string())
}

async fn fetch_google_font_repo_file(
    client: &reqwest::Client,
    family: &str,
) -> Result<GoogleFontRepoEntry, String> {
    let slug = google_font_repo_slug(family);
    for license_dir in GOOGLE_FONT_LICENSE_DIRS {
        let url = format!("{GOOGLE_FONTS_REPO_CONTENTS_URL}/{license_dir}/{slug}");
        let response = client
            .get(url)
            .header(reqwest::header::USER_AGENT, "TraduzAi")
            .send()
            .await
            .map_err(|e| format!("Falha ao localizar fonte no repositorio Google Fonts: {e}"))?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            continue;
        }
        if !response.status().is_success() {
            continue;
        }
        let entries = response
            .json::<Vec<GoogleFontRepoEntry>>()
            .await
            .map_err(|e| format!("Falha ao ler repositorio Google Fonts: {e}"))?;
        if let Some(selected) = select_google_font_repo_file(&entries) {
            return Ok(selected.clone());
        }
    }

    Err(format!(
        "Nao foi encontrado arquivo TTF/OTF para a fonte Google: {family}"
    ))
}

async fn cache_google_font_in_dir(
    request: CacheGoogleFontRequest,
    cache_dir: &Path,
) -> Result<CachedGoogleFont, String> {
    let filename = sanitize_google_font_filename(&request.filename)?;
    let target_path = cache_dir.join(&filename);

    if let Ok(metadata) = std::fs::metadata(&target_path) {
        if metadata.is_file() && metadata.len() > 0 {
            return Ok(CachedGoogleFont {
                family: request.family,
                css_family: request.css_family,
                variant: request.variant,
                filename,
                path: target_path.to_string_lossy().to_string(),
            });
        }
    }

    let parsed_url = reqwest::Url::parse(&request.url)
        .map_err(|e| format!("URL de fonte Google invalida: {e}"))?;
    if parsed_url.scheme() != "https" && parsed_url.scheme() != "http" {
        return Err("URL de fonte Google deve usar http ou https".to_string());
    }

    std::fs::create_dir_all(cache_dir)
        .map_err(|e| format!("Falha ao criar cache de fontes Google: {e}"))?;

    let response = reqwest::Client::new()
        .get(parsed_url)
        .send()
        .await
        .map_err(|e| format!("Falha ao baixar fonte Google: {e}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "Falha ao baixar fonte Google: HTTP {}",
            response.status()
        ));
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| format!("Falha ao ler fonte Google baixada: {e}"))?;
    if bytes.is_empty() {
        return Err("Fonte Google baixada esta vazia".to_string());
    }

    std::fs::write(&target_path, &bytes)
        .map_err(|e| format!("Falha ao gravar fonte Google em cache: {e}"))?;

    Ok(CachedGoogleFont {
        family: request.family,
        css_family: request.css_family,
        variant: request.variant,
        filename,
        path: target_path.to_string_lossy().to_string(),
    })
}

#[tauri::command]
pub async fn open_file_dialog(app: tauri::AppHandle) -> Result<Option<String>, String> {
    let file = app
        .dialog()
        .file()
        .add_filter("Mangá", &["zip", "cbz", "jpg", "jpeg", "png", "webp"])
        .blocking_pick_file();

    Ok(file.map(|f| f.to_string()))
}

#[tauri::command]
pub async fn open_source_dialog(app: tauri::AppHandle) -> Result<Option<String>, String> {
    let file = app
        .dialog()
        .file()
        .add_filter("Mangá", &["zip", "cbz", "jpg", "jpeg", "png", "webp"])
        .blocking_pick_file();

    Ok(file.map(|f| f.to_string()))
}
#[tauri::command]
pub async fn open_multiple_sources_dialog(app: tauri::AppHandle) -> Result<Vec<String>, String> {
    let files = app
        .dialog()
        .file()
        .add_filter("Mangá", &["zip", "cbz", "jpg", "jpeg", "png", "webp"])
        .blocking_pick_files();

    Ok(files
        .unwrap_or_default()
        .into_iter()
        .map(|f| f.to_string())
        .collect())
}
#[tauri::command]
pub async fn open_project_dialog(app: tauri::AppHandle) -> Result<Option<String>, String> {
    let folder = app.dialog().file().blocking_pick_folder();
    Ok(folder.map(|f| f.to_string()))
}

#[tauri::command]
pub async fn save_file_dialog(
    app: tauri::AppHandle,
    format: Option<String>,
    suggested_name: Option<String>,
) -> Result<Option<String>, String> {
    let format = format.unwrap_or_else(|| "zip_full".to_string());
    let (filter_name, filter_exts, default_name): (&str, &[&str], &str) = match format.as_str() {
        "cbz" => ("CBZ", &["cbz"], "traduzido.cbz"),
        "jpg_only" => ("ZIP", &["zip"], "paginas-traduzidas.zip"),
        "lab_patch_json" => ("JSON", &["json"], "lab-patch.json"),
        "log" => ("Log", &["log", "txt"], "traduzai-log.log"),
        _ => ("ZIP", &["zip"], "traduzido.zip"),
    };
    let file_name = suggested_name
        .as_deref()
        .filter(|name| !name.trim().is_empty())
        .unwrap_or(default_name);

    let file = app
        .dialog()
        .file()
        .add_filter(filter_name, filter_exts)
        .set_file_name(file_name)
        .blocking_save_file();

    Ok(file.map(|f| f.to_string()))
}

#[tauri::command]
pub async fn export_text_file(output_path: String, content: String) -> Result<String, String> {
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
pub async fn validate_import(path: String) -> Result<ValidationResult, String> {
    let path = normalize_path(&path);

    if !path.exists() {
        return Ok(ValidationResult {
            valid: false,
            pages: 0,
            has_project_json: false,
            error: Some("Arquivo não encontrado".into()),
        });
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "zip" | "cbz" => validate_archive(&path),
        "jpg" | "jpeg" | "png" | "webp" => Ok(ValidationResult {
            valid: true,
            pages: 1,
            has_project_json: false,
            error: None,
        }),
        _ => {
            if path.is_dir() {
                validate_directory(&path)
            } else {
                Ok(ValidationResult {
                    valid: false,
                    pages: 0,
                    has_project_json: false,
                    error: Some("Formato não suportado. Use .zip, .cbz, imagem ou pasta.".into()),
                })
            }
        }
    }
}

pub fn normalize_path(raw_path: &str) -> PathBuf {
    let trimmed = raw_path.trim();

    #[cfg(windows)]
    let normalized = {
        if let Some(stripped) = trimmed.strip_prefix("file:///") {
            stripped.to_string()
        } else if let Some(stripped) = trimmed.strip_prefix("file://") {
            stripped.to_string()
        } else {
            trimmed.to_string()
        }
    };

    #[cfg(not(windows))]
    let normalized = {
        if let Some(stripped) = trimmed.strip_prefix("file://") {
            stripped.to_string()
        } else {
            trimmed.to_string()
        }
    };

    PathBuf::from(normalized)
}

fn validate_archive(path: &PathBuf) -> Result<ValidationResult, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| format!("ZIP inválido: {}", e))?;

    let mut page_count = 0;
    let mut has_project_json = false;
    let image_exts = ["jpg", "jpeg", "png", "webp"];

    for i in 0..archive.len() {
        if let Ok(entry) = archive.by_index(i) {
            let name = entry.name().to_lowercase();
            if name.ends_with("project.json") {
                has_project_json = true;
            }
            if image_exts.iter().any(|ext| name.ends_with(ext)) {
                page_count += 1;
            }
        }
    }

    if has_project_json {
        return Ok(ValidationResult {
            valid: false,
            pages: 0,
            has_project_json,
            error: Some(traduzai_project_source_error("arquivo")),
        });
    }

    if page_count == 0 {
        return Ok(ValidationResult {
            valid: false,
            pages: 0,
            has_project_json,
            error: Some("Nenhuma imagem encontrada no arquivo.".into()),
        });
    }

    Ok(ValidationResult {
        valid: true,
        pages: page_count,
        has_project_json,
        error: None,
    })
}

fn validate_directory(path: &PathBuf) -> Result<ValidationResult, String> {
    let image_exts = ["jpg", "jpeg", "png", "webp"];
    let mut page_count = 0;
    let has_project_json = path.join("project.json").exists();

    if has_project_json {
        return Ok(ValidationResult {
            valid: false,
            pages: 0,
            has_project_json,
            error: Some(traduzai_project_source_error("pasta")),
        });
    }

    for entry in walkdir::WalkDir::new(path).max_depth(3) {
        if let Ok(entry) = entry {
            if entry.file_type().is_file() {
                let name = entry.file_name().to_string_lossy().to_lowercase();
                if image_exts.iter().any(|ext| name.ends_with(ext)) {
                    page_count += 1;
                }
            }
        }
    }

    Ok(ValidationResult {
        valid: page_count > 0,
        pages: page_count,
        has_project_json,
        error: if page_count == 0 {
            Some("Nenhuma imagem encontrada na pasta.".into())
        } else {
            None
        },
    })
}

fn traduzai_project_source_error(kind: &str) -> String {
    if kind == "arquivo" {
        "Este arquivo ja e um projeto/exportacao do TraduzAi. Extraia o ZIP e use Abrir projeto para continuar, nao Nova traducao."
            .to_string()
    } else {
        "Esta pasta ja e um projeto/exportacao do TraduzAi. Use Abrir projeto para continuar, nao Nova traducao."
            .to_string()
    }
}

#[tauri::command]
pub async fn load_project_json(path: String) -> Result<Value, String> {
    let normalized = normalize_path(&path);
    let project_path = project_schema::resolve_project_file(&normalized);
    project_schema::load_project_value(&project_path)
}

#[derive(Debug, Deserialize)]
pub struct SaveProjectConfig {
    pub project_path: String,
    pub project_json: Value,
}

#[derive(Debug, Deserialize)]
pub struct LoadEditorPageConfig {
    pub project_path: String,
    pub page_index: usize,
}

#[derive(Debug, Deserialize)]
pub struct CreateTextLayerConfig {
    pub project_path: String,
    pub page_index: usize,
    pub layout_bbox: [i64; 4],
}

#[derive(Debug, Deserialize)]
pub struct PatchTextLayerConfig {
    pub project_path: String,
    pub page_index: usize,
    pub layer_id: String,
    pub patch: Value,
}

#[derive(Debug, Deserialize)]
pub struct DeleteTextLayerConfig {
    pub project_path: String,
    pub page_index: usize,
    pub layer_id: String,
}

#[derive(Debug, Deserialize)]
pub struct SetLayerVisibilityConfig {
    pub project_path: String,
    pub page_index: usize,
    pub layer_kind: String,
    pub layer_key: Option<String>,
    pub layer_id: Option<String>,
    pub visible: bool,
}

#[derive(Debug, Deserialize)]
pub struct BitmapLayerUpdateConfig {
    pub project_path: String,
    pub page_index: usize,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub brush_size: u32,
    #[serde(default)]
    pub clear: bool,
    #[serde(default)]
    pub erase: bool,
    #[serde(default)]
    pub strokes: Vec<Vec<[i32; 2]>>,
    #[serde(default)]
    pub color: Option<String>,
    #[serde(default)]
    pub opacity: Option<f32>,
    #[serde(default)]
    pub hardness: Option<f32>,
    #[serde(default)]
    pub dirty_bbox: Option<[u32; 4]>,
    #[serde(default)]
    pub clip_mask_png: Option<String>,
}

#[tauri::command]
pub async fn save_project_json(config: SaveProjectConfig) -> Result<(), String> {
    let base_path = normalize_path(&config.project_path);
    let project_file = project_schema::resolve_project_file(&base_path);
    let mut project_json = config.project_json;
    project_schema::save_project_value(&project_file, &mut project_json)?;
    Ok(())
}

fn load_project_for_editing(project_path: &str) -> Result<(PathBuf, Value), String> {
    let base_path = normalize_path(project_path);
    let project_file = project_schema::resolve_project_file(&base_path);
    let project = project_schema::load_project_value(&project_file)?;
    Ok((project_file, project))
}

fn resolve_project_file(project_path: &str) -> PathBuf {
    let base_path = normalize_path(project_path);
    project_schema::resolve_project_file(&base_path)
}

#[tauri::command]
pub async fn load_editor_page(config: LoadEditorPageConfig) -> Result<Value, String> {
    let (project_file, project) = load_project_for_editing(&config.project_path)?;
    let pages = project
        .get("paginas")
        .and_then(|value| value.as_array())
        .ok_or_else(|| "Projeto sem páginas".to_string())?;
    let page = pages
        .get(config.page_index)
        .cloned()
        .ok_or_else(|| "Página inválida".to_string())?;

    Ok(json!({
        "project_file": project_file.to_string_lossy(),
        "project_dir": project_file.parent().unwrap_or_else(|| Path::new("")).to_string_lossy(),
        "page_index": config.page_index,
        "total_pages": pages.len(),
        "page": page,
    }))
}

#[tauri::command]
pub async fn create_text_layer(config: CreateTextLayerConfig) -> Result<Value, String> {
    let project_file = resolve_project_file(&config.project_path);
    project_schema::edit_project_value(&project_file, |project| {
        project_schema::create_text_layer(project, config.page_index, config.layout_bbox)
    })
}

#[tauri::command]
pub async fn patch_text_layer(config: PatchTextLayerConfig) -> Result<Value, String> {
    let project_file = resolve_project_file(&config.project_path);
    project_schema::edit_project_value(&project_file, |project| {
        project_schema::patch_text_layer(
            project,
            config.page_index,
            &config.layer_id,
            &config.patch,
        )
    })
}

#[tauri::command]
pub async fn delete_text_layer(config: DeleteTextLayerConfig) -> Result<(), String> {
    let project_file = resolve_project_file(&config.project_path);
    project_schema::edit_project_value(&project_file, |project| {
        project_schema::delete_text_layer(project, config.page_index, &config.layer_id)
    })
}

#[tauri::command]
pub async fn set_layer_visibility(config: SetLayerVisibilityConfig) -> Result<(), String> {
    let project_file = resolve_project_file(&config.project_path);
    project_schema::edit_project_value(&project_file, |project| {
        project_schema::set_layer_visibility(
            project,
            config.page_index,
            &config.layer_kind,
            config.layer_key.as_deref(),
            config.layer_id.as_deref(),
            config.visible,
        )
    })
}

fn paint_circle(bitmap: &mut GrayImage, center_x: i32, center_y: i32, radius: i32, value: u8) {
    let radius_sq = radius * radius;
    for y in (center_y - radius)..=(center_y + radius) {
        if y < 0 || y >= bitmap.height() as i32 {
            continue;
        }
        for x in (center_x - radius)..=(center_x + radius) {
            if x < 0 || x >= bitmap.width() as i32 {
                continue;
            }
            let dx = x - center_x;
            let dy = y - center_y;
            if dx * dx + dy * dy <= radius_sq {
                bitmap.put_pixel(x as u32, y as u32, Luma([value]));
            }
        }
    }
}

fn paint_stroke(bitmap: &mut GrayImage, stroke: &[[i32; 2]], radius: i32, value: u8) {
    if stroke.is_empty() {
        return;
    }
    if stroke.len() == 1 {
        paint_circle(bitmap, stroke[0][0], stroke[0][1], radius, value);
        return;
    }

    for window in stroke.windows(2) {
        let [x1, y1] = window[0];
        let [x2, y2] = window[1];
        let dx = x2 - x1;
        let dy = y2 - y1;
        let steps = dx.abs().max(dy.abs()).max(1);
        for step in 0..=steps {
            let t = step as f32 / steps as f32;
            let x = x1 as f32 + dx as f32 * t;
            let y = y1 as f32 + dy as f32 * t;
            paint_circle(bitmap, x.round() as i32, y.round() as i32, radius, value);
        }
    }
}

fn load_or_create_bitmap_layer(
    path: &Path,
    width: u32,
    height: u32,
    clear: bool,
) -> Result<GrayImage, String> {
    if !clear && path.exists() {
        let existing = ImageReader::open(path)
            .map_err(|e| format!("Erro ao abrir bitmap da layer: {e}"))?
            .decode()
            .map_err(|e| format!("Erro ao decodificar bitmap da layer: {e}"))?
            .to_luma8();
        if existing.width() == width && existing.height() == height {
            return Ok(existing);
        }
    }

    Ok(ImageBuffer::from_pixel(width, height, Luma([0])))
}

fn save_bitmap_layer(path: &Path, bitmap: &GrayImage) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Erro ao preparar diretório da layer bitmap: {e}"))?;
    }
    // Sufixo .tmp.png garante que image-rs reconheça como PNG ao gravar
    let mut temp_path = path.to_path_buf();
    let original_name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "layer.png".to_string());
    temp_path.set_file_name(format!("{original_name}.tmp.png"));
    bitmap
        .save(&temp_path)
        .map_err(|e| format!("Erro ao salvar layer bitmap temporária: {e}"))?;
    if path.exists() {
        std::fs::remove_file(path)
            .map_err(|e| format!("Erro ao substituir layer bitmap anterior: {e}"))?;
    }
    std::fs::rename(&temp_path, path)
        .map_err(|e| format!("Erro ao finalizar gravação da layer bitmap: {e}"))?;
    Ok(())
}

fn parse_hex_rgb(value: Option<&str>) -> [u8; 3] {
    let raw = value.unwrap_or("#000000").trim().trim_start_matches('#');
    if raw.len() != 6 {
        return [0, 0, 0];
    }
    let r = u8::from_str_radix(&raw[0..2], 16).unwrap_or(0);
    let g = u8::from_str_radix(&raw[2..4], 16).unwrap_or(0);
    let b = u8::from_str_radix(&raw[4..6], 16).unwrap_or(0);
    [r, g, b]
}

fn looks_like_legacy_grayscale_brush(image: &RgbaImage) -> bool {
    image
        .pixels()
        .all(|pixel| pixel[3] == 255 && pixel[0] == pixel[1] && pixel[1] == pixel[2])
}

fn load_or_create_rgba_brush_layer(
    path: &Path,
    width: u32,
    height: u32,
    clear: bool,
) -> Result<RgbaImage, String> {
    if !clear && path.exists() {
        let decoded = ImageReader::open(path)
            .map_err(|e| format!("Erro ao abrir brush bitmap: {e}"))?
            .decode()
            .map_err(|e| format!("Erro ao decodificar brush bitmap: {e}"))?;
        let mut existing = decoded.to_rgba8();
        if existing.width() == width && existing.height() == height {
            if looks_like_legacy_grayscale_brush(&existing) {
                for pixel in existing.pixels_mut() {
                    let alpha = pixel[0];
                    *pixel = image::Rgba([0, 0, 0, alpha]);
                }
            }
            return Ok(existing);
        }
    }
    Ok(ImageBuffer::from_pixel(
        width,
        height,
        image::Rgba([0, 0, 0, 0]),
    ))
}

fn paint_circle_rgba(
    bitmap: &mut RgbaImage,
    center_x: i32,
    center_y: i32,
    radius: i32,
    color: [u8; 3],
    alpha: u8,
    erase: bool,
) {
    let radius_sq = radius * radius;
    for y in (center_y - radius)..=(center_y + radius) {
        if y < 0 || y >= bitmap.height() as i32 {
            continue;
        }
        for x in (center_x - radius)..=(center_x + radius) {
            if x < 0 || x >= bitmap.width() as i32 {
                continue;
            }
            let dx = x - center_x;
            let dy = y - center_y;
            if dx * dx + dy * dy <= radius_sq {
                if erase {
                    bitmap.put_pixel(x as u32, y as u32, image::Rgba([0, 0, 0, 0]));
                } else {
                    bitmap.put_pixel(
                        x as u32,
                        y as u32,
                        image::Rgba([color[0], color[1], color[2], alpha]),
                    );
                }
            }
        }
    }
}

fn paint_stroke_rgba(
    bitmap: &mut RgbaImage,
    stroke: &[[i32; 2]],
    radius: i32,
    color: [u8; 3],
    alpha: u8,
    erase: bool,
) {
    if stroke.is_empty() {
        return;
    }
    if stroke.len() == 1 {
        paint_circle_rgba(
            bitmap,
            stroke[0][0],
            stroke[0][1],
            radius,
            color,
            alpha,
            erase,
        );
        return;
    }
    for window in stroke.windows(2) {
        let [x1, y1] = window[0];
        let [x2, y2] = window[1];
        let dx = x2 - x1;
        let dy = y2 - y1;
        let steps = dx.abs().max(dy.abs()).max(1);
        for step in 0..=steps {
            let t = step as f32 / steps as f32;
            let x = x1 as f32 + dx as f32 * t;
            let y = y1 as f32 + dy as f32 * t;
            paint_circle_rgba(
                bitmap,
                x.round() as i32,
                y.round() as i32,
                radius,
                color,
                alpha,
                erase,
            );
        }
    }
}

fn apply_rgba_stroke_mask(
    bitmap: &mut RgbaImage,
    stroke_mask: &GrayImage,
    color: [u8; 3],
    alpha: u8,
    erase: bool,
) {
    let width = bitmap.width().min(stroke_mask.width());
    let height = bitmap.height().min(stroke_mask.height());
    for y in 0..height {
        for x in 0..width {
            if stroke_mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            if erase {
                bitmap.put_pixel(x, y, image::Rgba([0, 0, 0, 0]));
            } else {
                bitmap.put_pixel(x, y, image::Rgba([color[0], color[1], color[2], alpha]));
            }
        }
    }
}

fn save_rgba_bitmap_layer(path: &Path, bitmap: &RgbaImage) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Erro ao preparar diretório da layer bitmap: {e}"))?;
    }
    let mut temp_path = path.to_path_buf();
    let original_name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "layer.png".to_string());
    temp_path.set_file_name(format!("{original_name}.tmp.png"));
    bitmap
        .save_with_format(&temp_path, ImageFormat::Png)
        .map_err(|e| format!("Erro ao salvar layer brush temporária: {e}"))?;
    if path.exists() {
        std::fs::remove_file(path)
            .map_err(|e| format!("Erro ao substituir layer brush anterior: {e}"))?;
    }
    std::fs::rename(&temp_path, path)
        .map_err(|e| format!("Erro ao finalizar gravação da layer brush: {e}"))?;
    Ok(())
}

fn update_brush_rgba_layer(config: BitmapLayerUpdateConfig) -> Result<String, String> {
    if config.width == 0 || config.height == 0 {
        return Err("Dimensões da layer bitmap inválidas".to_string());
    }
    let project_file = resolve_project_file(&config.project_path);
    project_schema::edit_project_value(&project_file, |project| {
        let relative_layer_path =
            project_schema::ensure_bitmap_layer_path(project, config.page_index, "brush")?;
        let project_dir = project_file
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let absolute_layer_path = project_dir.join(&relative_layer_path);
        let mut bitmap = load_or_create_rgba_brush_layer(
            &absolute_layer_path,
            config.width,
            config.height,
            config.clear,
        )?;
        let color = parse_hex_rgb(config.color.as_deref());
        let alpha = ((config.opacity.unwrap_or(1.0).clamp(0.0, 1.0)) * 255.0).round() as u8;
        let _hardness = config.hardness.unwrap_or(1.0).clamp(0.0, 1.0);
        let stroke_mask = build_clipped_stroke_mask(
            config.width,
            config.height,
            config.brush_size,
            &config.strokes,
            config.clip_mask_png.as_deref(),
        )?;
        apply_rgba_stroke_mask(&mut bitmap, &stroke_mask, color, alpha, config.erase);
        save_rgba_bitmap_layer(&absolute_layer_path, &bitmap)?;
        project_schema::set_layer_visibility(
            project,
            config.page_index,
            "image",
            Some("brush"),
            None,
            true,
        )?;
        Ok(absolute_layer_path.to_string_lossy().replace('\\', "/"))
    })
}

fn update_bitmap_layer(config: BitmapLayerUpdateConfig, layer_key: &str) -> Result<String, String> {
    if config.width == 0 || config.height == 0 {
        return Err("Dimensões da layer bitmap inválidas".to_string());
    }

    let project_file = resolve_project_file(&config.project_path);
    project_schema::edit_project_value(&project_file, |project| {
        let relative_layer_path =
            project_schema::ensure_bitmap_layer_path(project, config.page_index, layer_key)?;
        let project_dir = project_file
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let absolute_layer_path = project_dir.join(&relative_layer_path);

        let mut bitmap = load_or_create_bitmap_layer(
            &absolute_layer_path,
            config.width,
            config.height,
            config.clear,
        )?;

        let value = if config.erase { 0 } else { 255 };
        let stroke_mask = build_clipped_stroke_mask(
            config.width,
            config.height,
            config.brush_size,
            &config.strokes,
            config.clip_mask_png.as_deref(),
        )?;
        apply_gray_stroke_mask(&mut bitmap, &stroke_mask, value);

        save_bitmap_layer(&absolute_layer_path, &bitmap)?;
        project_schema::set_layer_visibility(
            project,
            config.page_index,
            "image",
            Some(layer_key),
            None,
            !config.erase || !config.strokes.is_empty(),
        )?;

        Ok(absolute_layer_path.to_string_lossy().replace('\\', "/"))
    })
}

fn resolve_project_relative_path(project_dir: &Path, rel_path: &str) -> PathBuf {
    let path = Path::new(rel_path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        project_dir.join(path)
    }
}

fn page_image_layer_path(page: &Value, layer_key: &str) -> Option<String> {
    page.get("image_layers")
        .and_then(|value| value.as_object())
        .and_then(|layers| layers.get(layer_key))
        .and_then(|layer| layer.get("path"))
        .and_then(|path| path.as_str())
        .filter(|path| !path.trim().is_empty())
        .map(str::to_owned)
}

fn default_inpaint_cache_rel(page: &Value, page_index: usize) -> String {
    let page_number = page
        .get("numero")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .unwrap_or((page_index + 1) as u64);
    format!("layers/inpaint-cache/{page_number:03}.png")
}

fn page_editor_cache_path(page: &Value, key: &str) -> Option<String> {
    page.get("editor_cache")
        .and_then(Value::as_object)
        .and_then(|cache| cache.get(key))
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .map(str::to_owned)
}

fn ensure_inpaint_cache_rel(page: &mut Value, page_index: usize) -> Result<String, String> {
    if let Some(existing) = page_editor_cache_path(page, "inpaint") {
        return Ok(existing);
    }
    let rel = default_inpaint_cache_rel(page, page_index);
    let page_obj = page
        .as_object_mut()
        .ok_or_else(|| "pagina invalida para cache de inpaint".to_string())?;
    let cache_value = page_obj
        .entry("editor_cache".to_string())
        .or_insert_with(|| json!({}));
    let cache = cache_value
        .as_object_mut()
        .ok_or_else(|| "editor_cache invalido".to_string())?;
    cache.insert("inpaint".to_string(), Value::from(rel.clone()));
    Ok(rel)
}

fn ensure_inpaint_cache_snapshot(
    page: &mut Value,
    page_index: usize,
    project_dir: &Path,
    source: &RgbaImage,
) -> Result<PathBuf, String> {
    let cache_rel = ensure_inpaint_cache_rel(page, page_index)?;
    let cache_path = resolve_project_relative_path(project_dir, &cache_rel);
    if !cache_path.exists() {
        save_rgba_image_preserving_format(&cache_path, source)?;
    }
    Ok(cache_path)
}

fn load_rgba_image(path: &Path, label: &str) -> Result<RgbaImage, String> {
    Ok(ImageReader::open(path)
        .map_err(|e| format!("Erro ao abrir {label}: {e}"))?
        .decode()
        .map_err(|e| format!("Erro ao decodificar {label}: {e}"))?
        .to_rgba8())
}

pub fn clear_inpaint_cache_for_page(project_file: &Path, page_index: usize) -> Result<(), String> {
    let project_dir = project_file
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let cache_rel = project_schema::edit_project_value(project_file, |project| {
        let pages = project
            .get_mut("paginas")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| "Projeto sem paginas".to_string())?;
        let Some(page) = pages.get_mut(page_index) else {
            return Ok(None);
        };
        let rel = page_editor_cache_path(page, "inpaint")
            .unwrap_or_else(|| default_inpaint_cache_rel(page, page_index));
        let Some(page_obj) = page.as_object_mut() else {
            return Ok(Some(rel));
        };
        let remove_editor_cache = if let Some(cache_value) = page_obj.get_mut("editor_cache") {
            if let Some(cache) = cache_value.as_object_mut() {
                cache.remove("inpaint");
                cache.is_empty()
            } else {
                false
            }
        } else {
            false
        };
        if remove_editor_cache {
            page_obj.remove("editor_cache");
        }
        Ok(Some(rel))
    })?;
    if let Some(rel) = cache_rel {
        let cache_path = resolve_project_relative_path(&project_dir, &rel);
        if cache_path.exists() {
            std::fs::remove_file(&cache_path)
                .map_err(|e| format!("Erro ao limpar cache de inpaint: {e}"))?;
        }
    }
    Ok(())
}

fn save_rgba_image_preserving_format(path: &Path, image: &RgbaImage) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Erro ao preparar imagem de destino: {e}"))?;
    }

    let format = ImageFormat::from_path(path).unwrap_or(ImageFormat::Png);
    let dynamic = match format {
        ImageFormat::Jpeg => {
            DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(image.clone()).to_rgb8())
        }
        _ => DynamicImage::ImageRgba8(image.clone()),
    };

    let mut temp_path = path.to_path_buf();
    let original_name = path
        .file_name()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_else(|| "inpaint.png".to_string());
    temp_path.set_file_name(format!("{original_name}.tmp"));

    dynamic
        .save_with_format(&temp_path, format)
        .map_err(|e| format!("Erro ao salvar imagem temporária recuperada: {e}"))?;
    if path.exists() {
        std::fs::remove_file(path)
            .map_err(|e| format!("Erro ao substituir inpaint anterior: {e}"))?;
    }
    std::fs::rename(&temp_path, path)
        .map_err(|e| format!("Erro ao finalizar inpaint recuperado: {e}"))?;
    Ok(())
}

fn load_inpaint_or_history_source(inpaint_path: &Path, history_source: &RgbaImage) -> RgbaImage {
    if !inpaint_path.exists() {
        return history_source.clone();
    }
    let decoded = ImageReader::open(inpaint_path)
        .map_err(|error| error.to_string())
        .and_then(|reader| reader.decode().map_err(|error| error.to_string()));
    match decoded {
        Ok(image) => image.to_rgba8(),
        Err(error) => {
            eprintln!(
                "[recovery] inpaint inválido em {}; usando snapshot base como fallback: {error}",
                inpaint_path.display()
            );
            history_source.clone()
        }
    }
}

fn build_stroke_mask(
    width: u32,
    height: u32,
    brush_size: u32,
    strokes: &[Vec<[i32; 2]>],
) -> GrayImage {
    let mut mask = ImageBuffer::from_pixel(width, height, Luma([0]));
    let radius = (brush_size.max(1) as i32 / 2).max(1);
    for stroke in strokes {
        paint_stroke(&mut mask, stroke, radius, 255);
    }
    mask
}

fn decode_clip_mask_png(data: &str, width: u32, height: u32) -> Result<GrayImage, String> {
    let bytes = decode_png_data_url(data)?;
    let decoded = image::load_from_memory_with_format(&bytes, ImageFormat::Png)
        .map_err(|e| format!("Mascara de recorte invalida: {e}"))?
        .to_luma8();
    if decoded.width() == width && decoded.height() == height {
        return Ok(decoded);
    }
    Ok(image::imageops::resize(
        &decoded,
        width,
        height,
        FilterType::Nearest,
    ))
}

fn apply_clip_mask(stroke_mask: &mut GrayImage, clip_mask: &GrayImage) {
    let width = stroke_mask.width().min(clip_mask.width());
    let height = stroke_mask.height().min(clip_mask.height());
    for y in 0..height {
        for x in 0..width {
            if clip_mask.get_pixel(x, y)[0] == 0 {
                stroke_mask.put_pixel(x, y, Luma([0]));
            }
        }
    }
}

fn build_clipped_stroke_mask(
    width: u32,
    height: u32,
    brush_size: u32,
    strokes: &[Vec<[i32; 2]>],
    clip_mask_png: Option<&str>,
) -> Result<GrayImage, String> {
    let mut stroke_mask = build_stroke_mask(width, height, brush_size, strokes);
    if let Some(raw_clip) = clip_mask_png.filter(|value| !value.trim().is_empty()) {
        let clip_mask = decode_clip_mask_png(raw_clip, width, height)?;
        apply_clip_mask(&mut stroke_mask, &clip_mask);
    }
    Ok(stroke_mask)
}

fn apply_gray_stroke_mask(bitmap: &mut GrayImage, stroke_mask: &GrayImage, value: u8) {
    let width = bitmap.width().min(stroke_mask.width());
    let height = bitmap.height().min(stroke_mask.height());
    for y in 0..height {
        for x in 0..width {
            if stroke_mask.get_pixel(x, y)[0] > 0 {
                bitmap.put_pixel(x, y, Luma([value]));
            }
        }
    }
}

fn apply_history_brush_to_inpaint(
    project: &mut Value,
    page_index: usize,
    project_dir: &Path,
    stroke_mask: &GrayImage,
    dirty_bbox: Option<[u32; 4]>,
) -> Result<PathBuf, String> {
    let pages = project
        .get_mut("paginas")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "Projeto sem páginas".to_string())?;
    let page = pages
        .get_mut(page_index)
        .ok_or_else(|| "Página inválida".to_string())?;

    let base_rel = page_image_layer_path(page, "base")
        .or_else(|| {
            page.get("arquivo_original")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .ok_or_else(|| "Layer base não encontrada para recovery".to_string())?;
    let base_path = resolve_project_relative_path(project_dir, &base_rel);
    if !base_path.exists() {
        return Err(format!(
            "Imagem original não encontrada para recovery: {}",
            base_path.display()
        ));
    }

    let inpaint_rel = resolve_inpaint_rel(page)
        .map(str::to_owned)
        .unwrap_or_else(|| {
            let name = Path::new(&base_rel)
                .file_name()
                .map(|value| value.to_string_lossy().to_string())
                .unwrap_or_else(|| "001.png".to_string());
            format!("images/{name}")
        });
    let inpaint_path = resolve_project_relative_path(project_dir, &inpaint_rel);
    if let Some(parent) = inpaint_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Erro ao preparar layer inpaint: {e}"))?;
    }

    let original = ImageReader::open(&base_path)
        .map_err(|e| format!("Erro ao abrir imagem original para recovery: {e}"))?
        .decode()
        .map_err(|e| format!("Erro ao decodificar imagem original para recovery: {e}"))?
        .to_rgba8();
    let mut inpaint = load_inpaint_or_history_source(&inpaint_path, &original);

    if inpaint.dimensions() != original.dimensions() {
        inpaint = image::imageops::resize(
            &inpaint,
            original.width(),
            original.height(),
            FilterType::Lanczos3,
        );
    }
    ensure_inpaint_cache_snapshot(page, page_index, project_dir, &inpaint)?;
    let mask = if stroke_mask.dimensions() == original.dimensions() {
        stroke_mask.clone()
    } else {
        image::imageops::resize(
            stroke_mask,
            original.width(),
            original.height(),
            FilterType::Nearest,
        )
    };

    let [min_x, min_y, max_x, max_y] =
        dirty_bbox.unwrap_or([0, 0, original.width(), original.height()]);
    let min_x = min_x.min(original.width());
    let min_y = min_y.min(original.height());
    let max_x = max_x.min(original.width()).max(min_x);
    let max_y = max_y.min(original.height()).max(min_y);

    for y in min_y..max_y {
        for x in min_x..max_x {
            if mask.get_pixel(x, y)[0] > 0 {
                inpaint.put_pixel(x, y, *original.get_pixel(x, y));
            }
        }
    }
    if page_image_layer_path(page, "inpaint").is_none() {
        let layers = page
            .get_mut("image_layers")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| "image_layers inválido".to_string())?;
        layers.insert(
            "inpaint".to_string(),
            json!({
                "key": "inpaint",
                "path": inpaint_rel,
                "visible": true,
                "locked": false
            }),
        );
    }
    save_rgba_image_preserving_format(&inpaint_path, &inpaint)?;

    project_schema::set_layer_visibility(
        project,
        page_index,
        "image",
        Some("inpaint"),
        None,
        true,
    )?;
    Ok(inpaint_path)
}

fn apply_cached_inpaint_brush_to_inpaint(
    project: &mut Value,
    page_index: usize,
    project_dir: &Path,
    stroke_mask: &GrayImage,
    dirty_bbox: Option<[u32; 4]>,
) -> Result<PathBuf, String> {
    let pages = project
        .get_mut("paginas")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "Projeto sem paginas".to_string())?;
    let page = pages
        .get_mut(page_index)
        .ok_or_else(|| "Pagina invalida".to_string())?;

    let inpaint_rel = resolve_inpaint_rel(page)
        .map(str::to_owned)
        .or_else(|| {
            page.get("arquivo_original")
                .and_then(Value::as_str)
                .and_then(|original| {
                    Path::new(original)
                        .file_name()
                        .map(|name| format!("images/{}", name.to_string_lossy()))
                })
        })
        .ok_or_else(|| "Layer inpaint nao encontrada para reinpaint".to_string())?;
    let inpaint_path = resolve_project_relative_path(project_dir, &inpaint_rel);
    if !inpaint_path.exists() {
        return Err(format!(
            "Imagem inpaint atual nao encontrada para reinpaint: {}",
            inpaint_path.display()
        ));
    }

    let mut inpaint = load_rgba_image(&inpaint_path, "imagem inpaint atual")?;
    let cache_rel = ensure_inpaint_cache_rel(page, page_index)?;
    let cache_path = resolve_project_relative_path(project_dir, &cache_rel);
    if !cache_path.exists() {
        save_rgba_image_preserving_format(&cache_path, &inpaint)?;
    }
    let mut inpaint_cache = load_rgba_image(&cache_path, "cache de inpaint")?;
    if inpaint_cache.dimensions() != inpaint.dimensions() {
        inpaint_cache = image::imageops::resize(
            &inpaint_cache,
            inpaint.width(),
            inpaint.height(),
            FilterType::Lanczos3,
        );
    }
    let mask = if stroke_mask.dimensions() == inpaint.dimensions() {
        stroke_mask.clone()
    } else {
        image::imageops::resize(
            stroke_mask,
            inpaint.width(),
            inpaint.height(),
            FilterType::Nearest,
        )
    };

    let [min_x, min_y, max_x, max_y] =
        dirty_bbox.unwrap_or([0, 0, inpaint.width(), inpaint.height()]);
    let min_x = min_x.min(inpaint.width());
    let min_y = min_y.min(inpaint.height());
    let max_x = max_x.min(inpaint.width()).max(min_x);
    let max_y = max_y.min(inpaint.height()).max(min_y);

    for y in min_y..max_y {
        for x in min_x..max_x {
            if mask.get_pixel(x, y)[0] > 0 {
                inpaint.put_pixel(x, y, *inpaint_cache.get_pixel(x, y));
            }
        }
    }

    save_rgba_image_preserving_format(&inpaint_path, &inpaint)?;
    project_schema::set_layer_visibility(
        project,
        page_index,
        "image",
        Some("inpaint"),
        None,
        true,
    )?;
    Ok(inpaint_path)
}

#[tauri::command]
pub async fn update_mask_region(config: BitmapLayerUpdateConfig) -> Result<String, String> {
    update_bitmap_layer(config, "mask")
}

#[tauri::command]
pub async fn update_brush_region(config: BitmapLayerUpdateConfig) -> Result<String, String> {
    update_brush_rgba_layer(config)
}

#[tauri::command]
pub async fn update_recovery_region(config: BitmapLayerUpdateConfig) -> Result<String, String> {
    if config.width == 0 || config.height == 0 {
        return Err("Dimensões da layer bitmap inválidas".to_string());
    }

    let project_file = resolve_project_file(&config.project_path);
    project_schema::edit_project_value(&project_file, |project| {
        let project_dir = project_file
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        let updated_path = if config.erase {
            let relative_layer_path =
                project_schema::ensure_bitmap_layer_path(project, config.page_index, "recovery")?;
            let absolute_layer_path = project_dir.join(&relative_layer_path);
            let mut bitmap = load_or_create_bitmap_layer(
                &absolute_layer_path,
                config.width,
                config.height,
                config.clear,
            )?;
            let stroke_mask = build_clipped_stroke_mask(
                config.width,
                config.height,
                config.brush_size,
                &config.strokes,
                config.clip_mask_png.as_deref(),
            )?;
            apply_gray_stroke_mask(&mut bitmap, &stroke_mask, 0);
            save_bitmap_layer(&absolute_layer_path, &bitmap)?;
            absolute_layer_path
        } else {
            let stroke_mask = build_clipped_stroke_mask(
                config.width,
                config.height,
                config.brush_size,
                &config.strokes,
                config.clip_mask_png.as_deref(),
            )?;
            apply_history_brush_to_inpaint(
                project,
                config.page_index,
                &project_dir,
                &stroke_mask,
                config.dirty_bbox,
            )?
        };
        project_schema::set_layer_visibility(
            project,
            config.page_index,
            "image",
            Some("recovery"),
            None,
            false,
        )?;

        Ok(updated_path.to_string_lossy().replace('\\', "/"))
    })
}

#[tauri::command]
pub async fn update_reinpaint_region(config: BitmapLayerUpdateConfig) -> Result<String, String> {
    if config.width == 0 || config.height == 0 {
        return Err("Dimensoes da layer bitmap invalidas".to_string());
    }

    let project_file = resolve_project_file(&config.project_path);
    project_schema::edit_project_value(&project_file, |project| {
        let project_dir = project_file
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let stroke_mask = build_clipped_stroke_mask(
            config.width,
            config.height,
            config.brush_size,
            &config.strokes,
            config.clip_mask_png.as_deref(),
        )?;
        let updated_path = apply_cached_inpaint_brush_to_inpaint(
            project,
            config.page_index,
            &project_dir,
            &stroke_mask,
            config.dirty_bbox,
        )?;

        Ok(updated_path.to_string_lossy().replace('\\', "/"))
    })
}

#[derive(Debug, Deserialize)]
pub struct WriteMaskFromPngConfig {
    pub project_path: String,
    pub page_index: usize,
    /// PNG em base64 (com ou sem prefixo data:image/png;base64,)
    pub png_data: String,
    /// "mask" ou "brush"
    pub layer_key: String,
    /// Operação: "replace" (substitui), "add" (max), "subtract" (min do inverso).
    /// Para compatibilidade: se ausente, usa compose=true → "add", compose=false → "replace".
    #[serde(default)]
    pub op: String,
    /// Legado — use op em vez disso.
    #[serde(default)]
    pub compose: bool,
}

#[derive(Debug, Deserialize)]
pub struct SnapshotImageLayerConfig {
    pub project_path: String,
    pub page_index: usize,
    pub layer_key: String,
    #[serde(default)]
    pub source_path: Option<String>,
}

/// Escreve uma imagem PNG (passada como base64) diretamente na camada mask/brush da página.
/// Usado pelo lasso tool (Fase 8) para polygon fill.
#[derive(Debug, Deserialize)]
pub struct WriteHealingMaskConfig {
    pub project_path: String,
    pub page_index: usize,
    pub png_data: String,
    #[serde(default)]
    pub bbox: Option<[u32; 4]>,
}

#[derive(Debug, Deserialize)]
pub struct HealInpaintRegionConfig {
    pub project_path: String,
    pub page_index: usize,
    pub bbox: [i64; 4],
    pub mask_path: String,
}

#[derive(Debug, Serialize)]
pub struct RegionalInpaintResult {
    pub page_index: usize,
    pub inpaint_path: String,
    pub before_inpaint_path: Option<String>,
    pub bbox: [u32; 4],
}

const MAX_HEALING_MASK_BYTES: usize = 32 * 1024 * 1024;

fn healing_mask_root(project_dir: &Path) -> PathBuf {
    project_dir.join("editor_cache").join("healing_masks")
}

fn healing_inpaint_root(project_dir: &Path) -> PathBuf {
    project_dir.join("editor_cache").join("healing_inpaint")
}

fn project_relative_string(project_dir: &Path, path: &Path) -> String {
    path.strip_prefix(project_dir)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn decode_png_data_url(data: &str) -> Result<Vec<u8>, String> {
    let b64 = data
        .trim_start_matches("data:image/png;base64,")
        .trim_start_matches("data:image/webp;base64,");
    let bytes = BASE64
        .decode(b64)
        .map_err(|e| format!("Base64 invalido: {e}"))?;
    if bytes.len() > MAX_HEALING_MASK_BYTES {
        return Err("Mascara do pincel corretor muito grande".to_string());
    }
    Ok(bytes)
}

fn validate_bbox_against_dimensions(
    bbox: [i64; 4],
    width: u32,
    height: u32,
) -> Result<[u32; 4], String> {
    if bbox[0] < 0 || bbox[1] < 0 || bbox[2] < 0 || bbox[3] < 0 {
        return Err("BBox do pincel corretor fora dos limites da pagina".to_string());
    }
    let bbox = [
        bbox[0] as u32,
        bbox[1] as u32,
        bbox[2] as u32,
        bbox[3] as u32,
    ];
    if bbox[0] >= bbox[2] || bbox[1] >= bbox[3] {
        return Err("BBox do pincel corretor invalida".to_string());
    }
    if bbox[2] > width || bbox[3] > height {
        return Err("BBox do pincel corretor excede a pagina".to_string());
    }
    Ok(bbox)
}

fn resolve_healing_mask_path(project_dir: &Path, mask_path: &str) -> Result<PathBuf, String> {
    let raw = Path::new(mask_path);
    let path = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        project_dir.join(raw)
    };
    let canonical_root = healing_mask_root(project_dir)
        .canonicalize()
        .map_err(|e| format!("Cache de mascaras do pincel corretor invalido: {e}"))?;
    let canonical_path = path
        .canonicalize()
        .map_err(|e| format!("Mascara do pincel corretor nao encontrada: {e}"))?;
    if !canonical_path.starts_with(&canonical_root) {
        return Err("Mascara do pincel corretor fora do projeto".to_string());
    }
    Ok(canonical_path)
}

fn copy_inpaint_snapshot(source: &Path, version_dir: &Path) -> Result<PathBuf, String> {
    let extension = source
        .extension()
        .and_then(|extension| extension.to_str())
        .filter(|extension| !extension.trim().is_empty())
        .unwrap_or("png");
    let destination = version_dir.join(format!("{}-before.{extension}", uuid::Uuid::new_v4()));
    std::fs::copy(source, &destination)
        .map_err(|e| format!("Erro ao copiar snapshot anterior do inpaint: {e}"))?;
    Ok(destination)
}

fn path_is_inside(root: &Path, path: &Path) -> bool {
    let root = root
        .to_string_lossy()
        .replace('\\', "/")
        .trim_end_matches('/')
        .to_ascii_lowercase();
    let path = path
        .to_string_lossy()
        .replace('\\', "/")
        .to_ascii_lowercase();
    path == root || path.starts_with(&format!("{root}/"))
}

fn stable_healing_inpaint_path(
    project_dir: &Path,
    page: &Value,
    page_index: usize,
    output_path: &Path,
) -> PathBuf {
    let page_number = page
        .get("numero")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .unwrap_or((page_index + 1) as u64);
    let stem = page_image_layer_path(page, "base")
        .or_else(|| {
            page.get("arquivo_original")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .and_then(|path| {
            Path::new(&path)
                .file_stem()
                .map(|stem| stem.to_string_lossy().to_string())
        })
        .filter(|stem| !stem.trim().is_empty())
        .unwrap_or_else(|| format!("{page_number:03}"));
    let extension = output_path
        .extension()
        .and_then(|extension| extension.to_str())
        .filter(|extension| !extension.trim().is_empty())
        .unwrap_or("png");
    project_dir
        .join("images")
        .join(format!("{stem}.{extension}"))
}

fn normalize_healing_visible_output(
    project_dir: &Path,
    page: &Value,
    page_index: usize,
    output_path: &Path,
) -> Result<PathBuf, String> {
    if !path_is_inside(&healing_inpaint_root(project_dir), output_path) {
        return Ok(output_path.to_path_buf());
    }

    let stable_path = stable_healing_inpaint_path(project_dir, page, page_index, output_path);
    if stable_path == output_path {
        return Ok(stable_path);
    }
    let image = load_rgba_image(output_path, "resultado do pincel corretor")?;
    save_rgba_image_preserving_format(&stable_path, &image)?;
    Ok(stable_path)
}

fn validate_healing_mask_bbox(
    bbox: [i64; 4],
    mask_width: u32,
    mask_height: u32,
) -> Result<[u32; 4], String> {
    let bbox =
        validate_bbox_against_dimensions(bbox, bbox[2].max(0) as u32, bbox[3].max(0) as u32)?;
    let bbox_width = bbox[2].saturating_sub(bbox[0]);
    let bbox_height = bbox[3].saturating_sub(bbox[1]);
    if mask_width == bbox_width && mask_height == bbox_height {
        return Ok(bbox);
    }
    if bbox[2] > mask_width || bbox[3] > mask_height {
        return Err("BBox do pincel corretor excede a mascara".to_string());
    }
    Ok(bbox)
}

#[tauri::command]
pub async fn write_healing_mask(config: WriteHealingMaskConfig) -> Result<String, String> {
    let started = std::time::Instant::now();
    let project_file = resolve_project_file(&config.project_path);
    let project = project_schema::load_project_value(&project_file)?;
    let pages = project
        .get("paginas")
        .and_then(Value::as_array)
        .ok_or_else(|| "Projeto sem paginas validas".to_string())?;
    if config.page_index >= pages.len() {
        return Err("Pagina do pincel corretor invalida".to_string());
    }
    let project_dir = project_file
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "Caminho do projeto invalido".to_string())?;
    let png_bytes = decode_png_data_url(&config.png_data)?;
    let decoded = image::load_from_memory_with_format(&png_bytes, ImageFormat::Png)
        .map_err(|e| format!("Mascara PNG invalida: {e}"))?
        .to_luma8();
    if decoded.width() == 0 || decoded.height() == 0 {
        return Err("Mascara do pincel corretor vazia".to_string());
    }
    if let Some(bbox) = config.bbox {
        validate_healing_mask_bbox(
            [
                i64::from(bbox[0]),
                i64::from(bbox[1]),
                i64::from(bbox[2]),
                i64::from(bbox[3]),
            ],
            decoded.width(),
            decoded.height(),
        )?;
    }

    let page_dir =
        healing_mask_root(&project_dir).join(format!("page-{:04}", config.page_index + 1));
    std::fs::create_dir_all(&page_dir)
        .map_err(|e| format!("Erro ao criar cache do pincel corretor: {e}"))?;
    let mask_path = page_dir.join(format!("{}.png", uuid::Uuid::new_v4()));
    let mut temp_path = mask_path.clone();
    temp_path.set_extension("tmp.png");
    decoded
        .save_with_format(&temp_path, ImageFormat::Png)
        .map_err(|e| format!("Erro ao salvar mascara temporaria: {e}"))?;
    if mask_path.exists() {
        std::fs::remove_file(&mask_path)
            .map_err(|e| format!("Erro ao substituir mascara temporaria: {e}"))?;
    }
    std::fs::rename(&temp_path, &mask_path)
        .map_err(|e| format!("Erro ao finalizar mascara temporaria: {e}"))?;
    eprintln!(
        "[EditorAction] timing healing mask page={} elapsed={:.3}s bbox={:?} mask={}x{}",
        config.page_index,
        started.elapsed().as_secs_f64(),
        config.bbox,
        decoded.width(),
        decoded.height()
    );
    Ok(mask_path.to_string_lossy().replace('\\', "/"))
}

#[tauri::command]
pub async fn heal_inpaint_region(
    app: tauri::AppHandle,
    config: HealInpaintRegionConfig,
) -> Result<RegionalInpaintResult, String> {
    use crate::commands::pipeline::{reinpaint_page_with_region, PageRegionConfig};

    let command_started = std::time::Instant::now();
    let project_file = resolve_project_file(&config.project_path);
    let project = project_schema::load_project_value(&project_file)?;
    let pages = project
        .get("paginas")
        .and_then(Value::as_array)
        .ok_or_else(|| "Projeto sem paginas validas".to_string())?;
    if config.page_index >= pages.len() {
        return Err("Pagina do pincel corretor invalida".to_string());
    }
    let project_dir = project_file
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "Caminho do projeto invalido".to_string())?;
    let current_inpaint_path = pages
        .get(config.page_index)
        .and_then(|page| page_image_layer_path(page, "inpaint"))
        .map(|rel| resolve_project_relative_path(&project_dir, &rel));
    let mask_path = resolve_healing_mask_path(&project_dir, &config.mask_path)?;
    if !mask_has_nonzero_pixels(&mask_path)? {
        return Err("Mascara do pincel corretor sem pixels ativos".to_string());
    }
    let mask = image::open(&mask_path)
        .map_err(|e| format!("Mascara do pincel corretor invalida: {e}"))?
        .to_luma8();
    let bbox = validate_healing_mask_bbox(config.bbox, mask.width(), mask.height())?;

    let version_dir =
        healing_inpaint_root(&project_dir).join(format!("page-{:04}", config.page_index + 1));
    std::fs::create_dir_all(&version_dir)
        .map_err(|e| format!("Erro ao criar cache de inpaint do pincel corretor: {e}"))?;
    let before_snapshot_started = std::time::Instant::now();
    let before_inpaint_path =
        if let Some(path) = current_inpaint_path.as_ref().filter(|path| path.exists()) {
            Some(copy_inpaint_snapshot(path, &version_dir)?)
        } else {
            None
        };
    let before_snapshot_seconds = before_snapshot_started.elapsed().as_secs_f64();

    let worker_started = std::time::Instant::now();
    let output = reinpaint_page_with_region(
        app,
        config.project_path.clone(),
        config.page_index as u32,
        Some(PageRegionConfig {
            bbox: Some(bbox),
            mask_path: Some(mask_path.to_string_lossy().to_string()),
        }),
    )
    .await?;
    let worker_seconds = worker_started.elapsed().as_secs_f64();
    let output_path = if output.trim().is_empty() {
        current_inpaint_path
            .clone()
            .ok_or_else(|| "Inpaint regional nao retornou imagem".to_string())?
    } else {
        resolve_project_relative_path(&project_dir, &output)
    };
    if !output_path.exists() {
        return Err("Inpaint regional nao gerou imagem valida".to_string());
    }
    let visible_output_path = normalize_healing_visible_output(
        &project_dir,
        &pages[config.page_index],
        config.page_index,
        &output_path,
    )?;
    let visible_rel = project_relative_string(&project_dir, &visible_output_path);
    project_schema::edit_project_value(&project_file, |project| {
        let page = project
            .get_mut("paginas")
            .and_then(Value::as_array_mut)
            .and_then(|pages| pages.get_mut(config.page_index))
            .ok_or_else(|| "Pagina do pincel corretor invalida".to_string())?;
        let layers = page
            .get_mut("image_layers")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| "image_layers invalido".to_string())?;
        let entry = layers
            .entry("inpaint".to_string())
            .or_insert_with(|| json!({}));
        let entry_obj = entry
            .as_object_mut()
            .ok_or_else(|| "image_layers.inpaint invalido".to_string())?;
        entry_obj.insert("key".to_string(), json!("inpaint"));
        entry_obj.insert("path".to_string(), json!(visible_rel));
        entry_obj.insert("visible".to_string(), json!(true));
        entry_obj
            .entry("locked".to_string())
            .or_insert(json!(false));
        Ok(())
    })?;
    eprintln!(
        "[EditorAction] timing healing command page={} total={:.3}s before_snapshot={:.3}s worker={:.3}s finalize={:.3}s out={}",
        config.page_index,
        command_started.elapsed().as_secs_f64(),
        before_snapshot_seconds,
        worker_seconds,
        command_started.elapsed().as_secs_f64() - before_snapshot_seconds - worker_seconds,
        visible_output_path.display()
    );
    Ok(RegionalInpaintResult {
        page_index: config.page_index,
        inpaint_path: visible_output_path.to_string_lossy().replace('\\', "/"),
        before_inpaint_path: before_inpaint_path
            .map(|path| path.to_string_lossy().replace('\\', "/")),
        bbox,
    })
}

#[tauri::command]
pub async fn write_mask_from_png(config: WriteMaskFromPngConfig) -> Result<String, String> {
    let project_file = resolve_project_file(&config.project_path);
    project_schema::edit_project_value(&project_file, |project| {
        let project_dir = project_file
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        let layer_key = config.layer_key.as_str();

        // Obtém (ou cria) o path relativo da camada
        let relative_layer_path =
            project_schema::ensure_bitmap_layer_path(project, config.page_index, layer_key)?;
        let absolute_layer_path = project_dir.join(&relative_layer_path);
        if let Some(parent) = absolute_layer_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("Erro ao criar diretório: {e}"))?;
        }

        // Decodificar base64
        let b64 = config
            .png_data
            .trim_start_matches("data:image/png;base64,")
            .trim_start_matches("data:image/webp;base64,");
        let png_bytes = BASE64
            .decode(b64)
            .map_err(|e| format!("Base64 inválido: {e}"))?;

        // Resolver op: campo "op" tem prioridade; senão interpreta "compose"
        let effective_op = if !config.op.is_empty() {
            config.op.as_str()
        } else if config.compose {
            "add"
        } else {
            "replace"
        };

        if (effective_op == "add" || effective_op == "subtract") && absolute_layer_path.exists() {
            let existing = image::load_from_memory(
                &std::fs::read(&absolute_layer_path).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?
            .into_luma8();

            let incoming = image::load_from_memory(&png_bytes)
                .map_err(|e| e.to_string())?
                .into_luma8();

            let (w, h) = (existing.width(), existing.height());
            let mut composed = existing.clone();
            for y in 0..h.min(incoming.height()) {
                for x in 0..w.min(incoming.width()) {
                    let ev = existing.get_pixel(x, y)[0];
                    let iv = incoming.get_pixel(x, y)[0];
                    let result = if effective_op == "add" {
                        ev.max(iv) // union
                    } else {
                        // subtract: apaga pixels onde incoming é branco
                        if iv > 127 {
                            0u8
                        } else {
                            ev
                        }
                    };
                    composed.put_pixel(x, y, Luma([result]));
                }
            }
            composed
                .save(&absolute_layer_path)
                .map_err(|e| format!("Erro ao salvar máscara composta: {e}"))?;
        } else {
            // replace (ou arquivo não existe): escrita direta
            std::fs::write(&absolute_layer_path, &png_bytes)
                .map_err(|e| format!("Erro ao escrever PNG: {e}"))?;
        }

        // Atualizar visibilidade no project.json
        project_schema::set_layer_visibility(
            project,
            config.page_index,
            "image",
            Some(layer_key),
            None,
            true,
        )?;

        Ok(absolute_layer_path.to_string_lossy().replace('\\', "/"))
    })
}

#[tauri::command]
pub async fn snapshot_image_layer(
    config: SnapshotImageLayerConfig,
) -> Result<Option<String>, String> {
    let project_file = resolve_project_file(&config.project_path);
    let project = project_schema::load_project_value(&project_file)?;
    let pages = project
        .get("paginas")
        .and_then(Value::as_array)
        .ok_or_else(|| "Projeto sem paginas validas".to_string())?;
    let page = pages
        .get(config.page_index)
        .ok_or_else(|| "Pagina do snapshot invalida".to_string())?;
    let project_dir = project_file
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let source = config
        .source_path
        .as_deref()
        .filter(|path| !path.trim().is_empty())
        .map(str::to_owned)
        .or_else(|| page_image_layer_path(page, &config.layer_key));
    let Some(source) = source else {
        return Ok(None);
    };
    let source_path = resolve_project_relative_path(&project_dir, &source);
    if !source_path.exists() {
        return Ok(None);
    }
    let extension = source_path
        .extension()
        .and_then(|extension| extension.to_str())
        .filter(|extension| !extension.trim().is_empty())
        .unwrap_or("png");
    let snapshot_dir = project_dir
        .join("editor_cache")
        .join("history")
        .join(format!("page-{:04}", config.page_index + 1));
    std::fs::create_dir_all(&snapshot_dir)
        .map_err(|e| format!("Erro ao criar cache de historico do editor: {e}"))?;
    let destination = snapshot_dir.join(format!(
        "{}-{}.{}",
        config.layer_key,
        uuid::Uuid::new_v4(),
        extension
    ));
    std::fs::copy(&source_path, &destination)
        .map_err(|e| format!("Erro ao copiar snapshot da camada: {e}"))?;
    Ok(Some(destination.to_string_lossy().replace('\\', "/")))
}

#[derive(Debug, Deserialize)]
pub struct ExportConfig {
    pub project_path: String,
    pub format: String, // "zip_full", "jpg_only", "cbz"
    pub output_path: String,
    #[serde(default)]
    pub export_mode: Option<String>,
}

struct ExportQualityBundle {
    status: String,
    files: Vec<(String, Vec<u8>)>,
}

#[tauri::command]
pub async fn export_project(config: ExportConfig) -> Result<Value, String> {
    let project_base = normalize_path(&config.project_path);
    let project_file = project_schema::resolve_project_file(&project_base);
    let project_dir = project_file
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or(project_base);
    let output_path = normalize_path(&config.output_path);

    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    let translated_dir = project_dir.join("translated");
    let originals_dir = project_dir.join("originals");
    let legacy_images_dir = project_dir.join("images");
    let layers_dir = project_dir.join("layers");
    let project_json = project_dir.join("project.json");
    let project_json_data = if project_json.exists() {
        Some(std::fs::read(&project_json).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let project_value = project_json_data
        .as_ref()
        .and_then(|data| serde_json::from_slice::<Value>(data).ok());
    let quality_bundle = if matches!(config.format.as_str(), "zip_full") {
        project_value
            .as_ref()
            .map(|project| {
                build_export_quality_bundle(&project_dir, project, config.export_mode.as_deref())
            })
            .transpose()?
    } else {
        None
    };

    let file =
        std::fs::File::create(&output_path).map_err(|e| format!("Erro ao criar arquivo: {}", e))?;
    let mut zip_writer = zip::ZipWriter::new(file);

    match config.format.as_str() {
        "jpg_only" | "cbz" => {
            add_directory_to_zip(&mut zip_writer, &translated_dir, "", options)?;
        }
        _ => {
            let mut manifest_files = Vec::new();
            add_directory_to_zip_tracked(
                &mut zip_writer,
                &translated_dir,
                "translated/",
                options,
                Some(&mut manifest_files),
            )?;

            if originals_dir.exists() {
                add_directory_to_zip_tracked(
                    &mut zip_writer,
                    &originals_dir,
                    "originals/",
                    options,
                    Some(&mut manifest_files),
                )?;
            }
            if legacy_images_dir.exists() {
                add_directory_to_zip_tracked(
                    &mut zip_writer,
                    &legacy_images_dir,
                    "images/",
                    options,
                    Some(&mut manifest_files),
                )?;
            }
            if layers_dir.exists() {
                add_directory_to_zip_tracked(
                    &mut zip_writer,
                    &layers_dir,
                    "layers/",
                    options,
                    Some(&mut manifest_files),
                )?;
            }

            if let Some(data) = project_json_data.as_ref() {
                add_file_to_zip_tracked(
                    &mut zip_writer,
                    "project.json",
                    data,
                    options,
                    Some(&mut manifest_files),
                )?;
            }

            if let Some(bundle) = quality_bundle {
                for (path, data) in bundle.files {
                    add_file_to_zip_tracked(
                        &mut zip_writer,
                        &path,
                        &data,
                        options,
                        Some(&mut manifest_files),
                    )?;
                }
                let manifest = json!({
                    "run_id": uuid::Uuid::new_v4().to_string(),
                    "created_at": current_iso_timestamp(),
                    "status": bundle.status,
                    "files": manifest_files,
                });
                let manifest_data =
                    serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
                add_file_to_zip_tracked(
                    &mut zip_writer,
                    "export_manifest.json",
                    &manifest_data,
                    options,
                    None,
                )?;
            }
        }
    }

    zip_writer.finish().map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "path": output_path.to_string_lossy()
    }))
}

#[derive(Debug, Deserialize)]
pub struct ExportPagePsdConfig {
    pub project_path: String,
    pub page_index: usize,
    pub output_path: String,
}

#[tauri::command]
pub async fn export_page_psd(config: ExportPagePsdConfig) -> Result<String, String> {
    let project_base = normalize_path(&config.project_path);
    let project_file = project_schema::resolve_project_file(&project_base);
    let project_dir = project_file
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "Caminho do projeto inválido".to_string())?;

    let project = project_schema::load_project_value(&project_file)?;
    let pages = project
        .get("paginas")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "Projeto sem páginas".to_string())?;
    let page = pages
        .get(config.page_index)
        .ok_or_else(|| "Página não encontrada".to_string())?;

    let original_rel = page
        .get("arquivo_original")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Caminho original não encontrado na página".to_string())?;

    // Tenta arquivo_final, se não houver tenta arquivo_traduzido (que costuma ser o final renderizado)
    let inpaint_rel = resolve_inpaint_rel(page);

    let original_path = project_dir.join(original_rel);
    let original_img = ImageReader::open(&original_path)
        .map_err(|e| format!("Falha ao abrir imagem original: {e}"))?
        .decode()
        .map_err(|e| format!("Falha ao decodificar imagem original: {e}"))?
        .to_rgba8();

    let width = original_img.width();
    let height = original_img.height();

    let mut psd_layers = Vec::new();

    // 3. Text Layers (Prep)
    if let Some(text_layers) = page.get("text_layers").and_then(|v| v.as_array()) {
        for (i, layer_val) in text_layers.iter().enumerate() {
            let texto = layer_val
                .get("translated")
                .and_then(|v| v.as_str())
                .or_else(|| layer_val.get("traduzido").and_then(|v| v.as_str()))
                .or_else(|| layer_val.get("texto").and_then(|v| v.as_str()))
                .unwrap_or("");

            let bbox = resolve_text_layer_bbox(layer_val);

            if texto.trim().is_empty() || bbox.is_none() {
                continue;
            }
            let [x1, y1, x2, y2] = bbox.unwrap();
            let w = (x2 - x1).max(1);
            let h = (y2 - y1).max(1);

            let x = x1;
            let y = y1;

            let style = layer_val.get("style").or_else(|| layer_val.get("estilo"));
            let font_size = style
                .and_then(|s| s.get("tamanho").and_then(|v| v.as_f64()))
                .unwrap_or(28.0);
            let color_hex = style
                .and_then(|s| s.get("cor").and_then(|v| v.as_str()))
                .unwrap_or("#FFFFFF");
            let negrito = style
                .and_then(|s| s.get("bold").and_then(|v| v.as_bool()))
                .unwrap_or(false);
            let italico = style
                .and_then(|s| s.get("italico").and_then(|v| v.as_bool()))
                .unwrap_or(false);
            let alinhamento = style
                .and_then(|s| s.get("alinhamento").and_then(|v| v.as_str()))
                .unwrap_or("center");

            let vertical = layer_val
                .get("vertical")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let color = parse_hex_color(color_hex);
            let orientation = if vertical {
                TextOrientation::Vertical
            } else {
                TextOrientation::Horizontal
            };
            let justification = match alinhamento {
                "left" => TextJustification::Left,
                "right" => TextJustification::Right,
                _ => TextJustification::Center,
            };

            let text_pixels = RgbaImage::new(w.max(1) as u32, h.max(1) as u32);

            psd_layers.push(PsdLayer {
                name: format!("Texto {}", i + 1),
                x,
                y,
                pixels: text_pixels,
                hidden: false,
                text_spec: Some(TextEngineSpec {
                    text: texto.to_string(),
                    font_name: "ArialMT".to_string(),
                    font_size,
                    color,
                    faux_bold: negrito,
                    faux_italic: italico,
                    orientation,
                    justification,
                    box_width: w as f64,
                    box_height: h as f64,
                }),
            });
        }
    }

    // 4. Assemble final layers Bottom to Top
    let mut final_psd_layers = Vec::new();

    // Bottom: Original
    final_psd_layers.push(PsdLayer {
        name: "Original".to_string(),
        x: 0,
        y: 0,
        pixels: original_img,
        hidden: false,
        text_spec: None,
    });

    // Layer: Inpaint
    if let Some(rel) = inpaint_rel {
        let path = project_dir.join(rel);
        if path.exists() {
            if let Ok(img) = open(&path) {
                final_psd_layers.push(PsdLayer {
                    name: "Limpeza (Inpaint)".to_string(),
                    x: 0,
                    y: 0,
                    pixels: img.to_rgba8(),
                    hidden: false,
                    text_spec: None,
                });
            }
        }
    }

    // Utility Layers (Hidden)
    if let Some(image_layers) = page.get("image_layers").and_then(|v| v.as_object()) {
        for key in &["mask", "brush"] {
            if let Some(layer_val) = image_layers.get(*key) {
                if let Some(rel_path) = layer_val.get("path").and_then(|v| v.as_str()) {
                    let path = project_dir.join(rel_path);
                    if path.exists() {
                        if let Ok(img) = open(&path) {
                            final_psd_layers.push(PsdLayer {
                                name: if *key == "mask" {
                                    "Máscara de Detecção".to_string()
                                } else {
                                    "Pincel de Edição".to_string()
                                },
                                x: 0,
                                y: 0,
                                pixels: img.to_rgba8(),
                                hidden: true,
                                text_spec: None,
                            });
                        }
                    }
                }
            }
        }
    }

    // Top: Texts
    for layer in psd_layers {
        final_psd_layers.push(layer);
    }

    let psd_bytes = export_psd(width, height, &final_psd_layers)?;

    let output_path = normalize_path(&config.output_path);
    std::fs::write(&output_path, psd_bytes)
        .map_err(|e| format!("Falha ao gravar arquivo PSD: {e}"))?;

    Ok(output_path.to_string_lossy().to_string())
}

fn resolve_inpaint_rel(page: &Value) -> Option<&str> {
    page.get("image_layers")
        .and_then(|v| v.as_object())
        .and_then(|layers| layers.get("inpaint"))
        .and_then(|v| v.get("path"))
        .and_then(|v| v.as_str())
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            page.get("arquivo_final")
                .and_then(|v| v.as_str())
                .filter(|value| !value.trim().is_empty())
        })
}

fn parse_layer_bbox_array(value: Option<&Value>) -> Option<[i32; 4]> {
    let bbox = value?.as_array()?;
    if bbox.len() != 4 {
        return None;
    }

    let mut coords = [0_i32; 4];
    for (index, coord) in bbox.iter().enumerate() {
        let parsed = coord
            .as_i64()
            .or_else(|| coord.as_u64().and_then(|raw| i64::try_from(raw).ok()))?;
        let parsed = i32::try_from(parsed).ok()?;
        coords[index] = parsed;
    }

    if coords[2] <= coords[0] || coords[3] <= coords[1] {
        return None;
    }

    Some(coords)
}

fn resolve_text_layer_bbox(layer_val: &Value) -> Option<[i32; 4]> {
    parse_layer_bbox_array(layer_val.get("render_bbox"))
        .or_else(|| parse_layer_bbox_array(layer_val.get("layout_bbox")))
        .or_else(|| parse_layer_bbox_array(layer_val.get("bbox")))
        .or_else(|| parse_layer_bbox_array(layer_val.get("balloon_bbox")))
}

fn bboxes_intersect(a: [u32; 4], b: [i32; 4]) -> bool {
    let b = [
        b[0].max(0) as u32,
        b[1].max(0) as u32,
        b[2].max(0) as u32,
        b[3].max(0) as u32,
    ];
    a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]
}

fn process_text_layer_ids_in_bbox(page: &Value, bbox: [u32; 4]) -> Vec<String> {
    page.get("text_layers")
        .and_then(Value::as_array)
        .or_else(|| page.get("textos").and_then(Value::as_array))
        .map(|layers| {
            layers
                .iter()
                .filter(|layer| {
                    resolve_text_layer_bbox(layer)
                        .map(|layer_bbox| bboxes_intersect(bbox, layer_bbox))
                        .unwrap_or(false)
                })
                .filter_map(|layer| layer.get("id").and_then(Value::as_str).map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

fn append_process_overlay_to_page(
    page: &mut Value,
    page_index: usize,
    bbox: [u32; 4],
    crop_path: String,
) -> Result<ProcessRegionOverlay, String> {
    let text_layer_ids = process_text_layer_ids_in_bbox(page, bbox);
    let overlays_value = page
        .as_object_mut()
        .ok_or_else(|| "Pagina invalida para overlay de processo".to_string())?
        .entry("process_overlays".to_string())
        .or_insert_with(|| json!([]));
    let overlays = overlays_value
        .as_array_mut()
        .ok_or_else(|| "process_overlays invalido".to_string())?;
    let order = overlays.len();
    let overlay = ProcessRegionOverlay {
        id: uuid::Uuid::new_v4().to_string(),
        page_index,
        bbox,
        crop_path,
        text_layer_ids,
        visible: true,
        locked: false,
        order,
    };
    overlays.push(
        serde_json::to_value(&overlay)
            .map_err(|e| format!("Erro ao serializar overlay de processo: {e}"))?,
    );
    Ok(overlay)
}

fn page_number_for_cache(page: &Value, page_index: usize) -> u64 {
    page.get("numero")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .unwrap_or((page_index + 1) as u64)
}

fn crop_process_region_to_cache(
    project_dir: &Path,
    page: &Value,
    page_index: usize,
    bbox: [u32; 4],
) -> Result<String, String> {
    let source_rel = resolve_inpaint_rel(page)
        .or_else(|| page.get("arquivo_original").and_then(Value::as_str))
        .ok_or_else(|| "Pagina sem imagem limpa para criar crop de processo".to_string())?;
    let source_path = resolve_project_relative_path(project_dir, source_rel);
    let image = image::open(&source_path)
        .map_err(|e| format!("Falha ao abrir imagem limpa do processo: {e}"))?;
    let width = image.width();
    let height = image.height();
    let x1 = bbox[0].min(width);
    let y1 = bbox[1].min(height);
    let x2 = bbox[2].min(width);
    let y2 = bbox[3].min(height);
    if x2 <= x1 || y2 <= y1 {
        return Err("Area do processo fora da pagina".to_string());
    }

    let page_number = page_number_for_cache(page, page_index);
    let output_dir = project_dir
        .join("editor_cache")
        .join("process_regions")
        .join(format!("page-{page_number:04}"));
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("Erro ao criar cache de processo: {e}"))?;
    let output_path = output_dir.join(format!("{}.png", uuid::Uuid::new_v4()));
    let cropped = image.crop_imm(x1, y1, x2 - x1, y2 - y1);
    cropped
        .save_with_format(&output_path, ImageFormat::Png)
        .map_err(|e| format!("Erro ao salvar crop de processo: {e}"))?;
    Ok(project_relative_string(project_dir, &output_path))
}

fn process_bbox_from_config(bbox: [i64; 4]) -> Result<[u32; 4], String> {
    if bbox.iter().any(|value| *value < 0) {
        return Err("Area do processo nao pode ter coordenadas negativas".to_string());
    }
    if bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
        return Err("Area do processo invalida".to_string());
    }
    Ok([
        u32::try_from(bbox[0]).map_err(|_| "Area do processo excede u32".to_string())?,
        u32::try_from(bbox[1]).map_err(|_| "Area do processo excede u32".to_string())?,
        u32::try_from(bbox[2]).map_err(|_| "Area do processo excede u32".to_string())?,
        u32::try_from(bbox[3]).map_err(|_| "Area do processo excede u32".to_string())?,
    ])
}

fn parse_hex_color(hex: &str) -> [u8; 4] {
    let hex = hex.trim_start_matches('#');
    if hex.len() >= 6 {
        let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0);
        let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
        let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0);
        [r, g, b, 255]
    } else {
        [0, 0, 0, 255]
    }
}

fn current_iso_timestamp() -> String {
    Utc::now().to_rfc3339()
}

fn sha256_hex(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn export_flag_label(flag: &str) -> &'static str {
    match flag {
        "critical_error" => "erros criticos",
        "glossary_violation" | "forbidden_term" | "missing_protected_term" => "glossario violado",
        "visual_text_leak" | "page_not_processed" | "untranslated_english" => "ingles restante",
        "ocr_gibberish" | "ocr_suspect" | "suspected_ocr_error" => "ocr suspeito",
        "typesetting_overflow" | "text_too_large" => "texto grande demais",
        "inpaint_suspicious" | "inpaint_artifact" => "inpaint suspeito",
        "entity_mistranslated" => "nome proprio alterado",
        "scanlation_credit" => "creditos/scanlation",
        "invalid_mask" | "missing_mask" => "mascara ausente",
        _ => "warning",
    }
}

fn export_flag_severity(flag: &str) -> &'static str {
    match flag {
        "critical_error" | "visual_text_leak" | "page_not_processed" | "untranslated_english" => {
            "critical"
        }
        "glossary_violation"
        | "forbidden_term"
        | "missing_protected_term"
        | "entity_mistranslated"
        | "invalid_mask"
        | "missing_mask" => "high",
        "ocr_gibberish"
        | "ocr_suspect"
        | "suspected_ocr_error"
        | "typesetting_overflow"
        | "text_too_large"
        | "inpaint_suspicious"
        | "inpaint_artifact" => "medium",
        _ => "low",
    }
}

fn ignored_export_flag(layer: &Value, flag: &str) -> bool {
    layer
        .get("qa_actions")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
        .iter()
        .any(|action| {
            action.get("flag_id").and_then(Value::as_str) == Some(flag)
                && action.get("status").and_then(Value::as_str) == Some("ignored")
        })
}

fn normalized_export_text(text: &str) -> String {
    text.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_uppercase())
        .collect()
}

fn ascii_words_upper(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphabetic() {
            current.push(ch.to_ascii_uppercase());
        } else if !current.is_empty() {
            words.push(current.clone());
            current.clear();
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

fn is_scanlation_credit_text(text: &str) -> bool {
    let upper = text.to_ascii_uppercase();
    let compact: String = upper
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect();
    compact.contains("ASURA")
        || compact.contains("ASURASOANS")
        || compact.contains("ILEAFSKY")
        || upper.contains("FASTEST RELEASES")
        || upper.contains(". COM")
        || upper.ends_with(".COM")
        || upper.contains("DISCORD")
}

fn looks_like_untranslated_english(source: &str, translated: &str) -> bool {
    if is_scanlation_credit_text(source) || is_scanlation_credit_text(translated) {
        return false;
    }
    let source_norm = normalized_export_text(source);
    if source_norm.is_empty() || source_norm != normalized_export_text(translated) {
        return false;
    }
    const ENGLISH_LEAK_WORDS: &[&str] = &[
        "ALREADY",
        "COMMANDER",
        "UNCONTROLLABLE",
        "YOUNG",
        "MASTER",
        "WHAT",
        "WHY",
        "WHEN",
        "WHERE",
        "WHO",
        "THE",
        "YOU",
        "YOUR",
        "ARE",
        "THIS",
        "THAT",
        "DO",
        "NOT",
        "PANIC",
        "DEFENSE",
        "FORMATION",
        "KINGDOM",
        "ESTATE",
    ];
    ascii_words_upper(source)
        .iter()
        .any(|word| ENGLISH_LEAK_WORDS.contains(&word.as_str()))
}

fn looks_like_ocr_suspect(source: &str) -> bool {
    const OCR_SUSPECT_TOKENS: &[&str] = &[
        "CARBAGE",
        "TRAE",
        "SOUAD",
        "OEFENSE",
        "OOWN",
        "KINGDOME",
        "REJDICE",
        "HOUSEHOID",
        "TEDQNG",
        "AGOE",
        "DRCS",
        "RDC",
    ];
    let words = ascii_words_upper(source);
    words
        .iter()
        .any(|word| OCR_SUSPECT_TOKENS.contains(&word.as_str()))
        || source.to_ascii_uppercase().contains("TS TO EARLY")
}

fn is_common_source_word(word: &str) -> bool {
    const COMMON_WORDS: &[&str] = &[
        "ALREADY",
        "ATTACK",
        "COMMANDER",
        "CONTINENT",
        "CONTINENTS",
        "CRUSHED",
        "DAY",
        "DEFENSE",
        "DIE",
        "DUST",
        "ESTATE",
        "EVERYTHING",
        "FORMATION",
        "GARBAGE",
        "HOUSEHOLD",
        "INTO",
        "KINGDOM",
        "KNIGHT",
        "LOST",
        "MASTER",
        "MOST",
        "NOBLE",
        "PANIC",
        "PEOPLE",
        "RAID",
        "RELEASES",
        "RETURNED",
        "SOLDIER",
        "SQUAD",
        "THAT",
        "THE",
        "THIS",
        "UNCONTROLLABLE",
        "WHAT",
        "WHEN",
        "WHERE",
        "WHO",
        "WHY",
        "WITH",
        "YOUNG",
        "YOUR",
    ];
    COMMON_WORDS.contains(&word)
}

fn protected_source_tokens(source: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for word in ascii_words_upper(source) {
        if word.len() < 4 || word.len() > 24 || is_common_source_word(&word) {
            continue;
        }
        if !tokens.contains(&word) {
            tokens.push(word);
        }
    }
    tokens
}

fn looks_like_entity_mistranslated(source: &str, translated: &str) -> bool {
    if looks_like_ocr_suspect(source) {
        return false;
    }
    let tokens = protected_source_tokens(source);
    if tokens.len() < 2 && !tokens.iter().any(|token| token.ends_with("IUM")) {
        return false;
    }
    let translated_upper = normalized_export_text(translated);
    tokens
        .iter()
        .any(|token| !translated_upper.contains(&normalized_export_text(token)))
}

fn layer_background_luma(layer: &Value) -> Option<f64> {
    let values = layer.get("background_rgb")?.as_array()?;
    if values.len() != 3 {
        return None;
    }
    let mut sum = 0.0;
    for value in values {
        sum += value.as_f64()?;
    }
    Some(sum / 3.0)
}

fn looks_like_inpaint_artifact_risk(layer: &Value) -> bool {
    let balloon_type = layer
        .get("balloon_type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    if balloon_type != "textured" {
        return false;
    }
    let bbox = parse_layer_bbox_array(layer.get("text_pixel_bbox"))
        .or_else(|| parse_layer_bbox_array(layer.get("bbox")));
    let Some([x1, y1, x2, y2]) = bbox else {
        return false;
    };
    let area = i64::from(x2 - x1) * i64::from(y2 - y1);
    if area < 80_000 {
        return false;
    }
    if let Some(luma) = layer_background_luma(layer) {
        if !(24.0..=220.0).contains(&luma) {
            return false;
        }
    }
    let line_count = layer
        .get("line_polygons")
        .and_then(Value::as_array)
        .map(|items| items.len())
        .unwrap_or(0);
    line_count >= 3 || (y2 - y1) >= 220
}

fn infer_export_flags(layer: &Value) -> Vec<&'static str> {
    let source = layer.get("original").and_then(Value::as_str).unwrap_or("");
    let translated = layer
        .get("traduzido")
        .and_then(Value::as_str)
        .or_else(|| layer.get("translated").and_then(Value::as_str))
        .unwrap_or("");
    let mut flags = Vec::new();
    if is_scanlation_credit_text(source) {
        flags.push("scanlation_credit");
    }
    if looks_like_untranslated_english(source, translated) {
        flags.push("untranslated_english");
    }
    if looks_like_ocr_suspect(source) {
        flags.push("ocr_suspect");
    }
    if looks_like_entity_mistranslated(source, translated) {
        flags.push("entity_mistranslated");
    }
    if looks_like_inpaint_artifact_risk(layer) {
        flags.push("inpaint_artifact");
    }
    flags
}

fn collect_export_issues(project: &Value) -> Vec<Value> {
    let mut issues = Vec::new();
    for (page_index, page) in project
        .get("paginas")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
        .iter()
        .enumerate()
    {
        let page_number = page
            .get("numero")
            .and_then(Value::as_u64)
            .unwrap_or((page_index + 1) as u64);
        for layer in page
            .get("text_layers")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
        {
            let region_id = layer.get("id").and_then(Value::as_str).unwrap_or("region");
            let mut flags: Vec<String> = layer
                .get("qa_flags")
                .and_then(Value::as_array)
                .unwrap_or(&Vec::new())
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect();
            for inferred in infer_export_flags(layer) {
                if !flags.iter().any(|flag| flag == inferred) {
                    flags.push(inferred.to_string());
                }
            }
            for flag in flags.iter().map(String::as_str) {
                if ignored_export_flag(layer, flag) {
                    continue;
                }
                issues.push(json!({
                    "id": format!("{page_index}:{region_id}:{flag}"),
                    "page": page_number,
                    "region_id": region_id,
                    "type": flag,
                    "label": export_flag_label(flag),
                    "severity": export_flag_severity(flag),
                    "source_text": layer.get("original").and_then(Value::as_str).unwrap_or(""),
                    "translated_text": layer
                        .get("traduzido")
                        .and_then(Value::as_str)
                        .or_else(|| layer.get("translated").and_then(Value::as_str))
                        .unwrap_or(""),
                }));
            }
        }
    }
    issues
}

fn collect_user_actions(project: &Value) -> Vec<Value> {
    let mut actions = Vec::new();
    for page in project
        .get("paginas")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        for layer in page
            .get("text_layers")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
        {
            for action in layer
                .get("qa_actions")
                .and_then(Value::as_array)
                .unwrap_or(&Vec::new())
            {
                actions.push(action.clone());
            }
        }
    }
    actions
}

fn collect_glossary_hits(project: &Value) -> Vec<Value> {
    let mut hits = Vec::new();
    for page in project
        .get("paginas")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        for layer in page
            .get("text_layers")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
        {
            if let Some(layer_hits) = layer.get("glossary_hits").and_then(Value::as_array) {
                hits.extend(layer_hits.iter().cloned());
            }
        }
    }
    hits
}

fn collect_ocr_corrections(project: &Value) -> Vec<Value> {
    let mut corrections = Vec::new();
    for page in project
        .get("paginas")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        for layer in page
            .get("text_layers")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
        {
            if layer.get("normalization").is_some()
                || layer.get("raw_ocr").is_some()
                || layer.get("normalized_ocr").is_some()
            {
                corrections.push(json!({
                    "region_id": layer.get("id").and_then(Value::as_str).unwrap_or(""),
                    "raw_text": layer.get("raw_ocr").and_then(Value::as_str).unwrap_or(""),
                    "normalized_text": layer
                        .get("normalized_ocr")
                        .and_then(Value::as_str)
                        .or_else(|| layer.get("original").and_then(Value::as_str))
                        .unwrap_or(""),
                    "normalization": layer.get("normalization").cloned().unwrap_or(Value::Null),
                }));
            }
        }
    }
    corrections
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn build_issues_csv(issues: &[Value]) -> String {
    let mut rows = vec!["id,page,region_id,type,severity,label".to_string()];
    for issue in issues {
        rows.push(format!(
            "{},{},{},{},{},{}",
            csv_escape(issue.get("id").and_then(Value::as_str).unwrap_or("")),
            issue.get("page").and_then(Value::as_u64).unwrap_or(0),
            csv_escape(issue.get("region_id").and_then(Value::as_str).unwrap_or("")),
            csv_escape(issue.get("type").and_then(Value::as_str).unwrap_or("")),
            csv_escape(issue.get("severity").and_then(Value::as_str).unwrap_or("")),
            csv_escape(issue.get("label").and_then(Value::as_str).unwrap_or(""))
        ));
    }
    rows.join("\n")
}

fn build_qa_markdown_report(
    project: &Value,
    status: &str,
    issues: &[Value],
    user_actions: &[Value],
) -> String {
    let critical = issues
        .iter()
        .filter(|issue| issue.get("severity").and_then(Value::as_str) == Some("critical"))
        .count();
    let warnings = issues.len().saturating_sub(critical);
    let title = project
        .get("obra")
        .and_then(Value::as_str)
        .unwrap_or("Projeto");

    let mut lines = vec![
        format!("# Relatorio QA - {title}"),
        String::new(),
        "## resumo".to_string(),
        format!("- status: {status}"),
        format!("- erros criticos: {critical}"),
        format!("- warnings: {warnings}"),
        String::new(),
        "## erros criticos".to_string(),
    ];
    for issue in issues
        .iter()
        .filter(|issue| issue.get("severity").and_then(Value::as_str) == Some("critical"))
    {
        lines.push(format!(
            "- pagina {} / {}: {}",
            issue.get("page").and_then(Value::as_u64).unwrap_or(0),
            issue.get("region_id").and_then(Value::as_str).unwrap_or(""),
            issue.get("label").and_then(Value::as_str).unwrap_or("")
        ));
    }
    lines.extend([
        String::new(),
        "## warnings".to_string(),
        "## glossario usado".to_string(),
        "## violacoes".to_string(),
        "## correcoes OCR".to_string(),
        "## mascaras invalidas".to_string(),
        "## inpaint suspeito".to_string(),
        "## paginas bloqueadas".to_string(),
        "## acoes do usuario".to_string(),
    ]);
    for action in user_actions {
        lines.push(format!(
            "- {}: {}",
            action
                .get("flag_id")
                .and_then(Value::as_str)
                .unwrap_or("flag"),
            action
                .get("ignored_reason")
                .and_then(Value::as_str)
                .unwrap_or("sem motivo")
        ));
    }
    lines.push(String::new());
    lines.join("\n")
}

fn build_export_quality_bundle(
    project_dir: &Path,
    project: &Value,
    export_mode: Option<&str>,
) -> Result<ExportQualityBundle, String> {
    let mode = export_mode.unwrap_or("with_warnings");
    let issues = collect_export_issues(project);
    let critical_count = issues
        .iter()
        .filter(|issue| issue.get("severity").and_then(Value::as_str) == Some("critical"))
        .count();
    let high_count = issues
        .iter()
        .filter(|issue| issue.get("severity").and_then(Value::as_str) == Some("high"))
        .count();

    if mode == "clean" && critical_count + high_count > 0 {
        return Err("Export clean bloqueado: ha flags critical/high ativas.".to_string());
    }
    if mode != "debug" && critical_count > 0 {
        return Err("Export bloqueado: ha flags critical ativas. Use modo debug.".to_string());
    }

    let status = if critical_count > 0 {
        "blocked_debug_export"
    } else if issues.is_empty() {
        "clean"
    } else {
        "with_warnings"
    }
    .to_string();

    let user_actions = collect_user_actions(project);
    let qa_report_json = json!({
        "summary": {
            "status": status,
            "total": issues.len(),
            "critical": critical_count,
            "high": high_count,
        },
        "issues": issues,
        "user_actions": user_actions,
    });
    let glossary_used = collect_glossary_hits(project);
    let ocr_corrections = collect_ocr_corrections(project);
    let issues_ref = qa_report_json
        .get("issues")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let actions_ref = qa_report_json
        .get("user_actions")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let markdown = build_qa_markdown_report(project, &status, &issues_ref, &actions_ref);
    let structured_log = std::fs::read(project_dir.join("structured_log.jsonl"))
        .unwrap_or_else(|_| b"{\"event\":\"export\",\"status\":\"generated\"}\n".to_vec());

    Ok(ExportQualityBundle {
        status,
        files: vec![
            ("qa_report.md".to_string(), markdown.into_bytes()),
            (
                "qa_report.json".to_string(),
                serde_json::to_vec_pretty(&qa_report_json).map_err(|e| e.to_string())?,
            ),
            (
                "issues.csv".to_string(),
                build_issues_csv(&issues_ref).into_bytes(),
            ),
            (
                "glossary_used.json".to_string(),
                serde_json::to_vec_pretty(&glossary_used).map_err(|e| e.to_string())?,
            ),
            (
                "ocr_corrections.json".to_string(),
                serde_json::to_vec_pretty(&ocr_corrections).map_err(|e| e.to_string())?,
            ),
            ("structured_log.jsonl".to_string(), structured_log),
        ],
    })
}

fn add_file_to_zip_tracked(
    zip_writer: &mut zip::ZipWriter<std::fs::File>,
    name: &str,
    data: &[u8],
    options: zip::write::SimpleFileOptions,
    mut manifest: Option<&mut Vec<Value>>,
) -> Result<(), String> {
    zip_writer
        .start_file(name, options)
        .map_err(|e| e.to_string())?;
    zip_writer.write_all(data).map_err(|e| e.to_string())?;
    if let Some(entries) = manifest.as_deref_mut() {
        entries.push(json!({
            "path": name,
            "sha256": sha256_hex(data),
        }));
    }
    Ok(())
}

fn add_directory_to_zip(
    zip_writer: &mut zip::ZipWriter<std::fs::File>,
    dir: &PathBuf,
    prefix: &str,
    options: zip::write::SimpleFileOptions,
) -> Result<(), String> {
    add_directory_to_zip_tracked(zip_writer, dir, prefix, options, None)
}

fn add_directory_to_zip_tracked(
    zip_writer: &mut zip::ZipWriter<std::fs::File>,
    dir: &PathBuf,
    prefix: &str,
    options: zip::write::SimpleFileOptions,
    mut manifest: Option<&mut Vec<Value>>,
) -> Result<(), String> {
    if !dir.exists() {
        return Ok(());
    }

    for entry in walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path().to_path_buf();
        let relative = path.strip_prefix(dir).map_err(|e| e.to_string())?;
        let name = format!("{prefix}{}", relative.to_string_lossy().replace('\\', "/"));
        let data = std::fs::read(&path).map_err(|e| e.to_string())?;
        add_file_to_zip_tracked(zip_writer, &name, &data, options, manifest.as_deref_mut())?;
    }

    Ok(())
}

// ─── Máscara regional — helpers + command ────────────────────────────────────

/// Pixels abaixo deste limiar são ignorados ao calcular se a máscara tem conteúdo.
const MASK_ACTIVE_THRESHOLD: u8 = 8;

/// Retorna `true` se pelo menos um pixel da imagem grayscale é >= MASK_ACTIVE_THRESHOLD.
fn mask_has_nonzero_pixels(path: &Path) -> Result<bool, String> {
    let img = image::open(path).map_err(|e| format!("mask_has_nonzero_pixels: {e}"))?;
    let gray = img.to_luma8();
    Ok(gray.pixels().any(|p| p[0] >= MASK_ACTIVE_THRESHOLD))
}

/// Retorna o bbox [x1, y1, x2, y2) **half-open** (x2/y2 exclusivos) da região ativa da máscara.
/// Retorna `None` se a máscara estiver vazia.
fn mask_bounding_box(path: &Path) -> Result<Option<[u32; 4]>, String> {
    let img = image::open(path).map_err(|e| format!("mask_bounding_box: {e}"))?;
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    let (mut x1, mut y1, mut x2, mut y2) = (w, h, 0u32, 0u32);
    for (x, y, p) in gray.enumerate_pixels() {
        if p[0] >= MASK_ACTIVE_THRESHOLD {
            x1 = x1.min(x);
            y1 = y1.min(y);
            x2 = x2.max(x + 1); // half-open: exclusivo
            y2 = y2.max(y + 1);
        }
    }
    if x1 >= x2 || y1 >= y2 {
        Ok(None)
    } else {
        Ok(Some([x1, y1, x2, y2]))
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PageActionMode {
    Global,
    Regional,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
// Brush/Preview ainda não populadas pelo dispatcher atual mas existem no contrato
#[allow(dead_code)]
pub enum ChangedAsset {
    Brush,
    Mask,
    Inpaint,
    Rendered,
    Preview,
    ProjectJson,
}

#[derive(Debug, Serialize)]
pub struct PageActionResult {
    pub action: String,
    pub mode: PageActionMode,
    pub bbox: Option<[u32; 4]>,
    pub changed_assets: Vec<ChangedAsset>,
    pub changed_layers: Vec<String>,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct PageActionConfig {
    pub project_path: String,
    pub page_index: usize,
    pub action: String,
    #[serde(default)]
    pub bbox: Option<[i64; 4]>,
    #[serde(default)]
    pub mask_path: Option<String>,
    #[serde(default)]
    pub engine_preset_id: Option<String>,
    #[serde(default)]
    pub idioma_origem: Option<String>,
    #[serde(default)]
    pub idioma_destino: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ProcessRegionConfig {
    pub project_path: String,
    pub page_index: usize,
    pub bbox: [i64; 4],
    #[serde(default)]
    pub mask_path: Option<String>,
    #[serde(default)]
    pub engine_preset_id: Option<String>,
    #[serde(default)]
    pub idioma_origem: Option<String>,
    #[serde(default)]
    pub idioma_destino: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProcessRegionOverlay {
    pub id: String,
    pub page_index: usize,
    pub bbox: [u32; 4],
    pub crop_path: String,
    pub text_layer_ids: Vec<String>,
    pub visible: bool,
    pub locked: bool,
    pub order: usize,
}

#[derive(Debug, Serialize)]
pub struct ProcessRegionResult {
    pub page_index: usize,
    pub overlay: ProcessRegionOverlay,
    pub changed_assets: Vec<ChangedAsset>,
    pub changed_layers: Vec<String>,
    pub message: String,
}

/// Detecta se a página corrente tem uma máscara com pixels ativos e executa a ação
/// no modo Regional (com bbox da máscara) ou Global (página inteira).
///
/// Fase 4A: delega para as ações globais existentes.
/// Fases 4B-4E: adicionar `--mask`/`--bbox` ao sidecar quando `mode == Regional`.
#[tauri::command]
pub async fn run_page_action_with_optional_mask(
    app: tauri::AppHandle,
    config: PageActionConfig,
) -> Result<PageActionResult, String> {
    use crate::commands::pipeline::{
        detect_boxes_page_with_region, detect_page_with_region, ocr_page_with_region,
        reinpaint_page_with_region, translate_page_with_region, PageRegionConfig,
    };

    // Carregar project.json pelo mesmo caminho normalizado dos demais commands do editor.
    let project_file = resolve_project_file(&config.project_path);
    let project = project_schema::load_project_value(&project_file)?;

    let mask_path_str = config.mask_path.clone().or_else(|| {
        if config.bbox.is_some() {
            return None;
        }
        project["paginas"]
            .get(config.page_index)
            .and_then(|p| p["image_layers"]["mask"]["path"].as_str())
            .map(str::to_owned)
    });
    let resolved_mask_path = mask_path_str.as_ref().map(|raw| {
        let path = Path::new(raw);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            project_file
                .parent()
                .unwrap_or_else(|| Path::new(""))
                .join(path)
        }
    });

    let (mode, bbox) = if let Some(bbox) = config.bbox {
        (
            PageActionMode::Regional,
            Some([
                bbox[0].max(0) as u32,
                bbox[1].max(0) as u32,
                bbox[2].max(0) as u32,
                bbox[3].max(0) as u32,
            ]),
        )
    } else if let Some(ref mp) = resolved_mask_path {
        if mp.exists() && mask_has_nonzero_pixels(mp)? {
            let b = mask_bounding_box(mp)?;
            (PageActionMode::Regional, b)
        } else {
            (PageActionMode::Global, None)
        }
    } else {
        (PageActionMode::Global, None)
    };

    let page_index_u32 = config.page_index as u32;
    let region = PageRegionConfig {
        bbox,
        mask_path: resolved_mask_path.map(|path| path.to_string_lossy().to_string()),
    };

    eprintln!(
        "[EditorAction] start  page_action action={} page={} mode={:?} bbox={:?}",
        config.action, config.page_index, mode, bbox
    );

    // Mapeia a ação para os assets que ela efetivamente modifica.
    // Sem isso o frontend não invalida o cache dos bitmaps recém gravados
    // pelo Python (bug histórico: changed_assets vinha sempre só ProjectJson).
    let changed_assets: Vec<ChangedAsset> = match config.action.as_str() {
        "detect" => {
            detect_page_with_region(
                app,
                config.project_path.clone(),
                page_index_u32,
                Some(region.clone()),
                config.engine_preset_id.clone(),
                config.idioma_origem.clone(),
            )
            .await?;
            // detect re-renderiza a página ao final (render_page_image em main.py)
            vec![ChangedAsset::ProjectJson, ChangedAsset::Rendered]
        }
        "detect_boxes" => {
            detect_boxes_page_with_region(
                app,
                config.project_path.clone(),
                page_index_u32,
                Some(region.clone()),
                config.engine_preset_id.clone(),
                config.idioma_origem.clone(),
            )
            .await?;
            vec![ChangedAsset::ProjectJson]
        }
        "ocr" => {
            ocr_page_with_region(
                app,
                config.project_path.clone(),
                page_index_u32,
                Some(region.clone()),
                config.engine_preset_id.clone(),
                config.idioma_origem.clone(),
            )
            .await?;
            // ocr_page também re-renderiza ao final
            vec![ChangedAsset::ProjectJson, ChangedAsset::Rendered]
        }
        "translate" => {
            translate_page_with_region(
                app,
                config.project_path.clone(),
                page_index_u32,
                Some(region.clone()),
                config.idioma_origem.clone(),
                config.idioma_destino.clone(),
            )
            .await?;
            // translate atualiza texto traduzido e re-renderiza
            vec![ChangedAsset::ProjectJson, ChangedAsset::Rendered]
        }
        "inpaint" => {
            reinpaint_page_with_region(
                app,
                config.project_path.clone(),
                page_index_u32,
                Some(region.clone()),
            )
            .await?;
            // inpaint regrava a layer inpaint e o render final
            vec![
                ChangedAsset::ProjectJson,
                ChangedAsset::Inpaint,
                ChangedAsset::Rendered,
            ]
        }
        other => {
            eprintln!("[EditorAction] error  page_action ação desconhecida: {other}");
            return Err(format!("Ação desconhecida: {other}"));
        }
    };

    eprintln!(
        "[EditorAction] success page_action action={} page={} changed={:?}",
        config.action, config.page_index, changed_assets
    );

    Ok(PageActionResult {
        action: config.action,
        mode,
        bbox,
        changed_assets,
        changed_layers: vec![],
        message: "Ação concluída".to_string(),
    })
}

#[tauri::command]
pub async fn run_paint_optional_mask(
    app: tauri::AppHandle,
    config: PageActionConfig,
) -> Result<PageActionResult, String> {
    run_page_action_with_optional_mask(app, config).await
}

#[tauri::command]
pub async fn run_process_region(
    app: tauri::AppHandle,
    config: ProcessRegionConfig,
) -> Result<ProcessRegionResult, String> {
    use crate::commands::pipeline::{
        detect_page_with_region, ocr_page_with_region, reinpaint_page_with_region,
        translate_page_with_region, PageRegionConfig,
    };

    let started = std::time::Instant::now();
    let project_file = resolve_project_file(&config.project_path);
    let project = project_schema::load_project_value(&project_file)?;
    let pages = project
        .get("paginas")
        .and_then(Value::as_array)
        .ok_or_else(|| "Projeto sem paginas validas".to_string())?;
    if config.page_index >= pages.len() {
        return Err("Pagina do processo invalida".to_string());
    }
    let project_dir = project_file
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "Caminho do projeto invalido".to_string())?;
    let bbox = process_bbox_from_config(config.bbox)?;
    let mask_path = config
        .mask_path
        .as_ref()
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|raw| {
            let path = Path::new(raw);
            if path.is_absolute() {
                path.to_path_buf()
            } else {
                project_dir.join(path)
            }
        });
    let region = PageRegionConfig {
        bbox: Some(bbox),
        mask_path: mask_path
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
    };
    let page_index_u32 = config.page_index as u32;
    let engine_preset_id = Some(
        config
            .engine_preset_id
            .clone()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "manga".to_string()),
    );

    eprintln!(
        "[EditorAction] start  process_region page={} bbox={:?} mask={:?}",
        config.page_index, bbox, region.mask_path
    );

    detect_page_with_region(
        app.clone(),
        config.project_path.clone(),
        page_index_u32,
        Some(region.clone()),
        engine_preset_id.clone(),
        config.idioma_origem.clone(),
    )
    .await?;
    ocr_page_with_region(
        app.clone(),
        config.project_path.clone(),
        page_index_u32,
        Some(region.clone()),
        engine_preset_id,
        config.idioma_origem.clone(),
    )
    .await?;
    translate_page_with_region(
        app.clone(),
        config.project_path.clone(),
        page_index_u32,
        Some(region.clone()),
        config.idioma_origem.clone(),
        config.idioma_destino.clone(),
    )
    .await?;
    reinpaint_page_with_region(
        app,
        config.project_path.clone(),
        page_index_u32,
        Some(region),
    )
    .await?;

    let overlay = project_schema::edit_project_value(&project_file, |project| {
        let page = project
            .get_mut("paginas")
            .and_then(Value::as_array_mut)
            .and_then(|pages| pages.get_mut(config.page_index))
            .ok_or_else(|| "Pagina do processo invalida".to_string())?;
        let crop_path = crop_process_region_to_cache(&project_dir, page, config.page_index, bbox)?;
        append_process_overlay_to_page(page, config.page_index, bbox, crop_path)
    })?;
    let changed_layers = overlay.text_layer_ids.clone();

    eprintln!(
        "[EditorAction] success process_region page={} elapsed={:.3}s overlay={} layers={:?}",
        config.page_index,
        started.elapsed().as_secs_f64(),
        overlay.id,
        changed_layers
    );

    Ok(ProcessRegionResult {
        page_index: config.page_index,
        overlay,
        changed_assets: vec![
            ChangedAsset::ProjectJson,
            ChangedAsset::Inpaint,
            ChangedAsset::Rendered,
        ],
        changed_layers,
        message: "Processo regional concluido".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    const WEBP_2X2_RED: &[u8] = &[
        82, 73, 70, 70, 60, 0, 0, 0, 87, 69, 66, 80, 86, 80, 56, 32, 48, 0, 0, 0, 208, 1, 0, 157,
        1, 42, 2, 0, 2, 0, 1, 64, 38, 37, 160, 2, 116, 186, 1, 248, 0, 3, 176, 0, 254, 242, 235,
        127, 252, 216, 21, 205, 115, 239, 247, 255, 210, 224, 253, 46, 15, 210, 224, 255, 210, 144,
        0, 0,
    ];

    fn unique_temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("traduzai-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_file(path: &PathBuf, bytes: &[u8]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn google_font_filename_accepts_only_simple_font_files() {
        assert_eq!(
            sanitize_google_font_filename("NotoSans-Regular.ttf").unwrap(),
            "NotoSans-Regular.ttf"
        );
        assert_eq!(
            sanitize_google_font_filename("ComicNeue_Bold.otf").unwrap(),
            "ComicNeue_Bold.otf"
        );

        assert!(sanitize_google_font_filename("NotoSans.woff2").is_err());
    }

    #[test]
    fn google_font_filename_rejects_path_traversal() {
        for filename in [
            "../NotoSans-Regular.ttf",
            "..\\NotoSans-Regular.ttf",
            "fonts/NotoSans-Regular.ttf",
            "C:\\fonts\\NotoSans-Regular.ttf",
        ] {
            assert!(
                sanitize_google_font_filename(filename).is_err(),
                "filename deveria ser rejeitado: {filename}"
            );
        }
    }

    #[tokio::test]
    async fn cache_google_font_uses_existing_non_empty_file_without_network() {
        let root = unique_temp_dir();
        let cache_dir = root.join("google-fonts");
        let filename = "NotoSans-Regular.ttf";
        let font_path = cache_dir.join(filename);
        write_file(&font_path, b"cached-font");

        let cached = cache_google_font_in_dir(
            CacheGoogleFontRequest {
                family: "Noto Sans".to_string(),
                css_family: "Noto+Sans".to_string(),
                variant: "regular".to_string(),
                url: "http://127.0.0.1:9/should-not-be-called.ttf".to_string(),
                filename: filename.to_string(),
            },
            &cache_dir,
        )
        .await
        .unwrap();

        assert_eq!(cached.family, "Noto Sans");
        assert_eq!(cached.css_family, "Noto+Sans");
        assert_eq!(cached.variant, "regular");
        assert_eq!(cached.filename, filename);
        assert_eq!(cached.path, font_path.to_string_lossy());
        assert_eq!(std::fs::read(&font_path).unwrap(), b"cached-font");

        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn google_fonts_metadata_search_filters_remote_families() {
        let metadata = r#"{
          "familyMetadataList": [
            { "family": "Roboto", "category": "Sans Serif", "popularity": 2 },
            { "family": "Bebas Neue", "category": "Display", "popularity": 1 },
            { "family": "Noto Sans", "category": "Sans Serif", "popularity": 3 }
          ]
        }"#;

        let results = search_google_fonts_metadata_json(metadata, "bebas", 5).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].family, "Bebas Neue");
        assert_eq!(results[0].category.as_deref(), Some("Display"));
    }

    #[test]
    fn google_font_cache_filename_is_stable_for_remote_results() {
        assert_eq!(
            google_font_cache_filename("Bebas Neue", "ttf"),
            "GoogleFont__Bebas_Neue__regular.ttf"
        );
    }

    #[test]
    fn google_font_directory_entry_prefers_regular_ttf() {
        let entries = vec![
            GoogleFontRepoEntry {
                name: "BebasNeue-Italic.ttf".to_string(),
                download_url: Some("https://example.test/italic.ttf".to_string()),
                entry_type: "file".to_string(),
            },
            GoogleFontRepoEntry {
                name: "BebasNeue-Regular.ttf".to_string(),
                download_url: Some("https://example.test/regular.ttf".to_string()),
                entry_type: "file".to_string(),
            },
        ];

        let selected = select_google_font_repo_file(&entries).unwrap();

        assert_eq!(selected.name, "BebasNeue-Regular.ttf");
        assert_eq!(
            selected.download_url.as_deref(),
            Some("https://example.test/regular.ttf")
        );
    }

    fn write_zip_entries(path: &PathBuf, entries: &[(&str, &[u8])]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let file = std::fs::File::create(path).unwrap();
        let mut zip_writer = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for (name, data) in entries {
            zip_writer.start_file(*name, options).unwrap();
            zip_writer.write_all(data).unwrap();
        }
        zip_writer.finish().unwrap();
    }

    fn zip_entries(zip_path: &PathBuf) -> Vec<String> {
        let file = std::fs::File::open(zip_path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let mut names = Vec::new();

        for i in 0..archive.len() {
            let entry = archive.by_index(i).unwrap();
            names.push(entry.name().to_string());
        }

        names
    }

    #[test]
    fn validate_directory_rejects_exported_traduzai_project_as_source() {
        let root = unique_temp_dir();
        let project_dir = root.join("project");
        write_file(&project_dir.join("project.json"), br#"{"obra":"Teste"}"#);
        write_file(&project_dir.join("originals").join("001.jpg"), b"original");
        write_file(
            &project_dir.join("translated").join("001.jpg"),
            b"translated",
        );

        let result = validate_directory(&project_dir).unwrap();

        assert!(!result.valid);
        assert_eq!(result.pages, 0);
        assert!(result.has_project_json);
        assert!(result.error.unwrap().contains("Abrir projeto"));

        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn validate_archive_rejects_exported_traduzai_project_as_source() {
        let root = unique_temp_dir();
        let archive_path = root.join("traduzido.zip");
        write_zip_entries(
            &archive_path,
            &[
                ("project.json", br#"{"obra":"Teste"}"#),
                ("originals/001.jpg", b"original"),
                ("translated/001.jpg", b"translated"),
            ],
        );

        let result = validate_archive(&archive_path).unwrap();

        assert!(!result.valid);
        assert_eq!(result.pages, 0);
        assert!(result.has_project_json);
        assert!(result.error.unwrap().contains("Abrir projeto"));

        std::fs::remove_dir_all(root).ok();
    }

    fn zip_entry_string(zip_path: &PathBuf, name: &str) -> String {
        let file = std::fs::File::open(zip_path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let mut contents = String::new();
        archive
            .by_name(name)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        contents
    }

    fn quality_project_json(flags: Vec<&str>) -> String {
        let layers: Vec<Value> = flags
            .iter()
            .map(|flag| {
                json!({
                    "id": format!("region-{flag}"),
                    "original": "YOUNG MASTER?!",
                    "traduzido": "Jovem mestre?!",
                    "qa_flags": [flag],
                    "qa_actions": []
                })
            })
            .collect();
        json!({
            "obra": "Teste",
            "capitulo": 1,
            "paginas": [{
                "numero": 1,
                "text_layers": layers,
                "textos": layers,
                "image_layers": {
                    "mask": {"path": "layers/masks/001.png"}
                }
            }],
            "qa": {"summary": {"total": flags.len()}},
            "log": {"summary": {"qa_flags": flags.len()}}
        })
        .to_string()
    }

    #[test]
    fn collect_export_issues_infers_untranslated_english_without_existing_flags() {
        let project = json!({
            "paginas": [{
                "numero": 48,
                "text_layers": [{
                    "id": "region-already",
                    "original": "ALREADY?",
                    "traduzido": "ALREADY?",
                    "qa_flags": []
                }]
            }]
        });

        let issues = collect_export_issues(&project);

        assert!(issues.iter().any(|issue| {
            issue["type"] == "untranslated_english"
                && issue["severity"] == "critical"
                && issue["page"] == 48
        }));
    }

    #[test]
    fn collect_export_issues_infers_ocr_and_entity_problems_without_existing_flags() {
        let project = json!({
            "paginas": [{
                "numero": 3,
                "text_layers": [
                    {
                        "id": "region-ocr",
                        "original": "DO NOT PANIC GET INTO OEFENSE FORMATION!",
                        "traduzido": "Nao entrem em panico!",
                        "qa_flags": []
                    },
                    {
                        "id": "region-name",
                        "original": "GHISLAIN PERDIUM.",
                        "traduzido": "PERDIO GHISLAIN.",
                        "qa_flags": []
                    }
                ]
            }]
        });

        let issues = collect_export_issues(&project);
        let issue_types: Vec<&str> = issues
            .iter()
            .filter_map(|issue| issue["type"].as_str())
            .collect();

        assert!(issue_types.contains(&"ocr_suspect"));
        assert!(issue_types.contains(&"entity_mistranslated"));
    }

    #[test]
    fn collect_export_issues_infers_textured_large_inpaint_risk() {
        let project = json!({
            "paginas": [{
                "numero": 18,
                "text_layers": [{
                    "id": "region-textured",
                    "original": "One of the continents top seven Noble Knight Idun",
                    "traduzido": "Um dos sete principais cavaleiros nobres do continente, Idun",
                    "balloon_type": "textured",
                    "background_rgb": [61, 71, 94],
                    "text_pixel_bbox": [270, 3629, 906, 4010],
                    "line_polygons": [
                        [[421, 3633], [754, 3629], [755, 3721], [422, 3726]],
                        [[271, 3733], [908, 3740], [907, 3819], [270, 3812]],
                        [[374, 3825], [800, 3830], [799, 3926], [373, 3921]]
                    ],
                    "qa_flags": []
                }]
            }]
        });

        let issues = collect_export_issues(&project);

        assert!(issues.iter().any(|issue| {
            issue["type"] == "inpaint_artifact"
                && issue["label"] == "inpaint suspeito"
                && issue["page"] == 18
        }));
    }

    #[test]
    fn collect_export_issues_does_not_treat_common_caps_words_as_entities() {
        let project = json!({
            "paginas": [{
                "numero": 43,
                "text_layers": [
                    {
                        "id": "region-raid",
                        "original": "IS THE DAY MOST PEOPLE IN THE RAID SQUAD WILL DIE",
                        "traduzido": "E o dia em que a maioria das pessoas do esquadrao de ataque vai morrer",
                        "qa_flags": []
                    },
                    {
                        "id": "region-lost",
                        "original": "LOST EVERYTHING",
                        "traduzido": "Perdi tudo",
                        "qa_flags": []
                    }
                ]
            }]
        });

        let issues = collect_export_issues(&project);

        assert!(!issues
            .iter()
            .any(|issue| issue["type"] == "entity_mistranslated"));
    }

    #[tokio::test]
    async fn export_zip_full_includes_originals_project_and_translated() {
        let root = unique_temp_dir();
        let project_dir = root.join("project");
        std::fs::create_dir_all(project_dir.join("translated")).unwrap();
        std::fs::create_dir_all(project_dir.join("originals")).unwrap();

        write_file(
            &project_dir.join("translated").join("001.jpg"),
            b"translated",
        );
        write_file(&project_dir.join("originals").join("001.jpg"), b"original");
        write_file(&project_dir.join("project.json"), br#"{"obra":"Teste"}"#);

        let output = root.join("out.zip");
        export_project(ExportConfig {
            project_path: project_dir.to_string_lossy().to_string(),
            format: "zip_full".to_string(),
            output_path: output.to_string_lossy().to_string(),
            export_mode: None,
        })
        .await
        .unwrap();

        let path = output.to_string_lossy().to_string();
        assert_eq!(output.to_string_lossy(), output.to_string_lossy());
        let entries = zip_entries(&PathBuf::from(path));
        assert!(entries.contains(&"translated/001.jpg".to_string()));
        assert!(entries.contains(&"originals/001.jpg".to_string()));
        assert!(entries.contains(&"project.json".to_string()));

        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn export_cbz_flattens_translated_pages() {
        let root = unique_temp_dir();
        let project_dir = root.join("project");
        std::fs::create_dir_all(project_dir.join("translated")).unwrap();
        write_file(
            &project_dir.join("translated").join("001.jpg"),
            b"translated",
        );

        let output = root.join("out.cbz");
        export_project(ExportConfig {
            project_path: project_dir.to_string_lossy().to_string(),
            format: "cbz".to_string(),
            output_path: output.to_string_lossy().to_string(),
            export_mode: None,
        })
        .await
        .unwrap();

        let entries = zip_entries(&output);
        assert_eq!(entries, vec!["001.jpg".to_string()]);

        let file = std::fs::File::open(&output).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let mut contents = String::new();
        archive
            .by_name("001.jpg")
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        assert_eq!(contents, "translated");

        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn export_zip_full_includes_quality_reports_and_manifest() {
        let root = unique_temp_dir();
        let project_dir = root.join("project");
        std::fs::create_dir_all(project_dir.join("translated")).unwrap();
        std::fs::create_dir_all(project_dir.join("layers").join("masks")).unwrap();

        write_file(
            &project_dir.join("translated").join("001.jpg"),
            b"translated",
        );
        write_file(
            &project_dir.join("layers").join("masks").join("001.png"),
            b"mask",
        );
        write_file(
            &project_dir.join("structured_log.jsonl"),
            br#"{"event":"done"}"#,
        );
        write_file(
            &project_dir.join("project.json"),
            quality_project_json(vec!["ocr_gibberish"]).as_bytes(),
        );

        let output = root.join("quality.zip");
        export_project(ExportConfig {
            project_path: project_dir.to_string_lossy().to_string(),
            format: "zip_full".to_string(),
            output_path: output.to_string_lossy().to_string(),
            export_mode: Some("with_warnings".to_string()),
        })
        .await
        .unwrap();

        let entries = zip_entries(&output);
        for required in [
            "project.json",
            "qa_report.md",
            "qa_report.json",
            "issues.csv",
            "glossary_used.json",
            "ocr_corrections.json",
            "export_manifest.json",
            "structured_log.jsonl",
            "layers/masks/001.png",
        ] {
            assert!(
                entries.contains(&required.to_string()),
                "missing {required}"
            );
        }

        let manifest: Value =
            serde_json::from_str(&zip_entry_string(&output, "export_manifest.json")).unwrap();
        assert_eq!(manifest["status"], "with_warnings");
        let project_sha = manifest["files"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["path"] == "project.json")
            .and_then(|entry| entry["sha256"].as_str())
            .unwrap();
        assert_eq!(project_sha.len(), 64);

        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn export_clean_blocks_critical_flags() {
        let root = unique_temp_dir();
        let project_dir = root.join("project");
        std::fs::create_dir_all(project_dir.join("translated")).unwrap();
        write_file(
            &project_dir.join("translated").join("001.jpg"),
            b"translated",
        );
        write_file(
            &project_dir.join("project.json"),
            quality_project_json(vec!["visual_text_leak"]).as_bytes(),
        );

        let output = root.join("blocked.zip");
        let err = export_project(ExportConfig {
            project_path: project_dir.to_string_lossy().to_string(),
            format: "zip_full".to_string(),
            output_path: output.to_string_lossy().to_string(),
            export_mode: Some("clean".to_string()),
        })
        .await
        .unwrap_err();

        assert!(err.contains("critical"));
        assert!(!output.exists());
        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn export_debug_allows_critical_with_manifest_marking() {
        let root = unique_temp_dir();
        let project_dir = root.join("project");
        std::fs::create_dir_all(project_dir.join("translated")).unwrap();
        write_file(
            &project_dir.join("translated").join("001.jpg"),
            b"translated",
        );
        write_file(
            &project_dir.join("project.json"),
            quality_project_json(vec!["visual_text_leak"]).as_bytes(),
        );

        let output = root.join("debug.zip");
        export_project(ExportConfig {
            project_path: project_dir.to_string_lossy().to_string(),
            format: "zip_full".to_string(),
            output_path: output.to_string_lossy().to_string(),
            export_mode: Some("debug".to_string()),
        })
        .await
        .unwrap();

        let manifest: Value =
            serde_json::from_str(&zip_entry_string(&output, "export_manifest.json")).unwrap();
        assert_eq!(manifest["status"], "blocked_debug_export");

        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    #[cfg(windows)]
    fn normalize_path_strips_file_uri_prefix_on_windows() {
        let path = normalize_path("file:///C:/traduzai/teste/project.json");
        assert_eq!(path, PathBuf::from("C:/traduzai/teste/project.json"));
    }

    #[test]
    fn resolve_inpaint_rel_prefers_image_layer_over_rendered_output() {
        let page = serde_json::json!({
            "arquivo_traduzido": "translated/001.jpg",
            "arquivo_final": "translated/001.jpg",
            "image_layers": {
                "inpaint": {
                    "path": "images/001.jpg"
                }
            }
        });

        assert_eq!(resolve_inpaint_rel(&page), Some("images/001.jpg"));
    }

    #[test]
    fn resolve_inpaint_rel_falls_back_to_legacy_final_path() {
        let page = serde_json::json!({
            "arquivo_final": "images/001.jpg"
        });

        assert_eq!(resolve_inpaint_rel(&page), Some("images/001.jpg"));
    }

    #[test]
    fn append_process_overlay_to_page_preserves_existing_overlays_and_links_text_layers() {
        let mut page = serde_json::json!({
            "numero": 1,
            "process_overlays": [
                {
                    "id": "old",
                    "page_index": 0,
                    "bbox": [1, 2, 3, 4],
                    "crop_path": "editor_cache/process_regions/page-0001/old.png",
                    "text_layer_ids": [],
                    "visible": true,
                    "locked": false,
                    "order": 0
                }
            ],
            "text_layers": [
                {
                    "id": "inside",
                    "layout_bbox": [10, 20, 80, 90]
                },
                {
                    "id": "outside",
                    "layout_bbox": [200, 200, 300, 300]
                }
            ]
        });

        let overlay = append_process_overlay_to_page(
            &mut page,
            0,
            [10, 20, 80, 90],
            "editor_cache/process_regions/page-0001/new.png".to_string(),
        )
        .unwrap();

        assert_eq!(overlay.page_index, 0);
        assert_eq!(overlay.bbox, [10, 20, 80, 90]);
        assert_eq!(overlay.text_layer_ids, vec!["inside".to_string()]);
        assert_eq!(overlay.order, 1);
        assert_eq!(page["process_overlays"].as_array().unwrap().len(), 2);
        assert_eq!(page["process_overlays"][1]["crop_path"], overlay.crop_path);
    }

    #[test]
    fn save_rgba_image_preserving_format_writes_jpeg_without_alpha_error() {
        let root = unique_temp_dir();
        let path = root.join("images").join("001.jpg");
        let image = ImageBuffer::from_pixel(3, 3, image::Rgba([255, 0, 0, 128]));

        save_rgba_image_preserving_format(&path, &image).unwrap();

        let decoded = ImageReader::open(&path).unwrap().decode().unwrap();
        assert_eq!(decoded.width(), 3);
        assert_eq!(decoded.height(), 3);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn history_brush_restores_only_masked_pixels_from_base_snapshot() {
        let root = unique_temp_dir();
        let base_path = root.join("originals").join("001.png");
        let inpaint_path = root.join("images").join("001.png");
        let base = ImageBuffer::from_pixel(4, 4, image::Rgba([240, 10, 10, 255]));
        let inpaint = ImageBuffer::from_pixel(4, 4, image::Rgba([10, 20, 240, 255]));
        save_rgba_image_preserving_format(&base_path, &base).unwrap();
        save_rgba_image_preserving_format(&inpaint_path, &inpaint).unwrap();

        let mut project = serde_json::json!({
            "paginas": [{
                "arquivo_original": "originals/001.png",
                "image_layers": {
                    "base": { "key": "base", "path": "originals/001.png", "visible": true },
                    "inpaint": { "key": "inpaint", "path": "images/001.png", "visible": true }
                }
            }]
        });
        let mut mask = ImageBuffer::from_pixel(4, 4, Luma([0]));
        mask.put_pixel(2, 1, Luma([255]));
        mask.put_pixel(3, 3, Luma([255]));

        apply_history_brush_to_inpaint(&mut project, 0, &root, &mask, Some([2, 1, 3, 2])).unwrap();

        let decoded = ImageReader::open(&inpaint_path)
            .unwrap()
            .decode()
            .unwrap()
            .to_rgba8();
        assert_eq!(decoded.get_pixel(2, 1), base.get_pixel(2, 1));
        assert_eq!(decoded.get_pixel(0, 0), inpaint.get_pixel(0, 0));
        assert_eq!(decoded.get_pixel(3, 3), inpaint.get_pixel(3, 3));
        assert_eq!(
            project["paginas"][0]["image_layers"]["inpaint"]["visible"],
            true
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn reinpaint_brush_reapplies_only_masked_pixels_from_inpaint_cache() {
        let root = unique_temp_dir();
        let base_path = root.join("originals").join("001.png");
        let inpaint_path = root.join("images").join("001.png");
        let base = ImageBuffer::from_pixel(4, 4, image::Rgba([240, 10, 10, 255]));
        let inpaint = ImageBuffer::from_pixel(4, 4, image::Rgba([10, 20, 240, 255]));
        save_rgba_image_preserving_format(&base_path, &base).unwrap();
        save_rgba_image_preserving_format(&inpaint_path, &inpaint).unwrap();

        let mut project = serde_json::json!({
            "paginas": [{
                "numero": 1,
                "arquivo_original": "originals/001.png",
                "image_layers": {
                    "base": { "key": "base", "path": "originals/001.png", "visible": true },
                    "inpaint": { "key": "inpaint", "path": "images/001.png", "visible": true }
                }
            }]
        });
        let mut recovery_mask = ImageBuffer::from_pixel(4, 4, Luma([0]));
        recovery_mask.put_pixel(1, 1, Luma([255]));
        apply_history_brush_to_inpaint(&mut project, 0, &root, &recovery_mask, Some([1, 1, 2, 2]))
            .unwrap();

        let after_recovery = ImageReader::open(&inpaint_path)
            .unwrap()
            .decode()
            .unwrap()
            .to_rgba8();
        assert_eq!(after_recovery.get_pixel(1, 1), base.get_pixel(1, 1));
        assert!(root
            .join("layers")
            .join("inpaint-cache")
            .join("001.png")
            .exists());

        let mut reinpaint_mask = ImageBuffer::from_pixel(4, 4, Luma([0]));
        reinpaint_mask.put_pixel(1, 1, Luma([255]));
        reinpaint_mask.put_pixel(2, 2, Luma([255]));
        apply_cached_inpaint_brush_to_inpaint(
            &mut project,
            0,
            &root,
            &reinpaint_mask,
            Some([1, 1, 2, 2]),
        )
        .unwrap();

        let decoded = ImageReader::open(&inpaint_path)
            .unwrap()
            .decode()
            .unwrap()
            .to_rgba8();
        assert_eq!(decoded.get_pixel(1, 1), inpaint.get_pixel(1, 1));
        assert_eq!(decoded.get_pixel(2, 2), inpaint.get_pixel(2, 2));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn rgba_brush_keeps_previous_stroke_color_after_color_change() {
        let root = unique_temp_dir();
        let path = root.join("layers").join("brush").join("001.png");
        let mut first = load_or_create_rgba_brush_layer(&path, 8, 8, false).unwrap();
        paint_stroke_rgba(&mut first, &[[2, 2]], 1, [255, 0, 0], 255, false);
        save_rgba_bitmap_layer(&path, &first).unwrap();

        let mut second = load_or_create_rgba_brush_layer(&path, 8, 8, false).unwrap();
        paint_stroke_rgba(&mut second, &[[5, 5]], 1, [0, 0, 255], 255, false);
        save_rgba_bitmap_layer(&path, &second).unwrap();

        let decoded = ImageReader::open(&path)
            .unwrap()
            .decode()
            .unwrap()
            .to_rgba8();
        assert_eq!(decoded.get_pixel(2, 2), &image::Rgba([255, 0, 0, 255]));
        assert_eq!(decoded.get_pixel(5, 5), &image::Rgba([0, 0, 255, 255]));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn clip_mask_limits_bitmap_stroke_mask_to_selected_area() {
        let mut clip = ImageBuffer::from_pixel(8, 8, Luma([0]));
        for y in 0..4 {
            for x in 0..4 {
                clip.put_pixel(x, y, Luma([255]));
            }
        }
        let mut png = std::io::Cursor::new(Vec::new());
        DynamicImage::ImageLuma8(clip)
            .write_to(&mut png, ImageFormat::Png)
            .unwrap();
        let clip_mask_png = format!("data:image/png;base64,{}", BASE64.encode(png.into_inner()));

        let mask =
            build_clipped_stroke_mask(8, 8, 2, &[vec![[1, 1], [7, 7]]], Some(&clip_mask_png))
                .unwrap();

        assert!(mask.get_pixel(1, 1)[0] > 0);
        assert_eq!(mask.get_pixel(6, 6)[0], 0);
    }

    #[test]
    fn resolve_text_layer_bbox_prefers_render_bbox_over_layout_bbox() {
        let layer = serde_json::json!({
            "render_bbox": [120, 240, 420, 520],
            "layout_bbox": [100, 200, 500, 700],
            "bbox": [90, 180, 530, 740],
            "balloon_bbox": [80, 170, 540, 760]
        });

        assert_eq!(resolve_text_layer_bbox(&layer), Some([120, 240, 420, 520]));
    }

    #[test]
    fn resolve_text_layer_bbox_falls_back_when_render_bbox_is_missing_or_invalid() {
        let invalid_render = serde_json::json!({
            "render_bbox": [300, 300, 300, 450],
            "layout_bbox": [100, 200, 500, 700],
            "bbox": [90, 180, 530, 740]
        });
        assert_eq!(
            resolve_text_layer_bbox(&invalid_render),
            Some([100, 200, 500, 700])
        );

        let without_render = serde_json::json!({
            "bbox": [90, 180, 530, 740],
            "balloon_bbox": [80, 170, 540, 760]
        });
        assert_eq!(
            resolve_text_layer_bbox(&without_render),
            Some([90, 180, 530, 740])
        );

        let only_balloon = serde_json::json!({
            "balloon_bbox": [80, 170, 540, 760]
        });
        assert_eq!(
            resolve_text_layer_bbox(&only_balloon),
            Some([80, 170, 540, 760])
        );
    }

    #[test]
    fn system_font_cache_filename_is_stable_and_safe() {
        assert_eq!(
            system_font_cache_filename("Arial", "Regular", "ttf").unwrap(),
            "SystemFont__Arial__Regular.ttf"
        );
        assert!(system_font_cache_filename("..", "Regular", "ttf").is_err());
    }

    #[test]
    fn normalizes_system_font_query() {
        assert_eq!(normalize_system_font_query("  Times-New  "), "times new");
    }

    #[tokio::test]
    async fn export_page_psd_supports_webp_original() {
        let root = unique_temp_dir();
        let project_dir = root.join("project");
        std::fs::create_dir_all(project_dir.join("originals")).unwrap();

        write_file(
            &project_dir.join("originals").join("001.webp"),
            WEBP_2X2_RED,
        );
        write_file(
            &project_dir.join("project.json"),
            br#"{
                "obra":"Teste",
                "paginas":[
                    {
                        "numero":1,
                        "arquivo_original":"originals/001.webp"
                    }
                ]
            }"#,
        );

        let output = root.join("page.psd");
        let result = export_page_psd(ExportPagePsdConfig {
            project_path: project_dir.to_string_lossy().to_string(),
            page_index: 0,
            output_path: output.to_string_lossy().to_string(),
        })
        .await;

        assert!(result.is_ok(), "resultado inesperado: {result:?}");
        let bytes = std::fs::read(&output).unwrap();
        assert!(bytes.starts_with(b"8BPS"));

        std::fs::remove_dir_all(root).ok();
    }
}
