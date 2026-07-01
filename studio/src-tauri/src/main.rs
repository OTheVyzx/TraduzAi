#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use dafont::{FcFontCache, PatternMatch};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[path = "../../../src-tauri/src/commands/studio_lite.rs"]
mod studio_lite;

#[derive(Debug, Deserialize)]
struct ProjectPathConfig {
    project_path: String,
}

#[derive(Debug, Deserialize)]
struct SaveProjectConfig {
    project_path: String,
    project_json: Value,
}

#[derive(Debug, Deserialize)]
struct BitmapLayerConfig {
    project_path: String,
    page_index: usize,
    layer_key: String,
    png_data: String,
}

#[derive(Debug, Deserialize)]
struct PsdExportConfig {
    project_path: String,
    file_name: String,
}

#[derive(Debug, Deserialize)]
struct CacheGoogleFontRequest {
    family: String,
    css_family: String,
    variant: String,
    url: String,
    filename: String,
}

#[derive(Debug, Serialize)]
struct CachedGoogleFont {
    family: String,
    css_family: String,
    variant: String,
    filename: String,
    path: String,
}

#[derive(Debug, Serialize)]
struct GoogleFontSearchResult {
    family: String,
    css_family: String,
    variant: String,
    filename: String,
    download_url: String,
    category: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct SystemFontInfo {
    family: String,
    full_name: String,
    filename: String,
    path: String,
    weight: String,
    style: String,
    monospace: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct SupportedLanguage {
    code: String,
    label: String,
    ocr_strategy: String,
}

#[derive(Debug, Deserialize)]
struct GoogleFontsMetadataResponse {
    #[serde(rename = "familyMetadataList", default)]
    family_metadata_list: Vec<GoogleFontFamilyMetadata>,
}

#[derive(Debug, Clone, Deserialize)]
struct GoogleFontFamilyMetadata {
    family: String,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    popularity: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
struct GoogleFontRepoEntry {
    name: String,
    #[serde(default)]
    download_url: Option<String>,
    #[serde(rename = "type")]
    entry_type: String,
}

const GOOGLE_FONTS_METADATA_URL: &str = "https://fonts.google.com/metadata/fonts";
const GOOGLE_FONTS_REPO_CONTENTS_URL: &str = "https://api.github.com/repos/google/fonts/contents";
const GOOGLE_FONT_LICENSE_DIRS: [&str; 3] = ["ofl", "apache", "ufl"];
const GOOGLE_FONT_SEARCH_LIMIT: usize = 12;

#[tauri::command]
fn studio_load_project(config: ProjectPathConfig) -> Result<Value, String> {
    let project_file = resolve_project_file(&config.project_path);
    let payload = std::fs::read_to_string(&project_file)
        .map_err(|error| format!("Falha ao ler project.json: {error}"))?;
    parse_project_payload(&payload)
}

fn parse_project_payload(payload: &str) -> Result<Value, String> {
    serde_json::from_str(payload.trim_start_matches('\u{feff}'))
        .map_err(|error| format!("project.json invalido: {error}"))
}

#[tauri::command]
fn studio_save_project(config: SaveProjectConfig) -> Result<(), String> {
    let project_file = resolve_project_file(&config.project_path);
    if let Some(parent) = project_file.parent() {
        std::fs::create_dir_all(parent).map_err(|error| format!("Falha ao criar pasta do projeto: {error}"))?;
    }
    let payload = serde_json::to_string_pretty(&config.project_json)
        .map_err(|error| format!("Falha ao serializar project.json: {error}"))?;
    std::fs::write(project_file, payload).map_err(|error| format!("Falha ao salvar project.json: {error}"))
}

#[tauri::command]
fn studio_write_bitmap_layer(config: BitmapLayerConfig) -> Result<String, String> {
    let project_file = resolve_project_file(&config.project_path);
    let project_dir = project_file
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "Caminho de projeto invalido".to_string())?;
    let layer = sanitize_layer_key(&config.layer_key)?;
    let rel = format!("layers/{}/{:03}.png", layer, config.page_index + 1);
    let output = project_dir.join(rel.replace('/', std::path::MAIN_SEPARATOR_STR));
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|error| format!("Falha ao criar pasta de camada: {error}"))?;
    }
    let raw = config
        .png_data
        .split_once(',')
        .map(|(_, value)| value)
        .unwrap_or(config.png_data.as_str());
    let bytes = BASE64
        .decode(raw)
        .map_err(|error| format!("PNG base64 invalido: {error}"))?;
    std::fs::write(&output, bytes).map_err(|error| format!("Falha ao salvar camada bitmap: {error}"))?;
    Ok(rel)
}

#[tauri::command]
fn studio_prepare_psd_export(config: PsdExportConfig) -> Result<String, String> {
    let project_file = resolve_project_file(&config.project_path);
    let project_dir = project_file
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "Caminho de projeto invalido".to_string())?;
    let safe_name = sanitize_file_name(&config.file_name)?;
    let rel = format!("exports/{}", safe_name);
    let output = project_dir.join(rel.replace('/', std::path::MAIN_SEPARATOR_STR));
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|error| format!("Falha ao criar pasta de exportacao: {error}"))?;
    }
    Ok(output.to_string_lossy().to_string())
}

#[tauri::command]
async fn search_google_fonts(query: String) -> Result<Vec<GoogleFontSearchResult>, String> {
    let normalized_query = normalize_google_font_query(&query);
    if normalized_query.len() < 2 {
        return Ok(Vec::new());
    }

    let client = reqwest::Client::new();
    let metadata = client
        .get(GOOGLE_FONTS_METADATA_URL)
        .header(reqwest::header::USER_AGENT, "TraduzAI Studio")
        .send()
        .await
        .map_err(|error| format!("Falha ao consultar Google Fonts: {error}"))?;

    if !metadata.status().is_success() {
        return Err(format!("Falha ao consultar Google Fonts: HTTP {}", metadata.status()));
    }

    let metadata_text = metadata
        .text()
        .await
        .map_err(|error| format!("Falha ao ler resposta do Google Fonts: {error}"))?;
    let families = search_google_fonts_metadata_json(&metadata_text, &query, GOOGLE_FONT_SEARCH_LIMIT)?;
    let mut results = Vec::new();

    for family in families {
        if let Ok(repo_file) = fetch_google_font_repo_file(&client, &family.family).await {
            if let Some(download_url) = repo_file.download_url {
                let extension = Path::new(&repo_file.name)
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("ttf");
                results.push(GoogleFontSearchResult {
                    family: family.family.clone(),
                    css_family: family.family.clone(),
                    variant: "regular".to_string(),
                    filename: google_font_cache_filename(&family.family, extension),
                    download_url,
                    category: family.category,
                });
            }
        }
    }

    Ok(results)
}

#[tauri::command]
async fn cache_google_font(
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
fn load_supported_languages() -> Vec<SupportedLanguage> {
    [
        ("en", "English", "dedicated"),
        ("pt-BR", "Portugues (Brasil)", "best_effort"),
        ("es", "Espanol", "best_effort"),
        ("ja", "Japanese", "dedicated"),
        ("ko", "Korean", "dedicated"),
        ("zh", "Chinese", "dedicated"),
        ("fr", "Francais", "best_effort"),
        ("de", "Deutsch", "best_effort"),
        ("it", "Italiano", "best_effort"),
    ]
    .into_iter()
    .map(|(code, label, ocr_strategy)| SupportedLanguage {
        code: code.to_string(),
        label: label.to_string(),
        ocr_strategy: ocr_strategy.to_string(),
    })
    .collect()
}

#[tauri::command]
async fn list_system_fonts(query: Option<String>) -> Result<Vec<SystemFontInfo>, String> {
    let normalized_query = normalize_system_font_query(query.as_deref().unwrap_or(""));
    let cache = FcFontCache::build();
    let mut fonts = Vec::new();

    for (pattern, font_path) in cache.list() {
        let family = pattern.family.clone().unwrap_or_default().trim().to_string();
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
async fn resolve_system_font(filename: String) -> Result<Option<SystemFontInfo>, String> {
    let wanted = sanitize_system_font_filename(&filename)?;
    Ok(list_system_fonts(None)
        .await?
        .into_iter()
        .find(|font| font.filename == wanted))
}

fn google_fonts_cache_dir() -> Result<PathBuf, String> {
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .ok_or_else(|| "Nao foi possivel localizar a pasta do usuario para cache de fontes".to_string())?;

    Ok(PathBuf::from(home).join(".traduzai").join("fonts").join("google"))
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
        matches!(ch, '/' | '\\' | ':' | '\0' | '<' | '>' | '"' | '|' | '?' | '*') || ch.is_control()
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
        .map(|ch| if ch.is_ascii_alphanumeric() { ch.to_ascii_lowercase() } else { ' ' })
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
    format!("GoogleFont__{}__regular.{}", google_font_cache_slug(family), normalized_extension)
}

fn normalize_system_font_query(value: &str) -> String {
    value
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch.to_ascii_lowercase() } else { ' ' })
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

fn system_font_cache_filename(family: &str, style: &str, extension: &str) -> Result<String, String> {
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
            matches!(ch, '/' | '\\' | ':' | '\0' | '<' | '>' | '"' | '|' | '?' | '*') || ch.is_control()
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
    if bold == PatternMatch::True { "700".to_string() } else { "400".to_string() }
}

fn system_font_style(italic: PatternMatch) -> String {
    if italic == PatternMatch::True { "italic".to_string() } else { "normal".to_string() }
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
    let parsed: GoogleFontsMetadataResponse =
        serde_json::from_str(metadata_json).map_err(|error| format!("Resposta invalida do Google Fonts: {error}"))?;
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
            .then_with(|| a.popularity.unwrap_or(i64::MAX).cmp(&b.popularity.unwrap_or(i64::MAX)))
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
        .min_by(|a, b| google_font_repo_file_rank(&a.name).cmp(&google_font_repo_file_rank(&b.name)))
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
            .header(reqwest::header::USER_AGENT, "TraduzAI Studio")
            .send()
            .await
            .map_err(|error| format!("Falha ao localizar fonte no repositorio Google Fonts: {error}"))?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            continue;
        }
        if !response.status().is_success() {
            continue;
        }
        let entries = response
            .json::<Vec<GoogleFontRepoEntry>>()
            .await
            .map_err(|error| format!("Falha ao ler repositorio Google Fonts: {error}"))?;
        if let Some(selected) = select_google_font_repo_file(&entries) {
            return Ok(selected.clone());
        }
    }

    Err(format!("Nao foi encontrado arquivo TTF/OTF para a fonte Google: {family}"))
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
        .map_err(|error| format!("URL de fonte Google invalida: {error}"))?;
    if parsed_url.scheme() != "https" && parsed_url.scheme() != "http" {
        return Err("URL de fonte Google deve usar http ou https".to_string());
    }

    std::fs::create_dir_all(cache_dir).map_err(|error| format!("Falha ao criar cache de fontes Google: {error}"))?;

    let response = reqwest::Client::new()
        .get(parsed_url)
        .send()
        .await
        .map_err(|error| format!("Falha ao baixar fonte Google: {error}"))?;

    if !response.status().is_success() {
        return Err(format!("Falha ao baixar fonte Google: HTTP {}", response.status()));
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|error| format!("Falha ao ler fonte Google baixada: {error}"))?;
    if bytes.is_empty() {
        return Err("Fonte Google baixada esta vazia".to_string());
    }

    std::fs::write(&target_path, &bytes).map_err(|error| format!("Falha ao gravar fonte Google em cache: {error}"))?;

    Ok(CachedGoogleFont {
        family: request.family,
        css_family: request.css_family,
        variant: request.variant,
        filename,
        path: target_path.to_string_lossy().to_string(),
    })
}

fn resolve_project_file(path: &str) -> PathBuf {
    let input = PathBuf::from(path);
    if input
        .extension()
        .and_then(|name| name.to_str())
        .map(|extension| extension.eq_ignore_ascii_case("json"))
        .unwrap_or(false)
    {
        input
    } else {
        input.join("project.json")
    }
}

fn sanitize_layer_key(value: &str) -> Result<&str, String> {
    match value {
        "mask" | "inpaint" | "brush" | "recovery" | "rendered" => Ok(value),
        _ => Err("Camada bitmap invalida".to_string()),
    }
}

fn sanitize_file_name(value: &str) -> Result<String, String> {
    let cleaned: String = value
        .chars()
        .map(|ch| match ch {
            '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*' => '-',
            ch if ch.is_control() => '-',
            ch => ch,
        })
        .collect::<String>()
        .trim()
        .trim_matches('.')
        .to_string();
    if cleaned.is_empty() {
        return Err("Nome de arquivo PSD invalido".to_string());
    }
    if !cleaned.to_lowercase().ends_with(".psd") {
        return Err("Exportacao PSD deve usar extensao .psd".to_string());
    }
    Ok(cleaned)
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![
            studio_load_project,
            studio_save_project,
            studio_write_bitmap_layer,
            studio_prepare_psd_export,
            search_google_fonts,
            cache_google_font,
            list_system_fonts,
            resolve_system_font,
            load_supported_languages,
            studio_lite::studio_lite_model_status,
            studio_lite::studio_lite_build_mask,
            studio_lite::studio_lite_inpaint_region,
            studio_lite::studio_lite_detect_page,
        ])
        .run(tauri::generate_context!())
        .expect("erro ao executar TraduzAI Studio");
}

#[cfg(test)]
mod tests {
    use super::parse_project_payload;

    #[test]
    fn parses_project_json_with_utf8_bom() {
        let payload = "\u{feff}{\"app\":\"traduzai\",\"paginas\":[]}";
        let value = parse_project_payload(payload).expect("project json should parse with BOM");
        assert_eq!(value["app"], "traduzai");
    }

    #[test]
    fn resolves_any_selected_json_file_as_project_file() {
        let path = super::resolve_project_file("N:\\TraduzAI\\qa\\project-saved.json");
        assert_eq!(path.file_name().and_then(|name| name.to_str()), Some("project-saved.json"));
    }

    #[test]
    fn resolves_directory_path_to_default_project_json() {
        let path = super::resolve_project_file("N:\\TraduzAI\\qa");
        assert_eq!(path.file_name().and_then(|name| name.to_str()), Some("project.json"));
    }

    #[test]
    fn system_font_cache_filename_is_stable_and_safe() {
        assert_eq!(
            super::system_font_cache_filename("Arial", "Regular", "ttf").unwrap(),
            "SystemFont__Arial__Regular.ttf"
        );
        assert!(super::system_font_cache_filename("..", "Regular", "ttf").is_err());
    }

    #[test]
    fn normalizes_system_font_query() {
        assert_eq!(super::normalize_system_font_query("  Times-New  "), "times new");
    }
}
