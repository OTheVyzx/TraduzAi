#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use dafont::{FcFontCache, PatternMatch};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::time::{timeout, Duration};

#[path = "../../../src-tauri/src/commands/studio_lite.rs"]
mod studio_lite;

mod library;

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
struct RecoverySnapshotConfig {
    project_path: String,
    snapshot: Value,
}

#[derive(Debug, Deserialize)]
struct BitmapLayerConfig {
    project_path: String,
    page_index: usize,
    layer_key: String,
    png_data: String,
}

#[derive(Debug, Deserialize)]
struct GeneratedAssetConfig {
    project_path: String,
    page_index: usize,
    asset_id: String,
    png_data: String,
}

#[derive(Debug, Deserialize)]
struct DeleteGeneratedAssetsConfig {
    project_path: String,
    page_index: usize,
    asset_ids: Vec<String>,
}

const FLUX_ADAPTER_CONTRACT_VERSION: &str = "1.0";
const DEFAULT_FLUX_MODEL: &str = "black-forest-labs/FLUX.1-Fill-dev";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FluxGenerateConfig {
    contract_version: String,
    job_id: String,
    prompt: String,
    negative_prompt: String,
    model: String,
    source_png_data: String,
    mask_png_data: String,
    width: u32,
    height: u32,
    variant_count: usize,
    seed: i64,
    steps: u32,
    guidance_scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FluxVariantOutput {
    id: String,
    seed: i64,
    #[serde(default)]
    png_data: Option<String>,
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FluxGenerateResult {
    contract_version: String,
    job_id: String,
    provider: String,
    model: String,
    variants: Vec<FluxVariantOutput>,
}

#[derive(Debug, Clone, Serialize)]
struct FluxProviderStatus {
    status: String,
    provider: String,
    model: Option<String>,
    message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FluxCommandSpec {
    program: String,
    args: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct FluxCancelConfig {
    job_id: String,
}

enum FluxWorkerMessage {
    Generate {
        config: FluxGenerateConfig,
        timeout_seconds: u64,
        response: oneshot::Sender<Result<FluxGenerateResult, String>>,
    },
    Cancel {
        job_id: String,
        response: oneshot::Sender<bool>,
    },
}

#[derive(Default)]
struct FluxWorkerManager {
    sender: Mutex<Option<mpsc::Sender<FluxWorkerMessage>>>,
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
    recover_project_backup_if_needed(&project_file)?;
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
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("Falha ao criar pasta do projeto: {error}"))?;
    }
    let payload = serde_json::to_string_pretty(&config.project_json)
        .map_err(|error| format!("Falha ao serializar project.json: {error}"))?;
    write_project_json_atomically(&project_file, &payload)
}

fn project_backup_path(project_file: &Path) -> PathBuf {
    let name = project_file
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("project.json");
    project_file.with_file_name(format!(".{name}.traduzai-backup"))
}

fn recover_project_backup_if_needed(project_file: &Path) -> Result<(), String> {
    let backup = project_backup_path(project_file);
    if project_file.exists() {
        if backup.exists() {
            let _ = std::fs::remove_file(&backup);
        }
        return Ok(());
    }
    if backup.exists() {
        std::fs::rename(&backup, project_file)
            .map_err(|error| format!("Falha ao recuperar project.json anterior: {error}"))?;
    }
    Ok(())
}

fn write_project_json_atomically(project_file: &Path, payload: &str) -> Result<(), String> {
    if let Some(parent) = project_file.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("Falha ao criar pasta do projeto: {error}"))?;
    }
    recover_project_backup_if_needed(project_file)?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| format!("Relogio invalido para save atomico: {error}"))?
        .as_nanos();
    let file_name = project_file
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("project.json");
    let temporary = project_file.with_file_name(format!(".{file_name}.{nonce}.tmp"));
    let backup = project_backup_path(project_file);

    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|error| format!("Falha ao criar project.json temporario: {error}"))?;
    if let Err(error) = file.write_all(payload.as_bytes()).and_then(|_| file.sync_all()) {
        let _ = std::fs::remove_file(&temporary);
        return Err(format!("Falha ao sincronizar project.json temporario: {error}"));
    }
    drop(file);

    if project_file.exists() {
        std::fs::rename(project_file, &backup)
            .map_err(|error| format!("Falha ao preparar troca atomica do project.json: {error}"))?;
    }
    if let Err(error) = std::fs::rename(&temporary, project_file) {
        if backup.exists() {
            let _ = std::fs::rename(&backup, project_file);
        }
        let _ = std::fs::remove_file(&temporary);
        return Err(format!("Falha ao concluir troca atomica do project.json: {error}"));
    }
    if backup.exists() {
        let _ = std::fs::remove_file(&backup);
    }
    Ok(())
}

fn recovery_directory(project_file: &Path) -> Result<PathBuf, String> {
    let project_dir = project_file
        .parent()
        .ok_or_else(|| "Caminho de projeto invalido para recuperacao".to_string())?;
    let canonical = std::fs::canonicalize(project_file).unwrap_or_else(|_| project_file.to_path_buf());
    let identity = canonical.to_string_lossy().replace('\\', "/").to_lowercase();
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in identity.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let file_name = project_file
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("project.json")
        .chars()
        .map(|character| if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') { character } else { '-' })
        .collect::<String>();
    Ok(project_dir
        .join(".traduzai-studio")
        .join("recovery")
        .join(format!("{file_name}-{hash:016x}")))
}

fn recovery_snapshot_paths(project_file: &Path) -> Result<Vec<PathBuf>, String> {
    let directory = recovery_directory(project_file)?;
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut paths = std::fs::read_dir(&directory)
        .map_err(|error| format!("Falha ao listar snapshots de recuperacao: {error}"))?
        .filter_map(|entry| entry.ok().map(|item| item.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("snapshot-") && name.ends_with(".json"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    paths.sort_by(|left, right| right.file_name().cmp(&left.file_name()));
    Ok(paths)
}

fn write_recovery_snapshot(project_file: &Path, snapshot: &Value) -> Result<PathBuf, String> {
    let directory = recovery_directory(project_file)?;
    std::fs::create_dir_all(&directory)
        .map_err(|error| format!("Falha ao criar pasta de recuperacao: {error}"))?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| format!("Relogio invalido para snapshot de recuperacao: {error}"))?
        .as_nanos();
    let stem = format!("snapshot-{nonce:032}-{}", std::process::id());
    let temporary = directory.join(format!("{stem}.json.tmp"));
    let target = directory.join(format!("{stem}.json"));
    let payload = serde_json::to_vec_pretty(snapshot)
        .map_err(|error| format!("Falha ao serializar snapshot de recuperacao: {error}"))?;

    let result = (|| -> Result<(), String> {
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .map_err(|error| format!("Falha ao criar snapshot temporario: {error}"))?;
        file.write_all(&payload)
            .map_err(|error| format!("Falha ao gravar snapshot temporario: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("Falha ao sincronizar snapshot temporario: {error}"))?;
        std::fs::rename(&temporary, &target)
            .map_err(|error| format!("Falha ao concluir snapshot atomico: {error}"))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result?;

    if let Ok(paths) = recovery_snapshot_paths(project_file) {
        for stale in paths.into_iter().skip(5) {
            let _ = std::fs::remove_file(stale);
        }
    }
    Ok(target)
}

fn load_latest_recovery_snapshot(project_file: &Path) -> Result<Option<Value>, String> {
    let current_project_is_valid = std::fs::read_to_string(project_file)
        .ok()
        .and_then(|payload| parse_project_payload(&payload).ok())
        .is_some();
    let project_modified = current_project_is_valid
        .then(|| std::fs::metadata(project_file).and_then(|metadata| metadata.modified()).ok())
        .flatten();
    for path in recovery_snapshot_paths(project_file)? {
        if let (Some(project_time), Ok(snapshot_time)) = (
            project_modified,
            std::fs::metadata(&path).and_then(|metadata| metadata.modified()),
        ) {
            if snapshot_time < project_time {
                continue;
            }
        }
        let payload = match std::fs::read_to_string(&path) {
            Ok(payload) => payload,
            Err(_) => continue,
        };
        if let Ok(snapshot) = parse_project_payload(&payload) {
            return Ok(Some(snapshot));
        }
    }
    Ok(None)
}

fn normalized_project_identity(value: &str) -> String {
    let normalized = value.replace('\\', "/").trim_end_matches('/').to_string();
    if normalized.as_bytes().get(1) == Some(&b':') {
        normalized.to_lowercase()
    } else {
        normalized
    }
}

fn snapshot_matches_project(snapshot: &Value, configured_project_path: &str) -> bool {
    snapshot
        .get("projectPath")
        .and_then(Value::as_str)
        .map(|snapshot_path| {
            normalized_project_identity(snapshot_path) == normalized_project_identity(configured_project_path)
        })
        .unwrap_or(false)
}

#[tauri::command]
fn studio_save_recovery_snapshot(config: RecoverySnapshotConfig) -> Result<(), String> {
    if !snapshot_matches_project(&config.snapshot, &config.project_path) {
        return Err("Snapshot de recuperacao pertence a outro projeto".to_string());
    }
    let project_file = resolve_project_file(&config.project_path);
    write_recovery_snapshot(&project_file, &config.snapshot).map(|_| ())
}

#[tauri::command]
fn studio_load_recovery_snapshot(config: ProjectPathConfig) -> Result<Option<Value>, String> {
    let project_file = resolve_project_file(&config.project_path);
    Ok(load_latest_recovery_snapshot(&project_file)?
        .filter(|snapshot| snapshot_matches_project(snapshot, &config.project_path)))
}

#[tauri::command]
fn studio_clear_recovery_snapshot(config: ProjectPathConfig) -> Result<(), String> {
    let project_file = resolve_project_file(&config.project_path);
    for path in recovery_snapshot_paths(&project_file)? {
        std::fs::remove_file(path)
            .map_err(|error| format!("Falha ao limpar snapshot de recuperacao: {error}"))?;
    }
    Ok(())
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
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("Falha ao criar pasta de camada: {error}"))?;
    }
    let raw = config
        .png_data
        .split_once(',')
        .map(|(_, value)| value)
        .unwrap_or(config.png_data.as_str());
    let bytes = BASE64
        .decode(raw)
        .map_err(|error| format!("PNG base64 invalido: {error}"))?;
    std::fs::write(&output, bytes)
        .map_err(|error| format!("Falha ao salvar camada bitmap: {error}"))?;
    Ok(rel)
}

#[tauri::command]
fn studio_write_generated_asset(config: GeneratedAssetConfig) -> Result<String, String> {
    let project_file = resolve_project_file(&config.project_path);
    let project_dir = project_file
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "Caminho de projeto invalido".to_string())?;
    let asset_id = sanitize_generated_asset_id(&config.asset_id)?;
    let rel = format!(
        "layers/generated/{:03}/{}.png",
        config.page_index + 1,
        asset_id
    );
    let output = project_dir.join(rel.replace('/', std::path::MAIN_SEPARATOR_STR));
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("Falha ao criar pasta de asset gerado: {error}"))?;
    }
    let raw = config
        .png_data
        .split_once(',')
        .map(|(_, value)| value)
        .unwrap_or(config.png_data.as_str());
    let bytes = BASE64
        .decode(raw)
        .map_err(|error| format!("PNG base64 invalido: {error}"))?;
    std::fs::write(&output, bytes)
        .map_err(|error| format!("Falha ao salvar asset gerado: {error}"))?;
    Ok(rel)
}

#[tauri::command]
fn studio_delete_generated_assets(config: DeleteGeneratedAssetsConfig) -> Result<(), String> {
    if config.asset_ids.len() > 4 {
        return Err("O FLUX pode limpar no maximo 4 variantes por job".to_string());
    }
    let project_file = resolve_project_file(&config.project_path);
    let project_dir = project_file
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "Caminho de projeto invalido".to_string())?;
    for asset_id in config.asset_ids {
        let asset_id = sanitize_generated_asset_id(&asset_id)?;
        let relative = format!(
            "layers/generated/{:03}/{}.png",
            config.page_index + 1,
            asset_id
        );
        let path = project_dir.join(relative.replace('/', std::path::MAIN_SEPARATOR_STR));
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("Falha ao limpar asset FLUX temporario: {error}")),
        }
    }
    Ok(())
}

fn flux_command_spec(
    command: Option<String>,
    args_json: Option<String>,
) -> Result<FluxCommandSpec, String> {
    let program = command
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Configure TRADUZAI_STUDIO_FLUX_COMMAND para habilitar o FLUX local".to_string())?;
    if program.contains('\0') {
        return Err("Comando do adaptador FLUX invalido".to_string());
    }
    let args = match args_json.map(|value| value.trim().to_string()) {
        Some(value) if !value.is_empty() => serde_json::from_str::<Vec<String>>(&value)
            .map_err(|error| format!("TRADUZAI_STUDIO_FLUX_ARGS_JSON invalido: {error}"))?,
        _ => Vec::new(),
    };
    if args.iter().any(|value| value.contains('\0')) {
        return Err("Argumentos do adaptador FLUX invalidos".to_string());
    }
    Ok(FluxCommandSpec { program, args })
}

fn configured_flux_command() -> Result<FluxCommandSpec, String> {
    flux_command_spec(
        std::env::var("TRADUZAI_STUDIO_FLUX_COMMAND").ok(),
        std::env::var("TRADUZAI_STUDIO_FLUX_ARGS_JSON").ok(),
    )
}

fn validate_flux_job_id(job_id: &str) -> Result<(), String> {
    if job_id.is_empty()
        || job_id.len() > 128
        || !job_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | ':'))
    {
        return Err("Id de job FLUX invalido".to_string());
    }
    Ok(())
}

fn validate_flux_generate_config(config: &FluxGenerateConfig) -> Result<(), String> {
    if config.contract_version != FLUX_ADAPTER_CONTRACT_VERSION {
        return Err("Versao de contrato FLUX incompatível".to_string());
    }
    validate_flux_job_id(&config.job_id)?;
    if config.model.trim().is_empty() || config.model.len() > 256 {
        return Err("Modelo FLUX invalido".to_string());
    }
    if config.prompt.len() > 4000 || config.negative_prompt.len() > 4000 {
        return Err("Prompt FLUX excede o limite de 4000 caracteres".to_string());
    }
    if config.width == 0 || config.height == 0 || config.width > 4096 || config.height > 4096 {
        return Err("Dimensoes do crop FLUX devem ficar entre 1 e 4096 pixels".to_string());
    }
    if !(2..=4).contains(&config.variant_count) {
        return Err("O FLUX deve gerar entre 2 e 4 variantes".to_string());
    }
    if !(1..=100).contains(&config.steps) || !config.guidance_scale.is_finite() {
        return Err("Configuracao de amostragem FLUX invalida".to_string());
    }
    for (label, value) in [
        ("imagem", config.source_png_data.as_str()),
        ("mascara", config.mask_png_data.as_str()),
    ] {
        if !value.starts_with("data:image/png;base64,") || value.len() > 128 * 1024 * 1024 {
            return Err(format!("{label} FLUX deve ser um PNG local em data URL"));
        }
    }
    Ok(())
}

fn validate_flux_generate_result(
    request: &FluxGenerateConfig,
    result: FluxGenerateResult,
) -> Result<FluxGenerateResult, String> {
    if result.contract_version != FLUX_ADAPTER_CONTRACT_VERSION {
        return Err("O adaptador retornou uma versao de contrato FLUX incompatível".to_string());
    }
    if result.job_id != request.job_id {
        return Err("O adaptador FLUX respondeu para outro job".to_string());
    }
    if result.variants.len() != request.variant_count || !(2..=4).contains(&result.variants.len()) {
        return Err("O adaptador FLUX retornou uma quantidade inesperada de variantes".to_string());
    }
    let mut ids = std::collections::HashSet::new();
    for variant in &result.variants {
        if variant.id.trim().is_empty() || !ids.insert(variant.id.as_str()) {
            return Err("O adaptador FLUX retornou ids de variante invalidos".to_string());
        }
        let png_data = variant.png_data.as_deref().unwrap_or_default().trim();
        let path = variant.path.as_deref().unwrap_or_default().trim();
        if png_data.is_empty() && path.is_empty() {
            return Err(format!("A variante {} nao possui imagem", variant.id));
        }
        if !png_data.is_empty()
            && (!png_data.starts_with("data:image/png;base64,") || png_data.len() > 128 * 1024 * 1024)
        {
            return Err(format!("A variante {} retornou PNG invalido ou muito grande", variant.id));
        }
        let lower_path = path.to_ascii_lowercase();
        if lower_path.starts_with("http://") || lower_path.starts_with("https://") {
            return Err("O adaptador FLUX deve retornar somente arquivos locais".to_string());
        }
    }
    Ok(result)
}

fn parse_flux_adapter_output(stdout: &[u8]) -> Result<FluxGenerateResult, String> {
    let payload = std::str::from_utf8(stdout)
        .map_err(|error| format!("Saida UTF-8 invalida do adaptador FLUX: {error}"))?;
    let candidate = payload
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .ok_or_else(|| "O adaptador FLUX nao retornou JSON".to_string())?;
    let value: Value = serde_json::from_str(candidate.trim())
        .map_err(|error| format!("Resposta JSON invalida do adaptador FLUX: {error}"))?;
    if let Some(message) = value.get("error").and_then(Value::as_str) {
        return Err(format!("O adaptador FLUX falhou: {}", message.chars().take(1000).collect::<String>()));
    }
    serde_json::from_value(value)
        .map_err(|error| format!("Contrato JSON invalido do adaptador FLUX: {error}"))
}

async fn read_limited_flux_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    max_bytes: usize,
) -> Result<Option<Vec<u8>>, String> {
    let mut output = Vec::new();
    loop {
        let available = reader
            .fill_buf()
            .await
            .map_err(|error| format!("Falha ao ler adaptador FLUX: {error}"))?;
        if available.is_empty() {
            return if output.is_empty() { Ok(None) } else { Ok(Some(output)) };
        }
        let end = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map(|index| index + 1)
            .unwrap_or(available.len());
        if output.len().saturating_add(end) > max_bytes {
            return Err("A resposta do adaptador FLUX excedeu o limite local".to_string());
        }
        let finished = available[end - 1] == b'\n';
        output.extend_from_slice(&available[..end]);
        reader.consume(end);
        if finished {
            while matches!(output.last(), Some(b'\n' | b'\r')) {
                output.pop();
            }
            return Ok(Some(output));
        }
    }
}

async fn run_flux_worker(
    mut child: tokio::process::Child,
    mut stdin: tokio::process::ChildStdin,
    stdout: tokio::process::ChildStdout,
    mut receiver: mpsc::Receiver<FluxWorkerMessage>,
) {
    let mut stdout = BufReader::new(stdout);
    while let Some(message) = receiver.recv().await {
        let (config, timeout_seconds, response) = match message {
            FluxWorkerMessage::Generate { config, timeout_seconds, response } => {
                (config, timeout_seconds, response)
            }
            FluxWorkerMessage::Cancel { response, .. } => {
                let _ = response.send(false);
                continue;
            }
        };

        let mut response = Some(response);
        let payload = match serde_json::to_vec(&config) {
            Ok(payload) => payload,
            Err(error) => {
                if let Some(response) = response.take() {
                    let _ = response.send(Err(format!("Falha ao serializar job FLUX: {error}")));
                }
                continue;
            }
        };
        if let Err(error) = stdin.write_all(&payload).await {
            if let Some(response) = response.take() {
                let _ = response.send(Err(format!("Falha ao enviar job ao adaptador FLUX: {error}")));
            }
            let _ = child.kill().await;
            return;
        }
        if let Err(error) = stdin.write_all(b"\n").await {
            if let Some(response) = response.take() {
                let _ = response.send(Err(format!("Falha ao finalizar job FLUX: {error}")));
            }
            let _ = child.kill().await;
            return;
        }
        if let Err(error) = stdin.flush().await {
            if let Some(response) = response.take() {
                let _ = response.send(Err(format!("Falha ao enviar job ao adaptador FLUX: {error}")));
            }
            let _ = child.kill().await;
            return;
        }

        let deadline = tokio::time::sleep(Duration::from_secs(timeout_seconds));
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                line = read_limited_flux_line(&mut stdout, 384 * 1024 * 1024) => {
                    let protocol_failed = matches!(&line, Err(_) | Ok(None));
                    let result = match line {
                        Ok(Some(line)) => parse_flux_adapter_output(&line)
                            .and_then(|result| validate_flux_generate_result(&config, result)),
                        Ok(None) => Err("O adaptador FLUX encerrou antes de responder".to_string()),
                        Err(error) => Err(error),
                    };
                    if let Some(response) = response.take() {
                        let _ = response.send(result);
                    }
                    if protocol_failed {
                        let _ = child.kill().await;
                        return;
                    }
                    break;
                }
                queued = receiver.recv() => {
                    match queued {
                        Some(FluxWorkerMessage::Cancel { job_id, response: cancel_response }) => {
                            if job_id == config.job_id {
                                let _ = child.kill().await;
                                if let Some(response) = response.take() {
                                    let _ = response.send(Err("Geração FLUX cancelada pelo usuário".to_string()));
                                }
                                let _ = cancel_response.send(true);
                                return;
                            }
                            let _ = cancel_response.send(false);
                        }
                        Some(FluxWorkerMessage::Generate { response, .. }) => {
                            let _ = response.send(Err("Já existe uma geração FLUX em andamento".to_string()));
                        }
                        None => {
                            let _ = child.kill().await;
                            return;
                        }
                    }
                }
                _ = &mut deadline => {
                    let _ = child.kill().await;
                    if let Some(response) = response.take() {
                        let _ = response.send(Err(format!("O adaptador FLUX excedeu o limite de {timeout_seconds}s")));
                    }
                    return;
                }
            }
        }
    }
    let _ = child.kill().await;
}

async fn spawn_flux_worker_from_spec(
    spec: FluxCommandSpec,
) -> Result<mpsc::Sender<FluxWorkerMessage>, String> {
    let mut child = Command::new(&spec.program)
        .args(&spec.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| format!("Falha ao iniciar adaptador FLUX local: {error}"))?;
    let stdin = child.stdin.take().ok_or_else(|| "Falha ao abrir stdin do adaptador FLUX".to_string())?;
    let stdout = child.stdout.take().ok_or_else(|| "Falha ao abrir stdout do adaptador FLUX".to_string())?;
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                eprintln!("[FLUX local] {line}");
            }
        });
    }
    let (sender, receiver) = mpsc::channel(8);
    tokio::spawn(run_flux_worker(child, stdin, stdout, receiver));
    Ok(sender)
}

async fn spawn_flux_worker() -> Result<mpsc::Sender<FluxWorkerMessage>, String> {
    spawn_flux_worker_from_spec(configured_flux_command()?).await
}

impl FluxWorkerManager {
    async fn worker_sender(&self) -> Result<mpsc::Sender<FluxWorkerMessage>, String> {
        let mut current = self.sender.lock().await;
        if let Some(sender) = current.as_ref().filter(|sender| !sender.is_closed()) {
            return Ok(sender.clone());
        }
        let sender = spawn_flux_worker().await?;
        *current = Some(sender.clone());
        Ok(sender)
    }

    async fn generate(&self, config: FluxGenerateConfig, timeout_seconds: u64) -> Result<FluxGenerateResult, String> {
        let sender = self.worker_sender().await?;
        let (response, result) = oneshot::channel();
        if sender.send(FluxWorkerMessage::Generate { config, timeout_seconds, response }).await.is_err() {
            self.sender.lock().await.take();
            return Err("O worker FLUX local foi encerrado".to_string());
        }
        match result.await {
            Ok(result) => result,
            Err(_) => {
                self.sender.lock().await.take();
                Err("O worker FLUX local foi encerrado sem resposta".to_string())
            }
        }
    }

    async fn cancel(&self, job_id: String) -> Result<bool, String> {
        let sender = self.sender.lock().await.as_ref().cloned();
        let Some(sender) = sender else { return Ok(false) };
        let (response, result) = oneshot::channel();
        if sender.send(FluxWorkerMessage::Cancel { job_id, response }).await.is_err() {
            self.sender.lock().await.take();
            return Ok(false);
        }
        let cancelled = timeout(Duration::from_secs(10), result)
            .await
            .map_err(|_| "O worker FLUX nao confirmou o cancelamento".to_string())?
            .map_err(|_| "O worker FLUX encerrou durante o cancelamento".to_string())?;
        if cancelled {
            self.sender.lock().await.take();
        }
        Ok(cancelled)
    }
}

#[tauri::command]
fn studio_flux_status() -> FluxProviderStatus {
    let model = std::env::var("TRADUZAI_STUDIO_FLUX_MODEL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_FLUX_MODEL.to_string());
    match configured_flux_command() {
        Ok(spec) => {
            let absolute_program = PathBuf::from(&spec.program);
            if absolute_program.is_absolute() && !absolute_program.exists() {
                return FluxProviderStatus {
                    status: "error".to_string(),
                    provider: "local-adapter".to_string(),
                    model: Some(model),
                    message: Some(format!("Adaptador FLUX nao encontrado: {}", spec.program)),
                };
            }
            FluxProviderStatus {
                status: "configured".to_string(),
                provider: "local-adapter".to_string(),
                model: Some(model),
                message: Some("Adaptador FLUX configurado; dependencias e modelo serao validados ao gerar".to_string()),
            }
        }
        Err(message) => FluxProviderStatus {
            status: "missing".to_string(),
            provider: "local-adapter".to_string(),
            model: Some(model),
            message: Some(message),
        },
    }
}

#[tauri::command]
async fn studio_flux_generate(
    config: FluxGenerateConfig,
    worker: tauri::State<'_, FluxWorkerManager>,
) -> Result<FluxGenerateResult, String> {
    validate_flux_generate_config(&config)?;
    let timeout_seconds = std::env::var("TRADUZAI_STUDIO_FLUX_TIMEOUT_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1800)
        .clamp(30, 7200);
    worker.generate(config, timeout_seconds).await
}

#[tauri::command]
async fn studio_flux_cancel(
    config: FluxCancelConfig,
    worker: tauri::State<'_, FluxWorkerManager>,
) -> Result<bool, String> {
    validate_flux_job_id(&config.job_id)?;
    worker.cancel(config.job_id).await
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
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("Falha ao criar pasta de exportacao: {error}"))?;
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
        return Err(format!(
            "Falha ao consultar Google Fonts: HTTP {}",
            metadata.status()
        ));
    }

    let metadata_text = metadata
        .text()
        .await
        .map_err(|error| format!("Falha ao ler resposta do Google Fonts: {error}"))?;
    let families =
        search_google_fonts_metadata_json(&metadata_text, &query, GOOGLE_FONT_SEARCH_LIMIT)?;
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
        .map_err(|error| format!("Resposta invalida do Google Fonts: {error}"))?;
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
            .header(reqwest::header::USER_AGENT, "TraduzAI Studio")
            .send()
            .await
            .map_err(|error| {
                format!("Falha ao localizar fonte no repositorio Google Fonts: {error}")
            })?;
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
        .map_err(|error| format!("URL de fonte Google invalida: {error}"))?;
    if parsed_url.scheme() != "https" && parsed_url.scheme() != "http" {
        return Err("URL de fonte Google deve usar http ou https".to_string());
    }

    std::fs::create_dir_all(cache_dir)
        .map_err(|error| format!("Falha ao criar cache de fontes Google: {error}"))?;

    let response = reqwest::Client::new()
        .get(parsed_url)
        .send()
        .await
        .map_err(|error| format!("Falha ao baixar fonte Google: {error}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "Falha ao baixar fonte Google: HTTP {}",
            response.status()
        ));
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|error| format!("Falha ao ler fonte Google baixada: {error}"))?;
    if bytes.is_empty() {
        return Err("Fonte Google baixada esta vazia".to_string());
    }

    std::fs::write(&target_path, &bytes)
        .map_err(|error| format!("Falha ao gravar fonte Google em cache: {error}"))?;

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

fn sanitize_generated_asset_id(value: &str) -> Result<String, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.len() > 128 {
        return Err("Id de asset gerado invalido".to_string());
    }
    if !trimmed
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        return Err("Id de asset gerado invalido".to_string());
    }
    Ok(trimmed.to_string())
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
        .manage(FluxWorkerManager::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![
            library::studio_load_library,
            library::studio_save_library,
            studio_load_project,
            studio_save_project,
            studio_save_recovery_snapshot,
            studio_load_recovery_snapshot,
            studio_clear_recovery_snapshot,
            studio_write_bitmap_layer,
            studio_write_generated_asset,
            studio_delete_generated_assets,
            studio_flux_status,
            studio_flux_generate,
            studio_flux_cancel,
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

    async fn run_fake_flux_job(
        sender: &tokio::sync::mpsc::Sender<super::FluxWorkerMessage>,
        config: super::FluxGenerateConfig,
    ) -> Result<super::FluxGenerateResult, String> {
        let (response, result) = tokio::sync::oneshot::channel();
        sender
            .send(super::FluxWorkerMessage::Generate {
                config,
                timeout_seconds: 10,
                response,
            })
            .await
            .expect("fake worker channel should stay open");
        result.await.expect("fake worker should answer")
    }

    #[test]
    fn parses_project_json_with_utf8_bom() {
        let payload = "\u{feff}{\"app\":\"traduzai\",\"paginas\":[]}";
        let value = parse_project_payload(payload).expect("project json should parse with BOM");
        assert_eq!(value["app"], "traduzai");
    }

    #[test]
    fn resolves_any_selected_json_file_as_project_file() {
        let path = super::resolve_project_file("N:\\TraduzAI\\qa\\project-saved.json");
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("project-saved.json")
        );
    }

    #[test]
    fn resolves_directory_path_to_default_project_json() {
        let path = super::resolve_project_file("N:\\TraduzAI\\qa");
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("project.json")
        );
    }

    #[test]
    fn writes_and_loads_recovery_snapshot_without_leaving_partial_files() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("traduzai-studio-recovery-{nonce}"));
        let project_file = root.join("project.json");
        std::fs::create_dir_all(&root).unwrap();
        let snapshot = serde_json::json!({
            "version": "1.0",
            "projectPath": project_file.to_string_lossy(),
            "savedAt": 1234,
            "project": { "app": "traduzai", "paginas": [] }
        });

        super::write_recovery_snapshot(&project_file, &snapshot).unwrap();
        assert_eq!(super::load_latest_recovery_snapshot(&project_file).unwrap(), Some(snapshot));

        let recovery_dir = super::recovery_directory(&project_file).unwrap();
        let names = std::fs::read_dir(&recovery_dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
            .collect::<Vec<_>>();
        assert!(names.iter().any(|name| name.starts_with("snapshot-") && name.ends_with(".json")));
        assert!(!names.iter().any(|name| name.ends_with(".tmp")));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn isolates_recovery_snapshots_for_json_files_in_the_same_directory() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("traduzai-studio-recovery-isolation-{nonce}"));
        std::fs::create_dir_all(&root).unwrap();
        let first_file = root.join("chapter-a.json");
        let second_file = root.join("chapter-b.json");
        let first = serde_json::json!({ "projectPath": first_file.to_string_lossy(), "project": { "app": "traduzai", "paginas": [] } });
        let second = serde_json::json!({ "projectPath": second_file.to_string_lossy(), "project": { "app": "traduzai", "paginas": [] } });

        super::write_recovery_snapshot(&first_file, &first).unwrap();
        super::write_recovery_snapshot(&second_file, &second).unwrap();

        assert_ne!(
            super::recovery_directory(&first_file).unwrap(),
            super::recovery_directory(&second_file).unwrap(),
        );
        assert_eq!(super::load_latest_recovery_snapshot(&first_file).unwrap(), Some(first));
        assert_eq!(super::load_latest_recovery_snapshot(&second_file).unwrap(), Some(second));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn keeps_valid_recovery_available_when_newer_project_json_is_corrupted() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("traduzai-studio-recovery-corruption-{nonce}"));
        let project_file = root.join("project.json");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(&project_file, r#"{"app":"traduzai","paginas":[]}"#).unwrap();
        let snapshot = serde_json::json!({
            "projectPath": project_file.to_string_lossy(),
            "project": { "app": "traduzai", "paginas": [] }
        });
        super::write_recovery_snapshot(&project_file, &snapshot).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&project_file, "{arquivo truncado").unwrap();

        assert_eq!(super::load_latest_recovery_snapshot(&project_file).unwrap(), Some(snapshot));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_recovery_identity_from_another_project() {
        let snapshot = serde_json::json!({ "projectPath": "N:/TraduzAI/chapter-a.json" });
        assert!(super::snapshot_matches_project(&snapshot, "N:\\TraduzAI\\chapter-a.json"));
        assert!(!super::snapshot_matches_project(&snapshot, "N:/TraduzAI/chapter-b.json"));
    }

    #[test]
    fn replaces_project_json_transactionally_without_partial_files() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("traduzai-studio-atomic-{nonce}"));
        let project_file = root.join("project.json");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(&project_file, "{\"version\":1}").unwrap();

        super::write_project_json_atomically(&project_file, "{\"version\":2}").unwrap();

        assert_eq!(std::fs::read_to_string(&project_file).unwrap(), "{\"version\":2}");
        let names = std::fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["project.json"]);

        std::fs::remove_dir_all(root).unwrap();
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
        assert_eq!(
            super::normalize_system_font_query("  Times-New  "),
            "times new"
        );
    }

    #[test]
    fn accepts_only_safe_generated_asset_ids() {
        assert_eq!(
            super::sanitize_generated_asset_id("retouch-123_ok").unwrap(),
            "retouch-123_ok"
        );
        assert!(super::sanitize_generated_asset_id("../retouch").is_err());
        assert!(super::sanitize_generated_asset_id("").is_err());
    }

    fn flux_config(variant_count: usize) -> super::FluxGenerateConfig {
        super::FluxGenerateConfig {
            contract_version: "1.0".to_string(),
            job_id: "flux-job".to_string(),
            prompt: "reconstruir textura".to_string(),
            negative_prompt: "texto".to_string(),
            model: "flux-fill-local".to_string(),
            source_png_data: "data:image/png;base64,c291cmNl".to_string(),
            mask_png_data: "data:image/png;base64,bWFzaw==".to_string(),
            width: 512,
            height: 512,
            variant_count,
            seed: 42,
            steps: 20,
            guidance_scale: 18.0,
        }
    }

    #[test]
    fn validates_flux_generation_bounds_and_local_png_inputs() {
        assert!(super::validate_flux_generate_config(&flux_config(2)).is_ok());
        assert!(super::validate_flux_generate_config(&flux_config(1)).is_err());
        assert!(super::validate_flux_generate_config(&flux_config(5)).is_err());

        let mut invalid = flux_config(2);
        invalid.source_png_data = "https://example.com/source.png".to_string();
        assert!(super::validate_flux_generate_config(&invalid).is_err());
    }

    #[test]
    fn parses_flux_adapter_command_without_shell_expansion() {
        let spec = super::flux_command_spec(
            Some("python".to_string()),
            Some("[\"worker.py\",\"--local\"]".to_string()),
        )
        .expect("command should parse");
        assert_eq!(spec.program, "python");
        assert_eq!(spec.args, vec!["worker.py", "--local"]);
        assert!(super::flux_command_spec(Some("python".to_string()), Some("not-json".to_string())).is_err());
        assert!(super::flux_command_spec(None, None).is_err());
    }

    #[tokio::test]
    async fn keeps_flux_worker_persistent_and_kills_it_on_cancel() {
        let script = r#"import json,sys,time
counter=0
for line in sys.stdin:
    request=json.loads(line)
    if request.get('prompt') == 'wait':
        time.sleep(60)
        continue
    counter += 1
    variants=[{'id':f'variant-{index+1}','seed':request['seed']+index,'png_data':'data:image/png;base64,YQ==','path':None} for index in range(request['variant_count'])]
    print(json.dumps({'contract_version':'1.0','job_id':request['job_id'],'provider':f'fake-{counter}','model':request['model'],'variants':variants}), flush=True)
"#;
        let sender = super::spawn_flux_worker_from_spec(super::FluxCommandSpec {
            program: "python".to_string(),
            args: vec!["-u".to_string(), "-c".to_string(), script.to_string()],
        })
        .await
        .expect("fake persistent worker should start");

        let first = run_fake_flux_job(&sender, flux_config(2)).await.expect("first job should pass");
        let mut second_config = flux_config(2);
        second_config.job_id = "flux-job-2".to_string();
        let second = run_fake_flux_job(&sender, second_config).await.expect("second job should pass");
        assert_eq!(first.provider, "fake-1");
        assert_eq!(second.provider, "fake-2");

        let mut slow_config = flux_config(2);
        slow_config.job_id = "flux-slow".to_string();
        slow_config.prompt = "wait".to_string();
        let (generation_response, generation_result) = tokio::sync::oneshot::channel();
        sender
            .send(super::FluxWorkerMessage::Generate {
                config: slow_config,
                timeout_seconds: 70,
                response: generation_response,
            })
            .await
            .expect("slow job should enter worker");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        let (cancel_response, cancel_result) = tokio::sync::oneshot::channel();
        sender
            .send(super::FluxWorkerMessage::Cancel {
                job_id: "flux-slow".to_string(),
                response: cancel_response,
            })
            .await
            .expect("cancel message should reach worker");
        assert!(cancel_result.await.expect("worker should confirm cancellation"));
        assert!(generation_result
            .await
            .expect("cancelled generation should answer")
            .expect_err("cancelled generation must fail")
            .contains("cancelada"));
    }
}
